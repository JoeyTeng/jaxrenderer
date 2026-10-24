from functools import partial
import json
import os
from pathlib import Path
from typing import cast

import jax
import jax.lax as lax
import jax.numpy as jnp

from renderer import (
    Buffers,
    Camera,
    CameraParameters,
    LightSource,
    Renderer,
    render,
)
from renderer._meta_utils import typed_array, typed_asarray, typed_grad, typed_zeros
from renderer.shaders.gouraud import GouraudExtraInput, GouraudShader
from renderer.types import FloatV


def _write_numeric_report(section: str, metrics: dict[str, float]) -> None:
    artifact_dir = Path(
        os.environ.get("JAXRENDERER_ARTIFACT_DIR", ".artifacts/render-regression")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "numeric-report.json"
    report: dict[str, object] = (
        cast(dict[str, object], json.loads(report_path.read_text(encoding="utf-8")))
        if report_path.exists()
        else {}
    )
    report[section] = metrics
    _ = report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


eye = typed_array((0.0, 0, 2))
center = typed_array((0.0, 0, 0))
up = typed_array((0.0, 1, 0))

width = 84
height = 84
lowerbound = typed_zeros(2, dtype=int)
dimension = typed_array((width, height))
depth = 1

camera: Camera = Camera.create(
    view=Camera.view_matrix(eye=eye, centre=center, up=up),
    projection=Camera.perspective_projection_matrix(
        fovy=90.0,
        aspect=1.0,
        z_near=-1.0,
        z_far=1.0,
    ),
    viewport=Camera.viewport_matrix(
        lowerbound=lowerbound,
        dimension=dimension,
        depth=depth,
    ),
)

buffers = Buffers(
    zbuffer=lax.full(
        (width, height),
        0.0,
    ),
    targets=(
        lax.full(
            (width, height, 3),
            0.0,
        ),
    ),
)
face_indices = typed_array(
    (
        (0, 1, 2),
        (1, 3, 2),
        (0, 2, 4),
        (0, 4, 3),
        (2, 5, 1),
    )
)
position = typed_array(
    (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1, -1, 1.0),
        (-2, 0.0, 0.0),
    )
)
light = LightSource()
normal = jnp.broadcast_to(
    light.direction,
    (position.shape[0], *light.direction.shape),
)
extra = GouraudExtraInput(
    position=position,
    colour=typed_array(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
        )
    ),
    normal=normal,
    light=light,
)

_render = partial(
    render,
    shader=GouraudShader,
    buffers=buffers,
    face_indices=face_indices,
)


def test_grad_over_camera():
    camera_parameters = CameraParameters(
        viewWidth=width,
        viewHeight=height,
        viewDepth=1.0,
        near=0.1,
        far=10.0,
        hfov=90.0,
        vfov=90.0,
        position=typed_array((0.0, 0.0, 2.0)),
        target=typed_array((0.0, 0.0, 0.0)),
        up=typed_array((0.0, 1.0, 0.0)),
    )
    triangle_extra = GouraudExtraInput(
        position=typed_array(
            (
                (-0.6, -0.6, 0.0),
                (0.7, -0.5, 0.0),
                (0.0, 0.7, 0.0),
            )
        ),
        colour=typed_array(
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        normal=jnp.broadcast_to(light.direction, (3, 3)),
        light=light,
    )
    triangle_faces = typed_array(((0, 1, 2),))

    def camera_depth_window(position_x: FloatV, far: FloatV) -> jax.Array:
        position = typed_array((position_x, 0.0, 2.0))
        camera_from_parameters = Renderer.create_camera_from_parameters(
            camera_parameters._replace(position=position, far=far)
        )
        depth_buffer = render(
            camera=camera_from_parameters,
            extra=triangle_extra,
            buffers=buffers,
            face_indices=triangle_faces,
            shader=GouraudShader,
        )[0]
        return depth_buffer[36:46, 31:41]

    def camera_depth_loss(position_x: FloatV, far: FloatV) -> FloatV:
        return jnp.sum(camera_depth_window(position_x, far))

    position_x = typed_asarray(0.0)
    far = typed_asarray(10.0)
    # CPU float32 gave 0.61% and 0.04% relative FD error at these steps;
    # smaller steps amplified rounding in the depth-buffer reductions.
    position_epsilon = typed_asarray(0.01)
    far_epsilon = typed_asarray(0.3)

    # These five samples must stay fully covered so the oracle excludes raster
    # edge and visibility changes rather than treating background as depth.
    coverage_windows = (
        camera_depth_window(position_x, far),
        camera_depth_window(position_x + position_epsilon, far),
        camera_depth_window(position_x - position_epsilon, far),
        camera_depth_window(position_x, far + far_epsilon),
        camera_depth_window(position_x, far - far_epsilon),
    )
    assert all(bool(jnp.all(window > 0.0)) for window in coverage_windows)

    position_gradient: FloatV = typed_grad(lambda value: camera_depth_loss(value, far))(
        position_x
    )
    position_difference = (
        camera_depth_loss(position_x + position_epsilon, far)
        - camera_depth_loss(position_x - position_epsilon, far)
    ) / (2 * position_epsilon)

    far_gradient: FloatV = typed_grad(
        lambda value: camera_depth_loss(position_x, value)
    )(far)
    far_difference = (
        camera_depth_loss(position_x, far + far_epsilon)
        - camera_depth_loss(position_x, far - far_epsilon)
    ) / (2 * far_epsilon)

    _ = position_gradient.block_until_ready()
    _ = position_difference.block_until_ready()
    _ = far_gradient.block_until_ready()
    _ = far_difference.block_until_ready()

    def record(
        gradient: FloatV, difference: FloatV, epsilon: FloatV
    ) -> dict[str, float]:
        gradient_value = float(gradient)
        difference_value = float(difference)
        absolute_error = abs(gradient_value - difference_value)
        return {
            "autodiff": gradient_value,
            "central_difference": difference_value,
            "absolute_error": absolute_error,
            "relative_error": absolute_error / max(abs(difference_value), 1e-12),
            "epsilon": float(epsilon),
        }

    position_metrics = record(position_gradient, position_difference, position_epsilon)
    far_metrics = record(far_gradient, far_difference, far_epsilon)
    _write_numeric_report(
        "camera_gradient",
        {
            **{f"position_x_{key}": value for key, value in position_metrics.items()},
            **{f"far_{key}": value for key, value in far_metrics.items()},
        },
    )

    assert bool(jnp.isfinite(position_gradient))
    assert bool(jnp.isfinite(position_difference))
    assert abs(float(position_gradient)) > 1e-3
    # A 3% bound leaves room above the measured float32 error while rejecting
    # regressions that zero or materially distort either sensitivity.
    assert jnp.allclose(position_gradient, position_difference, rtol=0.03, atol=1e-4)
    assert bool(jnp.isfinite(far_gradient))
    assert bool(jnp.isfinite(far_difference))
    assert abs(float(far_gradient)) > 1e-3
    assert jnp.allclose(far_gradient, far_difference, rtol=0.03, atol=1e-4)


def test_grad_over_light():
    def _render_light(light: LightSource) -> FloatV:
        _, (canvas,) = _render(
            camera=camera,
            extra=extra._replace(light=light),
        )

        return canvas.sum()

    def loss_for_x(x: FloatV) -> FloatV:
        light = LightSource(
            direction=typed_array((x, 0.2, -1.0)),
        )
        return _render_light(light)

    x = typed_asarray(0.3)
    epsilon = 0.01
    gradient: FloatV = typed_grad(loss_for_x)(x)
    finite_difference = (loss_for_x(x + epsilon) - loss_for_x(x - epsilon)) / (
        2 * epsilon
    )
    _ = gradient.block_until_ready()
    _ = finite_difference.block_until_ready()

    gradient_value = float(gradient)
    finite_difference_value = float(finite_difference)
    absolute_error = abs(gradient_value - finite_difference_value)
    relative_error = absolute_error / max(abs(finite_difference_value), 1e-12)
    _write_numeric_report(
        "light_gradient",
        {
            "autodiff": gradient_value,
            "central_difference": finite_difference_value,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "epsilon": float(epsilon),
        },
    )

    assert bool(jnp.isfinite(gradient))
    assert bool(jnp.isfinite(finite_difference))
    assert float(gradient) < -1e-3
    assert jnp.allclose(gradient, finite_difference, rtol=0.01, atol=0.01)
