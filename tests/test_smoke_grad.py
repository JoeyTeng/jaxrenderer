# pyright: basic
from functools import partial
import json
import os
from pathlib import Path
from typing import Dict, cast

import jax
import jax.lax as lax
import jax.numpy as jnp

from renderer import Buffers, Camera, LightSource, render
from renderer.shaders.gouraud import GouraudExtraInput, GouraudShader
from renderer.types import FloatV


def _write_gradient_report(metrics: Dict[str, float]) -> None:
    artifact_dir = Path(
        os.environ.get("JAXRENDERER_ARTIFACT_DIR", ".artifacts/render-regression")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "numeric-report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    report["light_gradient"] = metrics
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


eye = jnp.array((0.0, 0, 2))  # pyright: ignore[reportUnknownMemberType]
center = jnp.array((0.0, 0, 0))  # pyright: ignore[reportUnknownMemberType]
up = jnp.array((0.0, 1, 0))  # pyright: ignore[reportUnknownMemberType]

width = 84
height = 84
lowerbound = jnp.zeros(2, dtype=int)  # pyright: ignore[reportUnknownMemberType]
dimension = jnp.array((width, height))  # pyright: ignore[reportUnknownMemberType]
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
    zbuffer=lax.full(  # pyright: ignore[reportUnknownMemberType]
        (width, height),
        0.0,
    ),
    targets=(
        lax.full(  # pyright: ignore[reportUnknownMemberType]
            (width, height, 3),
            0.0,
        ),
    ),
)
face_indices = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    (
        (0, 1, 2),
        (1, 3, 2),
        (0, 2, 4),
        (0, 4, 3),
        (2, 5, 1),
    )
)
position = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1, -1, 1.0),
        (-2, 0.0, 0.0),
    )
)
extra = GouraudExtraInput(
    position=position,
    colour=jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
        )
    ),
    normal=jax.vmap(lambda _: LightSource().direction)(position),  # pyright: ignore
    light=LightSource(),
)

_render = partial(
    render,
    shader=GouraudShader,
    buffers=buffers,
    face_indices=face_indices,
)


def test_grad_over_camera():
    def camera_depth_loss(a: Camera) -> FloatV:
        depth = _render(camera=a, extra=extra)[0]

        return jnp.sum(depth)  # pyright: ignore[reportUnknownMemberType]

    grad_camera = cast(
        FloatV,
        jax.jit(  # pyright: ignore[reportUnknownMemberType]
            jax.grad(camera_depth_loss)  # pyright: ignore[reportUnknownMemberType]
        )(camera),
    )

    jax.tree_util.tree_map(lambda a: a.block_until_ready(), grad_camera)
    assert jax.tree_util.tree_leaves(grad_camera)


def test_grad_over_light():
    def _render_light(light: LightSource) -> FloatV:
        _, (canvas,) = _render(
            camera=camera,
            extra=extra._replace(light=light),
        )

        return canvas.sum()  # pyright: ignore[reportUnknownMemberType]

    def loss_for_x(x: FloatV) -> FloatV:
        light = LightSource(direction=jnp.array((x, 0.2, -1.0)))
        return _render_light(light)

    x = jnp.asarray(0.3)
    epsilon = 0.01
    gradient = jax.grad(loss_for_x)(x)
    finite_difference = (loss_for_x(x + epsilon) - loss_for_x(x - epsilon)) / (
        2 * epsilon
    )
    gradient.block_until_ready()
    finite_difference.block_until_ready()

    gradient_value = float(gradient)
    finite_difference_value = float(finite_difference)
    absolute_error = abs(gradient_value - finite_difference_value)
    relative_error = absolute_error / max(abs(finite_difference_value), 1e-12)
    _write_gradient_report(
        {
            "autodiff": gradient_value,
            "central_difference": finite_difference_value,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "epsilon": float(epsilon),
        }
    )

    assert bool(jnp.isfinite(gradient))
    assert bool(jnp.isfinite(finite_difference))
    assert float(gradient) < -1e-3
    assert jnp.allclose(gradient, finite_difference, rtol=0.01, atol=0.01)
