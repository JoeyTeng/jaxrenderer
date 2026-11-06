"""High-level input records preserve rendering, inheritance and buffer ownership."""

from dataclasses import replace
from typing import override

from jax import Array
import jax.numpy as jnp
import numpy as np
import pytest

from renderer import (
    Buffers,
    CameraImageInput,
    CameraParameters,
    LightParameters,
    Model,
    ModelObject,
    RenderInput,
    Renderer,
    merge_objects,
)
from renderer._meta_utils import (
    typed_array,
    typed_grad,
    typed_jit,
    typed_ones,
    typed_zeros,
)
from renderer.types import Canvas


def _image_input() -> CameraImageInput:
    faces = typed_array(((0, 1, 2),), dtype=jnp.int32)
    model = Model(
        verts=typed_array(((-0.8, -0.8, 0.0), (0.8, -0.8, 0.0), (0.0, 0.8, 0.0))),
        norms=jnp.broadcast_to(typed_array((0.0, 0.0, 1.0)), (3, 3)),
        uvs=typed_array(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))),
        faces=faces,
        faces_norm=faces,
        faces_uv=faces,
        diffuse_map=typed_array((((0.8, 0.4, 0.2),),)),
        specular_map=typed_array(((2.0,),)),
    )
    return CameraImageInput(
        objects=(ModelObject(model),),
        light=LightParameters(
            ambient=typed_ones(3),
            diffuse=typed_zeros(3),
            specular=typed_zeros(3),
        ),
        camera=CameraParameters(
            viewWidth=16,
            viewHeight=16,
            position=(0.0, 0.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            hfov=90.0,
            vfov=90.0,
        ),
        width=16,
        height=16,
    )


@pytest.mark.parametrize("outer_jit", (False, True))
def test_render_reuses_input_data_with_separately_donated_buffers(
    outer_jit: bool,
) -> None:
    image_inputs = _image_input()
    assert isinstance(image_inputs.camera, CameraParameters)
    inputs = RenderInput(
        model=merge_objects(image_inputs.objects),
        light=image_inputs.light,
        camera=Renderer.create_camera_from_parameters(image_inputs.camera),
    )
    vertices = np.asarray(inputs.model.verts).copy()
    camera_view = np.asarray(inputs.camera.view).copy()
    ambient = np.asarray(inputs.light.ambient).copy()
    render = (
        typed_jit(Renderer.render, donate_argnames=("buffers",))
        if outer_jit
        else Renderer.render
    )

    buffers = render(inputs, buffers=Renderer.create_buffers(16, 16))
    first_pixel = np.asarray(buffers.targets[0][8, 8]).copy()
    buffers = render(
        replace(
            inputs, light=inputs.light._replace(ambient=inputs.light.ambient * 0.5)
        ),
        buffers=buffers,
    )

    np.testing.assert_allclose(first_pixel, (0.8, 0.4, 0.2), atol=1e-6)
    np.testing.assert_allclose(buffers.targets[0][8, 8], first_pixel * 0.5, atol=1e-6)
    np.testing.assert_array_equal(inputs.model.verts, vertices)
    np.testing.assert_array_equal(inputs.camera.view, camera_view)
    np.testing.assert_array_equal(inputs.light.ambient, ambient)


class _AmbientRenderer(Renderer):
    @classmethod
    @override
    def render(
        cls, inputs: RenderInput, buffers: Buffers[tuple[Canvas]]
    ) -> Buffers[tuple[Canvas]]:
        return super().render(
            replace(inputs, light=inputs.light._replace(ambient=typed_ones(3))),
            buffers,
        )


def test_camera_image_dispatches_to_override_with_replaced_inputs() -> None:
    inputs = _image_input()
    inputs = replace(inputs, light=inputs.light._replace(ambient=typed_zeros(3)))

    dark = Renderer.get_camera_image(inputs)
    lit = _AmbientRenderer.get_camera_image(inputs)

    np.testing.assert_allclose(dark[8, 8], 0.0, atol=1e-6)
    np.testing.assert_allclose(lit[8, 8], (0.8, 0.4, 0.2), atol=1e-6)


def test_camera_image_inputs_support_gradients_and_distinct_framebuffer_size() -> None:
    inputs = _image_input()
    inputs = replace(inputs, width=20, height=18, loop_unroll=2)

    def loss(ambient: Array) -> Array:
        image = Renderer.get_camera_image(
            replace(inputs, light=inputs.light._replace(ambient=ambient))
        )
        return jnp.sum(image)

    image = Renderer.get_camera_image(inputs)
    gradient = typed_grad(loss)(inputs.light.ambient)

    assert image.shape == (20, 18, 3)
    np.testing.assert_allclose(image[-1, -1], 1.0)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(gradient > 0.0)
