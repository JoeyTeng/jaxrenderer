import jax.numpy as jnp

from renderer import (
    CameraParameters,
    LightParameters,
    Model,
    ModelObject,
    Renderer,
)
from renderer._meta_utils import typed_array, typed_full, typed_ones, typed_zeros


def _textured_triangle(
    colour: tuple[float, float, float],
    faces_uv: tuple[int, int, int],
) -> Model:
    verts = typed_array(((-0.5, -0.4, 0.0), (0.5, -0.4, 0.0), (0.0, 0.5, 0.0)))
    faces = typed_array(((0, 1, 2),), dtype=jnp.int32)
    uvs = typed_array(((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (1.0, 1.0)))
    diffuse_map = jnp.broadcast_to(typed_array(colour), (2, 2, 3))
    return Model(
        verts=verts,
        norms=jnp.broadcast_to(typed_array((0.0, 0.0, 1.0)), (3, 3)),
        uvs=uvs,
        faces=faces,
        faces_norm=faces,
        faces_uv=typed_array((faces_uv,), dtype=jnp.int32),
        diffuse_map=diffuse_map,
        specular_map=typed_full((2, 2), 2.0),
    )


def test_merged_objects_keep_texture_ownership_with_independent_uv_indices() -> None:
    # The first mesh has four UVs for three vertices; its first corner uses UV 3.
    first = ModelObject(
        _textured_triangle((1.0, 0.0, 0.0), (3, 1, 2))
    ).replace_with_position(typed_array((-0.7, 0.0, 0.0)))
    second = ModelObject(
        _textured_triangle((0.0, 1.0, 0.0), (0, 1, 2))
    ).replace_with_position(typed_array((0.7, 0.0, 0.0)))
    image = Renderer.get_camera_image(
        objects=(first, second),
        light=LightParameters(
            ambient=typed_ones(3),
            diffuse=typed_zeros(3),
            specular=typed_zeros(3),
        ),
        camera=CameraParameters(
            viewWidth=48,
            viewHeight=32,
            position=(0.0, 0.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            hfov=90.0,
            vfov=90.0,
        ),
        width=48,
        height=32,
    )

    # Restrict the checks to rendered pixels in each object's half of the image.
    left = image[:24]
    right = image[24:]
    left_pixels = left[jnp.any(left < 0.99, axis=-1)]
    right_pixels = right[jnp.any(right < 0.99, axis=-1)]
    assert left_pixels.size > 0
    assert right_pixels.size > 0
    assert jnp.all(left_pixels[:, 0] > left_pixels[:, 1])
    assert jnp.all(right_pixels[:, 1] > right_pixels[:, 0])
