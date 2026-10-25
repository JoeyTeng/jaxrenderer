import jax.numpy as jnp

from renderer import CameraParameters, LightParameters, Renderer
from renderer._meta_utils import typed_array


def test_empty_scene_returns_configured_background_canvas() -> None:
    background = typed_array((0.2, 0.4, 0.8), dtype=jnp.float16)

    image = Renderer.get_camera_image(
        objects=(),
        light=LightParameters(),
        camera=CameraParameters(),
        width=4,
        height=3,
        colour_default=background,
    )

    assert image.shape == (4, 3, 3)
    assert image.dtype == background.dtype
    assert jnp.array_equal(image, jnp.broadcast_to(background, image.shape))
