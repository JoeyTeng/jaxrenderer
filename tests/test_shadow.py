import jax.numpy as jnp
import numpy as np

from renderer import Camera, jit
from renderer._meta_utils import typed_asarray
from renderer.shadow import Shadow
from renderer.types import FloatV, ZBuffer


def test_shadow_get_rounding_fill_and_negative_index_wrapping() -> None:
    shadow_map: ZBuffer = typed_asarray(((0.0, 0.1, 0.2), (0.3, 0.4, 0.5)))
    camera = Camera.create(
        view=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
        projection=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
        viewport=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
    )
    shadow = Shadow(
        shadow_map=shadow_map,
        strength=typed_asarray((0.5, 0.5, 0.5)),
        camera=camera,
    )
    positions = (
        typed_asarray((0.6, 1.6)),
        typed_asarray((2.0, 1.0)),
        typed_asarray((-1.0, 0.0)),
    )

    eager_values = tuple(shadow.get(position) for position in positions)
    compiled_get = jit(shadow.get)
    compiled_values: tuple[FloatV, ...] = tuple(
        compiled_get(position) for position in positions
    )

    expected = (0.5, np.inf, 0.3)
    np.testing.assert_allclose(np.asarray(eager_values), expected)
    np.testing.assert_allclose(np.asarray(compiled_values), expected)
