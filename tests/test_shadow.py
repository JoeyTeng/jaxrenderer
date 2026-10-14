from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from renderer import Camera
from renderer.shadow import Shadow
from renderer.types import FloatV, Vec2f, ZBuffer


def test_shadow_get_rounding_fill_and_negative_index_wrapping() -> None:
    shadow_map: ZBuffer = jnp.asarray(((0.0, 0.1, 0.2), (0.3, 0.4, 0.5)))  # pyright: ignore[reportUnknownMemberType]
    camera = Camera.create(
        view=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
        projection=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
        viewport=jnp.eye(4),  # pyright: ignore[reportUnknownMemberType]
    )
    shadow = Shadow(
        shadow_map=shadow_map,
        strength=jnp.asarray((0.5, 0.5, 0.5)),  # pyright: ignore[reportUnknownMemberType]
        camera=camera,
    )
    positions = (
        jnp.asarray((0.6, 1.6)),  # pyright: ignore[reportUnknownMemberType]
        jnp.asarray((2.0, 1.0)),  # pyright: ignore[reportUnknownMemberType]
        jnp.asarray((-1.0, 0.0)),  # pyright: ignore[reportUnknownMemberType]
    )

    eager_values = tuple(shadow.get(position) for position in positions)
    compiled_get = cast(
        Callable[[Vec2f], FloatV],
        jax.jit(shadow.get),  # pyright: ignore[reportUnknownMemberType]
    )
    compiled_values: tuple[FloatV, ...] = tuple(
        compiled_get(position) for position in positions
    )

    expected = (0.5, np.inf, 0.3)
    np.testing.assert_allclose(np.asarray(eager_values), expected)
    np.testing.assert_allclose(np.asarray(compiled_values), expected)
