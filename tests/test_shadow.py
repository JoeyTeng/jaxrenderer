import jax.numpy as jnp
import numpy as np

from renderer import Camera, jit
from renderer._meta_utils import typed_asarray, typed_full
from renderer.geometry import Viewport
from renderer.shadow import Shadow
from renderer.types import FaceIndices, FloatV, Vec3f, Vertices, ZBuffer


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


def _render_small_shadow(light_direction: Vec3f, up: Vec3f) -> Shadow:
    width = 8
    height = 8
    vertices: Vertices = typed_asarray(
        ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)),
        dtype=jnp.float32,
    )
    faces: FaceIndices = typed_asarray(((0, 1, 2),), dtype=int)
    viewport: Viewport = Camera.viewport_matrix(
        lowerbound=typed_asarray((0, 0), dtype=int),
        dimension=typed_asarray((width, height), dtype=int),
        depth=1.0,
    )
    shadow_map: ZBuffer = typed_full((width, height), 1.0, dtype=jnp.float32)
    return Shadow.render_shadow_map(
        shadow_map=shadow_map,
        verts=vertices,
        faces=faces,
        light_direction=light_direction,
        viewport_matrix=viewport,
        centre=typed_asarray((0.0, 0.0, 0.0), dtype=jnp.float32),
        up=up,
        strength=typed_asarray((0.5, 0.5, 0.5), dtype=jnp.float32),
        distance=0.5,
    )


def test_shadow_camera_handles_parallel_and_nearly_parallel_up() -> None:
    up: Vec3f = typed_asarray((0.0, 0.0, 1.0), dtype=jnp.float32)
    light_directions = (
        typed_asarray((0.0, 0.0, -1.0), dtype=jnp.float32),
        typed_asarray((0.0, 0.0, 1.0), dtype=jnp.float32),
        typed_asarray((1e-6, 0.0, -1.0), dtype=jnp.float32),
    )

    shadows = tuple(
        _render_small_shadow(light_direction, up)
        for light_direction in light_directions
    )

    for shadow in shadows:
        view = np.asarray(shadow.camera.view)
        depth = np.asarray(shadow.shadow_map)
        assert np.isfinite(view).all()
        np.testing.assert_allclose(np.linalg.det(view), 1.0, atol=1e-5)
        assert np.isfinite(depth).all()
        assert np.any(depth < 1.0)

    parallel_depth = np.min(np.asarray(shadows[0].shadow_map))
    near_parallel_depth = np.min(np.asarray(shadows[2].shadow_map))
    np.testing.assert_allclose(parallel_depth, near_parallel_depth, atol=1e-5)


def test_shadow_camera_preserves_nonparallel_up_direction() -> None:
    light_direction: Vec3f = typed_asarray((0.0, -1.0, 0.0), dtype=jnp.float32)
    up: Vec3f = typed_asarray((0.0, 0.0, 1.0), dtype=jnp.float32)

    shadow = _render_small_shadow(light_direction, up)
    expected_view = Camera.view_matrix(
        eye=typed_asarray((0.0, -0.5, 0.0), dtype=jnp.float32),
        centre=typed_asarray((0.0, 0.0, 0.0), dtype=jnp.float32),
        up=up,
    )

    np.testing.assert_allclose(shadow.camera.view, expected_view)
