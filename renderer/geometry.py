import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import jaxtyped, Array, Float

from .types import World2Screen


@jaxtyped
@jax.jit
def world_to_screen_matrix(width: int, height: int) -> World2Screen:
    """Generate the projection matrix to convert model coordinates to
        screen/canvas coordinates.

    It assumes all model coordinates are in [-1...1] and will transform them
    into ([0...width], [0...height], [-1...1]).

    Return: World2Screen (Float[Array, "4 4"])
    """
    world2screen: World2Screen = (
        # 3. div by half to centering
        jnp.identity(4).at[0, 0].set(.5).at[1, 1].set(.5)
        # 2. mul by width, height
        @ jnp.identity(4).at[0, 0].set(width).at[1, 1].set(height)
        # 1. Add by 1 to make values positive
        @ jnp.identity(4).at[:2, -1].set(1))

    return world2screen


@jaxtyped
@jax.jit
def compute_normal(triangle_verts: Float[Array, "3 3"]) -> Float[Array, "3"]:
    normal: Float[Array, "3"] = jnp.cross(
        triangle_verts[2, :] - triangle_verts[0, :],
        triangle_verts[1, :] - triangle_verts[0, :],
    )
    normal = normal / jnp.linalg.norm(normal, keepdims=True)
    assert isinstance(normal, Float[Array, "3"])

    return normal


@jaxtyped
@jax.jit
def compute_normals(batch_verts: Float[Array, "b 3 3"]) -> Float[Array, "b 3"]:
    return jax.vmap(compute_normal)(batch_verts)


@jaxtyped
@jax.jit
def to_homogeneous(
    coordinates: Float[Array, "*batch dim"], ) -> Float[Array, "*batch dim+1"]:
    """Transform the coordinates to homogeneous coordinates by append a batch
        of 1s in the last axis."""
    ones: Float[Array, "*batch 1"] = jnp.ones(
        (*coordinates.shape[:-1], 1),
        dtype=jax.dtypes.result_type(coordinates),
    )
    homo_coords: Float[Array, "*batch dim+1"] = lax.concatenate(
        (coordinates, ones),
        len(coordinates.shape) - 1,
    )

    return homo_coords


@jaxtyped
@jax.jit
def scale_homogeneous(
    coordinates: Float[Array, "*batch dim"], ) -> Float[Array, "*batch dim"]:
    """Transform the homogenous coordinates to make the scale factor equals to
        either 1 or 0, by divide every element with the last element on the
        last axis.

    Noted that when a coordinate is 0 and divides by 0, it will produce a nan;
    for non-zero elements divides by 0, a inf will be produced.
    """
    return coordinates / coordinates[..., -1]


@jaxtyped
@jax.jit
def to_cartesian(
    coordinates: Float[Array, "*batch dim"], ) -> Float[Array, "*batch dim-1"]:
    """Transform the homogenous coordinates to cartesian coordinates by divide
        every element with the last element on the last axis, then drop them.

    Noted that when a coordinate is 0 and divides by 0, it will produce a nan;
    for non-zero elements divides by 0, a inf will be produced.
    """
    return scale_homogeneous(coordinates)[..., :-1]
