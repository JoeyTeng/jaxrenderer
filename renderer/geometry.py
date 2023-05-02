from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import jaxtyped, Array, Float, Integer, Num

from .types import ModelView, Projection, Vec3f, Viewport, World2Screen


@jaxtyped
@partial(jax.jit, donate_argnums=(0, ))
def normalise(vector: Float[Array, "dim"]) -> Float[Array, "dim"]:
    """normalise vector in-place."""
    return vector / jnp.linalg.norm(vector)


@jaxtyped
@jax.jit
def model_view_matrix(
    eye: Vec3f,
    centre: Vec3f,
    up: Vec3f,
) -> ModelView:
    """Compute ModelView matrix as defined by OpenGL / tinyrenderer.

    Same as `lookAt` in OpenGL / tinyrenderer.

    Parameters:
      - eye: the position of camera, in world space
      - centre: the centre of the frame, where the camera points to, in world
        space
      - up: the direction vector with start point at "eye", indicating the "up"
        direction of the camera frame.
    """
    z: Vec3f = normalise(eye - centre)
    x: Vec3f = normalise(jnp.cross(up, z))
    y: Vec3f = normalise(jnp.cross(x, z))

    # M inverse
    m_inv = jnp.identity(4).at[0, :].set(x).at[1, :].set(y).at[2, :].set(z)
    tr = jnp.identity(4).at[:, 3].set(-eye)

    model_view: ModelView = m_inv @ tr

    return model_view


@jaxtyped
@jax.jit
def perspective_projection_matrix(
    eye: Vec3f,
    centre: Vec3f,
    dtype: jnp.dtype = jnp.single,
) -> Projection:
    """Create a projection matrix to map the model in the camera frame (eye
        coordinates) onto the viewing volume (clip coordinates), using
        perspective transformation.

    Parameters:
      - eye: the position of camera, in world space
      - centre: the centre of the frame, where the camera points to, in world
        space
      - dtype: the dtype for the projection matrix.

    Return: Projection, (4, 4) matrix.
    """
    projection: Projection = (jnp.identity(4, dtype=dtype).at[3, 2].set(
        -1 / jnp.linalg.norm(eye - centre)))

    return projection


@jaxtyped
@jax.jit
def viewport_matrix(
    lowerbound: Num[Array, "2"],
    viewport_dimension: Integer[Array, "2"],
    depth: Num[Array, ""],
    dtype: jnp.dtype = jnp.single,
) -> Viewport:
    """Create a viewport matrix to map the model in bi-unit cube ([-1...1]^3)
        onto the screen cube ([x, x+w]*[y, y+h]*[0, d]). The result matrix is
        the viewport matrix as defined in OpenGL / tinyrenderer.

    Parameters:
      - lowerbound: x-y of the lower left corner of the viewport, in screen
        space.
      - viewport_dimension: width, height of the viewport, in screen space.
      - depth: the depth of the viewport in screen space, for zbuffer
      - dtype: the dtype for the viewport matrix.

    Return: Viewport, (4, 4) matrix.
    """
    width, height = viewport_dimension
    viewport: Viewport = (
        jnp.identity(4, dtype=dtype)  #
        .at[:2, 3].set(lowerbound + viewport_dimension / 2)  #
        .at[0, 0].set(width / 2).at[1, 1].set(height / 2)  #
        .at[2, 2:].set(depth / 2)  #
    )

    return viewport


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
def normalise_homogeneous(
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
    return normalise_homogeneous(coordinates)[..., :-1]
