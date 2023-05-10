import enum
from functools import partial
from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, Num, jaxtyped

from .types import Triangle2Df, Vec2f, Vec3f

# Transform matrix that takes a batch of homogeneous 3D vertices and transform
# them into 2D cartesian vertices in screen space + Z value (making it 3D)
#
# The result of x-y values in screen space may be float, and thus further
# conversion to integers are needed.
World2Screen = Float[Array, "4 4"]
# Transform all coordinates from model space to view space, with camera at
# origin. (Object Coordinates -> Eye Coordinates)
ModelView = Float[Array, "4 4"]
# Transform all coordinates from view space to viewing volume.
# (Eye Coordinates -> Clip Coordinates)
Projection = Float[Array, "4 4"]
# Transform all coordinates from model space in a bi-unit cube ([-1...1]^3) to
# a screen cube ([x, x+width] * [y, y+height] * [0, depth]) in view space.
# (Normalised Device Coordinates -> Window Coordinates)
Viewport = Float[Array, "4 4"]


@jaxtyped
@partial(jax.jit, donate_argnums=(0, ))
def normalise(vector: Float[Array, "dim"]) -> Float[Array, "dim"]:
    """normalise vector in-place."""
    return vector / jnp.linalg.norm(vector)


class Interpolation(enum.Enum):
    """Interpolation methods for rasterisation.

    References:
      - [Interpolation qualifiers](https://www.khronos.org/opengl/wiki/Type_Qualifier_(GLSL)#Interpolation_qualifiers)
    """
    # Flat shading: use the value of the first vertex of the primitive
    FLAT = 0
    # No perspective correction: linear interpolation in screen space
    NOPERSPECTIVE = 1
    # Perspective correction: linear interpolation in clip space
    SMOOTH = 2

    @jaxtyped
    @partial(jax.jit, static_argnames=("self", ))
    def __call__(
        self,
        barycentric_screen: Vec3f,
        barycentric_clip: Vec3f,
        values: Num[Array, "3 *valueDimensions"],
    ) -> Num[Array, "*valueDimensions"]:
        """Interpolation, using barycentric coordinates.

        Parameters:
          - barycentric_screen: barycentric coordinates in screen space of the
            point to interpolate
          - barycentric_clip: barycentric coordinates in clip space of the
            point to interpolate
          - values: values at the vertices of the triangle, with axis 0 being
            the batch axis.
        """
        dtype = jax.dtypes.result_type(
            barycentric_screen,
            barycentric_clip,
            values,
        )
        coef: Vec3f
        # branches are ok because `self` is static: decided at compile time
        if self is Interpolation.FLAT:
            with jax.ensure_compile_time_eval():
                coef = jnp.array([1, 0, 0], dtype=dtype)
        elif self is Interpolation.NOPERSPECTIVE:
            coef = barycentric_screen
        elif self is Interpolation.SMOOTH:
            coef = barycentric_clip
        else:
            raise ValueError(f"Unknown interpolation method {self}")

        interpolated = lax.dot_general(
            coef.astype(dtype),
            values.astype(dtype),
            (((0, ), (0, )), ([], [])),
        )

        return interpolated


class Camera(NamedTuple):
    model_view: ModelView
    viewport: Viewport
    projection: Projection

    @classmethod
    @jaxtyped
    @partial(jax.jit, static_argnames=("cls", ))
    def create(
        cls,
        eye: Vec3f,
        centre: Vec3f,
        up: Vec3f,
        lowerbound: Num[Array, "2"],
        viewport_dimension: Integer[Array, "2"],
        depth: Num[Array, ""],
    ) -> "Camera":
        """Create a camera with the given parameters.

        Parameters:
          - eye: the position of camera, in world space
          - centre: the centre of the frame, where the camera points to, in
            world space
          - up: the direction vector with start point at "eye", indicating the
            "up" direction of the camera frame.
          - lowerbound: the lowerbound of the viewport, in screen space
          - viewport_dimension: the dimension of the viewport, in screen space
          - depth: the depth of the viewport, in screen space
        """
        model_view: ModelView = cls.model_view_matrix(eye, centre, up)
        projection: Projection = cls.perspective_projection_matrix(eye, centre)
        viewport: Viewport = cls.viewport_matrix(
            lowerbound,
            viewport_dimension,
            depth,
        )
        assert isinstance(model_view, ModelView)
        assert isinstance(projection, Projection)
        assert isinstance(viewport, Viewport)

        return cls(
            model_view=model_view,
            viewport=viewport,
            projection=projection,
        )

    @staticmethod
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
          - centre: the centre of the frame, where the camera points to, in
            world space
          - up: the direction vector with start point at "eye", indicating the
            "up" direction of the camera frame.
        """
        z: Vec3f = normalise(eye - centre)
        x: Vec3f = normalise(jnp.cross(up, z))
        y: Vec3f = normalise(jnp.cross(x, z))

        # M inverse
        m_inv = (
            jnp.identity(4).at[0, :3].set(x).at[1, :3].set(y).at[2, :3].set(z))
        tr = jnp.identity(4).at[:3, 3].set(-eye)

        model_view: ModelView = m_inv @ tr

        return model_view

    @staticmethod
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
          - centre: the centre of the frame, where the camera points to, in
            world space
          - dtype: the dtype for the projection matrix.

        Return: Projection, (4, 4) matrix.
        """
        projection: Projection = (
            jnp.identity(4, dtype=dtype)  #
            .at[3, 2].set(-1 / jnp.linalg.norm(eye - centre))  #
        )

        return projection

    @staticmethod
    @jaxtyped
    @jax.jit
    def viewport_matrix(
        lowerbound: Num[Array, "2"],
        viewport_dimension: Integer[Array, "2"],
        depth: Num[Array, ""],
        dtype: jnp.dtype = jnp.single,
    ) -> Viewport:
        """Create a viewport matrix to map the model in bi-unit cube
            ([-1...1]^3) onto the screen cube ([x, x+w]*[y, y+h]*[0, d]). The
            result matrix is the viewport matrix as defined in OpenGL /
            tinyrenderer.

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

    @staticmethod
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


@jaxtyped
@jax.jit
def barycentric(pts: Triangle2Df, p: Vec2f) -> Vec3f:
    """Compute the barycentric coordinate of `p`.
        Returns u[-1] < 0 if `p` is outside of the triangle.
    """
    mat: Float[Array, "3 2"] = jnp.vstack((
        pts[2] - pts[0],
        pts[1] - pts[0],
        pts[0] - p,
    ))
    vec: Vec3f = jnp.cross(mat[:, 0], mat[:, 1])
    # `u[2]` is 0, that means triangle is degenerate, in this case
    # return something with negative coordinates
    vec = lax.cond(
        jnp.abs(vec[-1]) < 1e-10,
        lambda: jnp.array((-1., 1., 1.)),
        lambda: vec,
    )
    vec = vec / vec[-1]
    vec = jnp.array((1 - (vec[0] + vec[1]), vec[1], vec[0]))

    return vec
