import enum
from functools import partial
from typing import Any, NamedTuple

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
        values: Num[Array, "3 *valueDimensions"],
        barycentric_screen: Vec3f,
        barycentric_clip: Vec3f,
    ) -> Num[Array, "*valueDimensions"]:
        """Interpolation, using barycentric coordinates.

        Parameters:
          - values: values at the vertices of the triangle, with axis 0 being
            the batch axis.
          - barycentric_screen: barycentric coordinates in screen space of the
            point to interpolate
          - barycentric_clip: barycentric coordinates in clip space of the
            point to interpolate
        """
        dtype = jax.dtypes.result_type(
            barycentric_screen,
            barycentric_clip,
            values,
        )
        coef: Vec3f
        # branches are ok because `self` is static: decided at compile time
        if self == Interpolation.FLAT:
            with jax.ensure_compile_time_eval():
                coef = jnp.array([1, 0, 0], dtype=dtype)
        elif self == Interpolation.NOPERSPECTIVE:
            coef = barycentric_screen
        elif self == Interpolation.SMOOTH:
            coef = barycentric_clip
        else:
            raise ValueError(f"Unknown interpolation method {self}")

        interpolated = lax.dot_general(
            coef.astype(dtype),
            values.astype(dtype),
            (((0, ), (0, )), ([], [])),
        )

        return interpolated


@jaxtyped
@partial(jax.jit, static_argnames=("mode", ))
def interpolate(
    values: Num[Array, "3 *valueDimensions"],
    barycentric_screen: Vec3f,
    barycentric_clip: Vec3f,
    mode: Interpolation = Interpolation.SMOOTH,
) -> Num[Array, "*valueDimensions"]:
    """Convenient wrapper, see `Interpolation.__call__`.

    Default mode is `Interpolation.SMOOTH`.
    """
    interpolated: Num[Array, "*valueDimensions"]
    interpolated = mode(barycentric_screen, barycentric_clip, values)
    assert isinstance(interpolated, Num[Array, "*valueDimensions"])

    return interpolated


class Camera(NamedTuple):
    """Camera parameters.

    - model_view: transform from model space to view space
    - projection: transform from view space to clip space
    - viewport: transform from NDC (normalised device coordinate) space to
      screen space. Noticed that this is NDC space in OpenGL, which has range
      [-1, 1]^3.
    - world_to_clip: transform from model space to clip space
    - world_to_screen: transform from model space to screen space
    """
    model_view: ModelView
    projection: Projection
    viewport: Viewport
    world_to_clip: Projection
    world_to_screen: World2Screen

    @classmethod
    @jaxtyped
    @partial(jax.jit, static_argnames=("cls", ))
    def create(
        cls,
        model_view: ModelView,
        projection: Projection,
        viewport: Viewport,
    ) -> "Camera":
        """Create a camera with the given parameters.

        Parameters:
          - model_view: transform from model space to view space
          - projection: transform from view space to clip space
          - viewport: transform from NDC (normalised device coordinate) space to
        """
        return cls(
            model_view=model_view,
            viewport=viewport,
            projection=projection,
            world_to_clip=projection @ model_view,
            world_to_screen=viewport @ projection @ model_view,
        )

    @staticmethod
    @jaxtyped
    @jax.jit
    def apply(
        points: Num[Array, "*N 4"],
        matrix: Num[Array, "4 4"],
    ) -> Num[Array, "*N 4"]:
        """Transform points from model space to screen space.

        Parameters:
          - points: shape (4, ) or (N, 4). points in model space, with axis 0
            being the batch axis. Batch axis can be omitted. Points must be
            in homogeneous coordinate.
          - matrix: shape (4, 4) transformation matrix

        Returns: coordinates transformed
        """
        assert jnp.ndim(points) < 3
        assert (((jnp.ndim(points) == 2) and (points.shape[1] == 4))
                or ((jnp.ndim(points) == 1) and (points.shape[0] == 4)))

        with jax.ensure_compile_time_eval():
            lhs_contract_axis = 1 if jnp.ndim(points) == 2 else 0
            dtype = jax.dtypes.result_type(points, matrix)

        # put `points` at lhs to keep batch axis at axis 0 in the result.
        transformed: Num[Array, "*N 4"] = lax.dot_general(
            points.astype(dtype),
            matrix.astype(dtype),
            (((lhs_contract_axis, ), (1, )), ([], [])),
        )
        assert isinstance(transformed, Num[Array, "*N 4"])

        return transformed

    @jaxtyped
    @jax.jit
    def to_screen(
        self,
        points: Num[Array, "*N 4"],
    ) -> Num[Array, "*N 4"]:
        """Transform points from model space to screen space.

        Parameters:
          - points: shape (4, ) or (N, 4). points in model space, with axis 0
            being the batch axis. Batch axis can be omitted. Points must be
            in homogeneous coordinate.

        Returns: points in screen space, with axis 0 being the batch axis, if
            given in batch. The dtype may be promoted. The homogeneous
            coordinates are normalised.
        """
        screen_space = self.apply(points, self.world_to_screen)
        assert isinstance(screen_space, Num[Array, "*N 4"])

        normalised = normalise_homogeneous(screen_space)
        assert isinstance(normalised, Num[Array, "*N 4"])

        return normalised

    @jaxtyped
    @jax.jit
    def to_clip(
        self,
        points: Num[Array, "*N 4"],
    ) -> Num[Array, "*N 4"]:
        """Transform points from model space to screen space.

        Parameters:
          - points: shape (4, ) or (N, 4). points in model space, with axis 0
            being the batch axis. Batch axis can be omitted. Points must be
            in homogeneous coordinate.

        Returns: points in clip space, with axis 0 being the batch axis, if
            given in batch. The dtype may be promoted. The homogeneous
            coordinates are not normalized.
        """
        clip_space = self.apply(points, self.world_to_clip)
        assert isinstance(clip_space, Num[Array, "*N 4"])

        return clip_space

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

        Reference:
          - [gluLookAt](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml)
          - [glTranslate](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glTranslate.xml)
          - [GluLookAt Code](https://www.khronos.org/opengl/wiki/GluLookAt_code)
        """
        forward: Vec3f = normalise(centre - eye)
        up = normalise(up)
        side: Vec3f = normalise(jnp.cross(forward, up))
        up = jnp.cross(side, forward)

        m: ModelView = (
            jnp.identity(4)  #
            .at[0, :3].set(side)  #
            .at[1, :3].set(up)  #
            .at[2, :3].set(-forward)  #
        )
        translation: ModelView = jnp.identity(4).at[:3, 3].set(-eye)

        model_view: ModelView = m @ translation

        return model_view

    @staticmethod
    @jaxtyped
    @jax.jit
    def perspective_projection_matrix(
        fovy: jnp.floating[Any],
        aspect: jnp.floating[Any],
        z_near: jnp.floating[Any],
        z_far: jnp.floating[Any],
    ) -> Projection:
        """Create a projection matrix to map the model in the camera frame (eye
            coordinates) onto the viewing volume (clip coordinates), using
            perspective transformation. This follows the implementation in
            OpenGL (gluPerspective)

        Parameters:
          - fovy: Specifies the field of view angle, in degrees, in the y
            direction.
          - aspect: Specifies the aspect ratio that determines the field of
            view in the x direction. The aspect ratio is the ratio of x (width)
            to y (height).
          - z_near: Specifies the distance from the viewer to the near clipping
            plane (always positive).
          - z_far: Specifies the distance from the viewer to the far clipping
            plane (always positive).

        Return: Projection, (4, 4) matrix.

        Reference:
          - [gluPerspective](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml)
        """
        f: jnp.single = 1. / lax.tan(fovy.astype(jnp.single) / 2.)
        projection: Projection = (
            jnp.zeros((4, 4), dtype=jnp.single)  #
            .at[0, 0].set(f / aspect)  #
            .at[1, 1].set(f)  #
            .at[2, 2].set((z_far + z_near) / (z_near - z_far))  #
            # translate z
            .at[2, 3].set((2. * z_far * z_near) / (z_near - z_far))  #
            .at[3, 2].set(-1.)  # let \omega be -z
        )

        return projection

    @staticmethod
    @jaxtyped
    @jax.jit
    def perspective_projection_matrix_tinyrenderer(
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
        dimension: Integer[Array, "2"],
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
          - dimension: width, height of the viewport, in screen space.
          - depth: the depth of the viewport in screen space, for zbuffer
          - dtype: the dtype for the viewport matrix.

        Return: Viewport, (4, 4) matrix.
        """
        width, height = dimension
        viewport: Viewport = (
            jnp.identity(4, dtype=dtype)  #
            .at[:2, 3].set(lowerbound + dimension / 2)  #
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
        jnp.ndim(coordinates) - 1,
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
    return coordinates / coordinates[..., -1:]


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
    v: Vec3f = jnp.cross(mat[:, 0], mat[:, 1])
    # `u[2]` is 0, that means triangle is degenerate, in this case
    # return something with negative coordinates
    v = lax.cond(
        jnp.abs(v[-1]) < 1e-10,
        lambda: jnp.array((-1., 1., 1.)),
        lambda: jnp.array((
            1 - (v[0] + v[1]) / v[2],
            v[1] / v[2],
            v[0] / v[2],
        )),
    )
    assert isinstance(v, Vec3f)

    return v
