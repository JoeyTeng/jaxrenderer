from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

import enum
from functools import partial
from typing import NamedTuple, Optional, Union, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, Num
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from ._backport import Tuple, TypeAlias
from ._meta_utils import add_tracing_name
from ._meta_utils import typed_jit as jit
from .types import FloatV, JaxFloating, NumV, Triangle2Df, Vec2f, Vec2i, Vec3f, Vec4f

# Transform matrix that takes a batch of homogeneous 3D vertices and transform
# them into 2D cartesian vertices in screen space + Z value (making it 3D)
#
# The result of x-y values in screen space may be float, and thus further
# conversion to integers are needed.
World2Screen: TypeAlias = Float[Array, "4 4"]
# Transform all coordinates from world space to view space, with camera at
# origin. (Object Coordinates -> Eye Coordinates)
View: TypeAlias = Float[Array, "4 4"]
# Transform all coordinates from view space to viewing volume.
# (Eye Coordinates -> Clip Coordinates)
Projection: TypeAlias = Float[Array, "4 4"]
# Transform all coordinates from clip space in a bi-unit cube ([-1...1]^3) to
# a screen cube ([x, x+width] * [y, y+height] * [0, depth]) in view space.
# (Normalised Device Coordinates -> Window Coordinates)
Viewport: TypeAlias = Float[Array, "4 4"]


@jaxtyped
@partial(jit, donate_argnums=(0,), inline=True)
@add_tracing_name
def normalise(vector: Float[Array, "*a dim"]) -> Float[Array, "*a dim"]:
    """normalise vector in-place."""
    result: Float[Array, "*a dim"] = cast(
        Float[Array, "*a dim"],
        vector / jnp.linalg.norm(vector),
    )
    assert isinstance(result, Float[Array, "*a dim"])

    return result


class Interpolation(enum.Enum):
    """Interpolation methods for rasterisation.

    References:
      - [Interpolation qualifiers](https://www.khronos.org/opengl/wiki/Type_Qualifier_(GLSL)#Interpolation_qualifiers)
    """

    FLAT = 0
    """Flat shading: use the value of the first vertex of the primitive"""
    NOPERSPECTIVE = 1
    """No perspective correction: linear interpolation in screen space"""
    SMOOTH = 2
    """Perspective correction: linear interpolation in clip space"""

    @jaxtyped
    @partial(
        jit,
        static_argnames=("self",),
        inline=True,
    )
    @add_tracing_name
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
        dtype = jax.dtypes.result_type(  # pyright: ignore
            barycentric_screen,
            barycentric_clip,
            values,
        )
        coef: Vec3f
        # branches are ok because `self` is static: decided at compile time
        if self == Interpolation.FLAT:
            with jax.ensure_compile_time_eval():
                coef = jnp.array([1, 0, 0], dtype=dtype)  # pyright: ignore
        elif self == Interpolation.NOPERSPECTIVE:
            coef = barycentric_screen
        elif self == Interpolation.SMOOTH:
            coef = barycentric_clip
        else:
            raise ValueError(f"Unknown interpolation method {self}")

        interpolated = lax.dot_general(  # pyright: ignore[reportUnknownMemberType]
            coef.astype(dtype),  # pyright: ignore[reportUnknownMemberType]
            values.astype(dtype),  # pyright: ignore[reportUnknownMemberType]
            (((0,), (0,)), ([], [])),
        )

        return interpolated


@jaxtyped
@partial(
    jit,
    static_argnames=("mode",),
    inline=True,
)
@add_tracing_name
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
    interpolated = mode(
        values=values,
        barycentric_screen=barycentric_screen,
        barycentric_clip=barycentric_clip,
    )
    assert isinstance(interpolated, Num[Array, "*valueDimensions"])

    return interpolated


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def to_homogeneous(
    coordinates: Float[Array, "*batch dim"],
    value: Union[float, Float[Array, "*batch"]] = 1.0,
) -> Float[Array, "*batch dim+1"]:
    """Transform the coordinates to homogeneous coordinates by append a batch
    of `value`s (default 1.) in the last axis."""
    if not isinstance(value, Float[Array, "*batch"]):
        value = jnp.array(value)  # pyright: ignore[reportUnknownMemberType]

    target_dtype = jax.dtypes.result_type(coordinates)  # pyright: ignore
    paddings: Float[Array, "*batch 1"] = jnp.broadcast_to(  # pyright: ignore
        value.astype(target_dtype),  # pyright: ignore
        (*coordinates.shape[:-1], 1),
    )
    homo_coords: Float[Array, "*batch dim+1"] = lax.concatenate(  # pyright: ignore
        (coordinates, paddings),
        jnp.ndim(coordinates) - 1,
    )

    return homo_coords


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def normalise_homogeneous(
    coordinates: Float[Array, "*batch dim"],
) -> Float[Array, "*batch dim"]:
    """Transform the homogenous coordinates to make the scale factor equals to
        either 1 or 0, by divide every element with the last element on the
        last axis.

    Noted that when a coordinate is 0 and divides by 0, it will produce a nan;
    for non-zero elements divides by 0, a inf will be produced.
    """
    return coordinates / coordinates[..., -1:]


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def to_cartesian(
    coordinates: Float[Array, "*batch dim"],
) -> Float[Array, "*batch dim-1"]:
    """Transform the homogenous coordinates to cartesian coordinates by divide
        every element with the last element on the last axis, then drop them.

    When last component is 0, this function just discard the w-component
    without division.
    """
    result: Float[Array, "*batch dim-1"]
    result = jnp.where(  # pyright: ignore[reportUnknownMemberType]
        # if w component is 0, just discard it and return.
        coordinates[..., -1:] == 0.0,
        coordinates[..., :-1],
        normalise_homogeneous(coordinates)[..., :-1],
    )

    return result


class Camera(NamedTuple):
    """Camera parameters.

    - view: transform from model space to view space
    - projection: transform from view space to clip space
    - viewport: transform from NDC (normalised device coordinate) space to
      screen space. Noticed that this is NDC space in OpenGL, which has range
      [-1, 1]^3.
    - world_to_clip: transform from model space to clip space
    - world_to_eye_norm: transform normals from model space to eye space, without projection.
    - world_to_screen: transform from model space to screen space
    """

    view: View
    projection: Projection
    viewport: Viewport
    world_to_clip: Projection
    world_to_eye_norm: Projection
    world_to_screen: World2Screen
    view_inv: View
    screen_to_world: World2Screen

    @classmethod
    @jaxtyped
    @partial(
        jit,
        static_argnames=("cls",),
        inline=True,
    )
    @add_tracing_name
    def create(
        cls,
        view: View,
        projection: Projection,
        viewport: Viewport,
        view_inv: Optional[View] = None,
    ) -> "Camera":
        """Create a camera with the given parameters.

        Parameters:
          - view: transform from model space to view space
          - projection: transform from view space to clip space
          - viewport: transform from NDC (normalised device coordinate) space to
          - view_inv: inverse of view. If not given, it will be computed.
        """
        if view_inv is None:
            view_inv = cast(View, jnp.linalg.inv(view))
        assert isinstance(view_inv, View)

        projection_inv: Projection = lax.cond(  # pyright: ignore
            jnp.isclose(projection[3, 3], 0),  # pyright: ignore
            # is perspective projection matrix
            cls.perspective_projection_matrix_inv,
            # is orthographic projection matrix
            cls.orthographic_projection_matrix_inv,
            # arg
            projection,
        )
        assert isinstance(projection_inv, Projection), f"{projection_inv}"

        viewport_inv: Viewport = cls.viewport_matrix_inv(viewport)
        assert isinstance(viewport_inv, Viewport)

        return cls(
            view=view,
            viewport=viewport,
            projection=projection,
            world_to_clip=projection @ view,
            # inverse transpose of projection @ view
            world_to_eye_norm=view_inv.T,
            world_to_screen=viewport @ projection @ view,
            view_inv=view_inv,
            screen_to_world=view_inv @ projection_inv @ viewport_inv,
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def apply(
        points: Num[Array, "*N 4"],
        matrix: Num[Array, "4 4"],
    ) -> Num[Array, "*N 4"]:
        """Transform homogeneous points using given matrix.

        Parameters:
          - points: shape (4, ) or (N, 4). points in model space, with axis 0
            being the batch axis. Batch axis can be omitted. Points must be
            in homogeneous coordinate.
          - matrix: shape (4, 4) transformation matrix

        Returns: coordinates transformed
        """
        assert jnp.ndim(points) < 3
        assert ((jnp.ndim(points) == 2) and (points.shape[1] == 4)) or (
            (jnp.ndim(points) == 1) and (points.shape[0] == 4)
        )

        with jax.ensure_compile_time_eval():
            lhs_contract_axis = 1 if jnp.ndim(points) == 2 else 0
            dtype = jax.dtypes.result_type(points, matrix)  # pyright: ignore

        # put `points` at lhs to keep batch axis at axis 0 in the result.
        transformed: Num[Array, "*N 4"] = lax.dot_general(  # pyright: ignore
            points.astype(dtype),  # pyright: ignore[reportUnknownMemberType]
            matrix.astype(dtype),  # pyright: ignore[reportUnknownMemberType]
            (((lhs_contract_axis,), (1,)), ([], [])),
        )
        assert isinstance(transformed, Num[Array, "*N 4"])

        return transformed

    @classmethod
    @jaxtyped
    @partial(jit, static_argnames=("cls",), inline=True)
    @add_tracing_name
    def apply_pos(
        cls,
        points: Num[Array, "*N 3"],
        matrix: Num[Array, "4 4"],
    ) -> Num[Array, "*N 3"]:
        """Transform points representing 3D positions using given matrix.

        Parameters:
          - points: shape (3, ) or (N, 3). points in model space, with axis 0
            being the batch axis. Batch axis can be omitted. Points must be
            in cartesian coordinate. For coordinates in homogeneous coordinate,
            use `apply` instead.
          - matrix: shape (4, 4) transformation matrix

        Returns: coordinates transformed
        """
        points_homo = to_homogeneous(points)
        assert isinstance(points_homo, Num[Array, "*N 4"])

        transformed_homo = cls.apply(points_homo, matrix)
        assert isinstance(transformed_homo, Num[Array, "*N 4"])

        transformed = to_cartesian(transformed_homo)
        assert isinstance(transformed, Num[Array, "*N 3"])

        return transformed

    @classmethod
    @jaxtyped
    @partial(jit, static_argnames=("cls",), inline=True)
    @add_tracing_name
    def apply_vec(
        cls,
        vectors: Num[Array, "*N 3"],
        matrix: Num[Array, "4 4"],
    ) -> Num[Array, "*N 3"]:
        """Transform vectors representing 3D positions using given matrix.

        Parameters:
          - vectors: shape (3, ) or (N, 3). Directional Vectors in model
            space, with axis 0 being the batch axis. Batch axis can be omitted.
            Vectors must be in cartesian coordinate. For coordinates in
            homogeneous coordinate, use `apply` instead.
          - matrix: shape (4, 4) transformation matrix

        Returns: vectors transformed and normalised
        """
        normalised_vectors = normalise(vectors)
        assert isinstance(normalised_vectors, Num[Array, "*N 3"])

        points_homo = cast(
            Num[Array, "*N 4"],
            to_homogeneous(
                normalised_vectors,
                jnp.zeros((), dtype=vectors.dtype),  # pyright: ignore
            ),
        )
        assert isinstance(points_homo, Num[Array, "*N 4"])

        transformed_homo = cls.apply(points_homo, matrix)
        assert isinstance(transformed_homo, Num[Array, "*N 4"])

        transformed = transformed_homo[..., :3]
        assert isinstance(transformed, Num[Array, "*N 3"])

        transformed_normalised = normalise(transformed)
        assert isinstance(transformed_normalised, Num[Array, "*N 3"])

        return transformed_normalised

    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
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
    @partial(jit, inline=True)
    @add_tracing_name
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

    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def to_screen_inv(
        self,
        screen: Float[Array, "*N 4"],
    ) -> Float[Array, "*N 4"]:
        """Transform points from screen space to model space.

        This is an inverse process of `to_screen`, and provide higher precision
        then just multiplying the inverse. This may help solve NaN issue.

        Internally this is done by two `lax.linalg.triangular_solve` for
        viewport and projection, then a `@` for `view_inv`. If a good
        `view_inv` is provided during creation of this camera using
        `view_matrix_inv`, this should provide a much higher precision.
        """
        if screen.ndim == 1:
            _screen = screen[None, :]
        else:
            _screen = screen

        clip: Float[Array, "*N 4"] = lax.linalg.triangular_solve(  # pyright: ignore
            self.viewport,
            _screen,
        )
        assert isinstance(clip, Float[Array, "*N 4"])
        shuffle = lax.cond(  # pyright: ignore
            self.projection[3, 3] == 0,
            # perspective projection
            lambda: jnp.array([0, 1, 3, 2]),  # pyright: ignore[reportUnknownMemberType]
            # orthographic projection
            lambda: jnp.array([0, 1, 2, 3]),  # pyright: ignore[reportUnknownMemberType]
        )
        eye: Float[Array, "*N 4"] = lax.linalg.triangular_solve(  # pyright: ignore
            self.projection[..., shuffle],
            clip[..., shuffle],
        )[..., shuffle]
        assert isinstance(eye, Float[Array, "*N 4"])
        world = self.apply(eye, self.view_inv)
        assert isinstance(world, Float[Array, "*N 4"])

        if screen.ndim == 1:
            world = world[0]

        return world

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def inv_scale_translation_matrix(
        scale_translation_mat: Float[Array, "4 4"]
    ) -> Float[Array, "4 4"]:
        """Compute the inverse matrix of a (4, 4) matrix representing a scale and translation, in a form of:

            [[s_x, 0,   0,   t_x],
             [0,   s_y, 0,   t_y],
             [0,   0,   s_z, t_z],
             [0,   0,   0,   1]]

        where s is a scale vector and t is a translation vector. It is treated
        as a combination of a scale matrix and a translation matrix, as
        `scale @ translation`: translate first, then scale.

        This utilise the fact that the inverse of a scale operation is just the
        reciprocal of the scale factor, and the inverse of a translation is
        just the negative of the translation. It separates the scale and
        translation operations first, inverse them separately, then combine
        them back (in reverse order).
        """

        scale_inv = jnp.diag(1.0 / jnp.diag(scale_translation_mat))  # pyright: ignore
        assert isinstance(scale_inv, Float[Array, "4 4"])

        # scale_translation = scale @ translation;
        # thus  translation = scale_inv @ scale @ translation
        #                   = scale_inv @ scale_translation
        translation: Float[Array, "4 4"] = scale_inv @ scale_translation_mat
        assert isinstance(translation, Float[Array, "4 4"])

        # inverse of translation: negative of translation
        translation_inv = (
            jnp.identity(4).at[:3, 3].set(-translation[:3, 3])  # pyright: ignore
        )
        assert isinstance(translation_inv, Float[Array, "4 4"])

        scale_translation_inv = translation_inv @ scale_inv
        assert isinstance(scale_translation_inv, Float[Array, "4 4"])

        return scale_translation_inv

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def view_matrix(
        eye: Vec3f,
        centre: Vec3f,
        up: Vec3f,
    ) -> View:
        """Compute View matrix as defined by OpenGL / tinyrenderer.

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
        side: Vec3f = normalise(jnp.cross(forward, up))  # pyright: ignore
        up = cast(Vec3f, jnp.cross(side, forward))

        m: View = (
            jnp.identity(4)  # pyright: ignore[reportUnknownMemberType]
            .at[0, :3]
            .set(side)
            .at[1, :3]
            .set(up)
            .at[2, :3]
            .set(-forward)
        )
        translation: View = jnp.identity(4).at[:3, 3].set(-eye)  # pyright: ignore

        view: View = m @ translation

        return view

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def view_matrix_inv(
        eye: Vec3f,
        centre: Vec3f,
        up: Vec3f,
    ) -> View:
        """Compute the invert of View matrix as defined by OpenGL.

        Same as inverting `lookAt` in OpenGL, but more precise.

        Parameters:
          - eye: the position of camera, in world space
          - centre: the centre of the frame, where the camera points to, in
            world space
          - up: the direction vector with start point at "eye", indicating the
            "up" direction of the camera frame.

        Noticed that the view matrix contains only rotation and translation, and
        thus the inverse of it is just the inverse of translation multiplies the
        inverse (a simple transpose!) of rotation.

        Returns: View^{-1}, (4, 4) matrix.

        Reference:
          - [gluLookAt](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml)
          - [4B](http://graphics.stanford.edu/courses/cs248-98-fall/Final/q4.html)
        """
        forward: Vec3f = normalise(centre - eye)
        up = normalise(up)
        side: Vec3f = normalise(jnp.cross(forward, up))  # pyright: ignore
        up = cast(Vec3f, jnp.cross(side, forward))

        # inverse of rotation is just the transpose
        m: View = (
            jnp.identity(4)  # pyright: ignore[reportUnknownMemberType]
            .at[0, :3]
            .set(side)
            .at[1, :3]
            .set(up)
            .at[2, :3]
            .set(-forward)
        )
        m_inv: View = m.T
        assert isinstance(m_inv, View)

        # inverse of translation is just the negative of translation
        translation_inv: View = jnp.identity(4).at[:3, 3].set(eye)  # pyright: ignore
        assert isinstance(translation_inv, View)

        view_matrix_inv: View = translation_inv @ m_inv
        assert isinstance(view_matrix_inv, View)

        return view_matrix_inv

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def perspective_projection_matrix(
        fovy: Union[float, JaxFloating, FloatV],
        aspect: Union[float, JaxFloating, FloatV],
        z_near: Union[float, JaxFloating, FloatV],
        z_far: Union[float, JaxFloating, FloatV],
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
        deg: FloatV = jnp.asarray(fovy, dtype=jnp.single)  # pyright: ignore
        f: FloatV = 1.0 / lax.tan(  # pyright: ignore[reportUnknownMemberType]
            cast(FloatV, jnp.radians(deg) / 2.0)
        )
        projection: Projection = (
            jnp.zeros((4, 4), dtype=jnp.single)  # pyright: ignore
            .at[0, 0]
            .set(f / aspect)
            .at[1, 1]
            .set(f)
            .at[2, 2]
            .set((z_far + z_near) / (z_near - z_far))
            # translate z
            .at[2, 3]
            .set((2.0 * z_far * z_near) / (z_near - z_far))
            .at[3, 2]
            .set(-1.0)  # let \omega be -z
        )

        return projection

    @classmethod
    @jaxtyped
    @partial(jit, static_argnames=("cls",), inline=True)
    @add_tracing_name
    def perspective_projection_matrix_inv(cls, mat: Projection) -> Projection:
        """Create the inverse of a perspective projection matrix as defined in
            `perspective_projection_matrix`.

        Since the perspective projection matrix is formed as:

            [[a, 0,  0, 0],
             [0, b,  0, 0],
             [0, 0,  c, d],
             [0, 0, -1, 0]]

        it can be simply transformed into a scale-translation matrix by
        swapping the last two columns. Thus, the inverse is computed by
        swapping the last columns, then inverting using
        `inv_scale_translation_matrix`, and finally swapping back (last two
        rows, instead).
        """
        with jax.ensure_compile_time_eval():
            shuffle: Integer[Array, "4"] = jnp.array((0, 1, 3, 2))  # pyright: ignore
            assert isinstance(shuffle, Integer[Array, "4"])

        inv = cls.inv_scale_translation_matrix(mat[:, shuffle])[shuffle, :]
        assert isinstance(inv, Projection)

        return inv

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def orthographic_projection_matrix(
        left: Union[float, JaxFloating],
        right: Union[float, JaxFloating],
        bottom: Union[float, JaxFloating],
        top: Union[float, JaxFloating],
        z_near: Union[float, JaxFloating],
        z_far: Union[float, JaxFloating],
    ) -> Projection:
        """Create a projection matrix to map the model in the camera frame (eye
            coordinates) onto the viewing volume (clip coordinates), using
            orthographic transformation. This follows the implementation in
            OpenGL (glOrtho).

        Parameters:
          - left, right: Specifies the coordinates for the left and right
            vertical clipping planes.
          - bottom, top: Specifies the coordinates for the bottom and top
            horizontal clipping planes..
          - z_near, z_far: Specifies the distances from the viewer to the
            nearer and farther depth clipping planes. These values are negative
            if they are behind the viewer.

        Return: Projection, (4, 4) matrix.

        Reference:
          - [glOrtho](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml)
        """
        l_op: Float[Array, "3"] = jnp.array([right, top, z_far])  # pyright: ignore
        r_op: Float[Array, "3"] = jnp.array([left, bottom, z_near])  # pyright: ignore
        projection: Projection = (
            jnp.zeros((4, 4), dtype=jnp.single)  # pyright: ignore
            .at[0, 0]
            .set(2 / (right - left))
            .at[1, 1]
            .set(2 / (top - bottom))
            .at[2, 2]
            .set(-2 / (z_far - z_near))
            .at[3, 3]
            .set(1)  #
            .at[:3, 3]
            .set(-(l_op + r_op) / (l_op - r_op))
        )

        return projection

    @classmethod
    @jaxtyped
    @partial(jit, static_argnames=("cls",))
    @add_tracing_name
    def orthographic_projection_matrix_inv(cls, mat: Projection) -> Projection:
        """Create the inverse of a orthographic projection matrix as defined in
        `orthographic_projection_matrix`. Since orthographic projection
        matrix is a scale-translation matrix, the inverse is computed by
        `inv_scale_translation_matrix` directly.
        """
        inv = cls.inv_scale_translation_matrix(mat)
        assert isinstance(inv, Projection)

        return inv

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def perspective_projection_matrix_tinyrenderer(
        eye: Vec3f,
        centre: Vec3f,
        dtype: type = jnp.single,
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
            jnp.identity(4, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]
            .at[3, 2]
            .set(-1 / jnp.linalg.norm(eye - centre))
        )

        return projection

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def viewport_matrix(
        lowerbound: Num[Array, "2"],
        dimension: Vec2i,
        depth: Union[int, float, NumV],
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
            jnp.identity(4)  # pyright: ignore[reportUnknownMemberType]
            .at[:2, 3]
            .set(lowerbound + dimension / 2)
            .at[0, 0]
            .set(width / 2)
            .at[1, 1]
            .set(height / 2)
            .at[2, 2:]
            .set(depth / 2)
        )

        return viewport

    @classmethod
    @jaxtyped
    @partial(jit, static_argnames=("cls",), inline=True)
    @add_tracing_name
    def viewport_matrix_inv(cls, viewport: Viewport) -> Viewport:
        """Create the inverse of a viewport matrix as defined in `viewport_matrix`.

        Parameters:
          - viewport: Viewport matrix to invert.

        Return: Viewport^{-1}, (4, 4) matrix.
        """
        viewport_inv: Viewport = cls.inv_scale_translation_matrix(viewport)
        assert isinstance(viewport_inv, Viewport)

        return viewport_inv

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def world_to_screen_matrix(width: int, height: int) -> World2Screen:
        """Generate the projection matrix to convert model coordinates to
            screen/canvas coordinates.

        It assumes all model coordinates are in [-1...1] and will transform them
        into ([0...width], [0...height], [-1...1]).

        Return: World2Screen (Float[Array, "4 4"])
        """
        world2screen: World2Screen = (
            # 3. div by half to centering
            jnp.identity(4).at[0, 0].set(0.5).at[1, 1].set(0.5)  # pyright: ignore
            # 2. mul by width, height
            @ jnp.eye(4).at[0, 0].set(width).at[1, 1].set(height)  # pyright: ignore
            # 1. Add by 1 to make values positive
            @ jnp.identity(4).at[:2, -1].set(1)  # pyright: ignore
        )

        return world2screen


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def compute_normal(triangle_verts: Float[Array, "3 3"]) -> Float[Array, "3"]:
    normal: Float[Array, "3"] = cast(
        Float[Array, "3"],
        jnp.cross(
            triangle_verts[2, :] - triangle_verts[0, :],
            triangle_verts[1, :] - triangle_verts[0, :],
        ),
    )
    normal = cast(Float[Array, "3"], normal / jnp.linalg.norm(normal, keepdims=True))
    assert isinstance(normal, Float[Array, "3"])

    return normal


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def quaternion(
    rotation_axis: Union[Vec3f, Tuple[float, float, float]],
    rotation_angle: Union[FloatV, float],
) -> Vec4f:
    """Generate a quaternion rotation from a rotation axis and angle.

    The rotation axis is normalised internally. The angle is specified in
    degrees (NOT radian). The rotation is clockwise.

    The resultant quaternion is in order of (w, x, y, z).
    """
    axis = normalise(jnp.asarray(rotation_axis))  # pyright: ignore
    angle = jnp.radians(jnp.asarray(rotation_angle))  # pyright: ignore
    assert isinstance(axis, Vec3f), f"{rotation_axis}"
    assert isinstance(angle, FloatV), f"{rotation_angle}"

    quaternion: Vec4f = (
        jnp.zeros(4)  # pyright: ignore[reportUnknownMemberType]
        .at[0]
        .set(jnp.cos(angle / 2))  # pyright: ignore[reportUnknownMemberType]
        .at[1:]
        .set(axis * jnp.sin(angle / 2))  # pyright: ignore[reportUnknownMemberType]
    )

    return quaternion


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def quaternion_mul(quatA: Vec4f, quatB: Vec4f) -> Vec4f:
    """Multiply two quaternion rotations, as to composite them.

    Noticed that all quaternions here are in order of (w, x, y, z).

    References:
      - [Quaternion multiplication](https://www.mathworks.com/help/nav/ref/quaternion.mtimes.html)
    """
    assert isinstance(quatA, Vec4f)
    assert isinstance(quatB, Vec4f)

    with jax.ensure_compile_time_eval():
        idx103 = jnp.array((1, 0, 3))  # pyright: ignore[reportUnknownMemberType]
        idx230 = jnp.array((2, 3, 0))  # pyright: ignore[reportUnknownMemberType]
        idx013 = jnp.array((0, 1, 3))  # pyright: ignore[reportUnknownMemberType]
        idx320 = jnp.array((3, 2, 0))  # pyright: ignore[reportUnknownMemberType]

    return jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            quatA[0] * quatB[0] - quatA[1:] @ quatB[1:],
            quatA[:3] @ quatB[idx103] - quatA[3] * quatB[2],
            quatA[idx230] @ quatB[:3] - quatA[1] * quatB[3],
            quatA[idx013] @ quatB[idx320] - quatA[2] * quatB[1],
        )
    )


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def rotation_matrix(
    rotation_axis: Union[Vec3f, Tuple[float, float, float]],
    rotation_angle: Union[FloatV, float],
) -> Float[Array, "3 3"]:
    """Generate a rotation matrix from a rotation axis and angle.

    The rotation axis is normalised internally. The angle is specified in
    degrees (NOT radian). The rotation is clockwise.

    References:
      - [glRotated](https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glRotate.xml)
    """
    axis = normalise(jnp.asarray(rotation_axis))  # pyright: ignore
    angle = jnp.radians(jnp.asarray(rotation_angle))  # pyright: ignore
    assert isinstance(axis, Vec3f), f"{rotation_axis}"
    assert isinstance(angle, FloatV), f"{rotation_angle}"

    c = jnp.cos(angle)  # pyright: ignore[reportUnknownMemberType]

    rotation_matrix: Float[Array, "3 3"] = cast(
        Float[Array, "3 3"],
        jnp.identity(3) * c  # +c at main diagonal # pyright: ignore
        # second term
        - jnp.sin(angle) * jnp.cross(axis, jnp.identity(3))  # pyright: ignore
        + (1 - c) * jnp.outer(axis, axis),  # first term
    )
    assert isinstance(rotation_matrix, Float[Array, "3 3"])

    return rotation_matrix


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def transform_matrix_from_rotation(rotation: Vec4f) -> Float[Array, "3 3"]:
    """Generate a transform matrix from a quaternion rotation.

    Quaternion is specified in (w, x, y, z) order. Supports non-unit rotation.

    References:
          - [Quaternions and spatial rotation#Quaternion-derived rotation matrix](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix)
          - [TinySceneRenderer::set_object_orientation](https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/tinyrenderer.cpp#LL997C20-L997C20)
    """
    d = rotation @ rotation
    s = 2.0 / d  # here s is $2\times s$ in Wikipedia.

    rs: Vec3f = rotation[1:] * s  # x y z
    ((wx, wy, wz), (xx, xy, xz), (_, yy, yz)) = jnp.outer(  # pyright: ignore
        rotation[:3],
        rs,
    )
    zz = rotation[3] * rs[2]

    mat: Float[Array, "3 3"] = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            (1.0 - (yy + zz), xy - wz, xz + wy),
            (xy + wz, 1.0 - (xx + zz), yz - wx),
            (xz - wy, yz + wx, 1.0 - (xx + yy)),
        )
    )
    assert isinstance(mat, Float[Array, "3 3"])

    return mat


@jaxtyped
@partial(jit, inline=True)
@add_tracing_name
def barycentric(pts: Triangle2Df, p: Vec2f) -> Vec3f:
    """Compute the barycentric coordinate of `p`.
    Returns u[-1] < 0 if `p` is outside of the triangle.
    """
    mat: Float[Array, "3 2"] = jnp.vstack(  # pyright: ignore[reportUnknownMemberType]
        (
            pts[2] - pts[0],
            pts[1] - pts[0],
            pts[0] - p,
        )
    )
    v: Vec3f = cast(Vec3f, jnp.cross(mat[:, 0], mat[:, 1]))
    # `u[2]` is 0, that means triangle is degenerate, in this case
    # return something with negative coordinates
    v = cast(
        Vec3f,
        lax.cond(  # pyright: ignore[reportUnknownMemberType]
            jnp.abs(v[-1]) < 1e-10,  # pyright: ignore[reportUnknownArgumentType]
            lambda: jnp.array((-1.0, 1.0, 1.0)),  # pyright: ignore
            lambda: jnp.array(  # pyright: ignore[reportUnknownMemberType]
                (
                    1 - (v[0] + v[1]) / v[2],
                    v[1] / v[2],
                    v[0] / v[2],
                )
            ),
        ),
    )
    assert isinstance(v, Vec3f)

    return v
