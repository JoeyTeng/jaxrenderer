from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, TypeVar

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map
from jaxtyping import Array, Bool, Float, Integer, PyTree, Shaped, jaxtyped

from .geometry import Camera, Interpolation, interpolate
from .types import FALSE_ARRAY, INF_ARRAY, TRUE_ARRAY, Vec2f, Vec3f, Vec4f

jax.config.update('jax_array', True)

ID = Integer[Array, ""]

ShaderExtraInputT = TypeVar(
    'ShaderExtraInputT',
    bound=PyTree[Shaped[Array, "..."]],
)
"""Extra input for vertex shader & fragment shader, shared by all."""


class PerVertex(NamedTuple):
    """Built-in output from Vertex Shader.

    gl_Position is in clip-space.
    """
    gl_Position: Vec4f
    # gl_PointSize is meaningful only when rendering point primitives.
    # not supported for now
    # gl_PointSize: Float[Array, ""]


class PerFragment(NamedTuple):
    """Built-in Output from Fragment Shader.

    If use_default_depth is True (default False), gl_FragCoord[2] will be used
    later by default.
    """
    gl_FragDepth: Float[Array, ""] = INF_ARRAY
    # not discard
    keeps: Bool[Array, ""] = TRUE_ARRAY
    use_default_depth: Bool[Array, ""] = FALSE_ARRAY


VaryingT = TypeVar(
    "VaryingT",
    bound=tuple[Shaped[Array, "..."], ...],
)
"""The user-defined input and second (extra) output of fragment shader."""

MixedExtraT = TypeVar(
    "MixedExtraT",
    bound=tuple[Shaped[Array, "..."], ...],
)
"""The user-defined second (extra) output of mix shader."""


class MixerOutput(NamedTuple):
    """Built-in output from `Shader.mix`.

    keep: bool, whether the output should be used to update buffers
    zbuffer: store depth value, and the result is used to set zbuffer.
    """
    keep: Bool[Array, ""]
    zbuffer: Float[Array, ""]


class Shader(ABC, Generic[ShaderExtraInputT, VaryingT, MixedExtraT]):
    """Base class for customised shader.

    Since JAX is pure functional (stateless), the state will be passed by
    returned values (the second return value in each function) in each process.
    """

    @staticmethod
    @jaxtyped
    @jax.jit
    @abstractmethod
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: ShaderExtraInputT,
    ) -> tuple[PerVertex, VaryingT]:
        """Override this to implement the vertex shader as defined by OpenGL.

        The meaning of the inputs follows the definitions in OpenGL. Additional
        named parameters can be defined and passed in if you need, as defined
        in `extra`.

        Noticed that no internal state will be tracked, thus if there is any
        value to be passed to downstream process, it must be returned as the
        output `VaryingT` of this function.

        Relevant information from the original document is copied below.

        Parameters:
          - gl_VertexID
            the index of the vertex currently being processed. When using
            non-indexed rendering, it is the effective index of the current
            vertex (the number of vertices processed + the `first` value). For
            indexed rendering, it is the index used to fetch this vertex from
            the buffer.
          - gl_InstanceID
            the index of the current instance when doing some form of instanced
            rendering. The instance count always starts at 0, even when using
            base instance calls. When not using instanced rendering, this value
            will be 0.
          - camera: Camera [extra input, not in GLSL]
            contains view, viewport, and projection matrices.
          - extra: Camera [extra input, not in GLSL]
            User-defined extra input for vertex shader, shared by all. They are
            **not** split over batch axis 0, if any; but directly passed in.


        Return: PerVertex (used for internals) and ExtraPerVertexOutput to be
            interpolated and used by downstream pipelines.
          - gl_Position
            the clip-space output position of the current vertex.
          - gl_PointSize
            the pixel width/height of the point being rasterized. It only has a
            meaning when rendering point primitives. It will be clamped to the
            GL_POINT_SIZE_RANGE.

        Reference:
          - [Vertex Shader/Defined Inputs](https://www.khronos.org/opengl/wiki/Vertex_Shader/Defined_Inputs)
          - [Vertex Shader#Outputs](https://www.khronos.org/opengl/wiki/Vertex_Shader#Outputs)
        """
        raise NotImplementedError("vertex shader not implemented")

    @staticmethod
    @jaxtyped
    @jax.jit
    def interpolate(
        values: VaryingT,
        barycentric_screen: Vec3f,
        barycentric_clip: Vec3f,
    ) -> VaryingT:
        """Override this to customise the interpolation of user-defined inputs.

        The default implementation is to interpolate all the fields of given
        values as `smooth`, which is perspective interpolation, as defined in
        GLSL.

        Parameters:
          - values: values at the vertices of the triangle, with axis 0 being
            the batch axis. It is expected to be a tuple of multiple batched
            values.
          - barycentric_screen: barycentric coordinates in screen space of the
            point to interpolate
          - barycentric_clip: barycentric coordinates in clip space of the
            point to interpolate

        Return: interpolated values for fragment shader process, with same
            structure (order of members) as `values`
        """
        varying: VaryingT = tree_map(
            Partial(
                interpolate,
                barycentric_screen=barycentric_screen,
                barycentric_clip=barycentric_clip,
                mode=Interpolation.SMOOTH,
            ),
            values,
        )

        return varying

    @staticmethod
    @jaxtyped
    @jax.jit
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: Bool[Array, ""],
        gl_PointCoord: Vec2f,
        varying: VaryingT,
        extra: ShaderExtraInputT,
    ) -> tuple[PerFragment, VaryingT]:
        """Override this to implement the vertex shader as defined by OpenGL.

        This is optional. The default implementation writes nothing and thus
        `gl_FragDepth` further down the pipeline will use `gl_FragCoord[2]`.
        For the `varying` input, it will be returned directly untouched.

        If the output from this default implementation is re-used, noticed that
        `use_default_depth` needs to be updated to False, otherwise the default
        depth (`gl_FragCoord[2]`) will be used in further process.

        Parameters:
          - gl_FragCoord: homogeneous coordinates in screen device space.
          - gl_FrontFacing: True if the primitive is NOT back facing.
          - gl_PointCoord: 2d coordinates in screen device space.
          - varying: interpolated values from `Shader.interpolate`; these are
            generated from `Shader.vertex`.
          - extra: ShaderExtraInputT, same as `extra` in `Shader.vertex`.

        Return: PerFragment for depth test and further mixing process.
          - gl_FragDepth: if not set (remains None), gl_FragCoord[2] will be
            used later.
          - varying: defined by user, passed as `varying` in `Shader.mix`.
            **NOTE** the return type must be the same type as `values` in
            `Shader.interpolate`, as that will be used as the dummy value for
            this return value, when `PerFragment` suggests `keeps` is False.

        Reference:
          - [Fragment Shader/Defined Inputs](https://www.khronos.org/opengl/wiki/Fragment_Shader/Defined_Inputs)
          - [Fragment Shader#Outputs](https://www.khronos.org/opengl/wiki/Fragment_Shader#Outputs)
        """
        return PerFragment(use_default_depth=TRUE_ARRAY), varying

    @staticmethod
    @jaxtyped
    @jax.jit
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: VaryingT,
    ) -> tuple[MixerOutput, MixedExtraT]:
        """Override this to customise the mixing behaviour per fragment over
            different primitives (triangles).

        Use this to implement the `blending` behaviour if needed.

        For the default behaviour, the values from fragment with maximum
        `gl_FragDepth` value AND `keeps` being True will be used as the output.
        In the default implementation, if no fragment has `keeps` being True,
        then mixed value will be the first fragment's value for both
        `gl_FragDepth` and `extra`.

        Returns: Built-in MixerOutput and user-defined extras.
          - MixerOutput:
            - keep: bool, whether uses this value to set the corresponding
              pixel in the buffers
            - zbuffer, the value used to update the zbuffer
          - User-defined outputs, must be a tuple (can be NamedTuple)
            Each field must be defined as same order as in the class `Buffers`.
            The values will be directly set to the `Buffers` **in the same
            order of the given definition** as if a usual `tuple`, but not
            based on field name.

        Reference:
          - [Blending](https://www.khronos.org/opengl/wiki/Blending)
        """

        def has_kept_fragment() -> Integer[Array, ""]:
            depths: Float[Array, "primitives"]
            depths = jnp.where(keeps, gl_FragDepth, jnp.inf)
            assert isinstance(depths, Float[Array, "primitives"])

            idx: Integer[Array, ""] = jnp.argmin(depths)

            return idx

        has_valid_fragment = jnp.any(keeps)

        idx: Integer[Array, ""] = lax.cond(
            has_valid_fragment,
            has_kept_fragment,
            lambda: jnp.array(0),
        )
        depth: Float[Array, ""] = gl_FragDepth[idx]
        assert isinstance(depth, Float[Array, ""])

        return (
            MixerOutput(keep=has_valid_fragment, zbuffer=depth),
            tree_map(lambda x: x[idx], extra),
        )
