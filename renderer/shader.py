from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from abc import ABC, abstractmethod
from functools import partial
from typing import Generic, NamedTuple, TypeVar, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map
from jaxtyping import Array, Bool, Float, Shaped
from jaxtyping import PyTree  # pyright: ignore[reportUnknownVariableType]
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from ._backport import Tuple, TypeAlias
from ._meta_utils import add_tracing_name
from ._meta_utils import typed_jit as jit
from .geometry import Camera, Interpolation, interpolate
from .types import (
    FALSE_ARRAY,
    INF_ARRAY,
    TRUE_ARRAY,
    BoolV,
    FloatV,
    IntV,
    Vec2f,
    Vec3f,
    Vec4f,
)

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]

ID: TypeAlias = IntV

ShaderExtraInputT = TypeVar(
    "ShaderExtraInputT",
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
    # gl_PointSize: FloatV


class PerFragment(NamedTuple):
    """Built-in Output from Fragment Shader.

    If use_default_depth is True (default False), gl_FragCoord[2] will be used
    later by default.
    """

    gl_FragDepth: FloatV = INF_ARRAY
    # not discard
    keeps: BoolV = TRUE_ARRAY
    use_default_depth: BoolV = FALSE_ARRAY


VaryingT = TypeVar(
    "VaryingT",
    bound=Tuple[Shaped[Array, "..."], ...],
)
"""The user-defined input and second (extra) output of fragment shader."""

MixedExtraT = TypeVar(
    "MixedExtraT",
    bound=Tuple[Shaped[Array, "..."], ...],
)
"""The user-defined second (extra) output of mix shader."""


class MixerOutput(NamedTuple):
    """Built-in output from `Shader.mix`.

    keep: bool, whether the output should be used to update buffers
    zbuffer: store depth value, and the result is used to set zbuffer.
    """

    keep: BoolV
    zbuffer: FloatV


class Shader(ABC, Generic[ShaderExtraInputT, VaryingT, MixedExtraT]):
    """Base class for customised shader.

    Since JAX is pure functional (stateless), the state will be passed by
    returned values (the second return value in each function) in each process.
    """

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @abstractmethod
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: ShaderExtraInputT,
    ) -> Tuple[PerVertex, VaryingT]:
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
    @partial(jit, inline=True)
    @add_tracing_name
    def primitive_chooser(
        gl_FragCoord: Float[Array, "primitives 4"],
        gl_FrontFacing: Bool[Array, "primitives"],
        gl_PointCoord: Float[Array, "primitives 2"],
        keeps: Bool[Array, "primitives"],
        values: VaryingT,
        barycentric_screen: Float[Array, "primitives 3"],
        barycentric_clip: Float[Array, "primitives 3"],
    ) -> Tuple[  #
        Float[Array, "kept_primitives 4"],  # gl_FragCoord
        Bool[Array, "kept_primitives"],  # gl_FrontFacing
        Float[Array, "kept_primitives 2"],  # gl_PointCoord
        Bool[Array, "kept_primitives"],  # keeps
        VaryingT,  # values
        Float[Array, "kept_primitives 3"],  # barycentric_screen
        Float[Array, "kept_primitives 3"],  # barycentric_clip
    ]:
        """Override this to customise the primitive choosing stage.

        The default implementation is to only keep the primitive with minimum
        `gl_FragCoord[2]` and `gl_FrontFacing` and `keeps` (interpolated `z`
        value in window space is minimum), i.e., the closest primitive that is
        kept and is not back-facing.

        Parameters:
          - gl_FragCoord: batch of coordinates in screen space. (x, y, z, 1/w).
          - gl_FrontFacing: batch of bool, True if the primitive is NOT back
            facing.
          - gl_PointCoord: batch of 2d coordinates in screen space. Not supported for now.
          - keeps: batch of bool, whether the primitive is kept. This is used
            to filter out the primitives that are not visible, or with garbage
            values.

          The parameters below are batched values over primitives, with each
          value same as the input given to `Shader.interpolate`

          - values: values at the vertices of the triangle, with axis 0 being
            the batch axis. It is expected to be a tuple of multiple batched
            values.
          - barycentric_screen: barycentric coordinates in screen space of the
            point to interpolate
          - barycentric_clip: barycentric coordinates in clip space of the
            point to interpolate

        Return:
          tuple of values from kept primitives, in same order and structure of
          the input parameters. The returned fields must be batched.
        """
        depths: Float[Array, "primitives"]
        depths = jnp.where(  # pyright: ignore[reportUnknownMemberType]
            keeps & gl_FrontFacing,
            gl_FragCoord[:, 2],
            jnp.inf,
        )
        assert isinstance(depths, Float[Array, "primitives"])

        # when all keeps are false, all depths will be inf, and there will
        # still be a valid idx generated, as promised by argmin.
        idx: IntV = jnp.argmin(depths)  # pyright: ignore[reportUnknownMemberType]
        assert isinstance(idx, IntV)

        _get = partial(
            # use `dynamic_slice` instead of `slice` according to benchmark
            # https://colab.research.google.com/drive/1idBbgEDbxI6wi5kzlHF6kzWryoFSm8-p#scrollTo=-bHrz3kZ5A0p
            lax.dynamic_slice_in_dim,  # pyright: ignore[reportUnknownMemberType]
            start_index=idx,
            slice_size=1,
            axis=0,
        )

        _gl_FragCoord: Float[Array, "kept_primitives 4"] = _get(gl_FragCoord)
        assert isinstance(_gl_FragCoord, Float[Array, "kept_primitives 4"])
        _gl_FrontFacing: Bool[Array, "kept_primitives"] = _get(gl_FrontFacing)
        assert isinstance(_gl_FrontFacing, Bool[Array, "kept_primitives"])
        _gl_PointCoord: Float[Array, "kept_primitives 2"] = _get(gl_PointCoord)
        assert isinstance(_gl_PointCoord, Float[Array, "kept_primitives 2"])
        _keeps: Bool[Array, "kept_primitives"] = _get(keeps)
        assert isinstance(_keeps, Bool[Array, "kept_primitives"])
        _values: VaryingT = tree_map(_get, values)
        _screen: Float[Array, "kept_primitives 3"] = _get(barycentric_screen)
        assert isinstance(_screen, Float[Array, "kept_primitives 3"])
        _clip: Float[Array, "kept_primitives 3"] = _get(barycentric_clip)
        assert isinstance(_clip, Float[Array, "kept_primitives 3"])

        return (
            _gl_FragCoord,
            _gl_FrontFacing,
            _gl_PointCoord,
            _keeps,
            _values,
            _screen,
            _clip,
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
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
    @partial(jit, inline=True)
    @add_tracing_name
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: VaryingT,
        extra: ShaderExtraInputT,
    ) -> Tuple[PerFragment, VaryingT]:
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
    @partial(jit, inline=True)
    @add_tracing_name
    def mix(
        gl_FragDepth: Float[Array, "kept_primitives"],
        keeps: Bool[Array, "kept_primitives"],
        extra: VaryingT,
    ) -> Tuple[MixerOutput, Union[VaryingT, MixedExtraT]]:
        """Override this to customise the mixing behaviour per fragment over
            different primitives (triangles).

        Use this to implement the `blending` behaviour if needed.

        For the default behaviour, the values from fragment with maximum
        `gl_FragDepth` value AND `keeps` being True will be used as the output.
        In the default implementation, if no fragment has `keeps` being True,
        then mixed value will be the an arbitrary fragment's value for both
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
            This type must be `MixedExtraT` when override; `VaryingT` is used for the
            default implementation here simply due to the limitation that we cannot
            know how to create a MixedExtraT from a VaryingT at this time.
            TODO: figure out a better way to define these generics.

        Reference:
          - [Blending](https://www.khronos.org/opengl/wiki/Blending)
        """

        depths: Float[Array, "primitives"]
        depths = jnp.where(  # pyright: ignore[reportUnknownMemberType]
            keeps,
            gl_FragDepth,
            jnp.inf,
        )
        assert isinstance(depths, Float[Array, "primitives"])

        # when all keeps are false, all depths will be inf, and there will
        # still be a valid idx generated, as promised by argmin.
        idx: IntV
        idx = jnp.argmin(depths)  # pyright: ignore[reportUnknownMemberType]
        assert isinstance(idx, IntV)

        keep: BoolV = keeps[idx]
        assert isinstance(keep, BoolV)
        depth: FloatV = depths[idx]
        assert isinstance(depth, FloatV)

        return (
            MixerOutput(keep=keep, zbuffer=depth),
            tree_map(lambda x: x[idx], extra),
        )
