from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import ClassVar, Generic, NamedTuple, TypeAlias, TypeVar

from jax import Array
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map
from jaxtyping import (
    Bool,
    Float,
    PyTree,  # pyright: ignore[reportUnknownVariableType]
    Shaped,
)

from ._meta_utils import add_tracing_name, is_array_type, typed_array
from ._meta_utils import typed_jaxtyped as jaxtyped
from ._meta_utils import typed_jit as jit
from .geometry import Camera, Interpolation, interpolate
from .types import (
    FALSE_ARRAY,
    INF_ARRAY,
    TRUE_ARRAY,
    BoolV,
    FloatV,
    IntV,
    PixelState,
    Vec2f,
    Vec3f,
    Vec4f,
)

ID: TypeAlias = IntV
_ONE_FLOAT_ARRAY: FloatV = typed_array(1.0)

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
    bound=tuple[Shaped[Array, "..."], ...],
)
"""Interpolated vertex output consumed by the fragment shader."""

MixedExtraT = TypeVar(
    "MixedExtraT",
    bound=tuple[Shaped[Array, "..."], ...],
)
"""User-defined fragment target layout shared by fragment and mix stages."""


class Fragment(NamedTuple, Generic[MixedExtraT]):
    """Fragment targets and their opacity."""

    targets: MixedExtraT
    alpha: FloatV = _ONE_FLOAT_ARRAY


class FragmentCollection(Enum):
    """Primitive candidates retained for fragment processing."""

    NEAREST = "nearest"
    ALL = "all"


class MixerOutput(NamedTuple):
    """Independent target and depth writes produced by mix."""

    keep: BoolV
    zbuffer: FloatV
    write_depth: BoolV


class Shader(ABC, Generic[ShaderExtraInputT, VaryingT, MixedExtraT]):
    """Base class for customised shader.

    Since JAX is pure functional (stateless), the state will be passed by
    returned values (the second return value in each function) in each process.
    """

    fragment_collection: ClassVar[FragmentCollection] = FragmentCollection.NEAREST

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
    @partial(jit, static_argnames=("collection",), inline=True)
    @add_tracing_name
    def primitive_chooser(
        gl_FragCoord: Float[Array, "primitives 4"],
        gl_FrontFacing: Bool[Array, "primitives"],
        gl_PointCoord: Float[Array, "primitives 2"],
        keeps: Bool[Array, "primitives"],
        values: VaryingT,
        barycentric_screen: Float[Array, "primitives 3"],
        barycentric_clip: Float[Array, "primitives 3"],
        *,
        double_sided: Bool[Array, "primitives"] | BoolV = FALSE_ARRAY,
        collection: FragmentCollection = FragmentCollection.NEAREST,
    ) -> tuple[  #
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
        kept and is front-facing. The optional `double_sided` mask also allows
        marked back-facing primitives to compete.

        Parameters:
          - gl_FragCoord: batch of coordinates in screen space. (x, y, z, 1/w).
          - gl_FrontFacing: batch of bool, True if the primitive is NOT back
            facing.
          - gl_PointCoord: batch of 2d coordinates in screen space. Not supported for now.
          - keeps: batch of bool, whether the primitive is kept. This is used
            to filter out the primitives that are not visible, or with garbage
            values.
          - double_sided: optional scalar or per-primitive mask that allows
            back-facing primitives to compete for the closest visible depth.
            This affects chooser eligibility only; it does not change
            `gl_FrontFacing` or the supplied normals.
          The parameters below are batched values over primitives, with each
          value same as the input given to `Shader.interpolate`

          - values: values at the vertices of the triangle, with axis 0 being
            the batch axis. It is expected to be a tuple of multiple batched
            values.
          - barycentric_screen: barycentric coordinates in screen space of the
            point to interpolate
          - barycentric_clip: barycentric coordinates in clip space of the
            point to interpolate

        Candidate collection is selected statically by collection.

        Return:
          tuple of values from kept primitives, in same order and structure of
          the input parameters. The returned fields must be batched.
        """
        double_sided_per_primitive: Bool[Array, "primitives"] = jnp.broadcast_to(
            double_sided,
            gl_FrontFacing.shape,
        )
        assert is_array_type(
            double_sided_per_primitive,
            Bool[Array, "primitives"],
        )

        eligible: Bool[Array, "primitives"] = keeps & (
            gl_FrontFacing | double_sided_per_primitive
        )
        assert is_array_type(eligible, Bool[Array, "primitives"])

        if collection is FragmentCollection.ALL or gl_FragCoord.shape[0] == 0:
            return (
                gl_FragCoord,
                gl_FrontFacing,
                gl_PointCoord,
                eligible,
                values,
                barycentric_screen,
                barycentric_clip,
            )

        depths: Float[Array, "primitives"] = jnp.where(
            eligible,
            gl_FragCoord[:, 2],
            jnp.inf,
        )
        assert is_array_type(depths, Float[Array, "primitives"])

        # An all-false mask still selects a slot; its eligibility stays false.
        idx: IntV = jnp.argmin(depths)
        assert is_array_type(idx, IntV)

        _get = partial(
            # use `dynamic_slice` instead of `slice` according to benchmark
            # https://colab.research.google.com/drive/1idBbgEDbxI6wi5kzlHF6kzWryoFSm8-p#scrollTo=-bHrz3kZ5A0p
            lax.dynamic_slice_in_dim,
            start_index=idx,
            slice_size=1,
            axis=0,
        )

        _gl_FragCoord: Float[Array, "kept_primitives 4"] = _get(gl_FragCoord)
        assert is_array_type(_gl_FragCoord, Float[Array, "kept_primitives 4"])
        _gl_FrontFacing: Bool[Array, "kept_primitives"] = _get(gl_FrontFacing)
        assert is_array_type(_gl_FrontFacing, Bool[Array, "kept_primitives"])
        _gl_PointCoord: Float[Array, "kept_primitives 2"] = _get(gl_PointCoord)
        assert is_array_type(_gl_PointCoord, Float[Array, "kept_primitives 2"])
        _keeps: Bool[Array, "kept_primitives"] = _get(eligible)
        assert is_array_type(_keeps, Bool[Array, "kept_primitives"])
        _values: VaryingT = tree_map(_get, values)
        _screen: Float[Array, "kept_primitives 3"] = _get(barycentric_screen)
        assert is_array_type(_screen, Float[Array, "kept_primitives 3"])
        _clip: Float[Array, "kept_primitives 3"] = _get(barycentric_clip)
        assert is_array_type(_clip, Float[Array, "kept_primitives 3"])

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
    @abstractmethod
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: VaryingT,
        extra: ShaderExtraInputT,
    ) -> tuple[PerFragment, Fragment[MixedExtraT]]:
        """Convert interpolated values into fragment targets and opacity.

        Subclasses define this conversion. Use PassthroughShader when the
        interpolated values already have the target layout.
        """
        ...

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def mix(
        gl_FragDepth: Float[Array, "kept_primitives"],
        keeps: Bool[Array, "kept_primitives"],
        extra: Fragment[MixedExtraT],
        *,
        destination: PixelState[MixedExtraT],
    ) -> tuple[MixerOutput, MixedExtraT]:
        """Select the nearest kept fragment, preserving the destination otherwise."""
        if gl_FragDepth.shape[0] == 0:
            return (
                MixerOutput(
                    keep=FALSE_ARRAY,
                    zbuffer=destination.zbuffer,
                    write_depth=FALSE_ARRAY,
                ),
                destination.targets,
            )

        depths: Float[Array, "kept_primitives"] = jnp.where(
            keeps,
            gl_FragDepth,
            jnp.inf,
        )
        assert is_array_type(depths, Float[Array, "kept_primitives"])
        idx: IntV = jnp.argmin(depths)
        assert is_array_type(idx, IntV)

        keep: BoolV = keeps[idx]
        assert is_array_type(keep, BoolV)
        depth: FloatV = depths[idx]
        assert is_array_type(depth, FloatV)

        def select(value: Array) -> Array:
            return value[idx]

        selected_targets: MixedExtraT = tree_map(select, extra.targets)

        return (
            MixerOutput(keep=keep, zbuffer=depth, write_depth=keep),
            selected_targets,
        )


class PassthroughShader(
    Shader[ShaderExtraInputT, VaryingT, VaryingT],
    Generic[ShaderExtraInputT, VaryingT],
):
    """Shader base that uses interpolated values as fragment targets."""

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
    ) -> tuple[PerFragment, Fragment[VaryingT]]:
        """Keep interpolated target values and use fully opaque coverage."""
        del gl_FragCoord, gl_FrontFacing, gl_PointCoord, extra
        return (
            PerFragment(use_default_depth=TRUE_ARRAY),
            Fragment(targets=varying),
        )
