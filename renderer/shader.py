from abc import ABC, abstractmethod
from functools import partial
from typing import ClassVar, Generic

from jax import Array
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map
from jaxtyping import (
    Bool,
    Float,
)

from ._meta_utils import add_tracing_name, is_array_type
from ._meta_utils import typed_jaxtyped as jaxtyped
from ._meta_utils import typed_jit as jit
from .geometry import Interpolation, interpolate
from .shader_types import (
    ID,
    Fragment,
    FragmentCollection,
    FragmentInput,
    InterpolationInput,
    MixInput,
    MixedExtraT,
    MixerOutput,
    PerFragment,
    PerVertex,
    PrimitiveInput,
    ShaderExtraInputT,
    VaryingT,
    VertexInput,
)
from .types import (
    FALSE_ARRAY,
    TRUE_ARRAY,
    BoolV,
    FloatV,
    IntV,
)

__all__ = [
    "Fragment",
    "FragmentCollection",
    "FragmentInput",
    "ID",
    "InterpolationInput",
    "MixInput",
    "MixedExtraT",
    "MixerOutput",
    "PassthroughShader",
    "PerFragment",
    "PerVertex",
    "PrimitiveInput",
    "Shader",
    "ShaderExtraInputT",
    "VaryingT",
    "VertexInput",
]


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
    def vertex(inputs: VertexInput[ShaderExtraInputT]) -> tuple[PerVertex, VaryingT]:
        """Override this to implement the vertex shader as defined by OpenGL.

        The input fields retain their OpenGL meanings. `inputs.camera` contains
        view, viewport, and projection matrices. `inputs.extra` is shared by
        all invocations and is not split over the batch axis.

        Noticed that no internal state will be tracked, thus if there is any
        value to be passed to downstream process, it must be returned as the
        output `VaryingT` of this function.

        `inputs.gl_VertexID` is the effective index used to fetch the vertex;
        `inputs.gl_InstanceID` is the current instance index, or zero when
        instancing is unused. Return `PerVertex` and the values to interpolate
        for downstream stages. `gl_Position` is in clip space.

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
        inputs: PrimitiveInput[VaryingT],
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
        value in window space is minimum), i.e. the closest eligible primitive.
        `inputs.double_sided` lets selected back-facing primitives compete; it
        does not change `gl_FrontFacing` or supplied normals. Collection mode
        comes from the static `inputs.collection` field.

        `inputs.gl_FragCoord` contains screen coordinates `(x, y, z, 1/w)`;
        the facing, point-coordinate, keep, varying, and barycentric fields are
        batched over primitives. The returned tuple contains those values for
        retained primitives, in the same order and structure, with batched
        axes.

        Return:
          tuple of values from kept primitives, in same order and structure of
          the input parameters. The returned fields must be batched.
        """
        double_sided_per_primitive: Bool[Array, "primitives"] = jnp.broadcast_to(
            inputs.double_sided,
            inputs.gl_FrontFacing.shape,
        )
        assert is_array_type(
            double_sided_per_primitive,
            Bool[Array, "primitives"],
        )

        eligible: Bool[Array, "primitives"] = inputs.keeps & (
            inputs.gl_FrontFacing | double_sided_per_primitive
        )
        assert is_array_type(eligible, Bool[Array, "primitives"])

        if (
            inputs.collection is FragmentCollection.ALL
            or inputs.gl_FragCoord.shape[0] == 0
        ):
            return (
                inputs.gl_FragCoord,
                inputs.gl_FrontFacing,
                inputs.gl_PointCoord,
                eligible,
                inputs.values,
                inputs.barycentric_screen,
                inputs.barycentric_clip,
            )

        depths: Float[Array, "primitives"] = jnp.where(
            eligible,
            inputs.gl_FragCoord[:, 2],
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

        _gl_FragCoord: Float[Array, "kept_primitives 4"] = _get(inputs.gl_FragCoord)
        assert is_array_type(_gl_FragCoord, Float[Array, "kept_primitives 4"])
        _gl_FrontFacing: Bool[Array, "kept_primitives"] = _get(inputs.gl_FrontFacing)
        assert is_array_type(_gl_FrontFacing, Bool[Array, "kept_primitives"])
        _gl_PointCoord: Float[Array, "kept_primitives 2"] = _get(inputs.gl_PointCoord)
        assert is_array_type(_gl_PointCoord, Float[Array, "kept_primitives 2"])
        _keeps: Bool[Array, "kept_primitives"] = _get(eligible)
        assert is_array_type(_keeps, Bool[Array, "kept_primitives"])
        _values: VaryingT = tree_map(_get, inputs.values)
        _screen: Float[Array, "kept_primitives 3"] = _get(inputs.barycentric_screen)
        assert is_array_type(_screen, Float[Array, "kept_primitives 3"])
        _clip: Float[Array, "kept_primitives 3"] = _get(inputs.barycentric_clip)
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
    def interpolate(inputs: InterpolationInput[VaryingT]) -> VaryingT:
        """Override this to customise the interpolation of user-defined inputs.

        The default implementation applies GLSL smooth (perspective)
        interpolation to `inputs.values`, using its screen- and clip-space
        barycentric coordinates. The returned structure matches `values`.
        """
        varying: VaryingT = tree_map(
            Partial(
                interpolate,
                barycentric_screen=inputs.barycentric_screen,
                barycentric_clip=inputs.barycentric_clip,
                mode=Interpolation.SMOOTH,
            ),
            inputs.values,
        )

        return varying

    @staticmethod
    @abstractmethod
    def fragment(
        inputs: FragmentInput[ShaderExtraInputT, VaryingT],
    ) -> tuple[PerFragment, Fragment[MixedExtraT]]:
        """Convert interpolated values into fragment targets and opacity.

        `inputs` contains the GLSL fragment coordinates, facing and point
        coordinates, interpolated `varying`, and shared user `extra`.
        Subclasses define this conversion. Use PassthroughShader when the
        interpolated values already have the target layout.
        """
        ...

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def mix(inputs: MixInput[MixedExtraT]) -> tuple[MixerOutput, MixedExtraT]:
        """Select the nearest kept fragment, preserving the destination otherwise."""
        if inputs.gl_FragDepth.shape[0] == 0:
            return (
                MixerOutput(
                    keep=FALSE_ARRAY,
                    zbuffer=inputs.destination.zbuffer,
                    write_depth=FALSE_ARRAY,
                ),
                inputs.destination.targets,
            )

        depths: Float[Array, "kept_primitives"] = jnp.where(
            inputs.keeps,
            inputs.gl_FragDepth,
            jnp.inf,
        )
        assert is_array_type(depths, Float[Array, "kept_primitives"])
        idx: IntV = jnp.argmin(depths)
        assert is_array_type(idx, IntV)

        keep: BoolV = inputs.keeps[idx]
        assert is_array_type(keep, BoolV)
        depth: FloatV = depths[idx]
        assert is_array_type(depth, FloatV)

        def select(value: Array) -> Array:
            return value[idx]

        selected_targets: MixedExtraT = tree_map(select, inputs.extra.targets)

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
        inputs: FragmentInput[ShaderExtraInputT, VaryingT],
    ) -> tuple[PerFragment, Fragment[VaryingT]]:
        """Keep interpolated target values and use fully opaque coverage."""
        return (
            PerFragment(use_default_depth=TRUE_ARRAY),
            Fragment(targets=inputs.varying),
        )
