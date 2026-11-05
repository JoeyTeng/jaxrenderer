from dataclasses import replace
from functools import partial
from typing import NamedTuple, override

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Bool,
    Float,
)

from .._meta_utils import add_tracing_name, is_array_type
from .._meta_utils import typed_jaxtyped as jaxtyped
from .._meta_utils import typed_jit as jit
from ..geometry import to_homogeneous
from ..shader import Shader
from ..shader_types import (
    Fragment,
    FragmentInput,
    InterpolationInput,
    PerFragment,
    PerVertex,
    PrimitiveInput,
    VertexInput,
)
from ..types import FALSE_ARRAY, TRUE_ARRAY, BoolV, Vec4f


class DepthExtraInput(NamedTuple):
    """Extra input for Depth Shader.

    Attributes:
      - position: in world space, of each vertex.
    """

    position: Float[Array, "vertices 3"]  # in world space
    double_sided: Bool[Array, "vertices"] | BoolV = FALSE_ARRAY


class DepthExtraFragmentData(NamedTuple):
    double_sided: BoolV = FALSE_ARRAY


class DepthExtraMixerOutput(NamedTuple):
    pass


class DepthShader(
    Shader[DepthExtraInput, DepthExtraFragmentData, DepthExtraMixerOutput]
):
    """Depth Shading."""

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def vertex(
        inputs: VertexInput[DepthExtraInput],
    ) -> tuple[PerVertex, DepthExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(inputs.extra.position[inputs.gl_VertexID])
        gl_Position: Vec4f = inputs.camera.to_clip(position)
        assert is_array_type(gl_Position, Vec4f)

        return (
            PerVertex(gl_Position=gl_Position),
            DepthExtraFragmentData(
                double_sided=jnp.broadcast_to(
                    inputs.extra.double_sided,
                    inputs.extra.position.shape[:1],
                )[inputs.gl_VertexID],
            ),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def interpolate(
        inputs: InterpolationInput[DepthExtraFragmentData],
    ) -> DepthExtraFragmentData:
        return inputs.values._replace(double_sided=inputs.values.double_sided[0])

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def primitive_chooser(
        inputs: PrimitiveInput[DepthExtraFragmentData],
    ) -> tuple[
        Float[Array, "kept_primitives 4"],
        Bool[Array, "kept_primitives"],
        Float[Array, "kept_primitives 2"],
        Bool[Array, "kept_primitives"],
        DepthExtraFragmentData,
        Float[Array, "kept_primitives 3"],
        Float[Array, "kept_primitives 3"],
    ]:
        return Shader[
            DepthExtraInput,
            DepthExtraFragmentData,
            DepthExtraMixerOutput,
        ].primitive_chooser(
            inputs=replace(
                inputs,
                double_sided=(inputs.double_sided | inputs.values.double_sided[:, 0]),
            )
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def fragment(
        inputs: FragmentInput[DepthExtraInput, DepthExtraFragmentData],
    ) -> tuple[PerFragment, Fragment[DepthExtraMixerOutput]]:
        return (
            PerFragment(
                keeps=inputs.gl_FrontFacing | inputs.varying.double_sided,
                use_default_depth=TRUE_ARRAY,
            ),
            Fragment(targets=DepthExtraMixerOutput()),
        )
