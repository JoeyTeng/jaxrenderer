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
from ..geometry import Camera, to_homogeneous
from ..shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from ..types import FALSE_ARRAY, TRUE_ARRAY, BoolV, Vec2f, Vec3f, Vec4f


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
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: DepthExtraInput,
    ) -> tuple[PerVertex, DepthExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert is_array_type(gl_Position, Vec4f)

        return (
            PerVertex(gl_Position=gl_Position),
            DepthExtraFragmentData(
                double_sided=jnp.broadcast_to(
                    extra.double_sided,
                    extra.position.shape[:1],
                )[gl_VertexID],
            ),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def interpolate(
        values: DepthExtraFragmentData,
        barycentric_screen: Vec3f,
        barycentric_clip: Vec3f,
    ) -> DepthExtraFragmentData:
        return values._replace(double_sided=values.double_sided[0])

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def primitive_chooser(
        gl_FragCoord: Float[Array, "primitives 4"],
        gl_FrontFacing: Bool[Array, "primitives"],
        gl_PointCoord: Float[Array, "primitives 2"],
        keeps: Bool[Array, "primitives"],
        values: DepthExtraFragmentData,
        barycentric_screen: Float[Array, "primitives 3"],
        barycentric_clip: Float[Array, "primitives 3"],
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
        ]._primitive_chooser(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            keeps,
            values,
            barycentric_screen,
            barycentric_clip,
            double_sided=values.double_sided[:, 0],
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: DepthExtraFragmentData,
        extra: DepthExtraInput,
    ) -> tuple[PerFragment, DepthExtraFragmentData]:
        return (
            PerFragment(
                keeps=gl_FrontFacing | varying.double_sided,
                use_default_depth=TRUE_ARRAY,
            ),
            varying,
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def mix(
        gl_FragDepth: Float[Array, "kept_primitives"],
        keeps: Bool[Array, "kept_primitives"],
        extra: DepthExtraFragmentData,
    ) -> tuple[MixerOutput, DepthExtraMixerOutput]:
        mixed = Shader[
            DepthExtraInput,
            DepthExtraFragmentData,
            DepthExtraMixerOutput,
        ].mix(gl_FragDepth, keeps, extra)
        return mixed[0], DepthExtraMixerOutput()
