from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from .._backport import Tuple
from .._meta_utils import add_tracing_name
from .._meta_utils import typed_jit as jit
from ..geometry import Camera, normalise, to_homogeneous
from ..shader import ID, PerFragment, PerVertex, Shader
from ..types import BoolV, Colour, FloatV, LightSource, Vec2f, Vec3f, Vec4f

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]


class GouraudExtraInput(NamedTuple):
    """Extra input for Gouraud Shader.

    Attributes:
      - position: in world space, of each vertex.
      - colour: of each vertex
      - normal: in world space, of each vertex.
      - light: parallel light source, shared by all vertices.
    """

    position: Float[Array, "vertices 3"]  # in world space
    colour: Float[Array, "vertices 3"]
    normal: Float[Array, "vertices 3"]  # in world space
    light: LightSource


class GouraudExtraFragmentData(NamedTuple):
    colour: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        [0.0, 0.0, 0.0]
    )


class GouraudExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""

    canvas: Colour


class GouraudShader(
    Shader[GouraudExtraInput, GouraudExtraFragmentData, GouraudExtraMixerOutput]
):
    """Gouraud Shading with simple parallel lighting."""

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: GouraudExtraInput,
    ) -> Tuple[PerVertex, GouraudExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        # assume normal here is in world space. If it is in model space, it
        # must be transformed by the inverse transpose of the model matrix.
        # Ref: https://github.com/ssloy/tinyrenderer/wiki/Lesson-5:-Moving-the-camera#transformation-of-normal-vectors
        normal: Vec3f = normalise(extra.normal[gl_VertexID])
        intensity = cast(
            FloatV,
            jnp.dot(
                normal,
                normalise(extra.light.direction),
            ),
        )
        assert isinstance(intensity, FloatV)

        colour: Colour
        colour = extra.colour[gl_VertexID] * extra.light.colour * intensity

        return (
            PerVertex(gl_Position=gl_Position),
            GouraudExtraFragmentData(colour=colour),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: GouraudExtraFragmentData,
        extra: GouraudExtraInput,
    ) -> Tuple[PerFragment, GouraudExtraFragmentData]:
        built_in: PerFragment = Shader.fragment(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            varying,
            extra,
        )[0]
        assert isinstance(built_in, PerFragment)

        return (
            PerFragment(
                keeps=jnp.array(  # pyright: ignore[reportUnknownMemberType]
                    (
                        built_in.keeps,
                        gl_FrontFacing,
                        jnp.all(  # pyright: ignore[reportUnknownMemberType]
                            varying.colour >= 0
                        ),
                    )
                ).all(),
                use_default_depth=built_in.use_default_depth,
            ),
            varying,
        )
