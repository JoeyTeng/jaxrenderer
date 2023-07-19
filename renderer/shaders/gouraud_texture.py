from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from functools import partial
from typing import NamedTuple, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from .._backport import Tuple
from .._meta_utils import add_tracing_name
from .._meta_utils import typed_jit as jit
from ..geometry import Camera, normalise, to_homogeneous
from ..shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from ..types import BoolV, Colour, FloatV, LightSource, Texture, Vec2f, Vec3f, Vec4f

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]


class GouraudTextureExtraInput(NamedTuple):
    """Extra input for Gouraud Shader.

    Attributes:
      - position: in world space, of each vertex.
      - normal: in world space, of each vertex.
      - uv: in texture space, of each vertex.
      - light: parallel light source, shared by all vertices.
      - texture: texture, shared by all vertices.
    """

    position: Float[Array, "vertices 3"]  # in world space
    normal: Float[Array, "vertices 3"]  # in world space
    uv: Float[Array, "vertices 2"]  # in texture space
    light: LightSource
    texture: Texture


class GouraudTextureExtraFragmentData(NamedTuple):
    colour: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        [0.0, 0.0, 0.0]
    )
    """light colour when passing from vertex shader to fragment shader;
    canvas colour when passing from fragment shader to mixer."""
    uv: Vec2f = jnp.zeros(2)  # pyright: ignore[reportUnknownMemberType]


class GouraudTextureExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""

    canvas: Colour


class GouraudTextureShader(
    Shader[
        GouraudTextureExtraInput,
        GouraudTextureExtraFragmentData,
        GouraudTextureExtraMixerOutput,
    ]
):
    """Gouraud Shading with simple parallel lighting and texture."""

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: GouraudTextureExtraInput,
    ) -> Tuple[PerVertex, GouraudTextureExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        # assume normal here is in world space. If it is in model space, it
        # must be transformed by the inverse transpose of the model matrix.
        # Ref: https://github.com/ssloy/tinyrenderer/wiki/Lesson-5:-Moving-the-camera#transformation-of-normal-vectors
        normal: Vec3f = normalise(extra.normal[gl_VertexID])
        intensity: FloatV = cast(
            FloatV,
            jnp.dot(
                normal,
                normalise(extra.light.direction),
            ),
        )
        assert isinstance(intensity, FloatV)

        light_colour: Colour
        light_colour = extra.light.colour * intensity

        return (
            PerVertex(gl_Position=gl_Position),
            GouraudTextureExtraFragmentData(
                colour=light_colour,
                uv=extra.uv[gl_VertexID],
            ),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: GouraudTextureExtraFragmentData,
        extra: GouraudTextureExtraInput,
    ) -> Tuple[PerFragment, GouraudTextureExtraFragmentData]:
        built_in: PerFragment = Shader.fragment(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            varying,
            extra,
        )[0]
        assert isinstance(built_in, PerFragment)

        # repeat texture
        uv = lax.floor(varying.uv).astype(int)  # pyright: ignore
        uv = uv % jnp.asarray(extra.texture.shape[:2])  # pyright: ignore
        texture_colour: Colour = extra.texture[uv[0], uv[1]]
        light_colour: Colour = varying.colour

        return (
            PerFragment(
                keeps=jnp.array(  # pyright: ignore[reportUnknownMemberType]
                    (
                        built_in.keeps,
                        gl_FrontFacing,
                        (light_colour >= 0).all(),  # pyright: ignore
                    )
                ).all(),
                use_default_depth=built_in.use_default_depth,
            ),
            GouraudTextureExtraFragmentData(
                colour=texture_colour * light_colour,
                uv=varying.uv,
            ),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: GouraudTextureExtraFragmentData,
    ) -> Tuple[MixerOutput, GouraudTextureExtraMixerOutput]:
        mixer_output: MixerOutput
        extra_output: GouraudTextureExtraFragmentData
        mixer_output, extra_output = cast(
            Tuple[MixerOutput, GouraudTextureExtraFragmentData],
            Shader.mix(gl_FragDepth, keeps, extra),
        )
        assert isinstance(mixer_output, MixerOutput)
        assert isinstance(extra_output, GouraudTextureExtraFragmentData)

        return (
            mixer_output,
            GouraudTextureExtraMixerOutput(canvas=extra_output.colour),
        )
