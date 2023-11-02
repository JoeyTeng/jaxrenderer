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
from ..types import BoolV, Colour, LightSource, Texture, Vec2f, Vec3f, Vec4f

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]


class PhongTextureExtraInput(NamedTuple):
    """Extra input for Phong Shader.

    Attributes:
      - position: in world space, of each vertex.
      - normal: in world space, of each vertex.
      - uv: in texture space, of each vertex.
      - light: parallel `headlight` light source, shared by all vertices.
        It is in the eye/view space.
      - texture: texture, shared by all vertices.
    """

    position: Float[Array, "vertices 3"]  # in world space
    normal: Float[Array, "vertices 3"]  # in world space
    uv: Float[Array, "vertices 2"]  # in texture space
    light: LightSource
    texture: Texture


class PhongTextureExtraFragmentData(NamedTuple):
    """From vertex shader to fragment shader, and from fragment shader to mixer.

    Attributes:
      - normal: in clip space, of each fragment; From VS to FS.
      - uv: in texture space, of each fragment; From VS to FS.
      - colour: colour when passing from FS to mixer.
    """

    normal: Vec3f = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        [0.0, 0.0, 0.0]
    )
    uv: Vec2f = jnp.zeros(2)  # pyright: ignore[reportUnknownMemberType]
    colour: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        [0.0, 0.0, 0.0]
    )


class PhongTextureExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""

    canvas: Colour


class PhongTextureShader(
    Shader[
        PhongTextureExtraInput,
        PhongTextureExtraFragmentData,
        PhongTextureExtraMixerOutput,
    ]
):
    """Phong Shading with simple parallel lighting and texture."""

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: PhongTextureExtraInput,
    ) -> Tuple[PerVertex, PhongTextureExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        # assume normal here is in world space. If it is in model space, it
        # must be transformed by the inverse transpose of the model matrix.
        # Ref: https://github.com/ssloy/tinyrenderer/wiki/Lesson-5:-Moving-the-camera#transformation-of-normal-vectors
        normal: Vec3f = Camera.apply_vec(
            normalise(extra.normal[gl_VertexID]),
            camera.world_to_eye_norm,
        )
        assert isinstance(normal, Vec3f)

        return (
            PerVertex(gl_Position=gl_Position),
            PhongTextureExtraFragmentData(
                normal=normal,
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
        varying: PhongTextureExtraFragmentData,
        extra: PhongTextureExtraInput,
    ) -> Tuple[PerFragment, PhongTextureExtraFragmentData]:
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

        # light colour * intensity
        light_colour: Colour = (
            extra.light.colour
            * lax.dot(  # pyright: ignore[reportUnknownMemberType]
                normalise(varying.normal),
                normalise(extra.light.direction),
            )
        )

        return (
            PerFragment(
                keeps=jnp.logical_and(  # pyright: ignore[reportUnknownMemberType]
                    built_in.keeps,
                    gl_FrontFacing,
                ),
                use_default_depth=built_in.use_default_depth,
            ),
            PhongTextureExtraFragmentData(
                colour=cast(
                    Vec3f,
                    lax.cond(  # pyright: ignore[reportUnknownMemberType]
                        jnp.all(  # pyright: ignore[reportUnknownMemberType]
                            light_colour >= 0
                        ),
                        lambda: texture_colour * light_colour,
                        lambda: jnp.zeros(  # pyright: ignore[reportUnknownMemberType]
                            3
                        ),
                    ),
                ),
                uv=varying.uv,
                normal=varying.normal,
            ),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: PhongTextureExtraFragmentData,
    ) -> Tuple[MixerOutput, PhongTextureExtraMixerOutput]:
        mixer_output: MixerOutput
        extra_output: PhongTextureExtraFragmentData
        mixer_output, extra_output = cast(
            Tuple[MixerOutput, PhongTextureExtraFragmentData],
            Shader.mix(gl_FragDepth, keeps, extra),
        )
        assert isinstance(mixer_output, MixerOutput)
        assert isinstance(extra_output, PhongTextureExtraFragmentData)

        return (
            mixer_output,
            PhongTextureExtraMixerOutput(canvas=extra_output.colour),
        )
