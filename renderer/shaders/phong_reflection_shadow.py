from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from functools import partial
from typing import NamedTuple, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]
from typing_extensions import override

from .._backport import Tuple
from .._meta_utils import add_tracing_name
from .._meta_utils import typed_jit as jit
from ..geometry import Camera, normalise, normalise_homogeneous, to_homogeneous
from ..model import MergedModel
from ..shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from ..shadow import Shadow
from ..types import (
    BoolV,
    Colour,
    IntV,
    LightSource,
    SpecularMap,
    Texture,
    Vec2f,
    Vec2i,
    Vec3f,
    Vec4f,
)

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]


class PhongReflectionShadowTextureExtraInput(NamedTuple):
    """Extra input for Phong Reflection with Shadow Shader.

    Attributes:
      - position: in world space, of each vertex.
      - normal: in world space, of each vertex.
      - uv: in texture space, of each vertex.
      - light: parallel light source, shared by all vertices.
      - light_dir_eye: normalised light source direction in eye/view space.
      - texture_shape: shape of each texture map, shared by all vertices.
      - texture_index: index of texture map for each vertex.
      - offset_shape: offset of each texture map in the merged model.
      - texture: texture, shared by all vertices.
      - specular_map: specular map, shared by all vertices.
      - shadow: Shadow from first pass, shared by all vertices.
      - shadow_mat: shadow matrix from target screen space to shadow's screen
        space, shared by all vertices.
      - ambient: ambient strength, shared by all vertices.
      - diffuse: diffuse strength, shared by all vertices.
      - specular: specular strength, shared by all vertices.
    """

    position: Float[Array, "vertices 3"]  # in world space
    normal: Float[Array, "vertices 3"]  # in world space
    uv: Float[Array, "vertices 2"]  # in texture space
    light: LightSource
    light_dir_eye: Vec3f  # in eye/view space.
    texture_shape: Integer[Array, "objects 2"]
    texture_index: Integer[Array, "vertices"]
    texture_offset: IntV
    texture: Texture
    specular_map: SpecularMap
    shadow: Shadow
    camera: Camera  # so accessible in FS as well.
    ambient: Colour
    diffuse: Colour
    specular: Colour


class PhongReflectionShadowTextureExtraFragmentData(NamedTuple):
    """From vertex shader to fragment shader, and from fragment shader to mixer.

    Attributes:
      - normal: in clip space, of each fragment; From VS to FS.
      - uv: in texture space, of each fragment; From VS to FS.
      - texture_index: index of texture map for each fragment; From VS to FS.
      - shadow_coord: in shadow's clip space, of each fragment; From VS to FS.
      - colour: colour when passing from FS to mixer.
    """

    normal: Vec3f = jnp.zeros(3)  # pyright: ignore[reportUnknownMemberType]
    uv: Vec2f = jnp.zeros(2)  # pyright: ignore[reportUnknownMemberType]
    texture_index: IntV = jnp.array(0)  # pyright: ignore
    shadow_coord: Vec4f = jnp.zeros(4)  # pyright: ignore[reportUnknownMemberType]
    colour: Colour = jnp.zeros(3)  # pyright: ignore[reportUnknownMemberType]


class PhongReflectionShadowTextureExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""

    canvas: Colour


class PhongReflectionShadowTextureShader(
    Shader[
        PhongReflectionShadowTextureExtraInput,
        PhongReflectionShadowTextureExtraFragmentData,
        PhongReflectionShadowTextureExtraMixerOutput,
    ]
):
    """Phong Shading with simple parallel lighting, texture, Phong Reflection
    approximation and ShadowMap.
    """

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: PhongReflectionShadowTextureExtraInput,
    ) -> Tuple[PerVertex, PhongReflectionShadowTextureExtraFragmentData]:
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

        # shadow. Normalise here as it is not done implicitly in the pipeline.
        # the result is in shadow's clip space, as NDC.
        shadow_coord: Vec4f = normalise_homogeneous(
            extra.shadow.camera.to_clip(position)
        )
        assert isinstance(shadow_coord, Vec4f)

        return (
            PerVertex(gl_Position=gl_Position),
            PhongReflectionShadowTextureExtraFragmentData(
                normal=normal,
                uv=extra.uv[gl_VertexID],
                texture_index=extra.texture_index[gl_VertexID],
                shadow_coord=shadow_coord,
            ),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def interpolate(
        values: PhongReflectionShadowTextureExtraFragmentData,
        barycentric_screen: Vec3f,
        barycentric_clip: Vec3f,
    ) -> PhongReflectionShadowTextureExtraFragmentData:
        varying = Shader.interpolate(
            values=values,
            barycentric_screen=barycentric_screen,
            barycentric_clip=barycentric_clip,
        )
        # all three values should be the same, just pick the first.
        varying = varying._replace(texture_index=values.texture_index[0])
        assert isinstance(varying, PhongReflectionShadowTextureExtraFragmentData)

        return varying

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: PhongReflectionShadowTextureExtraFragmentData,
        extra: PhongReflectionShadowTextureExtraInput,
    ) -> Tuple[PerFragment, PhongReflectionShadowTextureExtraFragmentData]:
        built_in: PerFragment = Shader.fragment(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            varying,
            extra,
        )[0]
        assert isinstance(built_in, PerFragment)

        # shadow
        # from NDC to screen coordinates, in shadow's screen space.
        shadow_coord: Vec4f = normalise_homogeneous(
            extra.shadow.camera.viewport @ varying.shadow_coord
        )
        assert isinstance(shadow_coord, Vec4f)
        shadow_str: Colour = extra.shadow.strength
        assert isinstance(shadow_str, Colour)
        shadow: Colour = jnp.where(  # pyright: ignore[reportUnknownMemberType]
            # if before/at shadow
            shadow_coord[2] <= extra.shadow.get(shadow_coord[:2]),
            # when not in shadow, keeps all light.
            jnp.ones_like(shadow_str),  # pyright: ignore[reportUnknownMemberType]
            # if in shadow, only keep "1 - shadow_str" amount of light.
            1.0 - shadow_str,
        )

        # texture
        texture_index = varying.texture_index.astype(int)  # pyright: ignore
        texture_shape = extra.texture_shape[texture_index]
        uv = MergedModel.uv_repeat(
            uv=varying.uv,
            shape=texture_shape,
            map_index=texture_index,
            offset=extra.texture_offset,
        )
        uv = lax.floor(uv).astype(int)  # pyright: ignore[reportUnknownMemberType]
        assert isinstance(uv, Vec2i)
        texture_colour: Colour = extra.texture[uv[0], uv[1]]

        normal: Vec3f = normalise(varying.normal)
        # in Phong shading, the light direction is towards the light source;
        light_dir: Vec3f = normalise(extra.light_dir_eye)

        # Phong Reflection Model
        diffuse: Float[
            Array, ""
        ] = jnp.maximum(  # pyright: ignore[reportUnknownMemberType]
            lax.dot(normal, light_dir),  # pyright: ignore[reportUnknownMemberType]
            0,
        )
        # If using standard reflection formula
        # `light_dir - 2 * diffuse * normal`, need to use
        # `-1 * light_dir` as `light_dir` instead.
        reflected_light: Vec3f = normalise(
            2 * lax.dot(normal, light_dir) * normal - light_dir  # pyright: ignore
        )
        assert isinstance(reflected_light, Vec3f)

        specular: Float[
            Array, ""
        ] = lax.pow(  # pyright: ignore[reportUnknownMemberType]
            lax.max(reflected_light[2], 0.0),  # pyright: ignore
            extra.specular_map[uv[0], uv[1]],
        )

        # compute colour
        colour: Colour = (
            extra.ambient * texture_colour
            + shadow
            * (extra.diffuse * diffuse + extra.specular * specular)
            * texture_colour
            * extra.light.colour
        )

        return (
            PerFragment(
                keeps=jnp.logical_and(  # pyright: ignore[reportUnknownMemberType]
                    built_in.keeps, gl_FrontFacing
                ),
                use_default_depth=built_in.use_default_depth,
            ),
            PhongReflectionShadowTextureExtraFragmentData(
                colour=colour,
                uv=varying.uv,
                normal=varying.normal,
            ),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: PhongReflectionShadowTextureExtraFragmentData,
    ) -> Tuple[MixerOutput, PhongReflectionShadowTextureExtraMixerOutput]:
        mixer_output: MixerOutput
        extra_output: PhongReflectionShadowTextureExtraFragmentData
        mixer_output, extra_output = cast(
            Tuple[MixerOutput, PhongReflectionShadowTextureExtraFragmentData],
            Shader.mix(gl_FragDepth, keeps, extra),
        )
        assert isinstance(mixer_output, MixerOutput)
        assert isinstance(extra_output, PhongReflectionShadowTextureExtraFragmentData)

        return (
            mixer_output,
            PhongReflectionShadowTextureExtraMixerOutput(canvas=extra_output.colour),
        )
