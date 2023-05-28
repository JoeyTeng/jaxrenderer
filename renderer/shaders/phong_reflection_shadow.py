from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, jaxtyped

from ..geometry import Camera, normalise, normalise_homogeneous, to_homogeneous
from ..shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from ..shadow import Shadow
from ..types import (Colour, LightSource, SpecularMap, Texture, Vec2f, Vec3f,
                     Vec4f)

jax.config.update('jax_array', True)


class PhongReflectionShadowTextureExtraInput(NamedTuple):
    """Extra input for Phong Reflection with Shadow Shader.

    Attributes:
      - position: in world space, of each vertex.
      - normal: in world space, of each vertex.
      - uv: in texture space, of each vertex.
      - light: parallel light source, shared by all vertices.
      - light_dir_eye: normalised light source direction in eye/view space.
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
      - shadow_coord: in shadow's clip space, of each fragment; From VS to FS.
      - colour: colour when passing from FS to mixer.
    """
    normal: Vec3f = jnp.zeros(3)
    uv: Vec2f = jnp.zeros(2)
    shadow_coord: Vec4f = jnp.zeros(4)
    colour: Colour = jnp.zeros(3)


class PhongReflectionShadowTextureExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""
    canvas: Colour


class PhongReflectionShadowTextureShader(
        Shader[PhongReflectionShadowTextureExtraInput,
               PhongReflectionShadowTextureExtraFragmentData,
               PhongReflectionShadowTextureExtraMixerOutput]):
    """Phong Shading with simple parallel lighting, texture, Phong Reflection
        approximation and ShadowMap.
    """

    @staticmethod
    @jaxtyped
    @jax.jit
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: PhongReflectionShadowTextureExtraInput,
    ) -> tuple[PerVertex, PhongReflectionShadowTextureExtraFragmentData]:
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
            extra.shadow.camera.to_clip(position))
        assert isinstance(shadow_coord, Vec4f)

        return (
            PerVertex(gl_Position=gl_Position),
            PhongReflectionShadowTextureExtraFragmentData(
                normal=normal,
                # repeat texture
                uv=extra.uv[gl_VertexID] %
                jnp.asarray(extra.texture.shape[:2]),
                shadow_coord=shadow_coord,
            ),
        )

    @staticmethod
    @jaxtyped
    @jax.jit
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: Bool[Array, ""],
        gl_PointCoord: Vec2f,
        varying: PhongReflectionShadowTextureExtraFragmentData,
        extra: PhongReflectionShadowTextureExtraInput,
    ) -> tuple[PerFragment, PhongReflectionShadowTextureExtraFragmentData]:
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
            extra.shadow.camera.viewport @ varying.shadow_coord)
        assert isinstance(shadow_coord, Vec4f)
        shadow_str: Colour = extra.shadow.strength
        assert isinstance(shadow_str, Colour)
        shadow: Colour = jnp.where(
            # if before/at shadow
            shadow_coord[2] <= extra.shadow.get(shadow_coord[:2]),
            # when not in shadow, keeps all light.
            jnp.ones_like(shadow_str),
            # if in shadow, only keep "1 - shadow_str" amount of light.
            1. - shadow_str,
        )

        # texture
        uv = lax.round(varying.uv).astype(int)
        texture_colour: Colour = extra.texture[uv[0], uv[1]]

        normal: Vec3f = normalise(varying.normal)
        light_dir: Vec3f = extra.light_dir_eye

        # Phong Reflection Model
        diffuse: float = lax.dot(normal, light_dir)
        reflected_light: Vec3f = normalise(light_dir - 2 * diffuse * normal)
        assert isinstance(reflected_light, Vec3f)

        specular: float = lax.pow(
            lax.max(reflected_light[2], 0.),
            extra.specular_map[uv[0], uv[1]],
        )

        # compute colour
        colour: Colour = ((
            extra.ambient +
            (extra.diffuse * diffuse + extra.specular * specular) *
            # shadow * intensity * light colour * texture colour
            shadow) * extra.light.colour * texture_colour)

        return (
            PerFragment(
                keeps=jnp.logical_and(built_in.keeps, gl_FrontFacing),
                use_default_depth=built_in.use_default_depth,
            ),
            PhongReflectionShadowTextureExtraFragmentData(
                colour=lax.cond(
                    (colour >= 0).all(),
                    lambda: colour,
                    lambda: jnp.zeros(3),
                ),
                uv=varying.uv,
                normal=varying.normal,
            ),
        )

    @staticmethod
    @jaxtyped
    @jax.jit
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: PhongReflectionShadowTextureExtraFragmentData,
    ) -> tuple[MixerOutput, PhongReflectionShadowTextureExtraMixerOutput]:
        mixer_output: MixerOutput
        extra_output: PhongReflectionShadowTextureExtraFragmentData
        mixer_output, extra_output = Shader.mix(gl_FragDepth, keeps, extra)
        assert isinstance(mixer_output, MixerOutput)
        assert isinstance(extra_output,
                          PhongReflectionShadowTextureExtraFragmentData)

        return (
            mixer_output,
            PhongReflectionShadowTextureExtraMixerOutput(
                canvas=extra_output.colour),
        )
