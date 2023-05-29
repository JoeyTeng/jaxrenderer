from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer, jaxtyped

from ..geometry import Camera, normalise, to_homogeneous
from ..model import MergedModel
from ..shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from ..types import (Colour, LightSource, SpecularMap, Texture, Vec2f, Vec2i,
                     Vec3f, Vec4f)

jax.config.update('jax_array', True)


class PhongReflectionTextureExtraInput(NamedTuple):
    """Extra input for PhongReflection Shader.

    Attributes:
      - position: in world space, of each vertex.
      - normal: in world space, of each vertex.
      - uv: in texture space, of each vertex.
      - light: parallel light source, shared by all vertices.
      - light_dir_eye: normalised light source direction in eye space.
      - texture_shape: shape of each texture map, shared by all vertices.
      - texture_index: index of texture map for each vertex.
      - offset_shape: offset of each texture map in the merged model.
      - texture: texture, shared by all vertices.
      - specular_map: specular map, shared by all vertices.
      - ambient: ambient strength, shared by all vertices.
      - diffuse: diffuse strength, shared by all vertices.
      - specular: specular strength, shared by all vertices.
    """
    position: Float[Array, "vertices 3"]  # in world space
    normal: Float[Array, "vertices 3"]  # in world space
    uv: Float[Array, "vertices 2"]  # in texture space
    light: LightSource
    light_dir_eye: Vec3f  # in eye/view space
    texture_shape: Integer[Array, "objects 2"]
    texture_index: Integer[Array, "vertices"]
    texture_offset: Integer[Array, ""]
    texture: Texture
    specular_map: SpecularMap
    ambient: Colour
    diffuse: Colour
    specular: Colour


class PhongReflectionTextureExtraFragmentData(NamedTuple):
    """From vertex shader to fragment shader, and from fragment shader to mixer.

    Attributes:
      - normal: in clip space, of each fragment; From VS to FS.
      - uv: in texture space, of each fragment; From VS to FS.
      - texture_index: index of texture map for each fragment; From VS to FS.
      - colour: colour when passing from FS to mixer.
    """
    normal: Vec3f = jnp.zeros(3)
    uv: Vec2f = jnp.zeros(2)
    texture_index: Integer[Array, ""] = jnp.array(0)
    colour: Colour = jnp.zeros(3)


class PhongReflectionTextureExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""
    canvas: Colour


class PhongReflectionTextureShader(
        Shader[PhongReflectionTextureExtraInput,
               PhongReflectionTextureExtraFragmentData,
               PhongReflectionTextureExtraMixerOutput]):
    """PhongReflection Shading with simple parallel lighting and texture."""

    @staticmethod
    @jaxtyped
    @jax.jit
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: PhongReflectionTextureExtraInput,
    ) -> tuple[PerVertex, PhongReflectionTextureExtraFragmentData]:
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
            PhongReflectionTextureExtraFragmentData(
                normal=normal,
                # repeat texture
                uv=extra.uv[gl_VertexID],
                texture_index=extra.texture_index[gl_VertexID],
            ),
        )

    @staticmethod
    @jaxtyped
    @jax.jit
    def interpolate(
        values: PhongReflectionTextureExtraFragmentData,
        barycentric_screen: Vec3f,
        barycentric_clip: Vec3f,
    ) -> PhongReflectionTextureExtraFragmentData:
        varying = Shader.interpolate(
            values=values,
            barycentric_screen=barycentric_screen,
            barycentric_clip=barycentric_clip,
        )
        # all three values should be the same, just pick the first.
        varying = varying._replace(texture_index=values.texture_index[0])
        assert isinstance(varying, PhongReflectionTextureExtraFragmentData)

        return varying

    @staticmethod
    @jaxtyped
    @jax.jit
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: Bool[Array, ""],
        gl_PointCoord: Vec2f,
        varying: PhongReflectionTextureExtraFragmentData,
        extra: PhongReflectionTextureExtraInput,
    ) -> tuple[PerFragment, PhongReflectionTextureExtraFragmentData]:
        built_in: PerFragment = Shader.fragment(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            varying,
            extra,
        )[0]
        assert isinstance(built_in, PerFragment)

        # texture
        texture_index = varying.texture_index.astype(int)
        texture_shape = extra.texture_shape[texture_index]
        uv = MergedModel.uv_repeat(
            uv=varying.uv,
            shape=texture_shape,
            map_index=texture_index,
            offset=extra.texture_offset,
        )
        uv = lax.floor(uv).astype(int)
        assert isinstance(uv, Vec2i)
        texture_colour: Colour = extra.texture[uv[0], uv[1]]

        normal: Vec3f = normalise(varying.normal)
        light_dir: Vec3f = normalise(extra.light_dir_eye)

        # Phong Reflection Model
        diffuse: float = jnp.maximum(lax.dot(normal, light_dir), 0)
        # as `light_dir * -1` should be used here, if
        # using `light_dir - 2 * diffuse * normal`
        reflected_light: Vec3f = normalise(2 * lax.dot(normal, light_dir) *
                                           normal - light_dir)
        assert isinstance(reflected_light, Vec3f)

        specular: float = lax.pow(
            lax.max(reflected_light[2], 0.),
            extra.specular_map[uv[0], uv[1]],
        )

        # compute colour
        colour: Colour = (
            extra.ambient * texture_colour +
            (extra.diffuse * diffuse + extra.specular * specular) *
            # intensity * light colour * texture colour
            extra.light.colour * texture_colour)

        return (
            PerFragment(
                keeps=jnp.logical_and(built_in.keeps, gl_FrontFacing),
                use_default_depth=built_in.use_default_depth,
            ),
            PhongReflectionTextureExtraFragmentData(
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
        extra: PhongReflectionTextureExtraFragmentData,
    ) -> tuple[MixerOutput, PhongReflectionTextureExtraMixerOutput]:
        mixer_output: MixerOutput
        extra_output: PhongReflectionTextureExtraFragmentData
        mixer_output, extra_output = Shader.mix(gl_FragDepth, keeps, extra)
        assert isinstance(mixer_output, MixerOutput)
        assert isinstance(extra_output,
                          PhongReflectionTextureExtraFragmentData)

        return (
            mixer_output,
            PhongReflectionTextureExtraMixerOutput(canvas=extra_output.colour),
        )
