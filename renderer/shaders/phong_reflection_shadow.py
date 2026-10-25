from functools import partial
from typing import NamedTuple, override

from jax import Array
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import (
    Bool,
    Float,
    Integer,
)

from .._meta_utils import add_tracing_name, is_array_type, typed_array, typed_zeros
from .._meta_utils import typed_jaxtyped as jaxtyped
from .._meta_utils import typed_jit as jit
from ..geometry import Camera, normalise, normalise_homogeneous, to_homogeneous
from ..model import MergedModel
from ..shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from ..shadow import Shadow
from ..types import (
    FALSE_ARRAY,
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
      - double_sided: per-vertex visibility mask; defaults to false.
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
    double_sided: Bool[Array, "vertices"] | BoolV = FALSE_ARRAY


class PhongReflectionShadowTextureExtraFragmentData(NamedTuple):
    """From vertex shader to fragment shader, and from fragment shader to mixer.

    Attributes:
      - normal: in clip space, of each fragment; From VS to FS.
      - uv: in texture space, of each fragment; From VS to FS.
      - texture_index: index of texture map for each fragment; From VS to FS.
      - shadow_coord: in shadow's clip space, of each fragment; From VS to FS.
      - colour: colour when passing from FS to mixer.
    """

    normal: Vec3f = typed_zeros(3)
    uv: Vec2f = typed_zeros(2)
    texture_index: IntV = typed_array(0)
    shadow_coord: Vec4f = typed_zeros(4)
    colour: Colour = typed_zeros(3)
    double_sided: BoolV = FALSE_ARRAY


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
    ) -> tuple[PerVertex, PhongReflectionShadowTextureExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert is_array_type(gl_Position, Vec4f)

        # assume normal here is in world space. If it is in model space, it
        # must be transformed by the inverse transpose of the model matrix.
        # Ref: https://github.com/ssloy/tinyrenderer/wiki/Lesson-5:-Moving-the-camera#transformation-of-normal-vectors
        normal: Vec3f = Camera.apply_vec(
            normalise(extra.normal[gl_VertexID]),
            camera.world_to_eye_norm,
        )
        assert is_array_type(normal, Vec3f)

        # shadow. Normalise here as it is not done implicitly in the pipeline.
        # the result is in shadow's clip space, as NDC.
        shadow_coord: Vec4f = normalise_homogeneous(
            extra.shadow.camera.to_clip(position)
        )
        assert is_array_type(shadow_coord, Vec4f)

        return (
            PerVertex(gl_Position=gl_Position),
            PhongReflectionShadowTextureExtraFragmentData(
                normal=normal,
                uv=extra.uv[gl_VertexID],
                texture_index=extra.texture_index[gl_VertexID],
                shadow_coord=shadow_coord,
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
        values: PhongReflectionShadowTextureExtraFragmentData,
        barycentric_screen: Vec3f,
        barycentric_clip: Vec3f,
    ) -> PhongReflectionShadowTextureExtraFragmentData:
        varying: PhongReflectionShadowTextureExtraFragmentData = Shader[
            PhongReflectionShadowTextureExtraInput,
            PhongReflectionShadowTextureExtraFragmentData,
            PhongReflectionShadowTextureExtraFragmentData,
        ].interpolate(
            values=values,
            barycentric_screen=barycentric_screen,
            barycentric_clip=barycentric_clip,
        )
        # all three values should be the same, just pick the first.
        varying = varying._replace(
            texture_index=values.texture_index[0],
            double_sided=values.double_sided[0],
        )
        assert isinstance(varying, PhongReflectionShadowTextureExtraFragmentData)

        return varying

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
        values: PhongReflectionShadowTextureExtraFragmentData,
        barycentric_screen: Float[Array, "primitives 3"],
        barycentric_clip: Float[Array, "primitives 3"],
        *,
        double_sided: Bool[Array, "primitives"] | BoolV = FALSE_ARRAY,
    ) -> tuple[
        Float[Array, "kept_primitives 4"],
        Bool[Array, "kept_primitives"],
        Float[Array, "kept_primitives 2"],
        Bool[Array, "kept_primitives"],
        PhongReflectionShadowTextureExtraFragmentData,
        Float[Array, "kept_primitives 3"],
        Float[Array, "kept_primitives 3"],
    ]:
        return Shader[
            PhongReflectionShadowTextureExtraInput,
            PhongReflectionShadowTextureExtraFragmentData,
            PhongReflectionShadowTextureExtraFragmentData,
        ].primitive_chooser(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            keeps,
            values,
            barycentric_screen,
            barycentric_clip,
            double_sided=double_sided | values.double_sided[:, 0],
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
        varying: PhongReflectionShadowTextureExtraFragmentData,
        extra: PhongReflectionShadowTextureExtraInput,
    ) -> tuple[PerFragment, PhongReflectionShadowTextureExtraFragmentData]:
        built_in: PerFragment = Shader[
            PhongReflectionShadowTextureExtraInput,
            PhongReflectionShadowTextureExtraFragmentData,
            PhongReflectionShadowTextureExtraFragmentData,
        ].fragment(
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
        assert is_array_type(shadow_coord, Vec4f)
        shadow_str: Colour = extra.shadow.strength
        assert is_array_type(shadow_str, Colour)
        shadow: Colour = jnp.where(
            # if before/at shadow
            shadow_coord[2] <= extra.shadow.get(shadow_coord[:2]),
            # when not in shadow, keeps all light.
            jnp.ones_like(shadow_str),  # pyright: ignore[reportUnknownMemberType]
            # if in shadow, only keep "1 - shadow_str" amount of light.
            1.0 - shadow_str,
        )

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
        assert is_array_type(uv, Vec2i)
        texture_colour: Colour = extra.texture[uv[0], uv[1]]

        normal: Vec3f = normalise(varying.normal)
        # in Phong shading, the light direction is towards the light source;
        light_dir: Vec3f = normalise(extra.light_dir_eye)

        # Phong Reflection Model
        diffuse: Float[Array, ""] = jnp.maximum(
            lax.dot(normal, light_dir),  # pyright: ignore[reportUnknownMemberType]
            0,
        )
        # If using standard reflection formula
        # `light_dir - 2 * diffuse * normal`, need to use
        # `-1 * light_dir` as `light_dir` instead.
        reflected_light: Vec3f = normalise(
            2 * lax.dot(normal, light_dir) * normal - light_dir  # pyright: ignore[reportUnknownMemberType]
        )
        assert is_array_type(reflected_light, Vec3f)

        specular: Float[Array, ""] = lax.pow(
            lax.max(reflected_light[2], 0.0),
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
                keeps=jnp.logical_and(
                    built_in.keeps,
                    gl_FrontFacing | varying.double_sided,
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
    ) -> tuple[MixerOutput, PhongReflectionShadowTextureExtraMixerOutput]:
        mixer_output: MixerOutput
        extra_output: PhongReflectionShadowTextureExtraFragmentData
        mixer_output, extra_output = Shader[
            PhongReflectionShadowTextureExtraInput,
            PhongReflectionShadowTextureExtraFragmentData,
            PhongReflectionShadowTextureExtraFragmentData,
        ].mix(gl_FragDepth, keeps, extra)
        assert isinstance(mixer_output, MixerOutput)
        assert isinstance(extra_output, PhongReflectionShadowTextureExtraFragmentData)

        return (
            mixer_output,
            PhongReflectionShadowTextureExtraMixerOutput(canvas=extra_output.colour),
        )
