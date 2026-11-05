from dataclasses import replace
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
from ..geometry import Camera, normalise, to_homogeneous
from ..model import MergedModel
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
from ..types import (
    FALSE_ARRAY,
    TRUE_ARRAY,
    BoolV,
    Colour,
    FloatV,
    IntV,
    LightSource,
    SpecularMap,
    Texture,
    Vec2f,
    Vec2i,
    Vec3f,
    Vec4f,
)


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
      - double_sided: per-vertex visibility mask; defaults to false.
    """

    position: Float[Array, "vertices 3"]  # in world space
    normal: Float[Array, "vertices 3"]  # in world space
    uv: Float[Array, "vertices 2"]  # in texture space
    light: LightSource
    light_dir_eye: Vec3f  # in eye/view space
    texture_shape: Integer[Array, "objects 2"]
    texture_index: Integer[Array, "vertices"]
    texture_offset: IntV
    texture: Texture
    specular_map: SpecularMap
    ambient: Colour
    diffuse: Colour
    specular: Colour
    double_sided: Bool[Array, "vertices"] | BoolV = FALSE_ARRAY


class PhongReflectionTextureExtraFragmentData(NamedTuple):
    """From vertex shader to fragment shader, and from fragment shader to mixer.

    Attributes:
      - normal: in clip space, of each fragment; From VS to FS.
      - uv: in texture space, of each fragment; From VS to FS.
      - texture_index: index of texture map for each fragment; From VS to FS.
      - colour: colour when passing from FS to mixer.
    """

    normal: Vec3f = typed_zeros(3)
    uv: Vec2f = typed_zeros(2)
    texture_index: IntV = typed_array(0)
    colour: Colour = typed_zeros(3)
    double_sided: BoolV = FALSE_ARRAY


class PhongReflectionTextureExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""

    canvas: Colour


class PhongReflectionTextureShader(
    Shader[
        PhongReflectionTextureExtraInput,
        PhongReflectionTextureExtraFragmentData,
        PhongReflectionTextureExtraMixerOutput,
    ]
):
    """PhongReflection Shading with simple parallel lighting and texture."""

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def vertex(
        inputs: VertexInput[PhongReflectionTextureExtraInput],
    ) -> tuple[PerVertex, PhongReflectionTextureExtraFragmentData]:
        gl_VertexID = inputs.gl_VertexID
        camera = inputs.camera
        extra = inputs.extra
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

        return (
            PerVertex(gl_Position=gl_Position),
            PhongReflectionTextureExtraFragmentData(
                normal=normal,
                # repeat texture
                uv=extra.uv[gl_VertexID],
                texture_index=extra.texture_index[gl_VertexID],
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
    def interpolate(
        inputs: InterpolationInput[PhongReflectionTextureExtraFragmentData],
    ) -> PhongReflectionTextureExtraFragmentData:
        values = inputs.values
        varying: PhongReflectionTextureExtraFragmentData = Shader[
            PhongReflectionTextureExtraInput,
            PhongReflectionTextureExtraFragmentData,
            PhongReflectionTextureExtraFragmentData,
        ].interpolate(
            inputs=inputs,
        )
        # all three values should be the same, just pick the first.
        varying = varying._replace(
            texture_index=values.texture_index[0],
            double_sided=values.double_sided[0],
        )
        assert isinstance(varying, PhongReflectionTextureExtraFragmentData)

        return varying

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    @override
    def primitive_chooser(
        inputs: PrimitiveInput[PhongReflectionTextureExtraFragmentData],
    ) -> tuple[
        Float[Array, "kept_primitives 4"],
        Bool[Array, "kept_primitives"],
        Float[Array, "kept_primitives 2"],
        Bool[Array, "kept_primitives"],
        PhongReflectionTextureExtraFragmentData,
        Float[Array, "kept_primitives 3"],
        Float[Array, "kept_primitives 3"],
    ]:
        return Shader[
            PhongReflectionTextureExtraInput,
            PhongReflectionTextureExtraFragmentData,
            PhongReflectionTextureExtraFragmentData,
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
    def fragment(
        inputs: FragmentInput[
            PhongReflectionTextureExtraInput,
            PhongReflectionTextureExtraFragmentData,
        ],
    ) -> tuple[PerFragment, Fragment[PhongReflectionTextureExtraMixerOutput]]:
        varying = inputs.varying
        extra = inputs.extra

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
        light_dir: Vec3f = normalise(extra.light_dir_eye)

        # Phong Reflection Model
        diffuse: Float[Array, ""] = jnp.maximum(
            lax.dot(normal, light_dir),  # pyright: ignore[reportUnknownMemberType]
            0,
        )
        # as `light_dir * -1` should be used here, if
        # using `light_dir - 2 * diffuse * normal`
        reflected_light: Vec3f = normalise(
            2
            * lax.dot(normal, light_dir)  # pyright: ignore[reportUnknownMemberType]
            * normal
            - light_dir
        )
        assert is_array_type(reflected_light, Vec3f)

        specular: FloatV = lax.pow(
            lax.max(reflected_light[2], 0.0),
            extra.specular_map[uv[0], uv[1]],
        )

        # compute colour
        colour: Colour = (
            extra.ambient * texture_colour
            + (extra.diffuse * diffuse + extra.specular * specular)
            *
            # intensity * light colour * texture colour
            extra.light.colour
            * texture_colour
        )

        return (
            PerFragment(
                keeps=inputs.gl_FrontFacing | varying.double_sided,
                use_default_depth=TRUE_ARRAY,
            ),
            Fragment(
                targets=PhongReflectionTextureExtraMixerOutput(canvas=colour),
            ),
        )
