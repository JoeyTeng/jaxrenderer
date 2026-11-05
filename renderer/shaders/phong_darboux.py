from functools import partial
from typing import NamedTuple, TypeAlias, cast

from jax import Array
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Integer,
)

from .._meta_utils import (
    add_tracing_name,
    is_array_type,
    typed_array,
    typed_asarray,
    typed_zeros,
)
from .._meta_utils import typed_jaxtyped as jaxtyped
from .._meta_utils import typed_jit as jit
from ..geometry import (
    Camera,
    Interpolation,
    interpolate,
    normalise,
    to_cartesian,
    to_homogeneous,
)
from ..shader import (
    ID,
    Fragment,
    FragmentInput,
    InterpolationInput,
    PerFragment,
    PerVertex,
    Shader,
    VertexInput,
)
from ..types import (
    TRUE_ARRAY,
    Colour,
    FaceIndices,
    LightSource,
    NormalMap,
    Texture,
    Triangle,
    Vec2f,
    Vec3f,
    Vec4f,
)

Triangle3f: TypeAlias = Float[Array, "3 3"]
Triangle2f: TypeAlias = Float[Array, "3 2"]


class PhongTextureDarbouxExtraInput(NamedTuple):
    """Extra input for Phong Shader.

    Attributes:
      - position: in world space, of each vertex.
      - normal: in world space, of each vertex.
      - uv: in texture space, of each vertex.
      - light: parallel `headlight` light source, shared by all vertices.
        It is in the eye/view space.
      - texture: texture, shared by all vertices.
      - normal_map: normal map, shared by all vertices.
        This is in Darboux frame.
      - id_to_face: id of the face that each vertex belongs to.
      - faces_indices: id of the vertex that each face contains.
    """

    position: Float[Array, "vertices 3"]  # in world space
    normal: Float[Array, "vertices 3"]  # in world space
    uv: Float[Array, "vertices 2"]  # in texture space
    light: LightSource
    texture: Texture
    normal_map: NormalMap
    """In Darboux frame."""
    id_to_face: Integer[Array, "vertices"]
    """id of the face that each vertex belongs to."""
    faces_indices: FaceIndices
    """id of the vertex that each face contains."""


class PhongTextureDarbouxExtraFragmentData(NamedTuple):
    """From vertex shader to fragment shader, and from fragment shader to mixer.

    Attributes:
      - normal: in clip space, of each fragment; From VS to FS.
      - uv: in texture space, of each fragment; From VS to FS.
      - triangle: in normalised device coordinates (NDC), of each fragment;
        From VS to FS. This should not be interpolated.
      - triangle: in texture space, of each fragment; From VS to FS.
        This should not be interpolated.
      - colour: colour when passing from FS to mixer.
    """

    normal: Vec3f = typed_array([0.0, 0.0, 0.0])
    uv: Vec2f = typed_zeros(2)
    triangle: Triangle3f = typed_zeros((3, 3))
    triangle_uv: Triangle2f = typed_zeros((3, 2))
    """triangle in NDC, not interpolated."""
    colour: Colour = typed_array([0.0, 0.0, 0.0])


class PhongTextureDarbouxExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""

    canvas: Colour


class PhongTextureDarbouxShader(
    Shader[
        PhongTextureDarbouxExtraInput,
        PhongTextureDarbouxExtraFragmentData,
        PhongTextureDarbouxExtraMixerOutput,
    ]
):
    """Phong Shading with simple parallel lighting and texture, normals are
    represented in tangent space (Darboux frame)."""

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def vertex(
        inputs: VertexInput[PhongTextureDarbouxExtraInput],
    ) -> tuple[PerVertex, PhongTextureDarbouxExtraFragmentData]:
        gl_VertexID = inputs.gl_VertexID
        gl_InstanceID = inputs.gl_InstanceID
        camera = inputs.camera
        extra = inputs.extra
        # Use gl_VertexID to index in `extra` buffer.
        assert is_array_type(gl_VertexID, ID), gl_VertexID
        assert is_array_type(gl_InstanceID, ID), gl_InstanceID
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert is_array_type(gl_Position, Vec4f)

        assert is_array_type(extra.id_to_face, Integer[Array, "vertices"]), (
            f"Expected Integer array with shape {extra.position.shape[:1]}, "
            f"got {type(extra.id_to_face)} with shape "
            f"{extra.id_to_face.shape}"
        )
        assert is_array_type(extra.faces_indices, Integer[Array, "faces 3"]), (
            f"Expected Integer array with shape (faces, 3), "
            f"got {type(extra.faces_indices)} with shape "
            f"{extra.faces_indices.shape}"
        )

        face_indices = extra.faces_indices[extra.id_to_face[gl_VertexID]]
        triangle_model: Triangle = to_homogeneous(extra.position[face_indices])
        triangle_clip: Triangle = camera.to_clip(triangle_model)
        triangle_ndc: Triangle3f = to_cartesian(triangle_clip)
        assert is_array_type(triangle_ndc, Triangle3f)

        triangle_uv: Triangle2f = extra.uv[face_indices]
        assert is_array_type(triangle_uv, Triangle2f)

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
            PhongTextureDarbouxExtraFragmentData(
                normal=normal,
                uv=extra.uv[gl_VertexID],
                triangle=triangle_ndc,
                triangle_uv=triangle_uv,
            ),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def interpolate(
        inputs: InterpolationInput[PhongTextureDarbouxExtraFragmentData],
    ) -> PhongTextureDarbouxExtraFragmentData:
        values = inputs.values
        barycentric_screen = inputs.barycentric_screen
        barycentric_clip = inputs.barycentric_clip

        def smooth_interpolation(value: Array) -> Array:
            return interpolate(
                value,
                barycentric_screen=barycentric_screen,
                barycentric_clip=barycentric_clip,
                mode=Interpolation.SMOOTH,
            )

        normal: Vec3f = smooth_interpolation(values.normal)
        assert is_array_type(normal, Vec3f)

        uv: Vec2f = smooth_interpolation(values.uv)
        assert is_array_type(uv, Vec2f)

        varying: PhongTextureDarbouxExtraFragmentData = (
            PhongTextureDarbouxExtraFragmentData(
                normal=normal,
                uv=uv,
                # pick first of the 3, as they are the same
                # noticed that `values` are batches, so here values.triangle is
                # actually in the shape of (3, 3, 3)
                triangle=values.triangle[0],
                triangle_uv=values.triangle_uv[0],
            )
        )
        assert is_array_type(varying.triangle, Triangle3f)
        assert is_array_type(varying.triangle_uv, Triangle2f)

        return varying

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def fragment(
        inputs: FragmentInput[
            PhongTextureDarbouxExtraInput,
            PhongTextureDarbouxExtraFragmentData,
        ],
    ) -> tuple[PerFragment, Fragment[PhongTextureDarbouxExtraMixerOutput]]:
        gl_FrontFacing = inputs.gl_FrontFacing
        varying = inputs.varying
        extra = inputs.extra
        assert isinstance(varying, PhongTextureDarbouxExtraFragmentData), (
            f"Expected PhongTextureDarbouxExtraFragmentData, got {varying}"
        )

        # repeat texture
        uv = lax.floor(varying.uv).astype(int)
        uv = uv % typed_asarray(extra.texture.shape[:2])

        normal: Vec3f = normalise(varying.normal)
        A: Float[Array, "3 3"] = jnp.vstack(
            [
                varying.triangle[1, :] - varying.triangle[0, :],
                varying.triangle[2, :] - varying.triangle[0, :],
                normal,
            ]
        )
        AI = cast(Float[Array, "3 3"], jnp.linalg.inv(A))
        _uv: Triangle2f = varying.triangle_uv
        i: Vec3f = AI @ typed_array([_uv[1, 0] - _uv[0, 0], _uv[2, 0] - _uv[0, 0], 0])
        j: Vec3f = AI @ typed_array([_uv[1, 1] - _uv[0, 1], _uv[2, 1] - _uv[0, 1], 0])

        B = lax.concatenate(
            [
                normalise(i)[:, None],
                normalise(j)[:, None],
                normal[:, None],
            ],
            dimension=1,
        )
        assert is_array_type(B, Float[Array, "3 3"])

        normal = normalise(B @ extra.normal_map[uv[0], uv[1]])
        assert is_array_type(normal, Vec3f)

        texture_colour: Colour = extra.texture[uv[0], uv[1]]

        # light colour * intensity
        light_colour: Colour = extra.light.colour * lax.dot(  # pyright: ignore[reportUnknownMemberType]
            normal,
            normalise(extra.light.direction),
        )

        return (
            PerFragment(
                keeps=gl_FrontFacing,
                use_default_depth=TRUE_ARRAY,
            ),
            Fragment(
                targets=PhongTextureDarbouxExtraMixerOutput(
                    canvas=lax.cond(  # pyright: ignore[reportUnknownMemberType]
                        jnp.all(light_colour >= 0),
                        lambda: texture_colour * light_colour,
                        lambda: typed_zeros(3),
                    ),
                )
            ),
        )
