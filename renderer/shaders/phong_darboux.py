from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from functools import partial
from typing import NamedTuple, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Float, Integer
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from .._backport import Tuple, TypeAlias
from .._meta_utils import add_tracing_name
from .._meta_utils import typed_jit as jit
from ..geometry import (
    Camera,
    Interpolation,
    interpolate,
    normalise,
    to_cartesian,
    to_homogeneous,
)
from ..shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from ..types import (
    BoolV,
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

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]

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

    normal: Vec3f = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        [0.0, 0.0, 0.0]
    )
    uv: Vec2f = jnp.zeros(2)  # pyright: ignore[reportUnknownMemberType]
    triangle: Triangle3f = jnp.zeros((3, 3))  # pyright: ignore[reportUnknownMemberType]
    triangle_uv: Triangle2f = jnp.zeros(  # pyright: ignore[reportUnknownMemberType]
        (3, 2)
    )
    """triangle in NDC, not interpolated."""
    colour: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        [0.0, 0.0, 0.0]
    )


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
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: PhongTextureDarbouxExtraInput,
    ) -> Tuple[PerVertex, PhongTextureDarbouxExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        assert isinstance(gl_VertexID, ID), gl_VertexID
        assert isinstance(gl_InstanceID, ID), gl_InstanceID
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        assert isinstance(extra.id_to_face, Integer[Array, "vertices"]), (
            f"Expected Integer array with shape {extra.position.shape[:1]}, "
            f"got {type(extra.id_to_face)} with shape "
            f"{extra.id_to_face.shape}"
        )
        assert isinstance(extra.faces_indices, Integer[Array, "faces 3"]), (
            f"Expected Integer array with shape (faces, 3), "
            f"got {type(extra.faces_indices)} with shape "
            f"{extra.faces_indices.shape}"
        )

        face_indices = extra.faces_indices[extra.id_to_face[gl_VertexID]]
        triangle_model: Triangle = to_homogeneous(extra.position[face_indices])
        triangle_clip: Triangle = camera.to_clip(triangle_model)
        triangle_ndc: Triangle3f = to_cartesian(triangle_clip)
        assert isinstance(triangle_ndc, Triangle3f)

        triangle_uv: Triangle2f = extra.uv[face_indices]
        assert isinstance(triangle_uv, Triangle2f)

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
        values: PhongTextureDarbouxExtraFragmentData,
        barycentric_screen: Vec3f,
        barycentric_clip: Vec3f,
    ) -> PhongTextureDarbouxExtraFragmentData:
        smooth_interpolation = Partial(
            interpolate,
            barycentric_screen=barycentric_screen,
            barycentric_clip=barycentric_clip,
            mode=Interpolation.SMOOTH,
        )

        normal = cast(Vec3f, smooth_interpolation(values.normal))
        assert isinstance(normal, Vec3f)

        uv = cast(Vec2f, smooth_interpolation(values.uv))
        assert isinstance(uv, Vec2f)

        varying: PhongTextureDarbouxExtraFragmentData = PhongTextureDarbouxExtraFragmentData(
            normal=normal,
            uv=uv,
            # pick first of the 3, as they are the same
            # noticed that `values` are batches, so here values.triangle is
            # actually in the shape of (3, 3, 3)
            triangle=values.triangle[0],
            triangle_uv=values.triangle_uv[0],
        )
        assert isinstance(varying.triangle, Triangle3f)
        assert isinstance(varying.triangle_uv, Triangle2f)

        return varying

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: PhongTextureDarbouxExtraFragmentData,
        extra: PhongTextureDarbouxExtraInput,
    ) -> Tuple[PerFragment, PhongTextureDarbouxExtraFragmentData]:
        built_in: PerFragment = Shader.fragment(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            varying,
            extra,
        )[0]
        assert isinstance(built_in, PerFragment)
        assert isinstance(
            varying, PhongTextureDarbouxExtraFragmentData
        ), f"Expected PhongTextureDarbouxExtraFragmentData, got {varying}"

        # repeat texture
        uv = lax.floor(varying.uv).astype(int)  # pyright: ignore
        uv = uv % jnp.asarray(extra.texture.shape[:2])  # pyright: ignore

        normal: Vec3f = normalise(varying.normal)
        A: Float[Array, "3 3"] = jnp.vstack(  # pyright: ignore[reportUnknownMemberType]
            [
                varying.triangle[1, :] - varying.triangle[0, :],
                varying.triangle[2, :] - varying.triangle[0, :],
                normal,
            ]
        )
        AI = cast(Float[Array, "3 3"], jnp.linalg.inv(A))
        _uv: Triangle2f = varying.triangle_uv
        i: Vec3f = AI @ jnp.array(  # pyright: ignore[reportUnknownMemberType]
            [_uv[1, 0] - _uv[0, 0], _uv[2, 0] - _uv[0, 0], 0]
        )
        j: Vec3f = AI @ jnp.array(  # pyright: ignore[reportUnknownMemberType]
            [_uv[1, 1] - _uv[0, 1], _uv[2, 1] - _uv[0, 1], 0]
        )

        B = lax.concatenate(  # pyright: ignore[reportUnknownMemberType]
            [
                normalise(i)[:, None],
                normalise(j)[:, None],
                normal[:, None],
            ],
            dimension=1,
        )
        assert isinstance(B, Float[Array, "3 3"])

        normal = normalise(B @ extra.normal_map[uv[0], uv[1]])
        assert isinstance(normal, Vec3f)

        texture_colour: Colour = extra.texture[uv[0], uv[1]]

        # light colour * intensity
        light_colour: Colour = (
            extra.light.colour
            * lax.dot(  # pyright: ignore[reportUnknownMemberType]
                normal,
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
            varying._replace(
                colour=lax.cond(  # pyright: ignore[reportUnknownMemberType]
                    jnp.all(  # pyright: ignore[reportUnknownMemberType]
                        light_colour >= 0
                    ),
                    lambda: texture_colour * light_colour,
                    lambda: jnp.zeros(3),  # pyright: ignore[reportUnknownMemberType]
                )
            ),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: PhongTextureDarbouxExtraFragmentData,
    ) -> Tuple[MixerOutput, PhongTextureDarbouxExtraMixerOutput]:
        mixer_output: MixerOutput
        extra_output: PhongTextureDarbouxExtraFragmentData
        mixer_output, extra_output = cast(
            Tuple[MixerOutput, PhongTextureDarbouxExtraFragmentData],
            Shader.mix(gl_FragDepth, keeps, extra),
        )
        assert isinstance(mixer_output, MixerOutput)
        assert isinstance(extra_output, PhongTextureDarbouxExtraFragmentData)

        return (
            mixer_output,
            PhongTextureDarbouxExtraMixerOutput(canvas=extra_output.colour),
        )
