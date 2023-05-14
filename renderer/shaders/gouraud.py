from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, jaxtyped

from ..shader import ID, PerFragment, PerVertex, Shader
from ..geometry import Camera, to_homogeneous
from ..types import Colour, LightSource, Vec2f, Vec3f, Vec4f

jax.config.update('jax_array', True)


class GouraudExtraVertexInput(NamedTuple):
    position: Vec3f  # in world space
    colour: Colour
    normal: Vec3f  # in world space
    light: LightSource


class GouraudExtraFragmentData(NamedTuple):
    colour: Colour = jnp.array([0.0, 0.0, 0.0, 1.0])


class GouraudExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""
    canvas: Colour


class GouraudShader(Shader[GouraudExtraVertexInput, GouraudExtraFragmentData,
                           GouraudExtraMixerOutput]):
    """Gouraud Shading without lighting."""

    @staticmethod
    @jaxtyped
    @jax.jit
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: GouraudExtraVertexInput,
    ) -> tuple[PerVertex, GouraudExtraFragmentData]:
        position: Vec4f = to_homogeneous(extra.position)
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        # assume normal here is in world space. If it is in model space, it
        # must be transformed by the inverse transpose of the model matrix.
        # Ref: https://github.com/ssloy/tinyrenderer/wiki/Lesson-5:-Moving-the-camera#transformation-of-normal-vectors
        normal: Vec3f = extra.normal
        intensity: Float[Array, ""] = jnp.dot(normal, extra.light.direction)
        assert isinstance(intensity, Float[Array, ""])

        colour: Colour = extra.colour * extra.light.colour * intensity

        return (
            PerVertex(gl_Position=gl_Position),
            GouraudExtraFragmentData(colour=colour),
        )

    @staticmethod
    @jaxtyped
    @jax.jit
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: Bool[Array, ""],
        gl_PointCoord: Vec2f,
        extra: GouraudExtraFragmentData,
    ) -> tuple[PerFragment, GouraudExtraFragmentData]:
        built_in: PerFragment = Shader.fragment(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            extra,
        )[0]
        assert isinstance(built_in, PerFragment)

        return (
            PerFragment(
                gl_FragDepth=built_in.gl_FragDepth,
                keeps=jnp.logical_and(built_in.keeps, gl_FrontFacing),
            ),
            extra,
        )
