from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, jaxtyped

from ..shader import ID, PerFragment, PerVertex, Shader
from ..geometry import Camera, normalise, to_homogeneous
from ..types import Colour, LightSource, Vec2f, Vec3f, Vec4f

jax.config.update('jax_array', True)


class DepthExtraInput(NamedTuple):
    """Extra input for Depth Shader.

    Attributes:
      - position: in world space, of each vertex.
    """
    position: Float[Array, "vertices 3"]  # in world space


class DepthExtraFragmentData(NamedTuple):
    pass


class DepthExtraMixerOutput(NamedTuple):
    pass


class DepthShader(Shader[DepthExtraInput, DepthExtraFragmentData,
                         DepthExtraMixerOutput]):
    """Depth Shading."""

    @staticmethod
    @jaxtyped
    @jax.jit
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: DepthExtraInput,
    ) -> tuple[PerVertex, DepthExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        return (
            PerVertex(gl_Position=gl_Position),
            DepthExtraFragmentData(),
        )
