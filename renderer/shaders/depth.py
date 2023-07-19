from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from functools import partial
from typing import NamedTuple

import jax
from jaxtyping import Array, Float
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from .._backport import Tuple
from .._meta_utils import add_tracing_name
from .._meta_utils import typed_jit as jit
from ..geometry import Camera, to_homogeneous
from ..shader import ID, PerVertex, Shader
from ..types import Vec4f

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]


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


class DepthShader(
    Shader[DepthExtraInput, DepthExtraFragmentData, DepthExtraMixerOutput]
):
    """Depth Shading."""

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: DepthExtraInput,
    ) -> Tuple[PerVertex, DepthExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        return (
            PerVertex(gl_Position=gl_Position),
            DepthExtraFragmentData(),
        )
