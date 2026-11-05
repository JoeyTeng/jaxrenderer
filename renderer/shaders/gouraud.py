from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from .._meta_utils import add_tracing_name, is_array_type, typed_array
from .._meta_utils import typed_jaxtyped as jaxtyped
from .._meta_utils import typed_jit as jit
from ..geometry import normalise, to_homogeneous
from ..shader import (
    Fragment,
    FragmentInput,
    PerFragment,
    PerVertex,
    Shader,
    VertexInput,
)
from ..types import TRUE_ARRAY, Colour, FloatV, LightSource, Vec3f, Vec4f


class GouraudExtraInput(NamedTuple):
    """Extra input for Gouraud Shader.

    Attributes:
      - position: in world space, of each vertex.
      - colour: of each vertex
      - normal: in world space, of each vertex.
      - light: parallel light source, shared by all vertices.
    """

    position: Float[Array, "vertices 3"]  # in world space
    colour: Float[Array, "vertices 3"]
    normal: Float[Array, "vertices 3"]  # in world space
    light: LightSource


class GouraudExtraFragmentData(NamedTuple):
    colour: Colour = typed_array([0.0, 0.0, 0.0])


class GouraudExtraMixerOutput(NamedTuple):
    """When render to only one target, for simplicity."""

    canvas: Colour


class GouraudShader(
    Shader[GouraudExtraInput, GouraudExtraFragmentData, GouraudExtraMixerOutput]
):
    """Gouraud Shading with simple parallel lighting."""

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def vertex(
        inputs: VertexInput[GouraudExtraInput],
    ) -> tuple[PerVertex, GouraudExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(inputs.extra.position[inputs.gl_VertexID])
        gl_Position: Vec4f = inputs.camera.to_clip(position)
        assert is_array_type(gl_Position, Vec4f)

        # assume normal here is in world space. If it is in model space, it
        # must be transformed by the inverse transpose of the model matrix.
        # Ref: https://github.com/ssloy/tinyrenderer/wiki/Lesson-5:-Moving-the-camera#transformation-of-normal-vectors
        normal: Vec3f = normalise(inputs.extra.normal[inputs.gl_VertexID])
        intensity: FloatV = jnp.dot(
            normal,
            normalise(inputs.extra.light.direction),
        )
        assert is_array_type(intensity, FloatV)

        colour: Colour
        colour = (
            inputs.extra.colour[inputs.gl_VertexID]
            * inputs.extra.light.colour
            * intensity
        )

        return (
            PerVertex(gl_Position=gl_Position),
            GouraudExtraFragmentData(colour=colour),
        )

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def fragment(
        inputs: FragmentInput[GouraudExtraInput, GouraudExtraFragmentData],
    ) -> tuple[PerFragment, Fragment[GouraudExtraMixerOutput]]:
        return (
            PerFragment(
                keeps=typed_array(
                    (
                        inputs.gl_FrontFacing,
                        jnp.all(inputs.varying.colour >= 0),
                    )
                ).all(),
                use_default_depth=TRUE_ARRAY,
            ),
            Fragment(targets=GouraudExtraMixerOutput(canvas=inputs.varying.colour)),
        )
