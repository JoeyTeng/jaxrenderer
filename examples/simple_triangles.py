from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    jaxtyped,  # pyright: ignore[reportUnknownVariableType]
)
import matplotlib.pyplot as plt

from renderer import jit
from renderer.geometry import Camera, normalise, to_homogeneous
from renderer.pipeline import render
from renderer.shader import (
    Fragment,
    FragmentInput,
    PerFragment,
    PerVertex,
    Shader,
    VertexInput,
)
from renderer.shaders.gouraud import (
    GouraudExtraInput,
    GouraudExtraMixerOutput,
    GouraudShader,
)
from renderer.types import (
    TRUE_ARRAY,
    Buffers,
    Colour,
    FloatV,
    LightSource,
    Vec2f,
    Vec3f,
    Vec4f,
)
from renderer.utils import transpose_for_display

eye = jnp.array((0.0, 0, 2))  # pyright: ignore[reportUnknownMemberType]
center = jnp.array((0.0, 0, 0))  # pyright: ignore[reportUnknownMemberType]
up = jnp.array((0.0, 1, 0))  # pyright: ignore[reportUnknownMemberType]

width = 1920
height = 1080
lowerbound = jnp.zeros(2, dtype=int)  # pyright: ignore[reportUnknownMemberType]
dimension = jnp.array((width, height))  # pyright: ignore[reportUnknownMemberType]
depth = jnp.array(255)  # pyright: ignore[reportUnknownMemberType]

camera: Camera = Camera.create(
    view=Camera.view_matrix(eye=eye, centre=center, up=up),
    projection=Camera.perspective_projection_matrix(
        fovy=90.0,
        aspect=1.0,
        z_near=-1.0,
        z_far=1.0,
    ),
    viewport=Camera.viewport_matrix(
        lowerbound=lowerbound,
        dimension=dimension,
        depth=depth,
    ),
)

buffers = Buffers(
    zbuffer=lax.full((width, height), 1.0),
    targets=GouraudExtraMixerOutput(canvas=lax.full((width, height, 3), 0.0)),
)
face_indices = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    (
        (0, 1, 2),
        (1, 3, 2),
        (0, 2, 4),
        (0, 4, 3),
        (2, 5, 1),
    )
)
position = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1, -1, 1.0),
        (-2, 0.0, 0.0),
    )
)
extra = GouraudExtraInput(
    position=position,
    colour=jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
        )
    ),
    normal=jax.vmap(lambda _: LightSource().direction)(position),  # pyright: ignore
    light=LightSource(),
)

result = render(camera, GouraudShader, buffers, face_indices, extra)


# Test perspective interpolation
class ExtraInput(NamedTuple):
    position: Float[Array, "vertices 3"]  # in world space
    normal: Float[Array, "vertices 3"]  # in world space
    uv: Float[Array, "vertices 2"]  # in texture space
    light: LightSource
    colour: Float[Array, "vertices 3"]


class ExtraFragmentData(NamedTuple):
    colour: Colour = jnp.array([0.0, 0.0, 0.0])  # pyright: ignore[reportUnknownMemberType]
    uv: Vec2f = jnp.zeros(2)  # pyright: ignore[reportUnknownMemberType]


class ExtraMixerOutput(NamedTuple):
    canvas: Colour


class _Shader(Shader[ExtraInput, ExtraFragmentData, ExtraMixerOutput]):
    @staticmethod
    @jaxtyped
    @jit
    def vertex(
        inputs: VertexInput[ExtraInput],
    ) -> tuple[PerVertex, ExtraFragmentData]:
        gl_VertexID = inputs.gl_VertexID
        camera = inputs.camera
        extra = inputs.extra
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)  # pyright: ignore[reportArgumentType]

        normal: Vec3f = normalise(extra.normal[gl_VertexID])
        intensity: FloatV = jnp.dot(
            normal,
            normalise(extra.light.direction),
        )
        assert isinstance(intensity, FloatV)  # pyright: ignore[reportArgumentType]

        light_colour: Colour
        light_colour = extra.light.colour * intensity

        return (
            PerVertex(gl_Position=gl_Position),
            ExtraFragmentData(
                colour=light_colour * extra.colour[gl_VertexID],
                uv=extra.uv[gl_VertexID],
            ),
        )

    @staticmethod
    @jaxtyped
    @jit
    def fragment(
        inputs: FragmentInput[ExtraInput, ExtraFragmentData],
    ) -> tuple[PerFragment, Fragment[ExtraMixerOutput]]:
        texture_colour: Colour = inputs.varying.colour
        a = jnp.modf(inputs.varying.uv)[0] < 0.5  # pyright: ignore[reportUnknownMemberType]
        texture_colour = jnp.where(
            a[0] != a[1],
            texture_colour,
            texture_colour * 0.5,
        )

        return (
            PerFragment(use_default_depth=TRUE_ARRAY),
            Fragment(targets=ExtraMixerOutput(canvas=texture_colour)),
        )


eye = jnp.array((0.0, 0, 1))  # pyright: ignore[reportUnknownMemberType]
center = jnp.array((0.0, 0, 0))  # pyright: ignore[reportUnknownMemberType]
up = jnp.array((0.0, 1, 0))  # pyright: ignore[reportUnknownMemberType]

camera: Camera = Camera.create(
    view=Camera.view_matrix(eye=eye, centre=center, up=up),
    projection=Camera.perspective_projection_matrix(
        fovy=90.0,
        aspect=1.0,
        z_near=-1.0,
        z_far=1.0,
    ),
    viewport=Camera.viewport_matrix(
        lowerbound=lowerbound,
        dimension=dimension,
        depth=depth,
    ),
)

buffers = Buffers(
    zbuffer=lax.full((width, height), 1.0),
    targets=ExtraMixerOutput(canvas=lax.full((width, height, 3), 0.0)),
)
face_indices = jnp.array(((0, 1, 2),))  # pyright: ignore[reportUnknownMemberType]
position = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    (
        (-1.0, -1.0, -2.0),
        (1.0, -1.0, -1.0),
        (0.0, 1.0, -1.0),
    )
)
extra = ExtraInput(
    position=position,
    colour=jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    ),
    uv=jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            (0.0, 0.0),
            (10.0, 0.0),
            (0.0, 10.0),
        )
    ),
    normal=jax.vmap(lambda _: LightSource().direction)(position),  # pyright: ignore
    light=LightSource(),
)

perspective_interpolation = render(camera, _Shader, buffers, face_indices, extra)

# show

fig, axs = plt.subplots(  # pyright: ignore
    ncols=2,
    nrows=2,
    sharex=True,
    sharey=True,
    figsize=(16, 8),
)

axs[0][0].imshow(transpose_for_display(result.zbuffer))  # pyright: ignore[reportUnknownMemberType]
axs[0][1].imshow(transpose_for_display(result.targets[0]))  # pyright: ignore[reportUnknownMemberType]
axs[1][0].imshow(transpose_for_display(perspective_interpolation.zbuffer))  # pyright: ignore[reportUnknownMemberType]
axs[1][1].imshow(transpose_for_display(perspective_interpolation.targets[0]))  # pyright: ignore[reportUnknownMemberType]

plt.show()  # pyright: ignore[reportUnknownMemberType]
