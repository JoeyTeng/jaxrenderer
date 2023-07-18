from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, jaxtyped
import matplotlib.pyplot as plt

from renderer.geometry import Camera, normalise, to_homogeneous
from renderer.pipeline import render
from renderer.shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from renderer.shaders.gouraud import GouraudExtraInput, GouraudShader
from renderer.types import Buffers, Colour, LightSource, Vec2f, Vec3f, Vec4f
from renderer.utils import transpose_for_display

eye = jnp.array((0.0, 0, 2))
center = jnp.array((0.0, 0, 0))
up = jnp.array((0.0, 1, 0))

width = 1920
height = 1080
lowerbound = jnp.zeros(2, dtype=int)
dimension = jnp.array((width, height))
depth = 255

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
    targets=(lax.full((width, height, 3), 0.0),),
)
face_indices = jnp.array(
    (
        (0, 1, 2),
        (1, 3, 2),
        (0, 2, 4),
        (0, 4, 3),
        (2, 5, 1),
    )
)
position = jnp.array(
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
    colour=jnp.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 0.0),
        )
    ),
    normal=jax.vmap(lambda _: LightSource().direction)(position),
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
    colour: Colour = jnp.array([0.0, 0.0, 0.0])
    uv: Vec2f = jnp.zeros(2)


class ExtraMixerOutput(NamedTuple):
    canvas: Colour


class _Shader(Shader[ExtraInput, ExtraFragmentData, ExtraMixerOutput]):
    @staticmethod
    @jaxtyped
    @jax.jit
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: ExtraInput,
    ) -> tuple[PerVertex, ExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        normal: Vec3f = normalise(extra.normal[gl_VertexID])
        intensity: Float[Array, ""] = jnp.dot(
            normal,
            normalise(extra.light.direction),
        )
        assert isinstance(intensity, Float[Array, ""])

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
    @jax.jit
    def fragment(
        gl_FragCoord: Vec4f,
        gl_FrontFacing: Bool[Array, ""],
        gl_PointCoord: Vec2f,
        varying: ExtraFragmentData,
        extra: ExtraInput,
    ) -> tuple[PerFragment, ExtraFragmentData]:
        built_in: PerFragment = Shader.fragment(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            varying,
            extra,
        )[0]
        texture_colour: Colour = varying.colour
        a = jnp.modf(varying.uv)[0] < 0.5
        texture_colour = jnp.where(
            a[0] != a[1],
            texture_colour,
            texture_colour * 0.5,
        )

        return (
            built_in,
            ExtraFragmentData(colour=texture_colour, uv=varying.uv),
        )

    @staticmethod
    @jaxtyped
    @jax.jit
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: ExtraFragmentData,
    ) -> tuple[MixerOutput, ExtraMixerOutput]:
        mixer_output, extra_output = Shader.mix(gl_FragDepth, keeps, extra)

        return (
            mixer_output,
            ExtraMixerOutput(canvas=extra_output.colour),
        )


eye = jnp.array((0.0, 0, 1))
center = jnp.array((0.0, 0, 0))
up = jnp.array((0.0, 1, 0))

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
    targets=(lax.full((width, height, 3), 0.0),),
)
face_indices = jnp.array(((0, 1, 2),))
position = jnp.array(
    (
        (-1.0, -1.0, -2.0),
        (1.0, -1.0, -1.0),
        (0.0, 1.0, -1.0),
    )
)
extra = ExtraInput(
    position=position,
    colour=jnp.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    ),
    uv=jnp.array(
        (
            (0.0, 0.0),
            (10.0, 0.0),
            (0.0, 10.0),
        )
    ),
    normal=jax.vmap(lambda _: LightSource().direction)(position),
    light=LightSource(),
)

perspective_interpolation = render(camera, _Shader, buffers, face_indices, extra)

# show
fig, axs = plt.subplots(
    ncols=2,
    nrows=2,
    sharex=True,
    sharey=True,
    figsize=(16, 8),
)

axs[0][0].imshow(transpose_for_display(result.zbuffer))
axs[0][1].imshow(transpose_for_display(result.targets[0]))
axs[1][0].imshow(transpose_for_display(perspective_interpolation.zbuffer))
axs[1][1].imshow(transpose_for_display(perspective_interpolation.targets[0]))

plt.show()
