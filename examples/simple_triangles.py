from typing import NamedTuple, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from renderer import Tuple, jit
from renderer.geometry import Camera, normalise, to_homogeneous
from renderer.pipeline import render
from renderer.shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from renderer.shaders.gouraud import GouraudExtraInput, GouraudShader
from renderer.types import (
    BoolV,
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
    zbuffer=lax.full((width, height), 1.0),  # pyright: ignore[reportUnknownMemberType]
    targets=(
        lax.full((width, height, 3), 0.0),  # pyright: ignore[reportUnknownMemberType]
    ),
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
    colour: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        [0.0, 0.0, 0.0]
    )
    uv: Vec2f = jnp.zeros(2)  # pyright: ignore[reportUnknownMemberType]


class ExtraMixerOutput(NamedTuple):
    canvas: Colour


class _Shader(Shader[ExtraInput, ExtraFragmentData, ExtraMixerOutput]):
    @staticmethod
    @jaxtyped
    @jit
    def vertex(
        gl_VertexID: ID,
        gl_InstanceID: ID,
        camera: Camera,
        extra: ExtraInput,
    ) -> Tuple[PerVertex, ExtraFragmentData]:
        # Use gl_VertexID to index in `extra` buffer.
        position: Vec4f = to_homogeneous(extra.position[gl_VertexID])
        gl_Position: Vec4f = camera.to_clip(position)
        assert isinstance(gl_Position, Vec4f)

        normal: Vec3f = normalise(extra.normal[gl_VertexID])
        intensity = cast(
            FloatV,
            jnp.dot(
                normal,
                normalise(extra.light.direction),
            ),
        )
        assert isinstance(intensity, FloatV)

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
        gl_FragCoord: Vec4f,
        gl_FrontFacing: BoolV,
        gl_PointCoord: Vec2f,
        varying: ExtraFragmentData,
        extra: ExtraInput,
    ) -> Tuple[PerFragment, ExtraFragmentData]:
        built_in: PerFragment = Shader.fragment(
            gl_FragCoord,
            gl_FrontFacing,
            gl_PointCoord,
            varying,
            extra,
        )[0]
        texture_colour: Colour = varying.colour
        a = cast(BoolV, jnp.modf(varying.uv)[0] < 0.5)
        texture_colour = jnp.where(  # pyright: ignore[reportUnknownMemberType]
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
    @jit
    def mix(
        gl_FragDepth: Float[Array, "primitives"],
        keeps: Bool[Array, "primitives"],
        extra: ExtraFragmentData,
    ) -> Tuple[MixerOutput, ExtraMixerOutput]:
        mixer_output, extra_output = Shader.mix(gl_FragDepth, keeps, extra)
        assert isinstance(extra_output, ExtraFragmentData)

        return (
            mixer_output,
            ExtraMixerOutput(canvas=extra_output.colour),
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
    zbuffer=lax.full((width, height), 1.0),  # pyright: ignore[reportUnknownMemberType]
    targets=(
        lax.full((width, height, 3), 0.0),  # pyright: ignore[reportUnknownMemberType]
    ),
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

import matplotlib.pyplot as plt

fig, axs = plt.subplots(  # pyright: ignore
    ncols=2,
    nrows=2,
    sharex=True,
    sharey=True,
    figsize=(16, 8),
)

axs[0][0].imshow(  # pyright: ignore[reportUnknownMemberType]
    transpose_for_display(result.zbuffer)
)
axs[0][1].imshow(  # pyright: ignore[reportUnknownMemberType]
    transpose_for_display(result.targets[0])
)
axs[1][0].imshow(  # pyright: ignore[reportUnknownMemberType]
    transpose_for_display(perspective_interpolation.zbuffer)
)
axs[1][1].imshow(  # pyright: ignore[reportUnknownMemberType]
    transpose_for_display(perspective_interpolation.targets[0])
)

plt.show()  # pyright: ignore[reportUnknownMemberType]
