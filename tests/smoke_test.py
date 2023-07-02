from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, jaxtyped

from renderer.geometry import Camera, normalise, to_homogeneous
from renderer.pipeline import render
from renderer.shader import ID, MixerOutput, PerFragment, PerVertex, Shader
from renderer.shaders.gouraud import GouraudExtraInput, GouraudShader
from renderer.types import Buffers, Colour, LightSource, Vec2f, Vec3f, Vec4f
from renderer.utils import transpose_for_display


def test_render_batched_triangles():
    eye = jnp.array((0., 0, 2))
    center = jnp.array((0., 0, 0))
    up = jnp.array((0., 1, 0))

    width: int = 1920
    height: int = 1080
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth: int = 255
    default_z: float = 1.
    default_ch: float = 0.

    camera: Camera = Camera.create(
        view=Camera.view_matrix(eye=eye, centre=center, up=up),
        projection=Camera.perspective_projection_matrix(
            fovy=90.,
            aspect=1.,
            z_near=-1.,
            z_far=1.,
        ),
        viewport=Camera.viewport_matrix(
            lowerbound=lowerbound,
            dimension=dimension,
            depth=depth,
        ),
    )

    buffers = Buffers(
        zbuffer=lax.full((width, height), default_z),
        targets=(lax.full((width, height, 3), default_ch), ),
    )
    face_indices = jnp.array((
        (0, 1, 2),
        (1, 3, 2),
        (0, 2, 4),
        (0, 4, 3),
        (2, 5, 1),
    ))
    position = jnp.array((
        (0., 0., 0.),
        (2., 0., 0.),
        (0., 1., 0.),
        (1., 1., 0.),
        (-1, -1, 1.),
        (-2, 0., 0.),
    ))
    extra = GouraudExtraInput(
        position=position,
        colour=jnp.array((
            (1., 0., 0.),
            (0., 1., 0.),
            (0., 0., 1.),
            (0., 0., 0.),
            (1., 1., 1.),
            (1., 1., 0.),
        )),
        normal=jax.vmap(lambda _: LightSource().direction)(position),
        light=LightSource(),
    )

    result = render(camera, GouraudShader, buffers, face_indices, extra)

    assert len(result) == 2, ("The result should be a tuple of two elements")
    assert len(result[1]) == 1, (
        "The resultant attachment should has only one canvas")

    zbuffer, (canvas, ) = result
    zbuffer = transpose_for_display(zbuffer)
    canvas = transpose_for_display(canvas)

    # test shape
    assert zbuffer.shape == (height, width)
    assert canvas.shape == (height, width, 3)

    # test zbuffer
    assert (
        jnp.unique(zbuffer[293:528, 964:1423].astype(jnp.uint8)).shape == (1, )
    ), ("The depths of the triangle parallel to the camera should be uniform")
    assert ((zbuffer[590:1049, 964:1423] == default_z).all()), (
        "The depths of unrendered places should remain default")
    assert (zbuffer[551, 914] < zbuffer[1026, 92]), (
        "The depths of a triangle facing towards camera should be closer")

    # test canvas
    default_pixel = jnp.array([default_ch] * 3)
    empty_pixels = (canvas == default_pixel).all(axis=2).sum()
    assert empty_pixels > (width * height // 2), (
        "The canvas should be mostly empty")
    assert empty_pixels < (width * height), (
        "The canvas should not be all empty")


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


def test_perspective_interpolation():
    eye = jnp.array((0., 0, 1))
    center = jnp.array((0., 0, 0))
    up = jnp.array((0., 1, 0))

    width: int = 1920
    height: int = 1080
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth: int = 255
    default_z: float = 1.
    default_ch: float = 0.

    camera: Camera = Camera.create(
        view=Camera.view_matrix(eye=eye, centre=center, up=up),
        projection=Camera.perspective_projection_matrix(
            fovy=90.,
            aspect=1.,
            z_near=-1.,
            z_far=1.,
        ),
        viewport=Camera.viewport_matrix(
            lowerbound=lowerbound,
            dimension=dimension,
            depth=depth,
        ),
    )

    buffers = Buffers(
        zbuffer=lax.full((width, height), 1.),
        targets=(lax.full((width, height, 3), 0.), ),
    )
    face_indices = jnp.array(((0, 1, 2), ))
    position = jnp.array((
        (-1., -1., -2.),
        (1., -1., -1.),
        (0., 1., -1.),
    ))
    extra = ExtraInput(
        position=position,
        colour=jnp.array((
            (1., 0., 0.),
            (0., 1., 0.),
            (0., 0., 1.),
        )),
        uv=jnp.array((
            (0., 0.),
            (10., 0.),
            (0., 10.),
        )),
        normal=jax.vmap(lambda _: LightSource().direction)(position),
        light=LightSource(),
    )

    result = render(camera, _Shader, buffers, face_indices, extra)

    assert len(result) == 2, ("The result should be a tuple of two elements")
    assert len(result[1]) == 1, (
        "The resultant attachment should has only one canvas")

    zbuffer, (canvas, ) = result
    zbuffer = transpose_for_display(zbuffer)
    canvas = transpose_for_display(canvas)

    # test shape
    assert zbuffer.shape == (height, width)
    assert canvas.shape == (height, width, 3)

    # test zbuffer
    assert ((zbuffer == default_z).sum() > (width * height // 2)), (
        "The depths of unrendered places should remain default, "
        "which is the majority of the screen space")
    assert (zbuffer[679, 701] < zbuffer[779, 1388]), (
        "The depths of a triangle facing towards camera should be closer")

    # test canvas
    default_pixel = jnp.array([default_ch] * 3)
    empty_pixels = (canvas == default_pixel).all(axis=2).sum()
    assert empty_pixels > (width * height // 2), (
        "The canvas should be mostly empty")
    assert empty_pixels < (width * height), (
        "The canvas should not be all empty")

    # TODO: find a way to specify the effect of "perspective transform"
