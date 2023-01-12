from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.image as mpimg
from jax import lax

from renderer.renderer import Canvas, Colour, Vec2i, line, triangle
from test_resources.utils import Model, make_model

jax.config.update('jax_array', True)


def test_line():
    canvas: Canvas = jnp.zeros((100, 100, 3))

    red: Colour = jnp.array((1., 0, 0))
    white: Colour = jnp.array((1., 1., 1.))
    t0: Vec2i = jnp.array((13, 20))
    t1: Vec2i = jnp.array((80, 40))
    t2: Vec2i = jnp.array((20, 13))
    t3: Vec2i = jnp.array((40, 80))
    canvasA: Canvas = line(t0, t1, canvas, white)
    canvasB: Canvas = line(t2, t3, canvasA, red)
    canvasC: Canvas = line(t1, t0, canvasB, red)

    mpimg.imsave("test_line.png", canvasC, origin='lower')


def test_wireframe_basic():
    file_path: str = "test_resources/obj/african_head.obj"
    model: Model = make_model(open(file_path, 'r').readlines())

    width: int = 800
    height: int = 800

    wh_vec: Vec2i = jnp.array((width, height))

    canvas: Canvas = jnp.zeros((height, width, 3))
    white: Colour = jnp.ones(3)

    def g(j: int, state: Tuple[int, Canvas]) -> Tuple[int, Canvas]:
        canvas: Canvas
        i, canvas = state
        v0 = model.verts[model.faces[i, j]]
        v1 = model.verts[model.faces[i, (j + 1) % 3]]
        t0 = ((v0[:2] + 1.) * wh_vec / 2.).astype(int)
        t1 = ((v1[:2] + 1.) * wh_vec / 2.).astype(int)
        canvas = line(t0, t1, canvas, white)

        return i, canvas

    def f(i: int, canvas: Canvas) -> Canvas:
        canvas = lax.fori_loop(0, 3, g, (i, canvas))[1]

        return canvas

    canvas = lax.fori_loop(0, model.nfaces, f, canvas)
    canvas = lax.transpose(canvas, (1, 0, 2))

    mpimg.imsave("test_wireframe_basic.png", canvas, origin='lower')


def test_triangle():
    canvas: Canvas = jnp.zeros((200, 200, 3))

    red: Colour = jnp.array((1., .0, .0))
    white: Colour = jnp.ones(3)
    green: Colour = jnp.array((.0, 1., .0))

    triangles = [
        (((10, 70), (50, 160), (70, 80)), red),
        (((180, 50), (150, 1), (70, 180)), white),
        (((180, 150), (120, 160), (130, 180)), green),
    ]

    for _t, colour in triangles:
        v0, v1, v2 = tuple(map(jnp.array, _t))
        canvas = triangle(v0, v1, v2, canvas, colour)

    canvas = lax.transpose(canvas, (1, 0, 2))

    mpimg.imsave("test_triangle.png", canvas, origin='lower')


if __name__ == '__main__':
    test_line()
    test_wireframe_basic()
    test_triangle()
