from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.image as mpimg
from jax import lax

from renderer.renderer import Canvas, Colour, line
from test_resources.utils import Model, make_model

jax.config.update('jax_array', True)


def test_line():
    canvas: Canvas = jnp.zeros((100, 100, 3))

    red: Colour = jnp.array((1., 0, 0))
    white: Colour = jnp.array((1., 1., 1.))
    canvasA: Canvas = line(13, 20, 80, 40, canvas, white)
    canvasB: Canvas = line(20, 13, 40, 80, canvasA, red)
    canvasC: Canvas = line(80, 40, 13, 20, canvasB, red)

    mpimg.imsave("test_line.png", canvasC, origin='lower')


def test_wireframe_basic():
    file_path: str = "test_resources/obj/african_head.obj"
    model: Model = make_model(open(file_path, 'r').readlines())

    width: int = 800
    height: int = 800

    canvas: Canvas = jnp.zeros((height, width, 3))
    white: Colour = jnp.ones(3)

    def g(j: int, state: Tuple[int, Canvas]) -> Tuple[int, Canvas]:
        canvas: Canvas
        i, canvas = state
        v0 = model.verts[model.faces[i, j]]
        v1 = model.verts[model.faces[i, (j + 1) % 3]]
        x0: int = ((v0[0] + 1.) * width / 2.).astype(int)
        y0: int = ((v0[1] + 1.) * height / 2.).astype(int)
        x1: int = ((v1[0] + 1.) * width / 2.).astype(int)
        y1: int = ((v1[1] + 1.) * height / 2.).astype(int)
        canvas = line(x0, y0, x1, y1, canvas, white)

        return i, canvas

    def f(i: int, canvas: Canvas) -> Canvas:
        canvas = lax.fori_loop(0, 3, g, (i, canvas))[1]

        return canvas

    canvas = lax.fori_loop(0, model.nfaces, f, canvas)
    canvas = lax.transpose(canvas, (1, 0, 2))

    mpimg.imsave("test_wireframe_basic.png", canvas, origin='lower')


if __name__ == '__main__':
    test_line()
    test_wireframe_basic()
