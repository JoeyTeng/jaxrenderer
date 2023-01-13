from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.image as mpimg
from jax import lax, random

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
    # width, height since it will be transposed in final step
    canvas: Canvas = jnp.zeros((width, height, 3))
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
    width: int = 200
    height: int = 200
    # width, height since it will be transposed in final step
    canvas: Canvas = jnp.zeros((width, height, 3))

    red: Colour = jnp.array((1., .0, .0))
    white: Colour = jnp.ones(3)
    green: Colour = jnp.array((.0, 1., .0))

    triangles = [
        (((10, 70), (50, 160), (70, 80)), red),
        (((180, 50), (150, 1), (70, 180)), white),
        (((180, 150), (120, 160), (130, 180)), green),
    ]

    for _t, colour in triangles:
        canvas = triangle(jnp.array(_t), canvas, colour)

    canvas = lax.transpose(canvas, (1, 0, 2))

    mpimg.imsave("test_triangle.png", canvas, origin='lower')


def test_flat_shading_random_colour():
    file_path: str = "test_resources/obj/african_head.obj"
    model: Model = make_model(open(file_path, 'r').readlines())

    width: int = 800
    height: int = 800

    wh_vec: Vec2i = jnp.array((width, height))
    # width, height since it will be transposed in final step
    canvas: Canvas = jnp.zeros((width, height, 3))

    key = random.PRNGKey(20230113)
    key, *subkeys = random.split(key, num=model.nfaces + 1)
    rand_colour = partial(random.uniform, shape=(3, ))
    colours = jax.vmap(rand_colour)(jnp.array(subkeys))

    @jax.jit
    def f(i: int, _canvas: Canvas) -> Canvas:
        world_coords = model.verts[model.faces[i]]
        screen_coords = ((world_coords[:, :2] + 1) * wh_vec // 2).astype(int)

        _canvas = triangle(screen_coords, _canvas, colours[i])

        return _canvas

    canvas = lax.fori_loop(0, model.nfaces, f, canvas)
    canvas = lax.transpose(canvas, (1, 0, 2))

    mpimg.imsave("test_flat_shading_random_colour.png", canvas, origin='lower')


def test_flat_shading_simple_light():
    file_path: str = "test_resources/obj/african_head.obj"
    model: Model = make_model(open(file_path, 'r').readlines())

    width: int = 800
    height: int = 800

    wh_vec: Vec2i = jnp.array((width, height))
    # width, height since it will be transposed in final step
    canvas: Canvas = jnp.zeros((width, height, 3))

    light_dir = jnp.array((0., 0., -1.))

    @jax.jit
    def f(i: int, _canvas: Canvas) -> Canvas:
        world_coords = model.verts[model.faces[i]]
        screen_coords = ((world_coords[:, :2] + 1) * wh_vec // 2).astype(int)

        n = jnp.cross(
            world_coords[2, :] - world_coords[0, :],
            world_coords[1, :] - world_coords[0, :],
        )
        n = n / jnp.linalg.norm(n)
        intensity: float = jnp.dot(n, light_dir)
        colour = jnp.ones(3) * intensity

        # with back-face culling
        _canvas = lax.cond(
            intensity > 0,
            lambda: triangle(screen_coords, _canvas, colour),
            lambda: _canvas,
        )

        return _canvas

    canvas = lax.fori_loop(0, model.nfaces, f, canvas)
    canvas = lax.transpose(canvas, (1, 0, 2))

    mpimg.imsave("test_flat_shading_simple_light.png", canvas, origin='lower')


if __name__ == '__main__':
    test_line()
    test_wireframe_basic()
    test_triangle()
    test_flat_shading_random_colour()
    test_flat_shading_simple_light()
