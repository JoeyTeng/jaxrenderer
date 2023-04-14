from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.image as mpimg
from jax import lax, random

from renderer.original_implementation import (Canvas, Colour, Triangle,
                                              Triangle3D, TriangleColours,
                                              Vec2i, Vec3f, ZBuffer, line,
                                              triangle, triangle3d,
                                              triangle_texture)
from test_resources.utils import Model, Texture, load_tga, make_model

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

    light_dir: Vec3f = jnp.array((0., 0., -1.))

    @jax.jit
    def f(i: int, _canvas: Canvas) -> Canvas:
        world_coords: Triangle3D = model.verts[model.faces[i]]
        screen_coords: Triangle = ((world_coords[:, :2] + 1) * wh_vec //
                                   2).astype(int)

        n: Vec3f = jnp.cross(
            world_coords[2, :] - world_coords[0, :],
            world_coords[1, :] - world_coords[0, :],
        )
        n = n / jnp.linalg.norm(n)
        intensity: float = jnp.dot(n, light_dir)
        colour: Vec3f = jnp.ones(3) * intensity

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


def test_flat_shading_simple_light_with_zbuffer():
    file_path: str = "test_resources/obj/african_head.obj"
    model: Model = make_model(open(file_path, 'r').readlines())

    width: int = 800
    height: int = 800

    # 3 constants for world-to-screen coordinate conversions
    ws_add = jnp.array((1, 1, 0))
    ws_mul: Vec2i = jnp.array((width, height, 1))
    ws_div = jnp.array((2, 2, 1))
    # width, height since it will be transposed in final step
    canvas: Canvas = jnp.zeros((width, height, 3))
    zbuffer: ZBuffer = (jnp.ones(
        (width, height), dtype=jnp.single) * jnp.finfo(dtype=jnp.single).min)

    light_dir: Vec3f = jnp.array((0., 0., -1.))

    @jax.jit
    def f(i: int, state: Tuple[ZBuffer, Canvas]) -> Tuple[ZBuffer, Canvas]:
        _zbuffer: ZBuffer
        _canvas: Canvas
        _zbuffer, _canvas = state

        world_coords: Triangle3D = model.verts[model.faces[i]].astype(float)
        screen_coords: Triangle3D = ((world_coords + ws_add) * ws_mul //
                                     ws_div).at[:, 2].set(world_coords[:, 2])

        n: Vec3f = jnp.cross(
            world_coords[2, :] - world_coords[0, :],
            world_coords[1, :] - world_coords[0, :],
        )
        n = n / jnp.linalg.norm(n)
        intensity: float = jnp.dot(n, light_dir)
        colour: Vec3f = jnp.ones(3) * intensity

        # with back-face culling
        _zbuffer, _canvas = lax.cond(
            intensity > 0,
            lambda: triangle3d(screen_coords, _zbuffer, _canvas, colour),
            lambda: (_zbuffer, _canvas),
        )

        return _zbuffer, _canvas

    canvas = lax.fori_loop(0, model.nfaces, f, (zbuffer, canvas))[1]
    canvas = lax.transpose(canvas, (1, 0, 2))

    mpimg.imsave(
        "test_flat_shading_simple_light_with_zbuffer.png",
        canvas,
        origin='lower',
    )


def test_flat_shading_texture():
    obj_path: str = "test_resources/obj/african_head.obj"
    model: Model = make_model(open(obj_path, 'r').readlines())
    texture_path: str = "test_resources/tga/african_head_diffuse.tga"
    texture: Texture = load_tga(texture_path)

    width: int = 800
    height: int = 800

    # 3 constants for world-to-screen coordinate conversions
    ws_add = jnp.array((1, 1, 0))
    ws_mul: Vec2i = jnp.array((width, height, 1))
    ws_div = jnp.array((2, 2, 1))
    # 3 constants for world-to-texture coordinate conversions
    wt_add = jnp.array((1, 1))
    wt_mul = jnp.array(texture.shape[:2])
    wt_div = jnp.array((2, 2))
    # width, height since it will be transposed in final step
    canvas: Canvas = jnp.zeros((width, height, 3))
    zbuffer: ZBuffer = (jnp.ones(
        (width, height), dtype=jnp.single) * jnp.finfo(dtype=jnp.single).min)

    light_dir: Vec3f = jnp.array((0., 0., -1.))

    @jax.jit
    def f(i: int, state: Tuple[ZBuffer, Canvas]) -> Tuple[ZBuffer, Canvas]:
        _zbuff: ZBuffer
        _canvas: Canvas
        _zbuff, _canvas = state

        world_coords: Triangle3D = model.verts[model.faces[i]].astype(float)
        screen_coords: Triangle3D = ((world_coords + ws_add) * ws_mul //
                                     ws_div).at[:, 2].set(world_coords[:, 2])
        texture_coords: Triangle = ((world_coords[:, :2] + wt_add) * wt_mul //
                                    wt_div).astype(jnp.int32)

        n: Vec3f = jnp.cross(
            world_coords[2, :] - world_coords[0, :],
            world_coords[1, :] - world_coords[0, :],
        )
        n = n / jnp.linalg.norm(n)
        intensity: float = jnp.dot(n, light_dir)
        tex_xs: Vec2i
        tex_ys: Vec2i
        tex_xs, tex_ys = texture_coords[:, 0], texture_coords[:, 1]
        colours: TriangleColours = jnp.vstack((
            texture[tex_xs[0], tex_ys[0], :],
            texture[tex_xs[1], tex_ys[1], :],
            texture[tex_xs[2], tex_ys[2], :],
        )) * intensity / 255.0

        # with back-face culling
        _zbuff, _canvas = lax.cond(
            intensity > 0,
            lambda: triangle_texture(screen_coords, _zbuff, _canvas, colours),
            lambda: (_zbuff, _canvas),
        )

        return _zbuff, _canvas

    canvas = lax.fori_loop(0, model.nfaces, f, (zbuffer, canvas))[1]
    canvas = lax.transpose(canvas, (1, 0, 2))

    mpimg.imsave(
        "test_flat_shading_simple_light_with_zbuffer.png",
        canvas,
        origin='lower',
    )


if __name__ == '__main__':
    test_line()
    test_wireframe_basic()
    test_triangle()
    test_flat_shading_random_colour()
    test_flat_shading_simple_light()
    test_flat_shading_simple_light_with_zbuffer()
    test_flat_shading_texture()
