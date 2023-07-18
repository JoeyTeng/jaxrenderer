from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp

from renderer import Buffers, Camera, LightSource, render
from renderer.shaders.gouraud import GouraudExtraInput, GouraudShader

eye = jnp.array((0.0, 0, 2))
center = jnp.array((0.0, 0, 0))
up = jnp.array((0.0, 1, 0))

width = 84
height = 84
lowerbound = jnp.zeros(2, dtype=int)
dimension = jnp.array((width, height))
depth = 1

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
    zbuffer=lax.full((width, height), 0.0),
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

_render = partial(
    render,
    shader=GouraudShader,
    buffers=buffers,
    face_indices=face_indices,
)


def test_grad_over_camera():
    grad_camera = jax.jit(
        jax.grad(
            lambda a: _render(
                camera=a,
                extra=extra,
            )[0].sum()
        )
    )(camera)

    jax.tree_map(lambda a: a.block_until_ready(), grad_camera)

    assert True


def test_grad_over_light():
    def _render_light(light: LightSource):
        _, (canvas,) = _render(
            camera=camera,
            extra=extra._replace(light=light),
        )

        return canvas.sum()

    grad_light = jax.jit(jax.grad(_render_light))(LightSource())

    jax.tree_map(lambda a: a.block_until_ready(), grad_light)

    assert True
