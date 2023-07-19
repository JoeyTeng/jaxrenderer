from functools import partial
from typing import cast

import jax
import jax.lax as lax
import jax.numpy as jnp

from renderer import Buffers, Camera, LightSource, render
from renderer.shaders.gouraud import GouraudExtraInput, GouraudShader
from renderer.types import FloatV

eye = jnp.array((0.0, 0, 2))  # pyright: ignore[reportUnknownMemberType]
center = jnp.array((0.0, 0, 0))  # pyright: ignore[reportUnknownMemberType]
up = jnp.array((0.0, 1, 0))  # pyright: ignore[reportUnknownMemberType]

width = 84
height = 84
lowerbound = jnp.zeros(2, dtype=int)  # pyright: ignore[reportUnknownMemberType]
dimension = jnp.array((width, height))  # pyright: ignore[reportUnknownMemberType]
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
    zbuffer=lax.full(  # pyright: ignore[reportUnknownMemberType]
        (width, height),
        0.0,
    ),
    targets=(
        lax.full(  # pyright: ignore[reportUnknownMemberType]
            (width, height, 3),
            0.0,
        ),
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

_render = partial(
    render,
    shader=GouraudShader,
    buffers=buffers,
    face_indices=face_indices,
)


def test_grad_over_camera():
    def camera_depth_loss(a: Camera) -> FloatV:
        depth = _render(camera=a, extra=extra)[0]

        return jnp.sum(depth)  # pyright: ignore[reportUnknownMemberType]

    grad_camera = cast(
        FloatV,
        jax.jit(  # pyright: ignore[reportUnknownMemberType]
            jax.grad(camera_depth_loss)  # pyright: ignore[reportUnknownMemberType]
        )(camera),
    )

    jax.tree_map(lambda a: a.block_until_ready(), grad_camera)

    assert True


def test_grad_over_light():
    def _render_light(light: LightSource) -> FloatV:
        _, (canvas,) = _render(
            camera=camera,
            extra=extra._replace(light=light),
        )

        return canvas.sum()  # pyright: ignore[reportUnknownMemberType]

    grad_light = cast(
        FloatV,
        jax.jit(  # pyright: ignore[reportUnknownMemberType]
            jax.grad(_render_light)  # pyright: ignore[reportUnknownMemberType]
        )(LightSource()),
    )

    jax.tree_map(lambda a: a.block_until_ready(), grad_light)

    assert True
