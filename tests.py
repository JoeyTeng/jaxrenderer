import jax
import jax.lax as lax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from renderer.geometry import Camera
from renderer.pipeline import render
from renderer.types import Buffers, LightSource
from renderer.shaders.gouraud import GouraudExtraInput, GouraudShader
from renderer.utils import transpose_for_display

from test_resources.utils import Model, load_tga, make_model


def gouraud_shader_with_random_colour(model: Model):
    eye = jnp.array((0, 0, 1.))
    center = jnp.array((0, 0, 0))
    up = jnp.array((0, 1, 0))

    width = height = 800
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth = 1.

    camera: Camera = Camera.create(
        model_view=Camera.model_view_matrix(eye=eye, centre=center, up=up),
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
        zbuffer=lax.full(
            (width, height),
            -20.,
        ),
        targets=(lax.full((width, height, 3), 0.), ),
    )

    extra = GouraudExtraInput(
        # flatten so each vertex has its own "extra"
        position=model.verts[model.faces.reshape((-1, ))],
        colour=jnp.ones((model.nfaces * 3, 3), dtype=jnp.single),
        normal=model.norms[model.faces_norm.reshape((-1, ))],
        light=LightSource(direction=jnp.array((0., 0., 1.))),
    )

    result = render(
        camera,
        GouraudShader,
        buffers,
        # such that vertex (3i, 3i + 1, 3i + 2) corresponds to primitive i
        jnp.arange(model.nfaces * 3).reshape((model.nfaces, 3)),
        extra,
    )

    # show
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))

    axs[0].imshow(transpose_for_display(result.zbuffer), origin='lower')
    axs[1].imshow(transpose_for_display(result.targets[0]), origin='lower')


if __name__ == '__main__':
    model = make_model(open('test_resources/obj/african_head.obj').readlines())
    gouraud_shader_with_random_colour(model)
    texture = load_tga('test_resources/tga/african_head_diffuse.tga')
    plt.show()
