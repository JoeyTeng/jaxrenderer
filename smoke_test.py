import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from renderer.shaders.gouraud import GouraudShader, GouraudExtraInput
from renderer.pipeline import render
from renderer.geometry import Camera
from renderer.types import Buffers, LightSource
from renderer.utils import transpose_for_display

eye = jnp.array((0., 0, 2))
center = jnp.array((0., 0, 0))
up = jnp.array((0., 1, 0))

width = 1920
height = 1080
lowerbound = jnp.zeros(2, dtype=int)
dimension = jnp.array((width, height))
depth = 255

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
        # jnp.finfo(jnp.single).min,
        -1.,
    ),
    targets=(lax.full((width, height, 3), 0.), ),
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

# show
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))

axs[0].imshow(transpose_for_display(result.zbuffer), origin='lower')
axs[1].imshow(transpose_for_display(result.targets[0]), origin='lower')

plt.show()
