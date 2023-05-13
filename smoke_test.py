import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from renderer.shaders.gouraud import GouraudShader, GouraudExtraVertexInput
from renderer.pipeline import render
from renderer.geometry import Camera, to_homogeneous
from renderer.types import Buffers
from renderer.utils import transpose_for_display

eye = jnp.array((0, 0, 2))
center = jnp.array((0, 0, 0))
up = jnp.array((0, 1, 0))

width = height = 800
# width = height = 4
lowerbound = jnp.zeros(2, dtype=int)
dimension = jnp.array((width, height))
depth = 2

camera: Camera = Camera.create(
    model_view=Camera.model_view_matrix(eye=eye, centre=center, up=up),
    projection=Camera.perspective_projection_matrix(
        fovy=90.,
        aspect=1.,
        z_near=-1e-3,
        z_far=1e-3,
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
        jnp.finfo(jnp.single).min,
    ),
    targets=(lax.full((width, height, 3), 0.), ),
)
face_indices = jnp.array(((0, 1, 2), (1, 3, 2)))
extra = GouraudExtraVertexInput(
    position=jnp.array((
        (0., 0., 0.),
        (2., 0., 0.),
        (0., 1., 0.),
        (1., 1., 0.),
    )),
    colour=jnp.eye(4, 3, dtype=jnp.single),
)

print("screen space")
print(camera.to_screen(to_homogeneous(extra.position)))

result = render(camera, GouraudShader, buffers, face_indices, extra)
# print("render result")
# print(result)

# show
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))

axs[0].imshow(transpose_for_display(result.zbuffer), origin='lower')
axs[1].imshow(transpose_for_display(result.targets[0]), origin='lower')

plt.show()
