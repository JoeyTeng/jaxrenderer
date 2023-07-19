"""Example: Batch rendering a 12-frame animation of a rotating capsule."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from renderer import (
    GUID,
    CameraParameters,
    LightParameters,
    List,
    Renderer,
    Scene,
    ShadowParameters,
    Texture,
    UpAxis,
    batch_models,
    merge_objects,
    quaternion,
    rotation_matrix,
    transpose_for_display,
)

# PROCESS: Set up models and objects

scene: Scene = Scene()
texture: Texture = (
    jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (
            (255, 255, 255),  # White
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
        )
    ).reshape((2, 2, 3))[:, ::-1, :]
    / 255.0
)

scene, capsule_id = scene.add_capsule(
    radius=0.1,
    half_height=0.4,
    up_axis=UpAxis.Z,
    diffuse_map=texture,
)

capsule_obj_ids: List[GUID] = []
for i in range(12):
    scene, capsule_obj_id = scene.add_object_instance(capsule_id)
    capsule_obj_ids.append(capsule_obj_id)

    # to try both ways of setting orientation
    if i < 6:
        scene = scene.set_object_orientation(
            capsule_obj_id,
            # rotation_matrix=rotation_matrix((0., 1., 0.), 30 * (i - 6)),
            orientation=quaternion((0.0, 1.0, 0.0), 30 * (i - 6)),
        )
    else:
        scene = scene.set_object_orientation(
            capsule_obj_id,
            rotation_matrix=rotation_matrix((0.0, 1.0, 0.0), 30 * (i - 6)),
        )

# PROCESS: Set up camera and light

width = 640
height = 480

eye = (0.0, 4.0, 0.0)
target = (0.0, 0.0, 0.0)

light: LightParameters = LightParameters()
camera_params: CameraParameters = CameraParameters(
    viewWidth=width,
    viewHeight=height,
    position=eye,
    target=target,
)
shadow_param = ShadowParameters()

# PROCESS: Render

merged_models = [merge_objects([scene.objects[obj_id]]) for obj_id in capsule_obj_ids]
buffers = Renderer.create_buffers(width, height, len(capsule_obj_ids))
camera = Renderer.create_camera_from_parameters(camera_params)

_, (images,) = jax.vmap(  # pyright: ignore[reportUnknownVariableType]
    lambda model, buffer: Renderer.render(  # pyright: ignore
        model=model,  # pyright: ignore[reportUnknownArgumentType]
        light=light,
        camera=camera,
        buffers=buffer,  # pyright: ignore[reportUnknownArgumentType]
        shadow_param=shadow_param,
    )
)(batch_models(merged_models), buffers)

# PROCESS: show

from typing import cast

import matplotlib.animation as animation
import matplotlib.figure as figure
import matplotlib.image as mimage
import matplotlib.pyplot as plt

fig: figure.Figure
fig, ax = plt.subplots()  # pyright: ignore

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims: List[List[mimage.AxesImage]] = []
for i in range(cast(Float[Array, "_b _w _h _c"], images).shape[0]):
    img = cast(Float[Array, "_w _h _c"], images[i])
    im = cast(
        mimage.AxesImage,
        ax.imshow(  # pyright: ignore[reportUnknownMemberType]
            transpose_for_display(img), animated=True
        ),
    )
    if i == 0:
        # show an initial one first
        ax.imshow(  # pyright: ignore[reportUnknownMemberType]
            transpose_for_display(img)
        )

    ims.append([im])

ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=500,
    blit=True,
    repeat_delay=0,
)

plt.show()  # pyright: ignore[reportUnknownMemberType]
