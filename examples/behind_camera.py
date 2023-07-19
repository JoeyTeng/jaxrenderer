import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, UInt8

from renderer import (
    CameraParameters,
    LightParameters,
    List,
    Renderer,
    Scene,
    ShadowParameters,
    Texture,
    transpose_for_display,
)

# PROCESS: Set up models and objects

scene: Scene = Scene()
texture: Texture = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
).reshape((2, 2, 3))

scene, cube_model_1 = scene.add_cube(
    half_extents=(1.0, 1.0, 0.03),
    diffuse_map=texture,
    texture_scaling=(16.0, 16.0),
)
scene, cube_1 = scene.add_object_instance(cube_model_1)
scene = scene.set_object_position(cube_1, (0.0, 0.0, 0.0))
scene = scene.set_object_orientation(cube_1, (1.0, 0.0, 0.0, 0.0))

scene, cube_model_2 = scene.add_cube(
    half_extents=(10.0, 10.0, 0.03),
    diffuse_map=texture,
    texture_scaling=(160.0, 160.0),
)
scene, cube_2 = scene.add_object_instance(cube_model_2)
scene = scene.set_object_position(cube_2, (0.0, 0.0, 0.0))
scene = scene.set_object_orientation(cube_2, (1.0, 0.0, 0.0, 0.0))

# PROCESS: Set up camera and light

width = 640
height = 480
eye = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
    [2.5894797, -2.5876467, 1.9174135]
)
target = (0.0, 0.0, 0.0)

light: LightParameters = LightParameters()
camera: CameraParameters = CameraParameters(
    viewWidth=width,
    viewHeight=height,
    position=eye,
    target=target,
    hfov=58.0,
    vfov=32.625,
)
shadow_param = ShadowParameters()

# PROCESS: Render

images: List[UInt8[Array, "width height channels"]] = []

img = Renderer.get_camera_image(
    objects=[scene.objects[cube_1]],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow_param,
)
rgb_array = lax.clamp(  # pyright: ignore[reportUnknownMemberType]
    0.0, img * 255, 255.0
).astype(jnp.uint8)
images.append(rgb_array)

img = Renderer.get_camera_image(
    objects=[scene.objects[cube_2]],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow_param,
)
rgb_array = lax.clamp(  # pyright: ignore[reportUnknownMemberType]
    0.0, img * 255, 255.0
).astype(jnp.uint8)
images.append(rgb_array)

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
for i, img in enumerate(images):
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
