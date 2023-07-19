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
    TypeAlias,
    UpAxis,
    build_texture_from_PyTinyrenderer,
    transpose_for_display,
)

# PROCESS: Set up models and objects

scene: Scene = Scene()
texture: Texture = (
    build_texture_from_PyTinyrenderer(
        jnp.array(  # pyright: ignore[reportUnknownMemberType]
            (
                255,
                255,
                255,  # White
                255,
                0,
                0,  # Red
                0,
                255,
                0,  # Green
                0,
                0,
                255,  # Blue
            )
        ),
        2,
        2,
    )
    / 255.0
)

scene, capx_model_id = scene.add_capsule(
    radius=0.1,
    half_height=0.4,
    up_axis=UpAxis.X,
    diffuse_map=texture,
)
scene, capy_model_id = scene.add_capsule(
    radius=0.1,
    half_height=0.4,
    up_axis=UpAxis.Y,
    diffuse_map=texture,
)
scene, capz_model_id = scene.add_capsule(
    radius=0.1,
    half_height=0.4,
    up_axis=UpAxis.Z,
    diffuse_map=texture,
)

scene, cube_model = scene.add_cube(
    half_extents=(1.5, 1.5, 0.03),
    diffuse_map=jnp.ones((1, 1, 3)),  # pyright: ignore[reportUnknownMemberType]
    texture_scaling=(16.0, 16.0),
)

scene, cube_instance_id = scene.add_object_instance(cube_model)
scene = scene.set_object_position(cube_instance_id, (0.0, 0.0, -0.5))

# PROCESS: Set up objects

scene, capsulex_instance_id = scene.add_object_instance(capx_model_id)
scene, capsuley_instance_id = scene.add_object_instance(capy_model_id)
scene, capsulez_instance_id = scene.add_object_instance(capz_model_id)

# PROCESS: Set up camera and light

width = 640
height = 480
eye = (2.0, 4.0, 1.0)
target = (0.0, 0.0, 0.0)

light: LightParameters = LightParameters()
camera: CameraParameters = CameraParameters(
    viewWidth=width,
    viewHeight=height,
    position=eye,
    target=target,
)
shadow_param = ShadowParameters()

# PROCESS: Render

CanvasT: TypeAlias = UInt8[Array, "width height"]

images: List[CanvasT] = []

img = Renderer.get_camera_image(
    objects=[
        scene.objects[obj_id]
        for obj_id in [
            cube_instance_id,
            capsulex_instance_id,
        ]
    ],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow_param,
)
rgb_array = lax.clamp(0.0, img * 255, 255.0).astype(jnp.uint8)  # pyright: ignore
images.append(rgb_array)

img = Renderer.get_camera_image(
    objects=[
        scene.objects[obj_id]
        for obj_id in [
            cube_instance_id,
            capsulex_instance_id,
            capsuley_instance_id,
        ]
    ],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow_param,
)
rgb_array = lax.clamp(0.0, img * 255, 255.0).astype(jnp.uint8)  # pyright: ignore
images.append(rgb_array)

img = Renderer.get_camera_image(
    objects=[
        scene.objects[obj_id]
        for obj_id in [
            cube_instance_id,
            capsulex_instance_id,
            capsuley_instance_id,
            capsulez_instance_id,
        ]
    ],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow_param,
)
rgb_array = lax.clamp(0.0, img * 255, 255.0).astype(jnp.uint8)  # pyright: ignore
images.append(rgb_array)

img = Renderer.get_camera_image(
    objects=[
        scene.objects[obj_id]
        for obj_id in [
            cube_instance_id,
            capsulex_instance_id,
            capsuley_instance_id,
            capsulez_instance_id,
        ]
    ],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow_param,
)
rgb_array = lax.clamp(0.0, img * 255, 255.0).astype(jnp.uint8)  # pyright: ignore
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
            transpose_for_display(img),
            animated=True,
        ),
    )
    if i == 0:
        # show an initial one first
        ax.imshow(  # pyright: ignore[reportUnknownMemberType]
            transpose_for_display(img),
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
