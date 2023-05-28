import jax
import jax.lax as lax
import jax.numpy as jnp

from renderer import (CameraParameters, LightParameters, Renderer, Scene,
                      ShadowParameters, Texture, UpAxis, Vec3f,
                      transpose_for_display)

# PROCESS: Set up models and objects

scene: Scene = Scene()
texture: Texture = jnp.array((
    (255, 255, 255),  # White
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
)).reshape((2, 2, 3)).swapaxes(0, 1) / 255.0  # shape (2, 2, 3)

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
    diffuse_map=jnp.ones((1, 1, 3)),
    texture_scaling=(16., 16.),
)

scene, cube_instance_id = scene.add_object_instance(cube_model)
scene = scene.set_object_position(cube_instance_id, (0., 0., -0.5))

# PROCESS: Set up objects

scene, capsulex_instance_id = scene.add_object_instance(capx_model_id)
scene, capsuley_instance_id = scene.add_object_instance(capy_model_id)
scene, capsulez_instance_id = scene.add_object_instance(capz_model_id)

# PROCESS: Set up camera and light

width = 640
height = 480
eye: Vec3f = jnp.array([2., 4., 1.])
target: Vec3f = jnp.array([0., 0., 0.])

light: LightParameters = LightParameters()
camera: CameraParameters = CameraParameters(
    viewWidth=width,
    viewHeight=height,
    position=eye,
    target=target,
)

# PROCESS: Render

images = []

img = Renderer.get_camera_image(
    objects=[
        scene.objects[obj_id] for obj_id in [
            cube_instance_id,
            capsulex_instance_id,
        ]
    ],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=ShadowParameters(),
)
rgb_array = lax.clamp(0., img * 255, 255.).astype(jnp.uint8)
images.append(rgb_array)

img = Renderer.get_camera_image(
    objects=[
        scene.objects[obj_id] for obj_id in [
            cube_instance_id,
            capsulex_instance_id,
            capsuley_instance_id,
        ]
    ],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=ShadowParameters(),
)
rgb_array = lax.clamp(0., img * 255, 255.).astype(jnp.uint8)
images.append(rgb_array)

img = Renderer.get_camera_image(
    objects=[
        scene.objects[obj_id] for obj_id in [
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
    shadow_param=ShadowParameters(),
)
rgb_array = lax.clamp(0., img * 255, 255.).astype(jnp.uint8)
images.append(rgb_array)

img = Renderer.get_camera_image(
    objects=[
        scene.objects[obj_id] for obj_id in [
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
    shadow_param=ShadowParameters(),
)
rgb_array = lax.clamp(0., img * 255, 255.).astype(jnp.uint8)
images.append(rgb_array)

# PROCESS: show

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i, img in enumerate(images):
    im = ax.imshow(
        transpose_for_display(img),
        origin='lower',
        animated=True,
    )
    if i == 0:
        # show an initial one first
        ax.imshow(transpose_for_display(img), origin='lower')

    ims.append([im])

ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=500,
    blit=True,
    repeat_delay=0,
)

plt.show()
