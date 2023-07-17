import jax
import jax.lax as lax
import jax.numpy as jnp

from renderer import (
    CameraParameters,
    LightParameters,
    Renderer,
    Scene,
    ShadowParameters,
    Texture,
    UpAxis,
    build_texture_from_PyTinyrenderer,
    transpose_for_display,
)

# PROCESS: Set up models and objects

scene: Scene = Scene()
texture: Texture = jnp.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 1., 0.],
]).reshape((2, 2, 3))

scene, cube_model_1 = scene.add_cube(
    half_extents=(1., 1., 0.03),
    diffuse_map=texture,
    texture_scaling=(16., 16.),
)
scene, cube_1 = scene.add_object_instance(cube_model_1)
scene = scene.set_object_position(cube_1, (0., 0., 0.))
scene = scene.set_object_orientation(cube_1, (1., 0., 0., 0.))

scene, cube_model_2 = scene.add_cube(
    half_extents=(10., 10., 0.03),
    diffuse_map=texture,
    texture_scaling=(160., 160.),
)
scene, cube_2 = scene.add_object_instance(cube_model_2)
scene = scene.set_object_position(cube_2, (0., 0., 0.))
scene = scene.set_object_orientation(cube_2, (1., 0., 0., 0.))

# PROCESS: Set up camera and light

width = 640
height = 480
eye = jnp.asarray([2.5894797, -2.5876467, 1.9174135])
target = [0., 0., 0.]

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

images = []

img = Renderer.get_camera_image(
    objects=[scene.objects[cube_1]],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow_param,
)
rgb_array = lax.clamp(0., img * 255, 255.).astype(jnp.uint8)
images.append(rgb_array)

img = Renderer.get_camera_image(
    objects=[scene.objects[cube_2]],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow_param,
)
rgb_array = lax.clamp(0., img * 255, 255.).astype(jnp.uint8)
images.append(rgb_array)

# PROCESS: show

import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i, img in enumerate(images):
    im = ax.imshow(transpose_for_display(img), animated=True)
    if i == 0:
        # show an initial one first
        ax.imshow(transpose_for_display(img))

    ims.append([im])

ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=500,
    blit=True,
    repeat_delay=0,
)

plt.show()
