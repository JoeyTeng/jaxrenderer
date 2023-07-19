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
    Vec3f,
    build_texture_from_PyTinyrenderer,
    transpose_for_display,
)

scene: Scene = Scene()

width = 640
height = 480
eye: Vec3f = jnp.array([2.0, 4.0, 1.0])  # pyright: ignore[reportUnknownMemberType]
target: Vec3f = jnp.array([0.0, 0.0, 0.0])  # pyright: ignore[reportUnknownMemberType]

light: LightParameters = LightParameters(
    direction=jnp.array([2.0, 4.0, 1.0]),  # pyright: ignore[reportUnknownMemberType]
    ambient=jnp.zeros(3),  # pyright: ignore[reportUnknownMemberType]
    diffuse=jnp.full(3, 1.0),  # pyright: ignore[reportUnknownMemberType]
    specular=jnp.full(3, 0.0),  # pyright: ignore[reportUnknownMemberType]
)
camera: CameraParameters = CameraParameters(
    viewWidth=width,
    viewHeight=height,
    position=eye,
    target=target,
)

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

scene, sphere_model_id = scene.add_capsule(
    radius=1.0,
    half_height=0.0,
    up_axis=UpAxis.X,
    diffuse_map=texture,
)

scene, sphere_instance_id = scene.add_object_instance(sphere_model_id)
with jax.disable_jit(False):
    img = Renderer.get_camera_image(
        objects=[scene.objects[sphere_instance_id]],
        light=light,
        camera=camera,
        width=width,
        height=height,
        shadow_param=ShadowParameters(offset=0.05),
    )

rgb_array = lax.clamp(  # pyright: ignore[reportUnknownMemberType]
    0.0, img * 255, 255.0
).astype(jnp.uint8)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # pyright: ignore
ax.imshow(transpose_for_display(rgb_array))  # pyright: ignore[reportUnknownMemberType]

plt.show()  # pyright: ignore[reportUnknownMemberType]
