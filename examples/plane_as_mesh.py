import jax.lax as lax
import jax.numpy as jnp

from renderer import (
    CameraParameters,
    LightParameters,
    Model,
    Renderer,
    Scene,
    ShadowParameters,
    Texture,
    build_texture_from_PyTinyrenderer,
    transpose_for_display,
)

scene: Scene = Scene()

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

texture: Texture = (
    build_texture_from_PyTinyrenderer(
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
        ),
        2,
        2,
    )
    / 255.0
)

vertices = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [100.000000, -100.000000, 0.000000],
        [100.000000, 100.000000, 0.000000],
        [-100.000000, 100.000000, 0.000000],
        [-100.000000, -100.000000, 0.000000],
    ]
)
vertices = vertices * 0.01
normals = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
    ]
)

uvs = jnp.array(  # pyright: ignore[reportUnknownMemberType]
    [
        [1.000000, 0.000000],
        [1.000000, 1.000000],
        [0.000000, 1.000000],
        [0.000000, 0.000000],
    ]
)

indices = jnp.array([[0, 1, 2], [0, 2, 3]])  # pyright: ignore[reportUnknownMemberType]
model: Model = Model.create(
    verts=vertices,
    norms=normals,
    uvs=uvs,
    faces=indices,
    diffuse_map=texture,
)
scene, plane_model = scene.add_model(model)

scene, plane_instance_id = scene.add_object_instance(plane_model)
scene = scene.set_object_orientation(
    plane_instance_id,
    (1.0, 0, 0, 0),
)

img = Renderer.get_camera_image(
    objects=[scene.objects[plane_instance_id]],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=ShadowParameters(),
)

rgb_array = lax.clamp(  # pyright: ignore[reportUnknownMemberType]
    0.0, img * 255, 255.0
).astype(jnp.uint8)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # pyright: ignore
ax.imshow(transpose_for_display(rgb_array))  # pyright: ignore[reportUnknownMemberType]

plt.show()  # pyright: ignore[reportUnknownMemberType]
