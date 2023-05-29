import jax
import jax.lax as lax
import jax.numpy as jnp

from renderer import (CameraParameters, LightParameters, Model, Renderer,
                      Scene, ShadowParameters, Texture, Vec3f,
                      build_texture_from_PyTinyrenderer, transpose_for_display)

scene: Scene = Scene()

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

texture: Texture = build_texture_from_PyTinyrenderer(
    jnp.array((
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
        255  # Blue
    )),
    2,
    2,
) / 255.0

vertices = jnp.array([
    [100.000000, -100.000000, 0.000000],
    [100.000000, 100.000000, 0.000000],
    [-100.000000, 100.000000, 0.000000],
    [-100.000000, -100.000000, 0.000000],
])
vertices = vertices * 0.01
normals = jnp.array([
    [0.000000, 0.000000, 1.000000],
    [0.000000, 0.000000, 1.000000],
    [0.000000, 0.000000, 1.000000],
    [0.000000, 0.000000, 1.000000],
])

uvs = jnp.array([
    [1.000000, 0.000000],
    [1.000000, 1.000000],
    [0.000000, 1.000000],
    [0.000000, 0.000000],
])

indices = jnp.array([[0, 1, 2], [0, 2, 3]])
model: Model = Model(
    verts=vertices,
    norms=normals,
    uvs=uvs,
    faces=indices,
    faces_norm=indices,
    faces_uv=indices,
    diffuse_map=texture,
    # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
    specular_map=lax.full(texture.shape[:2], 2.0),
)
scene, plane_model = scene.add_model(model)

scene, plane_instance_id = scene.add_object_instance(plane_model)
scene = scene.set_object_orientation(
    plane_instance_id,
    jnp.array([0, 0, 0, 1.]),
)

img = Renderer.get_camera_image(
    objects=[scene.objects[plane_instance_id]],
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=ShadowParameters(),
)

rgb_array = lax.clamp(0., img * 255, 255.).astype(jnp.uint8)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(transpose_for_display(rgb_array), origin='lower')

plt.show()
