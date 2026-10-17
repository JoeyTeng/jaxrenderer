import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
from renderer._meta_utils import typed_array

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

vertices = typed_array(
    [
        [100.000000, -100.000000, 0.000000],
        [100.000000, 100.000000, 0.000000],
        [-100.000000, 100.000000, 0.000000],
        [-100.000000, -100.000000, 0.000000],
    ]
)
vertices = vertices * 0.01
normals = typed_array(
    [
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 0.000000, 1.000000],
    ]
)

uvs = typed_array(
    [
        [1.000000, 0.000000],
        [1.000000, 1.000000],
        [0.000000, 1.000000],
        [0.000000, 0.000000],
    ]
)

indices = typed_array([[0, 1, 2], [0, 2, 3]])
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

rgb_array = lax.clamp(0.0, img * 255, 255.0).astype(jnp.uint8)

fig, ax = plt.subplots()  # pyright: ignore
ax.imshow(transpose_for_display(rgb_array))  # pyright: ignore[reportUnknownMemberType]

plt.show()  # pyright: ignore[reportUnknownMemberType]
