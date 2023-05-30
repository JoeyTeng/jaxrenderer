import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer
from matplotlib import pyplot as plt

from renderer.geometry import Camera, normalise, to_cartesian, to_homogeneous
from renderer.pipeline import render
from renderer.shaders.depth import DepthExtraInput, DepthShader
from renderer.shaders.gouraud import GouraudExtraInput, GouraudShader
from renderer.shaders.gouraud_texture import (GouraudTextureExtraInput,
                                              GouraudTextureShader)
from renderer.shaders.phong import PhongTextureExtraInput, PhongTextureShader
from renderer.shaders.phong_darboux import (PhongTextureDarbouxExtraInput,
                                            PhongTextureDarbouxShader)
from renderer.shaders.phong_reflection import (
    PhongReflectionTextureExtraInput, PhongReflectionTextureShader)
from renderer.shaders.phong_reflection_shadow import (
    PhongReflectionShadowTextureExtraInput, PhongReflectionShadowTextureShader)
from renderer.shadow import Shadow
from renderer.types import (Buffers, FaceIndices, LightSource, NormalMap,
                            SpecularMap, Texture, Vec3f)
from renderer.utils import transpose_for_display
from test_resources.utils import Model, load_tga, make_model


def gouraud_shader_with_simple_light(model: Model):
    eye = jnp.array((0, 0, 1.))
    center = jnp.array((0, 0, 0))
    up = jnp.array((0, 1, 0))

    width = height = 800
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth = 1.

    camera: Camera = Camera.create(
        view=Camera.view_matrix(eye=eye, centre=center, up=up),
        projection=Camera.perspective_projection_matrix(
            fovy=90.,
            aspect=1.,
            z_near=-1.,
            z_far=1.,
        ),
        viewport=Camera.viewport_matrix(
            lowerbound=lowerbound,
            dimension=dimension,
            depth=depth,
        ),
    )

    buffers = Buffers(
        zbuffer=lax.full((width, height), 20.),
        targets=(lax.full((width, height, 3), 0.), ),
    )

    extra = GouraudExtraInput(
        # flatten so each vertex has its own "extra"
        position=model.verts[model.faces.reshape((-1, ))],
        colour=jnp.ones((model.nfaces * 3, 3), dtype=jnp.single),
        normal=model.norms[model.faces_norm.reshape((-1, ))],
        light=LightSource(direction=jnp.array((0., 0., 1.))),
    )

    result = render(
        camera,
        GouraudShader,
        buffers,
        # such that vertex (3i, 3i + 1, 3i + 2) corresponds to primitive i
        jnp.arange(model.nfaces * 3).reshape((model.nfaces, 3)),
        extra,
    )

    # show
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))

    axs[0].imshow(transpose_for_display(result.zbuffer))
    axs[1].imshow(transpose_for_display(result.targets[0]))


def gouraud_shader_with_texture(model: Model, texture: Texture):
    eye = jnp.array((0, 0, 2.))
    center = jnp.array((0, 0, 0))
    up = jnp.array((0, 1, 0))

    width = height = 800
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth = 1.

    camera: Camera = Camera.create(
        view=Camera.view_matrix(eye=eye, centre=center, up=up),
        projection=Camera.perspective_projection_matrix(
            fovy=60.,
            aspect=1.,
            z_near=-1.,
            z_far=1.,
        ),
        viewport=Camera.viewport_matrix(
            lowerbound=lowerbound,
            dimension=dimension,
            depth=depth,
        ),
    )

    buffers = Buffers(
        zbuffer=lax.full((width, height), 20.),
        targets=(lax.full((width, height, 3), 0.), ),
    )
    uv = (
        # reverse along y direction
        (model.uv[model.faces_uv.reshape((-1, ))].at[:, 1].multiply(-1))
        # scale to the dimension of texture map
        * jnp.array(texture.shape[:2])[None, ...])
    # swap x, y axis
    texture = transpose_for_display(texture / 255.)

    extra = GouraudTextureExtraInput(
        # flatten so each vertex has its own "extra"
        position=model.verts[model.faces.reshape((-1, ))],
        normal=model.norms[model.faces_norm.reshape((-1, ))],
        uv=uv,
        light=LightSource(direction=jnp.array((0., 0., 1.))),
        texture=texture,
    )

    result = render(
        camera,
        GouraudTextureShader,
        buffers,
        # such that vertex (3i, 3i + 1, 3i + 2) corresponds to primitive i
        jnp.arange(model.nfaces * 3).reshape((model.nfaces, 3)),
        extra,
    )

    # show
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))

    axs[0].imshow(transpose_for_display(result.zbuffer))
    axs[1].imshow(transpose_for_display(result.targets[0]))


def phong_shader_with_texture(model: Model, texture: Texture):
    eye = jnp.array((1, 1, 3.))
    center = jnp.array((0, 0, 0))
    up = jnp.array((0, 1, 0))

    width = height = 800
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth = 1.

    camera: Camera = Camera.create(
        view=Camera.view_matrix(eye=eye, centre=center, up=up),
        projection=Camera.perspective_projection_matrix(
            fovy=40.,
            aspect=1.,
            z_near=-1.,
            z_far=1.,
        ),
        viewport=Camera.viewport_matrix(
            lowerbound=lowerbound,
            dimension=dimension,
            depth=depth,
        ),
    )

    buffers = Buffers(
        zbuffer=lax.full((width, height), 20.),
        targets=(lax.full((width, height, 3), 0.), ),
    )
    uv = (
        # reverse along y direction
        (model.uv[model.faces_uv.reshape((-1, ))].at[:, 1].multiply(-1))
        # scale to the dimension of texture map
        * jnp.array(texture.shape[:2])[None, ...])
    # swap x, y axis
    texture = transpose_for_display(texture / 255.)

    extra = PhongTextureExtraInput(
        # flatten so each vertex has its own "extra"
        position=model.verts[model.faces.reshape((-1, ))],
        normal=model.norms[model.faces_norm.reshape((-1, ))],
        uv=uv,
        light=LightSource(direction=jnp.array((0., 0., 1.))),
        texture=texture,
    )

    result = render(
        camera,
        PhongTextureShader,
        buffers,
        # such that vertex (3i, 3i + 1, 3i + 2) corresponds to primitive i
        jnp.arange(model.nfaces * 3).reshape((model.nfaces, 3)),
        extra,
    )

    # show
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))

    axs[0].imshow(transpose_for_display(result.zbuffer))
    axs[1].imshow(transpose_for_display(result.targets[0]))


def phong_shader_with_texture_phong_reflection(
    model: Model,
    texture: Texture,
    specular_map: SpecularMap,
):
    # light goes along 0, 0, 1
    light = LightSource(direction=jnp.array((0., 0., -1.)))
    eye = jnp.array((1, 1, 3.))
    center = jnp.array((0, 0, 0))
    up = jnp.array((0, 1, 0))

    width = height = 800
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth = 1.

    camera: Camera = Camera.create(
        view=Camera.view_matrix(eye=eye, centre=center, up=up),
        projection=Camera.perspective_projection_matrix(
            fovy=40.,
            aspect=1.,
            z_near=-1,
            z_far=1.,
        ),
        viewport=Camera.viewport_matrix(
            lowerbound=lowerbound,
            dimension=dimension,
            depth=depth,
        ),
    )

    buffers = Buffers(
        zbuffer=lax.full((width, height), 20.),
        targets=(lax.full((width, height, 3), 0.), ),
    )
    uv = (
        # reverse along y direction
        (model.uv[model.faces_uv.reshape((-1, ))].at[:, 1].multiply(-1))
        # scale to the dimension of texture map
        * jnp.array(texture.shape[:2])[None, ...])
    # swap x, y axis
    texture = transpose_for_display(texture / 255.)
    assert isinstance(texture, Texture)
    # swap x, y axis
    specular_map = transpose_for_display(specular_map)
    assert isinstance(specular_map, SpecularMap)

    # is not affected by projection, thus only need to transform by view
    # into view space.
    light_dir_eye: Vec3f = normalise(
        to_cartesian(camera.view @ (to_homogeneous(
            normalise(light.direction.copy()),
            0.,
        ))))

    assert isinstance(light_dir_eye, Vec3f)

    extra = PhongReflectionTextureExtraInput(
        # flatten so each vertex has its own "extra"
        position=model.verts[model.faces.reshape((-1, ))],
        normal=model.norms[model.faces_norm.reshape((-1, ))],
        uv=uv,
        light=light,
        light_dir_eye=light_dir_eye,
        texture=texture,
        specular_map=specular_map,
        ambient=lax.full((3, ), 5. / 255),
        diffuse=lax.full((3, ), .4),
        specular=lax.full((3, ), .6),
    )

    result = render(
        camera,
        PhongReflectionTextureShader,
        buffers,
        # such that vertex (3i, 3i + 1, 3i + 2) corresponds to primitive i
        jnp.arange(model.nfaces * 3).reshape((model.nfaces, 3)),
        extra,
    )

    # show
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))

    axs[0].imshow(transpose_for_display(result.zbuffer))
    axs[1].imshow(
        transpose_for_display(jnp.clip(result.targets[0], 0., 1.))
    )


def phong_shader_with_texture_nm_tangent(
    model: Model,
    texture: Texture,
    normal_darboux: NormalMap,
):
    eye = jnp.array((1, 1, 3.))
    center = jnp.array((0, 0, 0))
    up = jnp.array((0, 1, 0))

    width = height = 800
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth = 1.

    camera: Camera = Camera.create(
        view=Camera.view_matrix(eye=eye, centre=center, up=up),
        projection=Camera.perspective_projection_matrix(
            fovy=40.,
            aspect=1.,
            z_near=-1.,
            z_far=1.,
        ),
        viewport=Camera.viewport_matrix(
            lowerbound=lowerbound,
            dimension=dimension,
            depth=depth,
        ),
    )

    buffers = Buffers(
        zbuffer=lax.full((width, height), 20.),
        targets=(lax.full((width, height, 3), 0.), ),
    )
    faces_uv: Integer[Array, "faces*3"] = model.faces_uv.reshape((-1, ))
    uv = (
        # reverse along y direction
        (model.uv.at[:, 1].multiply(-1))
        # scale to the dimension of texture map
        * jnp.array(texture.shape[:2])[None, ...])[faces_uv]

    # swap x, y axis
    texture = transpose_for_display(texture / 255.)
    # swap x, y axis
    normal_map: NormalMap = transpose_for_display(normal_darboux)
    face_indices: FaceIndices
    face_indices = jnp.arange(model.nfaces * 3).reshape((model.nfaces, 3))
    id_to_face: Integer[Array, "faces*3"]
    id_to_face = (jnp.arange(model.nfaces * 3) // 3).astype(int)

    extra = PhongTextureDarbouxExtraInput(
        # flatten so each vertex has its own "extra"
        position=model.verts[model.faces.reshape((-1, ))],
        normal=model.norms[model.faces_norm.reshape((-1, ))],
        uv=uv,
        light=LightSource(direction=jnp.array((0., 0., 1.))),
        texture=texture,
        normal_map=normal_map,
        id_to_face=id_to_face,
        faces_indices=face_indices,
    )

    result = render(
        camera,
        PhongTextureDarbouxShader,
        buffers,
        # such that vertex (3i, 3i + 1, 3i + 2) corresponds to primitive i
        face_indices,
        extra,
    )

    # show
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))

    axs[0].imshow(transpose_for_display(result.zbuffer))
    axs[1].imshow(transpose_for_display(result.targets[0]))


def depth_shader(model: Model):
    light = LightSource(direction=jnp.array((0., 0., 1.)))
    eye = jnp.array((1, 1, 3.))
    center = jnp.array((0, 0, 0))
    up = jnp.array((0, 1, 0))

    width = height = 800
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth = 1.

    camera: Camera = Camera.create(
        view=Camera.view_matrix(
            eye=light.direction,
            centre=center,
            up=up,
        ),
        projection=Camera.orthographic_projection_matrix(
            left=-1.,
            right=1.,
            bottom=-1.,
            top=1.,
            z_near=-1.,
            z_far=1.,
        ),
        viewport=Camera.viewport_matrix(
            lowerbound=lowerbound,
            dimension=dimension,
            depth=depth,
        ),
    )

    buffers = Buffers(
        zbuffer=lax.full((width, height), 1.),
        targets=tuple(),
    )

    extra = DepthExtraInput(position=model.verts)

    result = render(
        camera,
        DepthShader,
        buffers,
        model.faces,
        extra,
    )

    # show
    fig, axs = plt.subplots(figsize=(8, 8))

    axs.imshow(transpose_for_display(result.zbuffer))


def phong_shader_with_shadow(
    model: Model,
    texture: Texture,
    specular_map: SpecularMap,
):
    light = LightSource(direction=jnp.array((0., -1., -1.)))
    eye = jnp.array((0, 0, 3.))
    center = jnp.array((0, 0, 0))
    up = jnp.array((0, 1, 0))

    width = height = 800
    lowerbound = jnp.zeros(2, dtype=int)
    dimension = jnp.array((width, height))
    depth = 1.

    camera: Camera = Camera.create(
        view=Camera.view_matrix(eye=eye, centre=center, up=up),
        projection=Camera.perspective_projection_matrix(
            fovy=40.,
            aspect=1.,
            z_near=-1.,
            z_far=1.,
        ),
        viewport=Camera.viewport_matrix(
            lowerbound=lowerbound,
            dimension=dimension,
            depth=depth,
        ),
    )

    shadow: Shadow = Shadow.render_shadow_map(
        shadow_map=lax.full((width, height), 1e10.),
        verts=model.verts,
        faces=model.faces,
        light_direction=light.direction,
        viewport_matrix=camera.viewport,
        centre=center,
        up=up,
        strength=lax.full((3, ), .7),
        offset=0.01,
    )
    assert isinstance(shadow, Shadow)

    shadow_mat: Float[Array, "4 4"]
    shadow_mat = shadow.matrix @ jnp.linalg.inv(camera.world_to_screen)
    assert isinstance(shadow_mat, Float[Array, "4 4"])

    def second_pass(texture: Texture, shadow: Shadow) -> Buffers:
        buffers = Buffers(
            zbuffer=lax.full((width, height), 20.),
            targets=(lax.full((width, height, 3), 0.), ),
        )
        uv = (
            # reverse along y direction
            (model.uv[model.faces_uv.reshape((-1, ))].at[:, 1].multiply(-1))
            # scale to the dimension of texture map
            * jnp.array(texture.shape[:2])[None, ...])
        # swap x, y axis
        texture = transpose_for_display(texture / 255.)

        # is not affected by projection, thus only need to transform by
        # view into view space.
        light_dir_eye: Vec3f = normalise(
            to_cartesian(camera.view @ (to_homogeneous(
                normalise(light.direction.copy()),
                0.,
            ))))

        assert isinstance(light_dir_eye, Vec3f)

        extra = PhongReflectionShadowTextureExtraInput(
            # flatten so each vertex has its own "extra"
            position=model.verts[model.faces.reshape((-1, ))],
            normal=model.norms[model.faces_norm.reshape((-1, ))],
            uv=uv,
            light=light,
            light_dir_eye=light_dir_eye,
            texture=texture,
            specular_map=specular_map,
            shadow=shadow,
            shadow_mat=shadow_mat,
            ambient=lax.full((3, ), 5. / 255),
            diffuse=lax.full((3, ), .4),
            specular=lax.full((3, ), .6),
        )

        result = render(
            camera,
            PhongReflectionShadowTextureShader,
            buffers,
            # such that vertex (3i, 3i + 1, 3i + 2) corresponds to primitive i
            jnp.arange(model.nfaces * 3).reshape((model.nfaces, 3)),
            extra,
        )

        return result

    result = second_pass(texture, shadow)

    # show
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8))

    axs[0].imshow(transpose_for_display(shadow.shadow_map))
    axs[1].imshow(transpose_for_display(result.targets[0]))


if __name__ == '__main__':
    model = make_model(open('test_resources/obj/african_head.obj').readlines())
    # gouraud_shader_with_simple_light(model)
    # plt.show()
    texture = load_tga('test_resources/tga/african_head_diffuse.tga')
    # gouraud_shader_with_texture(model, texture)
    # plt.show()
    # phong_shader_with_texture(model, texture)
    # plt.show()
    specular_map = load_tga('test_resources/tga/african_head_spec.tga')[..., 0]
    # phong_shader_with_texture_phong_reflection(model, texture, specular_map)
    # plt.show()
    normal_darboux = load_tga('test_resources/tga/african_head_nm_tangent.tga')
    # phong_shader_with_texture_nm_tangent(model, texture, normal_darboux)
    # plt.show()
    # depth_shader(model)
    # plt.show()
    phong_shader_with_shadow(model, texture, specular_map)
    plt.show()
