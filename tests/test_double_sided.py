import jax.numpy as jnp
import pytest

from renderer import (
    CameraParameters,
    LightParameters,
    Model,
    ModelObject,
    Renderer,
    ShadowParameters,
)
from renderer._meta_utils import typed_array, typed_full
from renderer.shadow import Shadow


def _triangle(
    colour: tuple[float, float, float],
    *,
    reverse_winding: bool = False,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Model:
    face = (0, 2, 1) if reverse_winding else (0, 1, 2)
    faces = typed_array((face,), dtype=jnp.int32)
    return Model(
        verts=typed_array(((-0.6, -0.6, 0.0), (0.6, -0.6, 0.0), (0.0, 0.6, 0.0))),
        norms=jnp.broadcast_to(typed_array(normal), (3, 3)),
        uvs=typed_array(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))),
        faces=faces,
        faces_norm=faces,
        faces_uv=faces,
        diffuse_map=typed_array(((colour,),)),
        specular_map=typed_array(((2.0,),)),
    )


def _render(objects: tuple[ModelObject, ...], *, with_shadows: bool = False):
    return Renderer.get_camera_image(
        objects=objects,
        light=LightParameters(
            direction=typed_array((0.0, 0.0, 1.0)),
            ambient=typed_array((0.0, 0.0, 0.0)),
            diffuse=typed_array((1.0, 1.0, 1.0)),
            specular=typed_array((0.0, 0.0, 0.0)),
        ),
        camera=CameraParameters(
            viewWidth=32,
            viewHeight=32,
            position=(0.0, 0.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            hfov=90.0,
            vfov=90.0,
        ),
        width=32,
        height=32,
        shadow_param=(
            ShadowParameters(
                centre=typed_array((0.0, 0.0, 0.0)),
                up=typed_array((0.0, 1.0, 0.0)),
                strength=typed_array((0.0, 0.0, 0.0)),
                offset=0.0,
            )
            if with_shadows
            else None
        ),
    )


@pytest.mark.parametrize("with_shadows", (False, True))
def test_reversed_winding_is_culled_by_default_and_uses_supplied_normal_when_enabled(
    with_shadows: bool,
) -> None:
    back_face = ModelObject(_triangle((0.8, 0.2, 0.1), reverse_winding=True))

    culled = _render((back_face,), with_shadows=with_shadows)
    visible = _render(
        (back_face.replace_with_double_sided(typed_array(True)),),
        with_shadows=with_shadows,
    )

    assert jnp.allclose(culled, 1.0)
    assert jnp.allclose(visible[16, 16], typed_array((0.8, 0.2, 0.1)))


@pytest.mark.parametrize("with_shadows", (False, True))
def test_double_sided_front_face_keeps_the_existing_normal_lighting(
    with_shadows: bool,
) -> None:
    front_face = ModelObject(_triangle((0.2, 0.7, 0.4)))

    single_sided = _render((front_face,), with_shadows=with_shadows)
    double_sided = _render(
        (front_face.replace_with_double_sided(typed_array(True)),),
        with_shadows=with_shadows,
    )

    assert jnp.allclose(single_sided, double_sided)


@pytest.mark.parametrize("with_shadows", (False, True))
def test_double_sided_dark_near_face_occludes_far_colour_in_mixed_scene(
    with_shadows: bool,
) -> None:
    far_front = ModelObject(_triangle((0.1, 0.8, 0.2))).replace_with_position(
        typed_array((0.0, 0.0, -0.3))
    )
    near_back = ModelObject(
        _triangle(
            (0.9, 0.1, 0.1),
            reverse_winding=True,
            normal=(0.0, 0.0, -1.0),
        )
    ).replace_with_position(typed_array((0.0, 0.0, 0.3)))

    visible_far = _render(
        (far_front, near_back),
        with_shadows=with_shadows,
    )
    occluded_by_dark_back = _render(
        (far_front, near_back.replace_with_double_sided(typed_array(True))),
        with_shadows=with_shadows,
    )

    assert jnp.allclose(visible_far[16, 16], typed_array((0.1, 0.8, 0.2)))
    assert jnp.allclose(occluded_by_dark_back[16, 16], 0.0)


def test_shadow_depth_pass_accepts_opted_in_back_faces() -> None:
    verts = _triangle((0.4, 0.5, 0.6), reverse_winding=True).verts
    faces = typed_array(((0, 2, 1),), dtype=jnp.int32)
    viewport = Renderer.create_camera_from_parameters(
        CameraParameters(
            viewWidth=32,
            viewHeight=32,
            position=(0.0, 0.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
    ).viewport
    default_shadow = Shadow.render_shadow_map(
        shadow_map=typed_full((32, 32), jnp.inf),
        verts=verts,
        faces=faces,
        light_direction=typed_array((0.0, 0.0, 1.0)),
        viewport_matrix=viewport,
        centre=typed_array((0.0, 0.0, 0.0)),
        up=typed_array((0.0, 1.0, 0.0)),
        strength=typed_array((1.0, 1.0, 1.0)),
    )
    double_sided_shadow = Shadow.render_shadow_map(
        shadow_map=typed_full((32, 32), jnp.inf),
        verts=verts,
        faces=faces,
        light_direction=typed_array((0.0, 0.0, 1.0)),
        viewport_matrix=viewport,
        centre=typed_array((0.0, 0.0, 0.0)),
        up=typed_array((0.0, 1.0, 0.0)),
        strength=typed_array((1.0, 1.0, 1.0)),
        double_sided=typed_array((True, True, True)),
    )

    assert jnp.all(jnp.isinf(default_shadow.shadow_map))
    assert jnp.any(jnp.isfinite(double_sided_shadow.shadow_map))
