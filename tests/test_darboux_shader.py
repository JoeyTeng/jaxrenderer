from jax import Array
import jax.numpy as jnp

from renderer import Buffers, Camera, render
from renderer._meta_utils import typed_asarray, typed_zeros
from renderer.geometry import Viewport
from renderer.shaders.phong_darboux import (
    PhongTextureDarbouxExtraInput,
    PhongTextureDarbouxShader,
)
from renderer.types import (
    Canvas,
    FaceIndices,
    LightSource,
    NormalMap,
    Normals,
    Texture,
    UVCoordinates,
    Vec3f,
    Vertices,
    ZBuffer,
)


def _array(values: object) -> Array:
    return typed_asarray(values, dtype=jnp.float32)


def test_darboux_shader_renders_nontrivial_normal_map() -> None:
    width = 12
    height = 12
    eye: Vec3f = _array((0.0, 0.0, 2.0))
    centre: Vec3f = _array((0.0, 0.0, 0.0))
    up: Vec3f = _array((0.0, 1.0, 0.0))
    viewport: Viewport = Camera.viewport_matrix(
        lowerbound=typed_zeros(2, dtype=int),
        dimension=typed_asarray((width, height), dtype=int),
        depth=1,
    )
    camera = Camera.create(
        view=Camera.view_matrix(eye=eye, centre=centre, up=up),
        projection=Camera.perspective_projection_matrix(
            fovy=90.0,
            aspect=1.0,
            z_near=-1.0,
            z_far=1.0,
        ),
        viewport=viewport,
    )

    position: Vertices = _array(((-0.7, -0.7, 0.0), (0.7, -0.7, 0.0), (-0.7, 0.7, 0.0)))
    uv: UVCoordinates = _array(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)))
    normals: Normals = _array(((0.0, 0.0, 1.0),) * 3)
    face_indices: FaceIndices = typed_asarray(((0, 1, 2),), dtype=int)
    texture: Texture = _array((((1.0, 1.0, 1.0),) * 2,) * 2)
    normal_map: NormalMap = _array(
        (
            ((0.0, 0.0, 1.0), (0.8, 0.0, 0.6)),
            ((0.0, 0.8, 0.6), (-0.5, 0.5, 0.7)),
        )
    )
    extra = PhongTextureDarbouxExtraInput(
        position=position,
        normal=normals,
        uv=uv,
        light=LightSource(direction=_array((0.0, 0.0, 1.0))),
        texture=texture,
        normal_map=normal_map,
        id_to_face=typed_zeros(3, dtype=int),
        faces_indices=face_indices,
    )
    zbuffer: ZBuffer = typed_zeros((width, height))
    canvas: Canvas = typed_zeros((width, height, 3))
    buffers: Buffers[tuple[Canvas]] = Buffers(
        zbuffer=zbuffer,
        targets=(canvas,),
    )

    result = render(
        camera=camera,
        shader=PhongTextureDarbouxShader,
        buffers=buffers,
        face_indices=face_indices,
        extra=extra,
    )
    canvas = result.targets[0]
    lit_pixels = canvas[jnp.any(canvas > 0.0, axis=-1)]

    assert bool(jnp.isfinite(canvas).all())
    assert lit_pixels.shape[0] > 0
    assert float(jnp.ptp(lit_pixels[:, 0])) > 0.05
