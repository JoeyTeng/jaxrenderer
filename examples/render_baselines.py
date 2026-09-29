# pyright: basic
"""Reusable CPU render scenes for visual and numerical regression checks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np

import renderer
from renderer.geometry import rotation_matrix

_ROOT = Path(__file__).resolve().parents[1]
_HEAD_OBJ = _ROOT / "test_resources" / "obj" / "african_head.obj"
_HEAD_DIFFUSE = _ROOT / "test_resources" / "tga" / "african_head_diffuse.tga"
_HEAD_SPECULAR = _ROOT / "test_resources" / "tga" / "african_head_spec.tga"


def render_cube(width: int = 640, height: int = 480) -> np.ndarray:
    """Render the blue cube example as H×W×RGB float values in [0, 1]."""
    cube = renderer.create_cube(
        half_extents=jnp.ones(3, dtype=jnp.single),
        texture_scaling=jnp.ones(2, dtype=jnp.single),
        diffuse_map=jnp.zeros((2, 2, 3), dtype=jnp.single).at[..., 2].set(1),
        specular_map=jnp.ones((2, 2), dtype=jnp.single) * 2.0,
    )
    image = renderer.Renderer.get_camera_image(
        objects=[renderer.ModelObject(model=cube)],
        camera=renderer.CameraParameters(
            viewWidth=width,
            viewHeight=height,
            position=jnp.array([2.0, 4.0, 1.0], dtype=jnp.single),
        ),
        light=renderer.LightParameters(),
        width=width,
        height=height,
    )
    return np.asarray(jax.device_get(renderer.transpose_for_display(image)))


def _load_head_model() -> renderer.Model:
    """Load the checked-in Tiny Renderer mesh and textures."""
    import re

    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[list[int]] = []
    faces_norm: list[list[int]] = []
    faces_uv: list[list[int]] = []
    float_pattern = re.compile(r"(-?\d+\.?\d*(?:e[+-]\d+)?)")
    integer_pattern = re.compile(r"\d+")
    vertex_pattern = re.compile(r"\d+/\d*/\d*")

    with _HEAD_OBJ.open(encoding="utf-8") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                x, y, z = map(float, float_pattern.findall(line, 2)[:3])
                verts.append((x, y, z))
            elif line.startswith("vn "):
                x, y, z = map(float, float_pattern.findall(line, 2)[:3])
                norms.append((x, y, z))
            elif line.startswith("vt "):
                u, v = map(float, float_pattern.findall(line, 2)[:2])
                uvs.append((u, v))
            elif line.startswith("f "):
                v_face: list[int] = []
                uv_face: list[int] = []
                norm_face: list[int] = []
                vertices = vertex_pattern.findall(line)
                if len(vertices) != 3:
                    raise ValueError(f"Expected triangulated OBJ face, got {line!r}")
                for vertex in vertices:
                    indices = list(map(int, integer_pattern.findall(vertex)))
                    if len(indices) != 3:
                        raise ValueError(
                            f"Expected v/vt/vn OBJ indices, got {vertex!r}"
                        )
                    v, vt, vn = indices
                    v_face.append(v - 1)
                    uv_face.append(vt - 1)
                    norm_face.append(vn - 1)
                faces.append(v_face)
                faces_uv.append(uv_face)
                faces_norm.append(norm_face)

    diffuse = np.asarray(Image.open(_HEAD_DIFFUSE).convert("RGB"), dtype=np.float32)
    specular = np.asarray(Image.open(_HEAD_SPECULAR).convert("RGB"), dtype=np.float32)
    # Match the original notebook's map transpose and horizontal UV flip.
    diffuse_map = jnp.asarray(
        diffuse.swapaxes(0, 1)[:, ::-1, :] / 255.0, dtype=jnp.single
    )
    specular_map = jnp.asarray(specular.swapaxes(0, 1)[:, ::-1, 0], dtype=jnp.single)
    return renderer.Model(
        verts=jnp.asarray(verts),
        norms=jnp.asarray(norms),
        uvs=jnp.asarray(uvs),
        faces=jnp.asarray(faces),
        faces_norm=jnp.asarray(faces_norm),
        faces_uv=jnp.asarray(faces_uv),
        diffuse_map=diffuse_map,
        specular_map=specular_map,
    )


def render_head_frames(
    width: int = 320,
    height: int = 240,
    frame_count: int = 30,
) -> Iterator[np.ndarray]:
    """Yield sequential Y-axis head renders as H×W×RGB float values in [0, 1]."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    model = _load_head_model()
    centre = jnp.array((0.0, 0.0, 0.0), dtype=jnp.single)
    camera = renderer.CameraParameters(
        viewWidth=width,
        viewHeight=height,
        position=jnp.array((0.0, 0.0, 3.0), dtype=jnp.single),
        target=centre,
        up=jnp.array((0.0, 1.0, 0.0), dtype=jnp.single),
    )
    light = renderer.LightParameters(
        direction=jnp.array((0.57735, -0.57735, 0.57735), dtype=jnp.single),
        ambient=jnp.full(3, 0.1, dtype=jnp.single),
        diffuse=jnp.full(3, 0.85, dtype=jnp.single),
        specular=jnp.full(3, 0.05, dtype=jnp.single),
    )
    shadow = renderer.ShadowParameters(centre=centre)
    base_instance = renderer.ModelObject(model=model)
    axis = jnp.array((0.0, 1.0, 0.0), dtype=jnp.single)

    for frame_index in range(frame_count):
        degrees = frame_index * 360.0 / frame_count
        instance = base_instance.replace_with_orientation(
            rotation_matrix=rotation_matrix(axis, degrees)
        )
        image = renderer.Renderer.get_camera_image(
            objects=[instance],
            camera=camera,
            light=light,
            width=width,
            height=height,
            shadow_param=shadow,
            colour_default=jnp.zeros(3, dtype=jnp.single),
        )
        image = jnp.clip(image, 0.0, 1.0)
        yield np.asarray(jax.device_get(renderer.transpose_for_display(image)))
