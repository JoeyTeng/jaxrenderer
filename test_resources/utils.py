import re
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
from jaxtyping import jaxtyped
import numpy as np
from PIL import Image

from renderer.types import (FaceIndices, Normals, Texture, UVCoordinates,
                            Vec2f, Vec3f, Vertices)

jax.config.update('jax_array', True)


@dataclass(frozen=True)
class Model:
    verts: Vertices
    norms: Normals
    uv: UVCoordinates
    faces: FaceIndices
    faces_norm: FaceIndices
    faces_uv: FaceIndices

    @jaxtyped
    def __post_init__(self):
        assert isinstance(self.verts, Vertices), self.verts.shape
        assert isinstance(self.norms, Vertices), self.norms.shape
        assert isinstance(self.uv, UVCoordinates), self.uv.shape
        assert isinstance(self.faces, FaceIndices), self.faces.shape
        assert isinstance(self.faces_norm, FaceIndices), self.faces_norm.shape
        assert isinstance(self.faces_uv, FaceIndices), self.faces_uv.shape

    @property
    def nverts(self) -> int:
        return self.verts.shape[0]

    @property
    def nfaces(self) -> int:
        return self.faces.shape[0]


def make_model(fileContent: List[str]) -> Model:
    verts: List[Vec3f] = []
    norms: List[Vec3f] = []
    uv: List[Vec2f] = []
    faces: List[List[int]] = []
    faces_norm: List[List[int]] = []
    faces_uv: List[List[int]] = []

    _float = re.compile(r"(-?\d+\.?\d*(?:e[+-]\d+)?)")
    _integer = re.compile(r"\d+")
    _one_vertex = re.compile(r"\d+/\d*/\d*")
    for line in fileContent:
        if line.startswith("v "):
            vert: Vec3f = tuple(map(float, _float.findall(line, 2)[:3]))
            verts.append(vert)
        elif line.startswith("vn "):
            norm: Vec3f = tuple(map(float, _float.findall(line, 2)[:3]))
            norms.append(norm)
        elif line.startswith("vt "):
            uv_coord: Vec2f = tuple(map(float, _float.findall(line, 2)[:2]))
            uv.append(uv_coord)
        elif line.startswith("f "):
            face: List[int] = []
            face_norm: List[int] = []
            face_uv: List[int] = []

            vertices: List[str] = _one_vertex.findall(line)
            assert len(vertices) == 3, ("Expected 3 vertices, "
                                        f"(got {len(vertices)}")
            for vertex in _one_vertex.findall(line):
                indices: List[int] = list(map(int, _integer.findall(vertex)))
                assert len(indices) == 3, ("Expected 3 indices (v/vt/vn), "
                                           f"got {len(indices)}")
                v, vt, vn = indices
                # indexed from 1 in Wavefront Obj
                face.append(v - 1)
                face_norm.append(vn - 1)
                face_uv.append(vt - 1)

            faces.append(face)
            faces_norm.append(face_norm)
            faces_uv.append(face_uv)

    return Model(
        verts=jnp.array(verts),
        norms=jnp.array(norms),
        uv=jnp.array(uv),
        faces=jnp.array(faces),
        faces_norm=jnp.array(faces_norm),
        faces_uv=jnp.array(faces_uv),
    )


def load_tga(path: str) -> Texture:
    image: Image.Image = Image.open(path)
    width, height = image.size
    buffer = np.zeros((width, height, 3))

    for y in range(height):
        for x in range(width):
            buffer[y, x] = np.array(image.getpixel((x, y)))

    texture: Texture = jnp.array(buffer, dtype=jnp.single)

    return texture
