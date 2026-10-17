from dataclasses import dataclass
import re
from typing import TypeAlias

from PIL import Image
import jax
import jax.numpy as jnp
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]
import numpy as np

from renderer._meta_utils import is_array_type
from renderer.types import FaceIndices, Normals, Texture, UVCoordinates, Vertices

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]

T2f: TypeAlias = tuple[float, float]
T3f: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True)
class Model:
    verts: Vertices
    norms: Normals
    uv: UVCoordinates
    faces: FaceIndices
    faces_norm: FaceIndices
    faces_uv: FaceIndices

    @jaxtyped
    def __post_init__(self) -> None:
        assert is_array_type(self.verts, Vertices), self.verts.shape
        assert is_array_type(self.norms, Normals), self.norms.shape
        assert is_array_type(self.uv, UVCoordinates), self.uv.shape
        assert is_array_type(self.faces, FaceIndices), self.faces.shape
        assert is_array_type(self.faces_norm, FaceIndices), self.faces_norm.shape
        assert is_array_type(self.faces_uv, FaceIndices), self.faces_uv.shape

    @property
    def nverts(self) -> int:
        return self.verts.shape[0]

    @property
    def nfaces(self) -> int:
        return self.faces.shape[0]


def make_model(fileContent: list[str]) -> Model:
    verts: list[T3f] = []
    norms: list[T3f] = []
    uv: list[T2f] = []
    faces: list[list[int]] = []
    faces_norm: list[list[int]] = []
    faces_uv: list[list[int]] = []

    _float = re.compile(r"(-?\d+\.?\d*(?:e[+-]\d+)?)")
    _integer = re.compile(r"\d+")
    _one_vertex = re.compile(r"\d+/\d*/\d*")
    for line in fileContent:
        if line.startswith("v "):
            values = _float.findall(line, 2)[:3]
            assert len(values) == 3
            vert: T3f = (float(values[0]), float(values[1]), float(values[2]))
            verts.append(vert)
        elif line.startswith("vn "):
            values = _float.findall(line, 2)[:3]
            assert len(values) == 3
            norm: T3f = (float(values[0]), float(values[1]), float(values[2]))
            norms.append(norm)
        elif line.startswith("vt "):
            values = _float.findall(line, 2)[:2]
            assert len(values) == 2
            uv_coord: T2f = (float(values[0]), float(values[1]))
            uv.append(uv_coord)
        elif line.startswith("f "):
            face: list[int] = []
            face_norm: list[int] = []
            face_uv: list[int] = []

            vertices: list[str] = _one_vertex.findall(line)
            assert len(vertices) == 3, f"Expected 3 vertices, (got {len(vertices)}"
            for vertex in _one_vertex.findall(line):
                indices: list[int] = list(map(int, _integer.findall(vertex)))
                assert len(indices) == 3, (
                    f"Expected 3 indices (v/vt/vn), got {len(indices)}"
                )
                v, vt, vn = indices
                # indexed from 1 in Wavefront Obj
                face.append(v - 1)
                face_norm.append(vn - 1)
                face_uv.append(vt - 1)

            faces.append(face)
            faces_norm.append(face_norm)
            faces_uv.append(face_uv)

    return Model(
        verts=jnp.array(verts),  # pyright: ignore[reportUnknownMemberType]
        norms=jnp.array(norms),  # pyright: ignore[reportUnknownMemberType]
        uv=jnp.array(uv),  # pyright: ignore[reportUnknownMemberType]
        faces=jnp.array(faces),  # pyright: ignore[reportUnknownMemberType]
        faces_norm=jnp.array(faces_norm),  # pyright: ignore[reportUnknownMemberType]
        faces_uv=jnp.array(faces_uv),  # pyright: ignore[reportUnknownMemberType]
    )


def load_tga(path: str) -> Texture:
    image: Image.Image = Image.open(path)
    width, height = image.size
    buffer = np.zeros((width, height, 3))

    for y in range(height):
        for x in range(width):
            buffer[y, x] = np.array(image.getpixel((x, y)))

    texture: Texture = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        buffer,
        dtype=jnp.single,
    )

    return texture
