from dataclasses import dataclass
import re
from typing import List

from PIL import Image
import jax
import jax.numpy as jnp
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]
import numpy as np

from renderer import Tuple, TypeAlias
from renderer.types import FaceIndices, Normals, Texture, UVCoordinates, Vertices

if hasattr(jax.config, "jax_array"):
    jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]

T2f: TypeAlias = Tuple[float, float]
T3f: TypeAlias = Tuple[float, float, float]


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
    verts: List[T3f] = []
    norms: List[T3f] = []
    uv: List[T2f] = []
    faces: List[List[int]] = []
    faces_norm: List[List[int]] = []
    faces_uv: List[List[int]] = []

    _float = re.compile(r"(-?\d+\.?\d*(?:e[+-]\d+)?)")
    _integer = re.compile(r"\d+")
    _one_vertex = re.compile(r"\d+/\d*/\d*")
    for line in fileContent:
        if line.startswith("v "):
            vert: T3f = tuple(map(float, _float.findall(line, 2)[:3]))
            verts.append(vert)
        elif line.startswith("vn "):
            norm: T3f = tuple(map(float, _float.findall(line, 2)[:3]))
            norms.append(norm)
        elif line.startswith("vt "):
            uv_coord: T2f = tuple(map(float, _float.findall(line, 2)[:2]))
            uv.append(uv_coord)
        elif line.startswith("f "):
            face: List[int] = []
            face_norm: List[int] = []
            face_uv: List[int] = []

            vertices: List[str] = _one_vertex.findall(line)
            assert len(vertices) == 3, "Expected 3 vertices, " f"(got {len(vertices)}"
            for vertex in _one_vertex.findall(line):
                indices: List[int] = list(map(int, _integer.findall(vertex)))
                assert len(indices) == 3, (
                    "Expected 3 indices (v/vt/vn), " f"got {len(indices)}"
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
            buffer[y, x] = np.array(image.getpixel((x, y)))  # pyright: ignore

    texture: Texture = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        buffer,
        dtype=jnp.single,
    )

    return texture
