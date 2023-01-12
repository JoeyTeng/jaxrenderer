import re
from dataclasses import dataclass, field
from typing import List, NewType, Tuple

import jax
import jax.numpy as jnp

jax.config.update('jax_array', True)

# Vec3 = NewType("Vec3", jax.Array)
Vec3 = Tuple[float, float, float]


@dataclass()
class Model:
    verts: jax.Array
    faces: jax.Array

    def __post_init__(self):
        assert self.verts.shape[1] == 3
        assert len(self.faces.shape) == 2

    @property
    def nverts(self) -> int:
        return self.verts.shape[0]

    @property
    def nfaces(self) -> int:
        return self.faces.shape[0]


def make_model(fileContent: List[str]) -> Model:
    verts: List[Vec3] = []
    faces: List[List[int]] = []

    _float = re.compile(r"(-?\d+\.?\d*(?:e[+-]\d+)?)")
    _integer = re.compile(r"\d+")
    _one_vertex = re.compile(r"\d+/\d*/\d*")
    for line in fileContent:
        if line.startswith("v "):
            vert: Vec3 = tuple(map(float, _float.findall(line, 2)[:3]))
            verts.append(vert)
        elif line.startswith("f "):
            face: List[int] = list(
                map(
                    lambda index: int(index) - 1,  # indexed from 1 in .obj
                    map(
                        lambda x: _integer.search(x)[0],  # pick first int
                        _one_vertex.findall(line),  # split into vertices
                    ),
                ))
            faces.append(face)

    return Model(jnp.array(verts), jnp.array(faces))
