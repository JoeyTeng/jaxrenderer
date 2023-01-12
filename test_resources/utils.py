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
    verts: List[Vec3] = field(default_factory=list)
    faces: List[List[int]] = field(default_factory=list)

    @property
    def nverts(self) -> int:
        return len(self.verts)

    @property
    def nfaces(self) -> int:
        return len(self.faces)


def make_model(fileContent: List[str]) -> Model:
    verts: List[Vec3] = []
    faces: List[List[int]] = []

    _decimal = re.compile(r"-?\d+\.?\d*")
    _integer = re.compile(r"\d+")
    _one_vertex = re.compile(r"\d+/\d*/\d*")
    for line in fileContent:
        if line.startswith("v "):
            vert: Vec3 = tuple(map(float, _decimal.findall(line, 2)[:3]))
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

    return Model(verts, faces)
