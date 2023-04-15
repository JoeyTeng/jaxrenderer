from typing import NewType, Sequence, Tuple

import jax

jax.config.update('jax_array', True)

# CanvasMask = NewType("CanvasMask", jax.Array)
CanvasMask = Sequence[Sequence[bool]]
# BatchCanvasMask = NewType("CanvasMask", jax.Array)
BatchCanvasMask = Sequence[Sequence[Sequence[bool]]]
# Canvas = NewType("Canvas", jax.Array)
Canvas = Sequence[Sequence[Sequence[float]]]
# ZBuffer = NewType("ZBuffer", jax.Array)
ZBuffer = Sequence[Sequence[float]]
# Colour = NewType("Colour", jax.Array)
Colour = Tuple[float, float, float]
# TriangleColours = NewType("TriangleColours", jax.Array)
TriangleColours = Tuple[Colour, Colour, Colour]
# Vec2i = NewType("Vec2i", jax.Array)
Vec2i = Tuple[int, int]
# Vec3f = NewType("Vec3f", jax.Array)
Vec3f = Tuple[float, float, float]
# Triangle = NewType("Triangle", jax.Array)
Triangle = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
# Triangle3D = NewType("Triangle3D", jax.Array)
Triangle3D = Tuple[Tuple[float, float, float], Tuple[float, float, float],
                   Tuple[float, float, float]]
