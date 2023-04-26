import dataclasses
from jaxtyping import Array, Bool, Float, Integer

import jax

jax.config.update('jax_array', True)

Index = Integer[Array, ""]

CanvasMask = Bool[Array, "#width #height"]
BatchCanvasMask = Bool[Array, "#batch #width #height"]
Canvas = Float[Array, "width height channel"]
ZBuffer = Float[Array, "width height"]
Colour = Float[Array, "channel"]
# triangle thus 3 "colour"s
TriangleColours = Float[Array, "3 channel"]
Vec2i = Integer[Array, "2"]
Vec3i = Integer[Array, "3"]
Vec3f = Float[Array, "3"]
# 3 vertices, with each vertex defined in Vec2i in screen(canvas) space
Triangle = Integer[Array, "3 2"]
# 3 vertices, each vertex defined in Vec2i in 3d (world/model) space + Float z
Triangle3D = Float[Array, "3 3"]
# Barycentric coordinates has 3 components
TriangleBarycentric = Float[Array, "3 3"]

# Transform matrix that takes a batch of homogeneous 3D vertices and transform
# them into 2D cartesian vertices in screen space + Z value (making it 3D)
#
# The result of x-y values in screen space may be float, and thus further
# conversion to integers are needed.
World2Screen = Float[Array, "4 4"]

# each face has 3 vertices
FaceIndices = Integer[Array, "faces 3"]
# each vertex is defined by 3 float numbers, x-y-z
Vertices = Float[Array, "vertices 3"]
Texture = Float[Array, "textureWidth textureHeight channel"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LightSource:
    light_direction: Vec3f = jax.numpy.array((0., 0., -1.))
    light_colour: Colour = jax.numpy.ones(3)
