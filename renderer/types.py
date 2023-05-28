from typing import Any, NamedTuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer, Shaped, jaxtyped

jax.config.update('jax_array', True)

TRUE_ARRAY: Bool[Array, ""] = lax.full((), True, dtype=jnp.bool_)
FALSE_ARRAY: Bool[Array, ""] = lax.full((), False, dtype=jnp.bool_)
INF_ARRAY: Float[Array, ""] = lax.full((), jnp.inf)

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
Vec2f = Float[Array, "2"]
Vec3f = Float[Array, "3"]
# usually used only for 3D homogeneous coordinates
Vec4f = Float[Array, "4"]
# 3 vertices, with each vertex defined in Vec2i in screen(canvas) space
Triangle2D = Integer[Array, "3 2"]
# 3 vertices, with each vertex defined in Vec2f
Triangle2Df = Float[Array, "3 2"]
# 3 vertices, each vertex defined in Vec2i in 3d (world/model) space + Float z
Triangle = Float[Array, "3 4"]
# Barycentric coordinates has 3 components
TriangleBarycentric = Float[Array, "3 3"]

# each face has 3 vertices
FaceIndices = Integer[Array, "faces 3"]
# each vertex is defined by 3 float numbers, x-y-z
Vertices = Float[Array, "vertices 3"]
Normals = Float[Array, "normals 3"]
UVCoordinates = Float[Array, "uv_counts 2"]
Texture = Float[Array, "textureWidth textureHeight channel"]
SpecularMap = Float[Array, "textureWidth textureHeight"]
NormalMap = Float[Array, "textureWidth textureHeight 3"]


class DtypeInfo(NamedTuple):
    # TODO: use Generic NamedTuple when bump to Python 3.11
    min: Union[jnp.floating[Any], jnp.integer[Any]]
    max: Union[jnp.floating[Any], jnp.integer[Any]]
    bits: int
    dtype: type

    @classmethod
    @jaxtyped
    # cannot be jitted as `dtype` will not be a valid JAX type
    def create(cls, dtype: type) -> "DtypeInfo":
        with jax.ensure_compile_time_eval():
            if jnp.issubdtype(dtype, jnp.floating):
                finfo = jnp.finfo(dtype)

                return cls(
                    min=finfo.min,
                    max=finfo.max,
                    bits=finfo.bits,
                    dtype=dtype,
                )
            if jnp.issubdtype(dtype, jnp.integer):
                iinfo = jnp.iinfo(dtype)

                return cls(
                    min=iinfo.min,
                    max=iinfo.max,
                    bits=iinfo.bits,
                    dtype=dtype,
                )

        raise ValueError(f"Unexpected dtype {dtype}")


class LightSource(NamedTuple):
    direction: Vec3f = jax.numpy.array((0., 0., -1.))
    """in world space, as it goes from that position to origin (camera).

    For example, if direction = camera's eye, full specular will be applied to
    triangles with normal towards camera.
    """
    colour: Colour = jax.numpy.ones(3)


class Buffers(NamedTuple):
    """Use lax.full to create buffers and attach here."""
    # TODO: use Generic NamedTuple when bump to Python 3.11
    zbuffer: ZBuffer
    targets: tuple[Shaped[Array, "width height ..."]]
