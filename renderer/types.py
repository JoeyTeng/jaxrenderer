from typing import Any, Generic, TypeVar, Union, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer, Num
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from ._backport import JaxFloating, JaxInteger, NamedTuple, Tuple, Type, TypeAlias

__all__ = [
    "JaxFloating",
    "JaxInteger",
    "TRUE_ARRAY",
    "FALSE_ARRAY",
    "INF_ARRAY",
    "Index",
    "CanvasMask",
    "BatchCanvasMask",
    "Canvas",
    "ZBuffer",
    "Colour",
    "Vec2i",
    "Vec3i",
    "Vec2f",
    "Vec3f",
    "Vec4f",
    "Triangle2D",
    "Triangle",
    "TriangleBarycentric",
    "FaceIndices",
    "Vertices",
    "Normals",
    "UVCoordinates",
    "Texture",
    "SpecularMap",
    "NormalMap",
    "DtypeInfo",
    "LightSource",
    "Buffers",
]

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]

BoolV: TypeAlias = Bool[Array, ""]
"""JAX Array with single bool value.""" ""
FloatV: TypeAlias = Float[Array, ""]
"""JAX Array with single float value."""
IntV: TypeAlias = Integer[Array, ""]
"""JAX Array with single int value.""" ""
NumV: TypeAlias = Num[Array, ""]
"""JAX Array with single num value.""" ""


TRUE_ARRAY: BoolV = lax.full((), True, dtype=jnp.bool_)  # pyright: ignore
FALSE_ARRAY: BoolV = lax.full((), False, dtype=jnp.bool_)  # pyright: ignore
INF_ARRAY: FloatV = lax.full((), jnp.inf)  # pyright: ignore

Index: TypeAlias = Integer[Array, ""]

CanvasMask: TypeAlias = Bool[Array, "#width #height"]
BatchCanvasMask: TypeAlias = Bool[Array, "#batch #width #height"]
Canvas: TypeAlias = Float[Array, "width height channel"]
ZBuffer: TypeAlias = Float[Array, "width height"]
Colour: TypeAlias = Float[Array, "channel"]

Vec2i: TypeAlias = Integer[Array, "2"]
Vec3i: TypeAlias = Integer[Array, "3"]
Vec2f: TypeAlias = Float[Array, "2"]
Vec3f: TypeAlias = Float[Array, "3"]
# usually used only for 3D homogeneous coordinates
Vec4f: TypeAlias = Float[Array, "4"]
# 3 vertices, with each vertex defined in Vec2i in screen(canvas) space
Triangle2D: TypeAlias = Integer[Array, "3 2"]
# 3 vertices, with each vertex defined in Vec2f
Triangle2Df: TypeAlias = Float[Array, "3 2"]
# 3 vertices, each vertex defined in Vec2i in 3d (world/model) space + Float z
Triangle: TypeAlias = Float[Array, "3 4"]
# Barycentric coordinates has 3 components
TriangleBarycentric: TypeAlias = Float[Array, "3 3"]

# each face has 3 vertices
FaceIndices: TypeAlias = Integer[Array, "faces 3"]
# each vertex is defined by 3 float numbers, x-y-z
Vertices: TypeAlias = Float[Array, "vertices 3"]
Normals: TypeAlias = Float[Array, "normals 3"]
UVCoordinates: TypeAlias = Float[Array, "uv_counts 2"]
Texture: TypeAlias = Float[Array, "textureWidth textureHeight channel"]
SpecularMap: TypeAlias = Float[Array, "textureWidth textureHeight"]
NormalMap: TypeAlias = Float[Array, "textureWidth textureHeight 3"]

_DtypeT = TypeVar("_DtypeT", bound=Union[JaxFloating, JaxInteger, int])


class DtypeInfo(NamedTuple, Generic[_DtypeT]):
    min: _DtypeT
    max: _DtypeT
    bits: int
    dtype: Type

    @classmethod
    @jaxtyped
    # cannot be jitted as `dtype` will not be a valid JAX type
    def create(cls, dtype: Type[_DtypeT]) -> "DtypeInfo[_DtypeT]":
        with jax.ensure_compile_time_eval():
            if jnp.issubdtype(dtype, jnp.floating):  # pyright: ignore
                finfo = jnp.finfo(dtype)

                return cls(
                    min=cast(
                        _DtypeT,
                        finfo.min,  # pyright: ignore[reportUnknownMemberType]
                    ),
                    max=cast(
                        _DtypeT,
                        finfo.max,  # pyright: ignore[reportUnknownMemberType]
                    ),
                    bits=finfo.bits,
                    dtype=dtype,
                )
            if jnp.issubdtype(dtype, jnp.integer):  # pyright: ignore
                iinfo = jnp.iinfo(dtype)

                return cls(
                    min=cast(_DtypeT, iinfo.min),
                    max=cast(_DtypeT, iinfo.max),
                    bits=iinfo.bits,
                    dtype=dtype,
                )

        raise ValueError(f"Unexpected dtype {dtype}")


class LightSource(NamedTuple):
    direction: Vec3f = jax.numpy.array((0.0, 0.0, -1.0))  # pyright: ignore
    """in world space, as it goes from that position to origin (camera).

    For example, if direction = camera's eye, full specular will be applied to
    triangles with normal towards camera.
    """
    colour: Colour = jax.numpy.ones(3)  # pyright: ignore[reportUnknownMemberType]


_TargetsT = TypeVar("_TargetsT", bound=Tuple[Any, ...])
"""Extra target buffers, must be in shape of (width, height, ...)."""


class Buffers(NamedTuple, Generic[_TargetsT]):
    """Use lax.full to create buffers and attach here.

    targets must be a tuple of arrays with shape of (width, height, ...).
    """

    zbuffer: ZBuffer
    targets: _TargetsT
