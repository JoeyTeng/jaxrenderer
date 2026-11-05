from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, NamedTuple, TypeAlias, TypeVar

from jax import Array
from jax.tree_util import register_dataclass
from jaxtyping import (
    Bool,
    Float,
    PyTree,  # pyright: ignore[reportUnknownVariableType]
    Shaped,
)

from ._meta_utils import typed_array
from .geometry import Camera
from .types import (
    FALSE_ARRAY,
    INF_ARRAY,
    TRUE_ARRAY,
    BoolV,
    FloatV,
    IntV,
    PixelState,
    Vec2f,
    Vec3f,
    Vec4f,
)

__all__ = [
    "Fragment",
    "FragmentCollection",
    "FragmentInput",
    "ID",
    "InterpolationInput",
    "MixInput",
    "MixedExtraT",
    "MixerOutput",
    "PerFragment",
    "PerVertex",
    "PrimitiveInput",
    "ShaderExtraInputT",
    "VaryingT",
    "VertexInput",
]

ID: TypeAlias = IntV
_ONE_FLOAT_ARRAY: FloatV = typed_array(1.0)

ShaderExtraInputT = TypeVar(
    "ShaderExtraInputT",
    bound=PyTree[Shaped[Array, "..."]],
)
"""Extra input for vertex shader & fragment shader, shared by all."""


class PerVertex(NamedTuple):
    """Built-in output from Vertex Shader.

    gl_Position is in clip-space.
    """

    gl_Position: Vec4f
    # gl_PointSize is meaningful only when rendering point primitives.
    # not supported for now
    # gl_PointSize: FloatV


class PerFragment(NamedTuple):
    """Built-in Output from Fragment Shader.

    If use_default_depth is True (default False), gl_FragCoord[2] will be used
    later by default.
    """

    gl_FragDepth: FloatV = INF_ARRAY
    # not discard
    keeps: BoolV = TRUE_ARRAY
    use_default_depth: BoolV = FALSE_ARRAY


VaryingT = TypeVar(
    "VaryingT",
    bound=tuple[Shaped[Array, "..."], ...],
)
"""Interpolated vertex output consumed by the fragment shader."""

MixedExtraT = TypeVar(
    "MixedExtraT",
    bound=tuple[Shaped[Array, "..."], ...],
)
"""User-defined fragment target layout shared by fragment and mix stages."""


class Fragment(NamedTuple, Generic[MixedExtraT]):
    """Fragment targets and their opacity."""

    targets: MixedExtraT
    alpha: FloatV = _ONE_FLOAT_ARRAY


class FragmentCollection(Enum):
    """Primitive candidates retained for fragment processing."""

    NEAREST = "nearest"
    ALL = "all"


@register_dataclass
@dataclass(frozen=True, kw_only=True, eq=False)
class VertexInput(Generic[ShaderExtraInputT]):
    """Inputs for one vertex shader invocation."""

    gl_VertexID: ID
    """Vertex fetch index: effective draw index when non-indexed, buffer index otherwise."""
    gl_InstanceID: ID
    """OpenGL instance index; the current pipeline supplies zero."""
    camera: Camera
    """Camera view, viewport, and projection matrices (not a GLSL input)."""
    extra: ShaderExtraInputT
    """User input shared by invocations and not split over the batch axis."""


@register_dataclass
@dataclass(frozen=True, kw_only=True, eq=False)
class PrimitiveInput(Generic[VaryingT]):
    """Per-primitive inputs for candidate selection."""

    gl_FragCoord: Float[Array, "primitives 4"]
    """Window coordinates `(x, y, z, 1/w)` for each primitive candidate."""
    gl_FrontFacing: Bool[Array, "primitives"]
    """Whether each primitive is front-facing; supplied normals are unchanged."""
    gl_PointCoord: Float[Array, "primitives 2"]
    """Reserved point coordinates; point primitives are not supported."""
    keeps: Bool[Array, "primitives"]
    """Whether each candidate is kept before facing eligibility is applied."""
    values: VaryingT
    """Vertex varyings with candidate and per-triangle vertex axes."""
    barycentric_screen: Float[Array, "primitives 3"]
    """Screen-space barycentric weights for interpolation at each candidate."""
    barycentric_clip: Float[Array, "primitives 3"]
    """Clip-space barycentric weights for interpolation at each candidate."""
    double_sided: Bool[Array, "primitives"] | BoolV = field(
        default_factory=lambda: FALSE_ARRAY
    )
    """Scalar or per-primitive eligibility mask, false by default.

    Allows back faces to compete without changing facing flags or normals.
    """
    collection: FragmentCollection = field(
        default=FragmentCollection.NEAREST,
        metadata={"static": True},
    )
    """Static policy, NEAREST by default; ALL retains every candidate slot."""


@register_dataclass
@dataclass(frozen=True, kw_only=True, eq=False)
class InterpolationInput(Generic[VaryingT]):
    """Inputs for interpolation at one fragment location."""

    values: VaryingT
    """Selected vertex varyings, with the triangle's vertex index on axis 0."""
    barycentric_screen: Vec3f
    """Screen-space barycentric weights for the fragment location."""
    barycentric_clip: Vec3f
    """Clip-space barycentric weights for the fragment location."""


@register_dataclass
@dataclass(frozen=True, kw_only=True, eq=False)
class FragmentInput(Generic[ShaderExtraInputT, VaryingT]):
    """Inputs for one fragment shader invocation."""

    gl_FragCoord: Vec4f
    """Window coordinates `(x, y, z, 1/w)` for this fragment."""
    gl_FrontFacing: BoolV
    """Whether the source primitive is front-facing; normals are unchanged."""
    gl_PointCoord: Vec2f
    """Reserved point coordinates; point primitives are not supported."""
    varying: VaryingT
    """Interpolated values returned by the interpolation stage."""
    extra: ShaderExtraInputT
    """User input shared by fragment invocations and not split over the batch axis."""


@register_dataclass
@dataclass(frozen=True, kw_only=True, eq=False)
class MixInput(Generic[MixedExtraT]):
    """Fragment candidates and destination state for mixing."""

    gl_FragDepth: Float[Array, "kept_primitives"]
    """Resolved depth for each retained fragment candidate."""
    keeps: Bool[Array, "kept_primitives"]
    """Final geometry and fragment eligibility for each retained candidate."""
    extra: Fragment[MixedExtraT]
    """Batched fragment targets and their corresponding alpha values."""
    destination: PixelState[MixedExtraT]
    """Current pixel targets and depth for custom mixing and empty-batch fallback."""


class MixerOutput(NamedTuple):
    """Independent target and depth writes produced by mix."""

    keep: BoolV
    zbuffer: FloatV
    write_depth: BoolV
