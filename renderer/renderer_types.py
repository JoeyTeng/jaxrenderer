"""Public input records and renderer parameter types."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Union

import jax.numpy as jnp
from jax.tree_util import register_dataclass

from ._meta_utils import typed_array, typed_ones, typed_zeros
from .geometry import Camera, normalise
from .model import MergedModel, ModelObject
from .types import Colour, NumV, Vec3f


class CameraParameters(NamedTuple):
    """Parameters for rendering from camera.

    Default values come from [erwincoumans/tinyrenderer::TinyRendererCamera](https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/python/pytinyrenderer.cc#L56)
    """

    viewWidth: int = 640
    """width of the viewport."""
    viewHeight: int = 480
    """height of the viewport."""
    viewDepth: float = 1.0
    """depth of the rendered view."""
    near: float = 0.01
    """near clipping plane."""
    far: float = 1000.0
    """far clipping plane."""
    hfov: float = 58.0
    """horizontal field of view, in degrees."""
    vfov: float = 45.0
    """vertical field of view, in degrees."""
    position: Union[Vec3f, tuple[float, float, float]] = typed_ones(3)
    """position of the camera in world space."""
    target: Union[Vec3f, tuple[float, float, float]] = typed_zeros(3)
    """target of the camera."""
    up: Union[Vec3f, tuple[float, float, float]] = typed_array((0.0, 0.0, 1.0))
    """up direction of the camera."""


class LightParameters(NamedTuple):
    """Parameters for directional light in rendering.

    Default values come from [erwincoumans/tinyrenderer::TinyRenderLight](https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/python/pytinyrenderer.cc#L74).
    """

    direction: Vec3f = normalise(typed_array((0.57735, 0.57735, 0.57735)))
    """in world space, as it goes from that position to origin (camera).

    For example, if direction = camera's eye, full specular will be applied to
    triangles with normal towards camera.
    """
    colour: Colour = typed_ones(3)
    """Light source to render."""
    ambient: Colour = typed_array((0.6, 0.6, 0.6))
    """Ambient colour. This is added to the final colour of the object."""
    diffuse: Colour = typed_array((0.35, 0.35, 0.35))
    """Diffuse coefficient per colour channel. This is multiplied to the
        diffuse texture colour of the object.
    """
    specular: Colour = typed_array((0.05, 0.05, 0.05))
    """Specular coefficient per colour channel. This is multiplied to the
        computed specular light colour.
    """


class ShadowParameters(NamedTuple):
    """Parameters for rendering shadow map.

    Default values come from:
      - [erwincoumans/tinyrenderer::TinyRenderLight](https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/python/pytinyrenderer.cc#L74).
      - for `up`, [erwincoumans/tinyrenderer::renderObject](https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/tinyrenderer.cpp#L372).
    """

    centre: Vec3f = typed_zeros(3)
    """centre of the scene, same as object's camera's centre."""
    up: Vec3f = typed_array((0.0, 0.0, 1.0))
    """up direction of the scene, same as object's camera's up."""
    strength: Colour = 1 - typed_array((0.4, 0.4, 0.4))
    """Strength of shadow. Must be in [0, 1]. 0 means no shadow, 1 means fully
        black shadow.  See `Shadow.strength` for more details.
    """
    offset: float = 0.05
    """Offset to avoid self-shadowing / z-fighting. This will be subtracted to
        the shadow map, making the shadows further away from the light.
    """


@register_dataclass
@dataclass(frozen=True, kw_only=True, eq=False)
class RenderInput:
    """Inputs for rendering a merged model.

    The arrays in this record are referenced directly and are not copied when
    the record is created or replaced.
    """

    model: MergedModel
    """Merged model of the objects to render."""
    light: LightParameters
    """Light parameters for shading."""
    camera: Camera
    """Camera used to transform and project the model."""
    shadow_param: ShadowParameters | None = None
    """Shadow settings, or `None` to disable shadows."""
    loop_unroll: int = field(default=1, metadata={"static": True})
    """Number of rows rendered in one loop; static JAX metadata."""


@register_dataclass
@dataclass(frozen=True, kw_only=True, eq=False)
class CameraImageInput:
    """Inputs for creating a rendered camera image."""

    objects: Sequence[ModelObject]
    """Objects to render."""
    light: LightParameters
    """Light parameters for shading."""
    camera: Camera | CameraParameters
    """Camera transform or parameters used to create one."""
    width: int = field(metadata={"static": True})
    """Framebuffer width; independent of the camera viewport width."""
    height: int = field(metadata={"static": True})
    """Framebuffer height; independent of the camera viewport height."""
    colour_default: Colour = field(
        default_factory=lambda: typed_array((1.0, 1.0, 1.0), dtype=jnp.single)
    )
    """Background colour and canvas dtype."""
    zbuffer_default: NumV = field(
        default_factory=lambda: typed_array(1, dtype=jnp.single)
    )
    """Initial depth value and zbuffer dtype."""
    shadow_param: ShadowParameters | None = None
    """Shadow settings, or `None` to disable shadows."""
    loop_unroll: int = field(default=1, metadata={"static": True})
    """Number of rows rendered in one loop; static JAX metadata."""


__all__ = [
    "CameraImageInput",
    "CameraParameters",
    "LightParameters",
    "RenderInput",
    "ShadowParameters",
]
