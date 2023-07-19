from functools import partial
from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from ._backport import Tuple
from ._meta_utils import add_tracing_name
from .geometry import Camera, View, Viewport
from .pipeline import render
from .shaders.depth import DepthExtraInput, DepthShader
from .types import (
    Buffers,
    Colour,
    FaceIndices,
    FloatV,
    Vec2f,
    Vec2i,
    Vec3f,
    Vertices,
    ZBuffer,
)


class Shadow(NamedTuple):
    """Shadow map for one light source."""

    shadow_map: ZBuffer
    """Depth map of the scene from the light source's point of view."""
    strength: Colour
    """Strength of shadow. Must be in [0, 1]. 0 means no shadow, 1 means fully
    black shadow. (1 - strength) of the original colour will be added to the
    shadowed colour.
    """
    camera: Camera
    """Camera from world space to shadow map's screen space."""

    @staticmethod
    @jaxtyped
    @partial(
        jax.jit,  # pyright: ignore[reportUnknownMemberType]
        static_argnames=("loop_unroll",),
        donate_argnums=(0,),
        inline=True,
    )
    @add_tracing_name
    def render_shadow_map(
        shadow_map: ZBuffer,
        verts: Vertices,
        faces: FaceIndices,
        light_direction: Vec3f,
        viewport_matrix: Viewport,
        centre: Vec3f,
        up: Vec3f,
        strength: Colour,
        offset: float = 0.001,
        distance: float = 10.0,
        loop_unroll: int = 1,
    ) -> "Shadow":
        """Render shadow map from light source's point of view.

        Parameters:
          - shadow_map: ZBuffer to store the depth map.
          - verts: vertices of the object.
          - faces: face indices of the object.
          - light_direction: direction of **parallel** light source, where it
            goes towards, in world space.
          - viewport_matrix: viewport matrix for rendering the objects.
          - centre: centre of the scene, same as object's camera's centre.
          - up: up direction of the scene, same as object's camera's up.
          - strength: strength of shadow. For details, see `Shadow.strength`.
          - offset: Offset to avoid self-shadowing / z-fighting. This will be
            added to the shadow map, making the shadows further away from
            the light.
          - distance: Distance from the light source to the centre of the
            scene. This is mainly to avoid objects being clipped.
          - loop_unroll: passed directly to `render`. See `pipeline:render`.

        Returns: Updated `Shadow` object with shadow_map updated.
        """

        view: View = Camera.view_matrix(
            # keep "forward = -light_direction"
            eye=centre + light_direction * distance,
            centre=centre,
            up=up,
        )
        assert isinstance(view, View)

        _camera: Camera = Camera.create(
            view=view,
            projection=Camera.orthographic_projection_matrix(
                left=-1.0,
                right=1.0,
                bottom=-1.0,
                top=1.0,
                z_near=-1.0,
                z_far=1.0,
            ),
            viewport=viewport_matrix,
        )
        assert isinstance(_camera, Camera)

        buffers: Buffers[Tuple[()]] = Buffers(zbuffer=shadow_map, targets=tuple())
        extra = DepthExtraInput(position=verts)
        shadow_map, _ = render(
            _camera,
            DepthShader,
            buffers,
            faces,
            extra,
            loop_unroll=loop_unroll,
        )
        shadow_map = shadow_map + offset
        assert isinstance(shadow_map, ZBuffer)

        shadow: Shadow = Shadow(
            shadow_map=shadow_map,
            strength=strength,
            camera=_camera,
        )

        return shadow

    @jaxtyped
    @add_tracing_name
    def get(self, position: Vec2f) -> FloatV:
        """Get shadow depth at `position`.

        Parameters:
          - position: position in shadow buffer's screen space.
        """
        assert isinstance(position, Vec2f), f"{position} is not a Vec3f."

        pos: Vec2i = lax.round(  # pyright: ignore[reportUnknownMemberType]
            position[:2]
        ).astype(int)
        assert isinstance(pos, Vec2i)

        value: FloatV = self.shadow_map.at[
            pos[0], pos[1]
        ].get(  # pyright: ignore[reportUnknownMemberType]
            mode="fill",
            indices_are_sorted=True,
            unique_indices=True,
            # outside shadow map, no shadow
            fill_value=jnp.inf,
        )
        assert isinstance(value, FloatV)

        return value
