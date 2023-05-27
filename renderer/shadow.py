from functools import partial
from typing import NamedTuple

import jax
import jax.lax as lax
from jaxtyping import Array, Float, jaxtyped

from .geometry import Camera, ModelView, Viewport, World2Screen
from .pipeline import render
from .shaders.depth import DepthExtraInput, DepthShader
from .types import (Buffers, Colour, FaceIndices, Vec2i, Vec3f, Vertices,
                    ZBuffer)


class Shadow(NamedTuple):
    """Shadow map for one light source."""
    shadow_map: ZBuffer
    """Depth map of the scene from the light source's point of view."""
    strength: Colour
    """Strength of shadow. Must be in [0, 1]. 0 means no shadow, 1 means fully
    black shadow. (1 - strength) of the original colour will be added to the
    shadowed colour.
    """
    matrix: World2Screen
    """Transformation matrix from world space to light source's screen space."""

    @staticmethod
    @jaxtyped
    @partial(jax.jit, donate_argnums=(0, ))
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
            subtracted to the shadow map, making the shadows further away from
            the light.

        Returns: Updated `Shadow` object with shadow_map updated.
        """

        model_view: ModelView = Camera.model_view_matrix(
            eye=light_direction - centre,  # keep "forward = -light_direction"
            centre=centre,
            up=up,
        )
        assert isinstance(model_view, ModelView)

        _camera: Camera = Camera.create(
            model_view=model_view,
            projection=Camera.orthographic_projection_matrix(
                left=-1.,
                right=1.,
                bottom=-1.,
                top=1.,
                z_near=-1.,
                z_far=1.,
            ),
            viewport=viewport_matrix,
        )
        buffers = Buffers(zbuffer=shadow_map, targets=tuple())
        extra = DepthExtraInput(position=verts)
        shadow_map, _ = render(_camera, DepthShader, buffers, faces, extra)
        shadow_map = shadow_map - offset
        assert isinstance(shadow_map, ZBuffer)

        shadow: Shadow = Shadow(
            shadow_map=shadow_map,
            strength=strength,
            matrix=_camera.world_to_screen,
        )

        return shadow

    @jaxtyped
    @partial(jax.jit, inline=True)
    def get(self, position: Vec3f) -> Float[Array, ""]:
        """Get shadow depth at `position`.

        Parameters:
          - position: position in shadow buffer's screen space.
        """
        pos: Vec2i = lax.round(position[:2]).astype(int)
        assert isinstance(pos, Vec2i)

        return self.shadow_map[pos[0], pos[1]]
