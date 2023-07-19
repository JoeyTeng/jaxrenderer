from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from functools import partial
from typing import Any, NamedTuple, Optional, Sequence, TypeVar, Union, cast

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Integer
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from ._backport import Tuple, TypeAlias
from ._meta_utils import add_tracing_name
from ._meta_utils import typed_jit as jit
from .geometry import Camera, Projection, View, Viewport, normalise
from .model import MergedModel, ModelObject, merge_objects
from .pipeline import render
from .shaders.phong_reflection import (
    PhongReflectionTextureExtraInput,
    PhongReflectionTextureShader,
)
from .shaders.phong_reflection_shadow import (
    PhongReflectionShadowTextureExtraInput,
    PhongReflectionShadowTextureShader,
)
from .shadow import Shadow
from .types import (
    Buffers,
    Canvas,
    Colour,
    DtypeInfo,
    FloatV,
    LightSource,
    NumV,
    Vec3f,
    ZBuffer,
)

DoubleSidedFaces: TypeAlias = Bool[Array, "faces"]
"""Whether to render both sides of each face (triangle primitive)."""

TargetT = TypeVar("TargetT", bound=Tuple[Any])


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
    position: Union[Vec3f, Tuple[float, float, float]] = jnp.ones(3)  # pyright: ignore
    """position of the camera in world space."""
    target: Union[Vec3f, Tuple[float, float, float]] = jnp.zeros(3)  # pyright: ignore
    """target of the camera."""
    up: Union[Vec3f, Tuple[float, float, float]] = jnp.array(  # pyright: ignore
        (0.0, 0.0, 1.0)
    )
    """up direction of the camera."""


class LightParameters(NamedTuple):
    """Parameters for directional light in rendering.

    Default values come from [erwincoumans/tinyrenderer::TinyRenderLight](https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/python/pytinyrenderer.cc#L74).
    """

    direction: Vec3f = normalise(
        jnp.array(  # pyright: ignore[reportUnknownMemberType]
            (0.57735, 0.57735, 0.57735)
        )
    )
    """in world space, as it goes from that position to origin (camera).

    For example, if direction = camera's eye, full specular will be applied to
    triangles with normal towards camera.
    """
    colour: Colour = jnp.ones(3)  # pyright: ignore[reportUnknownMemberType]
    """Light source to render."""
    ambient: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (0.6, 0.6, 0.6)
    )
    """Ambient colour. This is added to the final colour of the object."""
    diffuse: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (0.35, 0.35, 0.35)
    )
    """Diffuse coefficient per colour channel. This is multiplied to the
        diffuse texture colour of the object.
    """
    specular: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (0.05, 0.05, 0.05)
    )
    """Specular coefficient per colour channel. This is multiplied to the
        computed specular light colour.
    """


class ShadowParameters(NamedTuple):
    """Parameters for rendering shadow map.

    Default values come from:
      - [erwincoumans/tinyrenderer::TinyRenderLight](https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/python/pytinyrenderer.cc#L74).
      - for `up`, [erwincoumans/tinyrenderer::renderObject](https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/tinyrenderer.cpp#L372).
    """

    centre: Vec3f = jnp.zeros(3)  # pyright: ignore[reportUnknownMemberType]
    """centre of the scene, same as object's camera's centre."""
    up: Vec3f = jnp.array((0.0, 0.0, 1.0))  # pyright: ignore[reportUnknownMemberType]
    """up direction of the scene, same as object's camera's up."""
    strength: Colour = 1 - jnp.array(  # pyright: ignore[reportUnknownMemberType]
        (0.4, 0.4, 0.4)
    )
    """Strength of shadow. Must be in [0, 1]. 0 means no shadow, 1 means fully
        black shadow.  See `Shadow.strength` for more details.
    """
    offset: float = 0.05
    """Offset to avoid self-shadowing / z-fighting. This will be subtracted to
        the shadow map, making the shadows further away from the light.
    """


class Renderer:
    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def create_camera_from_parameters(camera: CameraParameters) -> Camera:
        """Create a camera from camera parameters."""
        eye: Vec3f = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
            camera.position, dtype=float
        )
        assert isinstance(eye, Vec3f), f"{eye}"
        centre: Vec3f = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
            camera.target, dtype=float
        )
        assert isinstance(centre, Vec3f), f"{centre}"
        up: Vec3f = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
            camera.up, dtype=float
        )
        assert isinstance(up, Vec3f), f"{up}"

        view_mat: View = Camera.view_matrix(eye=eye, centre=centre, up=up)
        assert isinstance(view_mat, View), f"{view_mat}"
        view_inv: View = Camera.view_matrix_inv(eye=eye, centre=centre, up=up)
        assert isinstance(view_inv, View), f"{view_inv}"
        projection_mat: Projection = Camera.perspective_projection_matrix(
            fovy=camera.vfov,
            aspect=(
                lax.tan(  # pyright: ignore[reportUnknownMemberType]
                    cast(FloatV, jnp.radians(camera.hfov) / 2.0)
                )
                / lax.tan(  # pyright: ignore[reportUnknownMemberType]
                    cast(FloatV, jnp.radians(camera.vfov) / 2.0)
                )
            ),
            z_near=camera.near,
            z_far=camera.far,
        )
        assert isinstance(projection_mat, Projection), f"{projection_mat}"
        viewport_mat: Viewport = Camera.viewport_matrix(
            lowerbound=jnp.zeros(  # pyright: ignore[reportUnknownMemberType]
                2,
                dtype=int,
            ),
            dimension=jnp.array(  # pyright: ignore[reportUnknownMemberType]
                (camera.viewWidth, camera.viewHeight)
            ),
            depth=jnp.array(  # pyright: ignore[reportUnknownMemberType]
                camera.viewDepth
            ),
        )
        assert isinstance(viewport_mat, Viewport), f"{viewport_mat}"

        _camera: Camera = Camera.create(
            view=view_mat,
            projection=projection_mat,
            viewport=viewport_mat,
            view_inv=view_inv,
        )
        assert isinstance(_camera, Camera), f"{_camera}"

        return _camera

    @staticmethod
    @jaxtyped
    @add_tracing_name
    def create_buffers(
        width: int,
        height: int,
        batch: Optional[int] = None,
        colour_default: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType, reportCallInDefaultInitializer]
            (1.0, 1.0, 1.0),
            dtype=jnp.single,
        ),
        zbuffer_default: NumV = jnp.array(  # pyright: ignore[reportUnknownMemberType, reportCallInDefaultInitializer]
            1,
            dtype=jnp.single,
        ),
    ) -> Buffers[Tuple[Canvas]]:
        """Render the scene with the given camera.

        The dtype of `colour_default` and `zbuffer_default` will be used as the
        dtype of canvas and zbuffer.

        Parameters:
          - width, height: the size of the image to render.
          - batch: if specified, will produce a batch of buffers, with batch
            axis at axis 0.
          - colour_default: default colours to fill the image with.
          - zbuffer_default: default zbuffer values to fill with.
          - shadow_param: the shadow parameters to render the scene with. Keep

        Returns: Buffers, with zbuffer and (coloured image, ).
        """
        _batch = (batch,) if batch is not None else ()
        zbuffer: ZBuffer = lax.full(  # pyright: ignore[reportUnknownMemberType]
            (*_batch, width, height),
            zbuffer_default,
        )
        canvas: Canvas = jnp.full(  # pyright: ignore[reportUnknownMemberType]
            (*_batch, width, height, colour_default.size),
            colour_default,
        )
        buffers: Buffers[Tuple[Canvas]] = Buffers(
            zbuffer=zbuffer,
            targets=(canvas,),
        )

        return buffers

    @classmethod
    @jaxtyped
    @partial(
        jit,
        static_argnames=("cls", "loop_unroll"),
        donate_argnums=(4,),
        inline=True,
    )
    @add_tracing_name
    def render(
        cls,
        model: MergedModel,
        light: LightParameters,
        camera: Camera,
        buffers: Buffers[TargetT],
        shadow_param: Optional[ShadowParameters] = None,
        loop_unroll: int = 1,
    ) -> Buffers[TargetT]:
        """Render the scene with the given camera.

        Parameters:
          - model: merged model of all the objects to render.
          - light: the light to render the scene with.
          - camera: the camera to render the scene with.
          - buffers: the buffers to render the scene with.
          - shadow_param: the shadow parameters to render the scene with. Keep
            it None to disable shadows.
          - loop_unroll: passed directly to `render`. See `pipeline:render`.

        Returns: Buffers, with zbuffer and (coloured image, ).
        """
        # flatten so each vertex has its own "extra"
        position = model.verts[
            model.faces.reshape((-1,))  # pyright: ignore[reportUnknownMemberType]
        ]
        normal = model.norms[
            model.faces_norm.reshape((-1,))  # pyright: ignore[reportUnknownMemberType]
        ]
        uv = model.uvs[
            model.faces_uv.reshape((-1,))  # pyright: ignore[reportUnknownMemberType]
        ]

        texture_index = model.texture_index[
            model.faces_uv.reshape((-1,))  # pyright: ignore[reportUnknownMemberType]
        ]
        # double_sided = model.texture_index[model.faces_uv.reshape((-1,))]

        face_indices: Integer[Array, "_ 3"]
        face_indices = jnp.arange(  # pyright: ignore[reportUnknownMemberType]
            model.faces.size
        ).reshape(model.faces.shape)
        assert isinstance(face_indices, Integer[Array, "_ 3"])

        light_dir: Vec3f = normalise(light.direction.copy())
        assert isinstance(light_dir, Vec3f), f"{light_dir}"

        light_dir_eye: Vec3f = Camera.apply_vec(
            light_dir.copy(),
            camera.view,
        )
        assert isinstance(light_dir_eye, Vec3f), f"{light_dir_eye}"

        extra = PhongReflectionTextureExtraInput(
            position=position,
            normal=normal,
            uv=uv,
            light=LightSource(
                direction=light_dir,
                colour=light.colour,
            ),
            light_dir_eye=light_dir_eye,
            texture_shape=model.texture_shape,
            texture_index=texture_index,
            texture_offset=jnp.array(  # pyright: ignore[reportUnknownMemberType]
                model.offset,
                dtype=int,
            ),
            texture=model.diffuse_map,
            specular_map=model.specular_map,
            ambient=light.ambient,
            diffuse=light.diffuse,
            specular=light.specular,
        )

        if shadow_param is None:
            # no shadows
            buffers = render(
                camera=camera,
                shader=PhongReflectionTextureShader,
                buffers=buffers,
                face_indices=face_indices,
                extra=extra,
                loop_unroll=loop_unroll,
            )
            assert isinstance(buffers, Buffers), f"{buffers}"

            return buffers
        else:
            # with shadows
            assert isinstance(shadow_param, ShadowParameters), f"{shadow_param}"
            # first pass: render shadow map
            shadow = cast(
                Shadow,
                Shadow.render_shadow_map(
                    shadow_map=lax.full_like(  # pyright: ignore[reportUnknownMemberType]
                        buffers.zbuffer,
                        DtypeInfo.create(
                            jax.dtypes.result_type(buffers.zbuffer)  # pyright: ignore
                        ).max,
                    ),
                    verts=model.verts,
                    faces=model.faces,
                    light_direction=light.direction,
                    viewport_matrix=camera.viewport,
                    centre=shadow_param.centre,
                    up=shadow_param.up,
                    strength=shadow_param.strength,
                    offset=shadow_param.offset,
                    loop_unroll=loop_unroll,
                ),
            )
            assert isinstance(shadow, Shadow), f"{shadow}"

            _extra = PhongReflectionShadowTextureExtraInput(
                **extra._asdict(),
                shadow=shadow,
                camera=camera,
            )

            # second pass: actual rendering
            buffers = render(
                camera=camera,
                shader=PhongReflectionShadowTextureShader,
                buffers=buffers,
                face_indices=face_indices,
                extra=_extra,
                loop_unroll=loop_unroll,
            )
            assert isinstance(buffers, Buffers), f"{buffers}"

            return buffers

    @classmethod
    @jaxtyped
    @partial(
        jit,
        static_argnames=("cls", "width", "height", "loop_unroll"),
        inline=True,
    )
    @add_tracing_name
    def get_camera_image(
        cls,
        objects: Sequence[ModelObject],
        light: LightParameters,
        camera: Union[Camera, CameraParameters],
        width: int,
        height: int,
        colour_default: Colour = jnp.array(  # pyright: ignore[reportUnknownMemberType, reportCallInDefaultInitializer]
            (1.0, 1.0, 1.0),
            dtype=jnp.single,
        ),
        zbuffer_default: NumV = jnp.array(  # pyright: ignore[reportUnknownMemberType, reportCallInDefaultInitializer]
            1,
            dtype=jnp.single,
        ),
        shadow_param: Optional[ShadowParameters] = None,
        loop_unroll: int = 1,
    ) -> Canvas:
        """Render the scene with the given camera.

        The dtype of `colour_default` and `zbuffer_default` will be used as the
        dtype of canvas and zbuffer.

        Parameters:
          - objects: the objects to render.
          - light: the light to render the scene with.
          - camera: the camera to render the scene with.
          - width, height: the size of the image to render.
          - colour_default: default colours to fill the image with.
          - zbuffer_default: default zbuffer values to fill with.
          - shadow_param: the shadow parameters to render the scene with. Keep
            it None to disable shadows.
          - loop_unroll: passed directly to `render`. See `pipeline:render`.

        Returns: Buffers, with zbuffer and (coloured image, ).
        """
        _camera: Camera
        if isinstance(camera, CameraParameters):
            _camera = cls.create_camera_from_parameters(camera)
        else:
            _camera = camera

        assert isinstance(_camera, Camera), f"{_camera}"

        light = tree_map(
            jnp.asarray,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            light,
            # only flatten one layer
            is_leaf=lambda x: not isinstance(x, LightParameters),
        )
        assert isinstance(light, LightParameters), f"{light}"

        buffers: Buffers[Tuple[Canvas]] = cls.create_buffers(
            width=width,
            height=height,
            colour_default=colour_default,
            zbuffer_default=zbuffer_default,
        )

        model: MergedModel = merge_objects(objects)
        assert isinstance(model, MergedModel), f"{model}"

        if shadow_param is not None:
            shadow_param = tree_map(
                jnp.asarray,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                shadow_param,
                # only flatten one layer
                is_leaf=lambda x: not isinstance(x, ShadowParameters),
            )

        canvas: Canvas
        _, (canvas,) = cls.render(
            model=model,
            light=light,
            camera=_camera,
            buffers=buffers,
            shadow_param=shadow_param,
            loop_unroll=loop_unroll,
        )
        assert isinstance(canvas, Canvas), f"{canvas}"

        return canvas
