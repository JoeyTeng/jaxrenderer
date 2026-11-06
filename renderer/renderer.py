from functools import partial
from typing import Optional, TypeAlias

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import (
    Array,
    Bool,
    Integer,
)

from ._meta_utils import (
    add_tracing_name,
    is_array_type,
    typed_array,
    typed_asarray,
    typed_full,
    typed_zeros,
)
from ._meta_utils import typed_jaxtyped as jaxtyped
from ._meta_utils import typed_jit as jit
from .geometry import Camera, Projection, View, Viewport, normalise
from .model import MergedModel, merge_objects
from .pipeline import render
from .renderer_types import (
    CameraImageInput as CameraImageInput,
)
from .renderer_types import (
    CameraParameters as CameraParameters,
)
from .renderer_types import (
    LightParameters as LightParameters,
)
from .renderer_types import (
    RenderInput as RenderInput,
)
from .renderer_types import (
    ShadowParameters as ShadowParameters,
)
from .shaders.phong_reflection import (
    PhongReflectionTextureExtraInput,
    PhongReflectionTextureExtraMixerOutput,
    PhongReflectionTextureShader,
)
from .shaders.phong_reflection_shadow import (
    PhongReflectionShadowTextureExtraInput,
    PhongReflectionShadowTextureExtraMixerOutput,
    PhongReflectionShadowTextureShader,
)
from .shadow import Shadow
from .types import (
    Buffers,
    Canvas,
    Colour,
    DtypeInfo,
    JaxFloating,
    JaxInteger,
    LightSource,
    NumV,
    Vec3f,
    ZBuffer,
)

DoubleSidedFaces: TypeAlias = Bool[Array, "faces"]
"""Whether to render both sides of each face (triangle primitive)."""


class Renderer:
    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def create_camera_from_parameters(camera: CameraParameters) -> Camera:
        """Create a camera from camera parameters."""
        eye: Vec3f = typed_asarray(camera.position, dtype=float)
        assert is_array_type(eye, Vec3f), f"{eye}"
        centre: Vec3f = typed_asarray(camera.target, dtype=float)
        assert is_array_type(centre, Vec3f), f"{centre}"
        up: Vec3f = typed_asarray(camera.up, dtype=float)
        assert is_array_type(up, Vec3f), f"{up}"

        view_mat: View = Camera.view_matrix(eye=eye, centre=centre, up=up)
        assert is_array_type(view_mat, View), f"{view_mat}"
        view_inv: View = Camera.view_matrix_inv(eye=eye, centre=centre, up=up)
        assert is_array_type(view_inv, View), f"{view_inv}"
        projection_mat: Projection = Camera.perspective_projection_matrix(
            fovy=camera.vfov,
            aspect=(
                lax.tan(jnp.radians(camera.hfov) / 2.0)
                / lax.tan(jnp.radians(camera.vfov) / 2.0)
            ),
            z_near=camera.near,
            z_far=camera.far,
        )
        assert is_array_type(projection_mat, Projection), f"{projection_mat}"
        viewport_mat: Viewport = Camera.viewport_matrix(
            lowerbound=typed_zeros(
                2,
                dtype=int,
            ),
            dimension=typed_array((camera.viewWidth, camera.viewHeight)),
            depth=typed_array(camera.viewDepth),
        )
        assert is_array_type(viewport_mat, Viewport), f"{viewport_mat}"

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
        colour_default: Colour = typed_array(  # pyright: ignore[reportCallInDefaultInitializer]
            (1.0, 1.0, 1.0),
            dtype=jnp.single,
        ),
        zbuffer_default: NumV = typed_array(  # pyright: ignore[reportCallInDefaultInitializer]
            1,
            dtype=jnp.single,
        ),
    ) -> Buffers[tuple[Canvas]]:
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
        zbuffer: ZBuffer = lax.full(
            (*_batch, width, height),
            zbuffer_default,
        )
        canvas: Canvas = typed_full(
            (*_batch, width, height, colour_default.size),
            colour_default,
        )
        buffers: Buffers[tuple[Canvas]] = Buffers(
            zbuffer=zbuffer,
            targets=(canvas,),
        )

        return buffers

    @classmethod
    @jaxtyped
    @partial(
        jit,
        static_argnames=("cls",),
        donate_argnames=("buffers",),
        inline=True,
    )
    @add_tracing_name
    def render(
        cls,
        inputs: RenderInput,
        buffers: Buffers[tuple[Canvas]],
    ) -> Buffers[tuple[Canvas]]:
        """Render a merged model into the provided working buffers.

        Parameters:
          - inputs: render configuration and model arrays; creating or replacing
            this record does not copy its arrays.
          - buffers: initial zbuffer and canvas, donated to JAX and consumed by
            this call. Do not reuse the input buffers afterward.

        Returns: A new Buffers record with the rendered zbuffer and canvas.
        """
        model = inputs.model
        light = inputs.light
        camera = inputs.camera
        shadow_param = inputs.shadow_param
        loop_unroll = inputs.loop_unroll

        # flatten so each vertex has its own "extra"
        position = model.verts[model.faces.reshape((-1,))]
        normal = model.norms[model.faces_norm.reshape((-1,))]
        uv = model.uvs[model.faces_uv.reshape((-1,))]

        texture_index = model.texture_index[model.faces.reshape((-1,))]
        double_sided = model.double_sided[model.faces.reshape((-1,))]

        face_indices: Integer[Array, "_ 3"]
        face_indices = jnp.arange(  # pyright: ignore[reportUnknownMemberType]
            model.faces.size
        ).reshape(model.faces.shape)
        assert is_array_type(face_indices, Integer[Array, "_ 3"])

        light_dir: Vec3f = normalise(light.direction.copy())
        assert is_array_type(light_dir, Vec3f), f"{light_dir}"

        light_dir_eye: Vec3f = Camera.apply_vec(
            light_dir.copy(),
            camera.view,
        )
        assert is_array_type(light_dir_eye, Vec3f), f"{light_dir_eye}"

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
            texture_offset=typed_array(
                model.offset,
                dtype=int,
            ),
            texture=model.diffuse_map,
            specular_map=model.specular_map,
            ambient=light.ambient,
            diffuse=light.diffuse,
            specular=light.specular,
            double_sided=double_sided,
        )

        if shadow_param is None:
            # no shadows
            rendered = render(
                camera=camera,
                shader=PhongReflectionTextureShader,
                buffers=Buffers(
                    zbuffer=buffers.zbuffer,
                    targets=PhongReflectionTextureExtraMixerOutput(
                        canvas=buffers.targets[0]
                    ),
                ),
                face_indices=face_indices,
                extra=extra,
                loop_unroll=loop_unroll,
            )
            return Buffers(zbuffer=rendered.zbuffer, targets=(rendered.targets.canvas,))
        else:
            # with shadows
            assert isinstance(shadow_param, ShadowParameters), f"{shadow_param}"
            # first pass: render shadow map
            zbuffer_dtype: type[JaxFloating | JaxInteger | int] = jnp.dtype(
                jax.dtypes.result_type(buffers.zbuffer)
            ).type
            shadow = Shadow.render_shadow_map(
                shadow_map=lax.full_like(
                    buffers.zbuffer,
                    DtypeInfo[JaxFloating | JaxInteger | int].create(zbuffer_dtype).max,
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
                double_sided=model.double_sided,
            )
            assert isinstance(shadow, Shadow), f"{shadow}"

            _extra = PhongReflectionShadowTextureExtraInput(
                **extra._asdict(),
                shadow=shadow,
                camera=camera,
            )

            # second pass: actual rendering
            rendered_shadow = render(
                camera=camera,
                shader=PhongReflectionShadowTextureShader,
                buffers=Buffers(
                    zbuffer=buffers.zbuffer,
                    targets=PhongReflectionShadowTextureExtraMixerOutput(
                        canvas=buffers.targets[0]
                    ),
                ),
                face_indices=face_indices,
                extra=_extra,
                loop_unroll=loop_unroll,
            )
            return Buffers(
                zbuffer=rendered_shadow.zbuffer,
                targets=(rendered_shadow.targets.canvas,),
            )

    @classmethod
    @jaxtyped
    @partial(
        jit,
        static_argnames=("cls",),
        inline=True,
    )
    @add_tracing_name
    def get_camera_image(
        cls,
        inputs: CameraImageInput,
    ) -> Canvas:
        """Render the scene with the given camera.

        If `inputs.objects` is empty, return a canvas filled with
        `inputs.colour_default` without processing the camera, light, or
        rendering pipeline. Replacing the input record does not copy its arrays.

        The dtype of `colour_default` and `zbuffer_default` is used as the
        dtype of canvas and zbuffer.

        Parameters:
          - inputs: objects, camera, lighting, framebuffer size, and render
            options.

        Returns: The rendered canvas, or the configured background canvas when
          there are no objects.
        """
        width = inputs.width
        height = inputs.height
        colour_default = inputs.colour_default
        zbuffer_default = inputs.zbuffer_default

        buffers: Buffers[tuple[Canvas]] = cls.create_buffers(
            width=width,
            height=height,
            colour_default=colour_default,
            zbuffer_default=zbuffer_default,
        )
        if not inputs.objects:
            return buffers.targets[0]

        _camera: Camera
        if isinstance(inputs.camera, CameraParameters):
            _camera = cls.create_camera_from_parameters(inputs.camera)
        else:
            _camera = inputs.camera

        assert isinstance(_camera, Camera), f"{_camera}"

        light = tree_map(
            typed_asarray,
            inputs.light,
            # only flatten one layer
            is_leaf=lambda x: not isinstance(x, LightParameters),
        )
        assert isinstance(light, LightParameters), f"{light}"

        model: MergedModel = merge_objects(inputs.objects)
        assert isinstance(model, MergedModel), f"{model}"

        shadow_param = inputs.shadow_param
        if shadow_param is not None:
            shadow_param = tree_map(
                typed_asarray,
                shadow_param,
                # only flatten one layer
                is_leaf=lambda x: not isinstance(x, ShadowParameters),
            )

        canvas: Canvas
        _, (canvas,) = cls.render(
            inputs=RenderInput(
                model=model,
                light=light,
                camera=_camera,
                shadow_param=shadow_param,
                loop_unroll=inputs.loop_unroll,
            ),
            buffers=buffers,
        )
        assert is_array_type(canvas, Canvas), f"{canvas}"

        return canvas
