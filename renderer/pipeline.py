from functools import partial
from typing import Any, NamedTuple, TypeVar, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Float, Integer, Num, jaxtyped

from .geometry import (Camera, Interpolation, barycentric, interpolate,
                       normalise_homogeneous)
from .shader import (ID, MixedExtraT, MixerOutput, PerFragment, PerVertex,
                     Shader, ShaderExtraInputT, VaryingT)
from .types import (FALSE_ARRAY, Buffers, FaceIndices, Triangle, Triangle2Df,
                    Vec2f, Vec2i, Vec3f, Vec4f)

jax.config.update('jax_array', True)


class PerVertexInScreen(NamedTuple):
    """Built-in output from Vertex Shader after Viewport transformation.

    gl_Position is in screen-space.
    """
    gl_Position: Vec4f
    # gl_PointSize is meaningful only when rendering point primitives.
    # not supported for now
    # gl_PointSize: Float[Array, ""]


class PerPrimitive(NamedTuple):
    """Input for each primitive, using outputs from Vertex Shader.

    gl_Position is in screen-space, normalised homogeneous coordinate
    """
    gl_Position: Triangle
    # gl_PointSize is meaningful only when rendering point primitives.
    # not supported for now
    # gl_PointSize: Float[Array, "primitives"]


@jaxtyped
@partial(jax.jit, static_argnames=("shader", ), donate_argnums=(1, ))
def _postprocessing(
    shader: Union[Shader[ShaderExtraInputT, VaryingT, MixedExtraT],
                  type[Shader[ShaderExtraInputT, VaryingT, MixedExtraT]]],
    buffers: Buffers,
    per_primitive: tuple[Any, ...],  # Batch PerPrimitive
    varyings: VaryingT,
    extra: ShaderExtraInputT,
) -> Buffers:
    with jax.ensure_compile_time_eval():
        # loop along first axis, for memory efficiency
        # TODO: benchmark if this is actually faster
        loop_size: int = int(buffers[0].shape[0])
        # vmap batch along second axis
        batch_size: int = int(buffers[0].shape[1])

    @jaxtyped
    @jax.jit
    def _per_pixel(coord: Vec2i) -> tuple[MixerOutput, MixedExtraT]:

        assert isinstance(coord, Vec2i), f"expected Vec2i, got {coord}"

        @jaxtyped
        def _per_primitive_process(
            primitive: PerPrimitive,
            varying_per_primitive: VaryingT,
        ) -> tuple[PerFragment, VaryingT]:
            screen: Triangle = primitive.gl_Position

            # 2d screen coordinates
            screen2d: Triangle2Df = screen[:, :2]
            assert isinstance(screen2d, Triangle2Df)

            # PROCESS: Rasterisation (Interpolate)
            # barycentric coordinates
            bc_screen: Vec3f = barycentric(screen2d, coord)
            assert isinstance(bc_screen, Vec3f)

            def _when_keep_triangle() -> tuple[PerFragment, VaryingT]:
                # weight barycentric coordinates by 1/w to obtain
                # perspective-coorected barycentric coordinates
                bc_clip: Vec3f = bc_screen * screen[:, 3]

                # normalise barycentric coordinates so that it sums to 1.
                bc_clip = bc_clip / bc_clip.sum()

                # Prepare inputs for fragment shader
                # no need to "normalise" as barycentric coordinates here
                # ensures it sums to 1, thus result is normalised when inputs
                # are.
                # Mode: NOPERSPECTIVE: since inverse of the depth is linear,
                # the correct way to interpolate it is just to interpolate
                # under screen space using bc_screen,
                # or using `NONPERSPECTVIE` mode.
                gl_FragCoord: Vec4f = interpolate(
                    screen,
                    bc_screen,
                    bc_clip,
                    mode=Interpolation.NOPERSPECTIVE,
                )
                gl_FragCoord.at[:2].set(coord)
                assert isinstance(gl_FragCoord, Vec4f)

                # Ref: https://registry.khronos.org/OpenGL-Refpages/gl4/html/gl_FrontFacing.xhtml
                gl_FrontFacing: Bool[Array, ""] = jnp.cross(
                    screen[1, :3] - screen[0, :3],
                    screen[2, :3] - screen[0, :3],
                )[-1] > 0
                assert isinstance(gl_FrontFacing, Bool[Array, ""])

                gl_PointCoord: Vec2f
                with jax.ensure_compile_time_eval():
                    # TODO: implement Point primitive properly.
                    gl_PointCoord = lax.full((2, ), 0)

                varying: VaryingT = shader.interpolate(
                    varying_per_primitive,
                    bc_screen,
                    bc_clip,
                )
                assert isinstance(varying, tuple)

                # PROCESS: Fragment Processing
                per_frag: PerFragment
                extra_fragment_output: VaryingT
                per_frag, extra_fragment_output = shader.fragment(
                    gl_FragCoord=gl_FragCoord,
                    gl_FrontFacing=gl_FrontFacing,
                    gl_PointCoord=gl_PointCoord,
                    varying=varying,
                    extra=extra,
                )
                assert isinstance(per_frag, PerFragment)
                assert isinstance(extra_fragment_output, tuple)

                # enforce default `gl_FragDepth` when it is None
                per_frag = lax.cond(
                    per_frag.use_default_depth,
                    lambda: per_frag._replace(gl_FragDepth=gl_FragCoord[2]),
                    lambda: per_frag,
                )
                assert isinstance(per_frag, PerFragment)

                return per_frag, extra_fragment_output

            # END OF `_when_keep_triangle`

            in_triangle: Bool[Array, ""] = (bc_screen >= 0).all()
            assert isinstance(in_triangle, Bool[Array, ""])

            built_in: PerFragment
            attachments: VaryingT
            built_in, attachments = lax.cond(
                in_triangle,
                _when_keep_triangle,
                # discard out-of-triangle values
                lambda: (
                    PerFragment(keeps=FALSE_ARRAY),
                    # dummy values
                    tree_map(lambda field: field[0], varying_per_primitive),
                ),
            )
            assert isinstance(built_in, PerFragment)
            assert isinstance(attachments, tuple)

            return built_in, attachments

        # END OF `_per_primitive_process`

        built_in, extra_outputs = jax.vmap(_per_primitive_process)(
            per_primitive,
            varyings,
        )
        gl_Depths = built_in.gl_FragDepth
        keeps = built_in.keeps

        # PROCESS: Per-Sample Operations (Mixing: depth test + colour blending)
        mixed_output: MixerOutput
        attachments: MixedExtraT
        mixed_output, attachments = shader.mix(gl_Depths, keeps, extra_outputs)
        assert isinstance(mixed_output, MixerOutput)
        assert isinstance(attachments, tuple)

        return mixed_output, attachments

    @jaxtyped
    @partial(jax.jit, donate_argnums=(1, ))
    def loop_body(
        index: Integer[Array, ""],
        buffers: Buffers,
    ) -> Buffers:

        _valueT = TypeVar('_valueT', bound=tuple[Any, ...])

        @jaxtyped
        @partial(jax.jit, donate_argnums=(2, ))
        def select_value_per_pixel(
            keep: Bool[Array, ""],
            new_values: _valueT,
            old_values: _valueT,
        ) -> _valueT:
            """Choose new value of the pixel, or keep the previous."""
            FieldRowT = TypeVar("FieldRowT")

            def _select_per_field(
                new_field_value: FieldRowT,
                old_field_value: FieldRowT,
            ) -> FieldRowT:
                """Choose this pixel for this field in the PyTree."""
                return lax.cond(
                    keep,
                    lambda: new_field_value,
                    lambda: old_field_value,
                )

            # tree_map over each field in the PyTree
            result: _valueT = tree_map(
                _select_per_field,
                new_values,
                old_values,
            )

            return result

        keeps: Bool[Array, "height"]
        depths: Num[Array, "height"]
        extras: MixedExtraT
        # vmap over axis 1 (height) of the buffers. Axis 0 (width) is `index`.
        (keeps, depths), extras = jax.vmap(_per_pixel)(lax.concatenate(
            (
                lax.full((batch_size, 1), index),
                lax.broadcasted_iota(int, (batch_size, 1), 0),
            ),
            1,
        ))
        assert isinstance(keeps, Bool[Array, "height"])
        assert isinstance(depths, Num[Array, "height"])
        assert isinstance(extras, tuple)

        # vmap each pixel over axis 1 (height) of the buffers (per row in
        # matrix)
        buffers_row = jax.vmap(select_value_per_pixel)(
            keeps,
            Buffers(zbuffer=depths, targets=tuple(extras)),
            tree_map(lambda field: field[index], buffers),
        )

        # tree_map over each field in the PyTree to update all buffers
        return tree_map(
            lambda field, value: field.at[index].set(value),
            buffers,
            buffers_row,
        )

    # END OF `loop_body`

    # iterate over axis 0 (width) of the buffers (one row at a time)
    buffers = lax.fori_loop(
        0,
        loop_size,
        loop_body,
        buffers,
    )
    assert isinstance(buffers, Buffers)

    return buffers


@jaxtyped
@partial(jax.jit, static_argnames=("shader", ), donate_argnums=(2, ))
def render(
    camera: Camera,
    shader: Union[Shader[ShaderExtraInputT, VaryingT, MixedExtraT],
                  type[Shader[ShaderExtraInputT, VaryingT, MixedExtraT]]],
    buffers: Buffers,
    face_indices: FaceIndices,
    extra: ShaderExtraInputT,
) -> Buffers:
    vertices_count: int
    gl_InstanceID: ID
    with jax.ensure_compile_time_eval():
        vertices_count = extra[0].shape[0]
        gl_InstanceID = jnp.array(0, dtype=int)
        assert isinstance(vertices_count, int)
        assert isinstance(gl_InstanceID, ID)

    @jaxtyped
    @jax.jit
    def vertex_processing(
            gl_VertexID: Integer[Array, ""],  #
    ) -> tuple[PerVertexInScreen, VaryingT]:
        """Process one vertex into screen space, and keep varying values."""
        per_vertex: PerVertex
        varying: VaryingT
        per_vertex, varying = shader.vertex(
            gl_VertexID,
            gl_InstanceID,
            camera,
            extra,
        )
        assert isinstance(per_vertex, PerVertex)
        assert isinstance(varying, tuple)

        # TODO: add clipping in clip space.

        # NDC, normalised device coordinate
        # Ref: OpenGL Spec 4.6 (Core Profile), Section 15.2.2
        w: Float[Array, ""] = per_vertex.gl_Position[3]
        ndc: Vec4f = normalise_homogeneous(per_vertex.gl_Position)
        assert isinstance(ndc, Vec4f)

        # already normalised; result is still normalised
        # Ref: OpenGL Spec 4.6 (Core Profile), Section 15.2.2
        screen: Vec4f = (camera.viewport @ ndc).at[3].divide(w)
        assert isinstance(screen, Vec4f)

        return PerVertexInScreen(gl_Position=screen), varying

    # PROCESS: Vertex Processing
    per_vertices, varyings = jax.vmap(vertex_processing)(
        lax.iota(int, vertices_count),  # gl_VertexID
    )

    # everything after vertex processing, will directly update buffers
    buffers = _postprocessing(
        shader=shader,
        buffers=buffers,
        per_primitive=tree_map(lambda field: field[face_indices],
                               per_vertices),
        varyings=tree_map(lambda field: field[face_indices], varyings),
        extra=extra,
    )
    assert isinstance(buffers, Buffers)

    return buffers
