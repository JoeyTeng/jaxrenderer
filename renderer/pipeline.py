from functools import partial
from typing import Any, NamedTuple, TypeVar, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Integer, Num, jaxtyped

from .geometry import (Camera, Interpolation, barycentric, interpolate,
                       normalise_homogeneous)
from .shader import (ID, MixedExtraT, MixerOutput, PerFragment, PerVertex,
                     Shader, VaryingT, VertexShaderExtraInputT)
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
    shader: Union[Shader[VertexShaderExtraInputT, VaryingT, MixedExtraT],
                  type[Shader[VertexShaderExtraInputT, VaryingT,
                              MixedExtraT]]],
    buffers: Buffers,
    per_primitive: tuple[Any, ...],  # Batch PerPrimitive
    varyings: VaryingT,
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
                bc_clip: Vec3f = bc_screen / screen[:, 3]
                bc_clip = bc_clip / bc_clip.sum()

                # Prepare inputs for fragment shader
                # no need to "normalise" as barycentric coordinates here
                # ensures it sums to 1, thus result is normalised when inputs
                # are.
                gl_FragCoord: Vec4f = interpolate(
                    screen,
                    bc_screen,
                    bc_clip,
                    mode=Interpolation.SMOOTH,
                )
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
                    extra=varying,
                )
                assert isinstance(per_frag, PerFragment)
                assert isinstance(extra_fragment_output, tuple)

                # enforce default `gl_FragDepth` when it is None
                per_frag = lax.cond(
                    jnp.isnan(per_frag.gl_FragDepth),
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

        (gl_Depths, keeps), extra_outputs = jax.vmap(_per_primitive_process)(
            per_primitive,
            varyings,
        )

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
        def select_value_per_row(
            keep: Bool[Array, ""],
            new_values: _valueT,
            old_values: _valueT,
        ) -> _valueT:
            FieldRowT = TypeVar("FieldRowT")

            def _select_per_field(
                new_field_value: FieldRowT,
                old_field_value: FieldRowT,
            ) -> FieldRowT:
                """Either choose new value of this field for row `index`, or
                    keep the previous value."""
                return lax.cond(
                    keep,
                    lambda: new_field_value,
                    lambda: old_field_value,
                )

            result: _valueT = tree_map(
                _select_per_field,
                new_values,
                old_values,
            )

            return result

        keeps: Bool[Array, "height"]
        depths: Num[Array, "height"]
        extras: MixedExtraT
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

        buffers_row = jax.vmap(select_value_per_row)(
            keeps,
            Buffers(zbuffer=depths, targets=tuple(extras)),
            tree_map(lambda field: field[index], buffers),
        )

        return tree_map(
            lambda field, value: field.at[index].set(value),
            buffers,
            buffers_row,
        )

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
    shader: Union[Shader[VertexShaderExtraInputT, VaryingT, MixedExtraT],
                  type[Shader[VertexShaderExtraInputT, VaryingT,
                              MixedExtraT]]],
    buffers: Buffers,
    face_indices: FaceIndices,
    extra: VertexShaderExtraInputT,
) -> Buffers:
    vertices_count: int
    gl_InstanceID: ID
    with jax.ensure_compile_time_eval():
        vertices_count = extra[0].shape[0]
        gl_InstanceID = jnp.zeros((1, ))

    @jaxtyped
    @jax.jit
    def vertex_processing(
        gl_VertexID: Integer[Array, ""],
        _extra: VertexShaderExtraInputT,
    ) -> tuple[PerVertexInScreen, VaryingT]:
        per_vertex: PerVertex
        varying: VaryingT
        per_vertex, varying = shader.vertex(
            gl_VertexID,
            gl_InstanceID,
            camera,
            _extra,
        )
        assert isinstance(per_vertex, PerVertex)
        assert isinstance(varying, tuple)

        # TODO: add clipping in clip space.

        # NDC, normalised device coordinate
        ndc: Vec4f = normalise_homogeneous(per_vertex.gl_Position)
        assert isinstance(ndc, Vec4f)

        # already normalised; result is still normalised
        screen: Vec4f = camera.viewport @ ndc
        assert isinstance(screen, Vec4f)

        vertex_with_screen: PerVertexInScreen
        vertex_with_screen = PerVertexInScreen(gl_Position=screen)
        assert isinstance(vertex_with_screen, PerVertexInScreen)

        return vertex_with_screen, varying

    # PROCESS: Vertex Processing
    per_vertices, varyings = jax.vmap(vertex_processing)(
        lax.iota(int, vertices_count),  # gl_VertexID
        extra,  # extra
    )

    # everything after vertex processing, will directly update buffers
    buffers = _postprocessing(
        shader,
        buffers,
        tree_map(lambda field: field[face_indices], per_vertices),
        tree_map(lambda field: field[face_indices], varyings),
    )
    assert isinstance(buffers, Buffers)

    return buffers
