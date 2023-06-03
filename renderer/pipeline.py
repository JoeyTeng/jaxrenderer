from functools import partial
from typing import Any, NamedTuple, TypeVar

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Float, Integer, Num, jaxtyped

from .geometry import Camera, Interpolation, Viewport, interpolate
from .shader import (ID, MixedExtraT, MixerOutput, PerFragment, PerVertex,
                     Shader, ShaderExtraInputT, VaryingT)
from .types import (FALSE_ARRAY, Buffers, FaceIndices, Triangle, Vec2f, Vec2i,
                    Vec3f, Vec4f)

jax.config.update('jax_array', True)


class PerPrimitive(NamedTuple):
    """Input for each primitive, using outputs from Vertex Shader.

    gl_Position is in clip-space, not normalised.
    """
    gl_Position: Triangle
    # gl_PointSize is meaningful only when rendering point primitives.
    # not supported for now
    # gl_PointSize: Float[Array, "primitives"]
    keep: Bool[Array, ""]
    """Whether to keep this primitive for rasterisation.
        !!Never keep a primitive with a zero determinant.
    """
    determinant: Float[Array, ""]
    """determinant of the matrix with [x, y, w] of the three vertices in clip
        space, in a shape of

        [[x0, y0, w0],
         [x1, y1, w1],
         [x2, y2, w2]].

        When determinant is 0, the triangle will not be rendered for now.
    """
    matrix_inv: Float[Array, "3 3"]
    """inverse of the matrix described above (of [x, y, w])."""

    @classmethod
    @jaxtyped
    @partial(jax.jit, static_argnames=("cls", ))
    def create(cls, per_vertex: PerVertex) -> "PerPrimitive":
        """per_vertex is batched with size 3 (3 vertices per triangle)
            in clip-space, not normalised.
        """
        clip: Triangle = per_vertex.gl_Position
        assert isinstance(clip, Triangle)
        # matrix with x, y, w
        matrix: Float[Array, "3 3"] = clip[:, jnp.array((0, 1, 3))]
        # If == 0, the matrix inverse does not exist, should use another
        # interpolation method. Early exit for now.
        # `jnp.linalg.det` has built-in 3x3 det optimisation
        determinant: Float[Array, ""] = jnp.linalg.det(matrix)
        assert isinstance(determinant, Float[Array, ""])

        # an arbitrary number for numerical stability
        keep: Bool[Array, ""] = lax.abs(determinant) > 1e-6

        mat_inv: Float[Array, "3 3"] = lax.cond(
            # an arbitrary number for numerical stability
            keep,
            # may replace with custom implementation for higher precision
            lambda: jnp.linalg.inv(matrix),
            lambda: jnp.zeros((3, 3)),
        )
        assert isinstance(mat_inv, Float[Array, "3 3"])

        return cls(
            gl_Position=clip,
            keep=keep,
            determinant=determinant,
            matrix_inv=mat_inv,
        )


@jaxtyped
@partial(jax.jit, static_argnames=("shader", ), donate_argnums=(1, ))
def _postprocessing(
    shader: type[Shader[ShaderExtraInputT, VaryingT, MixedExtraT]],
    buffers: Buffers,
    per_primitive: tuple[Any, ...],  # Batch PerPrimitive
    varyings: VaryingT,
    extra: ShaderExtraInputT,
    viewport: Viewport,
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
            # PROCESS: Interpolation

            # For early exit when not keep primitive / determinant is 0
            def _when_keep_primitive() -> tuple[Vec3f, Float[Array, ""]]:
                """Returns clip_coef, w_reciprocal."""
                # x/w, y/w, with x, y, w in clip space.
                xy: Float[Array, "2"] = (
                    (coord - viewport[:2, 3]) /
                    viewport[jnp.arange(2), jnp.arange(2)])
                xy1_ndc: Float[Array, "3"] = jnp.array((xy[0], xy[1], 1))

                # As the interpolation formula is `xy1_ndc @ (mat_inv @ values)`
                # we can utilise associativity to generate a set of fixed Vec3f
                # coefficient for interpolation.
                # Noticed that this is also the "edge function" values, with
                # a pseudo-parameter that is zero at the two vertices on the
                # edge and one at the opposite vertex, as described
                # in [Olano and Greer, 1997].
                clip_coef: Vec3f = jnp.dot(xy1_ndc, primitive.matrix_inv)
                assert isinstance(clip_coef, Vec3f)
                # 1/w, w in clip space.
                w_reciprocal: Float[Array, ""] = clip_coef.sum()
                assert isinstance(w_reciprocal, Float[Array, ""])

                return clip_coef, w_reciprocal

            # END OF `_when_keep_primitive`

            # Prepare for interpolation parameters
            # clip_coef here interpolates to 1/w * target value
            clip_coef, w_reciprocal = lax.cond(
                # an arbitrary number for numerical stability
                primitive.keep,
                _when_keep_primitive,
                lambda: (lax.full((3, ), -1.), jnp.zeros(())),
            )

            def _when_in_triangle() -> tuple[PerFragment, VaryingT]:
                # Prepare inputs for fragment shader
                z: Float[Array, ""] = interpolate(
                    values=primitive.gl_Position[:, 2],
                    barycentric_screen=clip_coef,
                    barycentric_clip=clip_coef,
                    mode=Interpolation.SMOOTH,
                )
                # viewport transform for z, from clip space to window space
                z = z * viewport[2, 2] + viewport[2, 3]
                gl_FragCoord: Vec4f = jnp.array((
                    coord[0],
                    coord[1],
                    z,
                    w_reciprocal,
                ))
                assert isinstance(gl_FragCoord, Vec4f)

                # Ref: https://registry.khronos.org/OpenGL-Refpages/gl4/html/gl_FrontFacing.xhtml
                gl_FrontFacing: Bool[Array, ""] = primitive.determinant > 0
                assert isinstance(gl_FrontFacing, Bool[Array, ""])

                gl_PointCoord: Vec2f
                with jax.ensure_compile_time_eval():
                    # TODO: implement Point primitive properly.
                    gl_PointCoord = lax.full((2, ), 0)

                # this interpolates to target value u, not u/w
                true_clip_coef: Vec3f = clip_coef / w_reciprocal
                assert isinstance(true_clip_coef, Vec3f)

                varying: VaryingT = shader.interpolate(
                    values=varying_per_primitive,
                    barycentric_screen=true_clip_coef,
                    barycentric_clip=true_clip_coef,
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

            # END OF `_when_in_triangle`

            in_triangle: Bool[Array, ""] = (clip_coef >= 0).all()
            assert isinstance(in_triangle, Bool[Array, ""])

            built_in: PerFragment
            attachments: VaryingT
            built_in, attachments = lax.cond(
                jnp.logical_and(primitive.keep, in_triangle),
                _when_in_triangle,
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
    shader: type[Shader[ShaderExtraInputT, VaryingT, MixedExtraT]],
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
    ) -> tuple[PerVertex, VaryingT]:
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

        return per_vertex, varying

    # PROCESS: Vertex Processing
    per_vertices, varyings = jax.vmap(vertex_processing)(
        lax.iota(int, vertices_count),  # gl_VertexID
    )

    # everything after vertex processing, will directly update buffers
    buffers = _postprocessing(
        shader=shader,
        buffers=buffers,
        per_primitive=jax.vmap(PerPrimitive.create)(tree_map(
            lambda field: field[face_indices],
            per_vertices,
        )),
        varyings=tree_map(lambda field: field[face_indices], varyings),
        extra=extra,
        viewport=camera.viewport,
    )
    assert isinstance(buffers, Buffers)

    return buffers
