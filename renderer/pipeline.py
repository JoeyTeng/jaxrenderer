from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from functools import partial
from typing import Any, NamedTuple, TypeVar, cast

import jax
from jax import lax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Float, Integer, Num
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from ._backport import Tuple
from ._meta_utils import add_tracing_name
from ._meta_utils import typed_jit as jit
from .geometry import Camera, Interpolation, Viewport, interpolate
from .shader import (
    ID,
    MixedExtraT,
    MixerOutput,
    PerFragment,
    PerVertex,
    Shader,
    ShaderExtraInputT,
    VaryingT,
)
from .types import (
    BoolV,
    Buffers,
    CanvasMask,
    FaceIndices,
    FloatV,
    IntV,
    Triangle,
    Vec2f,
    Vec2i,
    Vec3f,
    Vec4f,
    ZBuffer,
)

jax.config.update("jax_array", True)  # pyright: ignore[reportUnknownMemberType]

RowIndices = Integer[Array, "row_batches row_batch_size"]
"""Indices of the rows in the buffers to be processed in this batch."""


class PerPrimitive(NamedTuple):
    """Input for each primitive, using outputs from Vertex Shader.

    gl_Position is in clip-space, not normalised.
    """

    gl_Position: Triangle
    # gl_PointSize is meaningful only when rendering point primitives.
    # not supported for now
    # gl_PointSize: Float[Array, "primitives"]
    keep: BoolV
    """Whether to keep this primitive for rasterisation.
        !!Never keep a primitive with a zero determinant.
    """
    determinant: FloatV
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
    @partial(jit, static_argnames=("cls",), inline=True)
    @add_tracing_name
    def create(cls, per_vertex: PerVertex) -> "PerPrimitive":
        """per_vertex is batched with size 3 (3 vertices per triangle)
        in clip-space, not normalised.
        """
        clip: Triangle = per_vertex.gl_Position
        assert isinstance(clip, Triangle)
        # matrix with x, y, w
        matrix: Float[Array, "3 3"] = clip[:, jnp.array((0, 1, 3))]  # pyright: ignore
        # If == 0, the matrix inverse does not exist, should use another
        # interpolation method. Early exit for now.
        # `jnp.linalg.det` has built-in 3x3 det optimisation
        determinant = cast(
            FloatV,
            jnp.linalg.det(matrix),  # pyright: ignore[reportUnknownMemberType]
        )
        assert isinstance(determinant, FloatV)

        # an arbitrary number for numerical stability
        keep: BoolV = (
            lax.abs(determinant) > 1e-6  # pyright: ignore[reportUnknownMemberType]
        )

        # although this may result in NaN or Inf when keep is False,
        # it will be discarded later.
        # Perf: Remove lax.cond to reduce extra operations `select_n` in HLO.
        mat_inv = cast(Float[Array, "3 3"], jnp.linalg.inv(matrix))
        assert isinstance(mat_inv, Float[Array, "3 3"])

        return cls(
            gl_Position=clip,
            keep=keep,
            determinant=determinant,
            matrix_inv=mat_inv,
        )


T = TypeVar("T", bound=Tuple[Any, ...])


@jaxtyped
@partial(
    jit,
    static_argnames=("shader", "loop_unroll"),
    donate_argnums=(1,),
    inline=True,
)
@add_tracing_name
def _postprocessing(
    shader: type[Shader[ShaderExtraInputT, VaryingT, MixedExtraT]],
    buffers: Buffers[T],
    per_primitive: Tuple[Any, ...],  # Batch PerPrimitive
    varyings: VaryingT,
    extra: ShaderExtraInputT,
    viewport: Viewport,
    loop_unroll: int,
) -> Buffers[T]:
    with jax.ensure_compile_time_eval():
        # vmap batch along second axis
        batch_size: int = int(buffers[0].shape[1])
        row_indices: Integer[Array, "width"]
        row_indices = lax.iota(  # pyright: ignore[reportUnknownMemberType]
            int, int(buffers[0].shape[0])
        )

    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def _per_pixel(coord: Vec2i) -> Tuple[MixerOutput, MixedExtraT]:
        assert isinstance(coord, Vec2i), f"expected Vec2i, got {coord}"

        ReturnT = Tuple[  #
            Float[Array, "kept_primitives 4"],  #
            Bool[Array, "kept_primitives"],  #
            Float[Array, "kept_primitives 2"],  #
            Bool[Array, "kept_primitives"],  #
            VaryingT,  #
            Float[Array, "kept_primitives 3"],  #
            Float[Array, "kept_primitives 3"],  #
        ]

        @jaxtyped
        @partial(jit, inline=True)
        @add_tracing_name
        def _per_primitive_preprocess(
            primitive: PerPrimitive,
            varying_per_primitive: VaryingT,
        ) -> ReturnT:
            # PROCESS: Early Culling (`primitive_chooser`)

            # For early exit when not keep primitive / determinant is 0
            @partial(jit, inline=True)
            @add_tracing_name
            def _when_keep_primitive() -> Tuple[Vec3f, FloatV]:
                """Returns clip_coef, w_reciprocal."""
                # x/w, y/w, with x, y, w in clip space.
                _idx: Integer[Array, "2"]
                _idx = jnp.arange(2)  # pyright: ignore[reportUnknownMemberType]
                xy: Float[Array, "2"] = (coord - viewport[:2, 3]) / viewport[_idx, _idx]
                xy1_ndc: Float[Array, "3"]
                xy1_ndc = jnp.array(  # pyright: ignore[reportUnknownMemberType]
                    (xy[0], xy[1], 1)
                )

                # As the interpolation formula is `xy1_ndc @ (mat_inv @ values)`
                # we can utilise associativity to generate a set of fixed Vec3f
                # coefficient for interpolation.
                # Noticed that this is also the "edge function" values, with
                # a pseudo-parameter that is zero at the two vertices on the
                # edge and one at the opposite vertex, as described
                # in [Olano and Greer, 1997].
                clip_coef = cast(Vec3f, jnp.dot(xy1_ndc, primitive.matrix_inv))
                assert isinstance(clip_coef, Vec3f)
                # 1/w, w in clip space.
                w_reciprocal: FloatV = clip_coef.sum()  # pyright: ignore
                assert isinstance(w_reciprocal, FloatV)

                return clip_coef, w_reciprocal

            # END OF `_when_keep_primitive`

            @partial(jit, inline=True)
            @add_tracing_name
            def _when_in_triangle(
                clip_coef: Vec3f,
                w_reciprocal: FloatV,
            ) -> Tuple[  #
                Float[Array, "kept_primitives 4"],  # gl_FragCoord
                Bool[Array, "kept_primitives"],  # gl_FrontFacing
                Float[Array, "kept_primitives 2"],  # gl_PointCoord
                Float[Array, "kept_primitives 3"],  # true_clip_coef
            ]:
                # Prepare inputs for fragment shader
                z: FloatV = interpolate(
                    values=primitive.gl_Position[:, 2],
                    barycentric_screen=clip_coef,
                    barycentric_clip=clip_coef,
                    mode=Interpolation.SMOOTH,
                )
                # viewport transform for z, from clip space to window space
                z = z * viewport[2, 2] + viewport[2, 3]
                gl_FragCoord: Vec4f = jnp.array(  # pyright: ignore
                    (
                        coord[0],
                        coord[1],
                        z,
                        w_reciprocal,
                    )
                )
                assert isinstance(gl_FragCoord, Vec4f)

                # Ref: https://registry.khronos.org/OpenGL-Refpages/gl4/html/gl_FrontFacing.xhtml
                # True if not back-facing.
                gl_FrontFacing: BoolV = primitive.determinant >= 0
                assert isinstance(gl_FrontFacing, BoolV)

                gl_PointCoord: Vec2f
                with jax.ensure_compile_time_eval():
                    # TODO: implement Point primitive properly.
                    gl_PointCoord = lax.full((2,), 0.0)  # pyright: ignore

                # this interpolates to target value u, not u/w
                true_clip_coef: Vec3f = clip_coef / w_reciprocal
                assert isinstance(true_clip_coef, Vec3f)

                return (gl_FragCoord, gl_FrontFacing, gl_PointCoord, true_clip_coef)

            # END OF `_when_in_triangle`

            # Prepare for interpolation parameters
            # clip_coef here interpolates to 1/w * target value
            # Perf: although this may result in garbage values (NaN or Inf)
            # when keep is False, since it will be discarded later, we can
            # remove the lax.cond to reduce extra operations `select_n` in HLO
            # as the computation is quite cheap.
            # also see google/brax#8409 for why `_when_keep_primitive` is
            # always executed.
            clip_coef, w_reciprocal = _when_keep_primitive()

            in_triangle: BoolV = (clip_coef >= 0).all()  # pyright: ignore
            assert isinstance(in_triangle, BoolV)

            # Perf: although this may result in garbage values (NaN or Inf)
            # when keep or in_triangle is False, since it will be discarded
            # later, we can remove the lax.cond to reduce extra operations
            # `select_n` in HLO.
            # See google/brax#8409 for why `_when_keep_primitive` is always
            # executed.
            # TODO: change back to `lax.cond` when it does not force execute both branches under vmap.
            r = _when_in_triangle(clip_coef, w_reciprocal)
            gl_FragCoord, gl_FrontFacing, gl_PointCoord, true_clip_coef = r

            return (
                gl_FragCoord,
                gl_FrontFacing,
                gl_PointCoord,
                primitive.keep & in_triangle,
                varying_per_primitive,
                true_clip_coef,
                true_clip_coef,
            )

        # END OF `_per_primitive_preprocess`

        @partial(jit, inline=True)
        @add_tracing_name
        def _interpolate_and_fragment_shading(
            gl_FragCoord: Vec4f,
            gl_FrontFacing: BoolV,
            gl_PointCoord: Vec2f,
            keeps: BoolV,
            values: VaryingT,
            barycentric_screen: Vec3f,
            barycentric_clip: Vec3f,
        ) -> Tuple[PerFragment, VaryingT]:
            # PROCESS: Interpolation
            varying: VaryingT = shader.interpolate(
                values=values,
                barycentric_screen=barycentric_screen,
                barycentric_clip=barycentric_clip,
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

            # enforce default `gl_FragDepth` when `use_default_depth`
            per_frag = cast(
                PerFragment,
                lax.cond(  # pyright: ignore[reportUnknownMemberType]
                    per_frag.use_default_depth,
                    lambda: per_frag._replace(gl_FragDepth=gl_FragCoord[2]),
                    lambda: per_frag,
                ),
            )
            assert isinstance(per_frag, PerFragment)

            per_frag = per_frag._replace(keeps=keeps & per_frag.keeps)

            return per_frag, extra_fragment_output

        # END OF `_interpolate_fragment_shading`

        args = jax.vmap(_per_primitive_preprocess)(
            per_primitive,  # pyright: ignore # need to find a way to express un-batch
            varyings,
        )
        chosen_args = shader.primitive_chooser(*args)

        built_in: PerFragment
        extra_outputs: VaryingT
        _f = jax.vmap(_interpolate_and_fragment_shading)
        built_in, extra_outputs = _f(*chosen_args)
        assert isinstance(built_in, PerFragment)

        gl_Depths = built_in.gl_FragDepth
        keeps = built_in.keeps
        assert isinstance(gl_Depths, Float[Array, "kept_primitives"])
        assert isinstance(keeps, Bool[Array, "kept_primitives"])

        # PROCESS: Per-Sample Operations (Mixing: depth test + colour blending)
        mixed_output: MixerOutput
        attachments: MixedExtraT
        mixed_output, attachments = cast(
            Tuple[MixerOutput, MixedExtraT],
            shader.mix(gl_Depths, keeps, extra_outputs),
        )
        assert isinstance(mixed_output, MixerOutput)
        assert isinstance(attachments, tuple)

        return mixed_output, attachments

    # END OF `_per_pixel`

    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def _per_row(
        i: IntV,
    ) -> Tuple[MixerOutput, MixedExtraT]:
        """Render one row.

        Parameters:
          - i: the index of the row to be rendered on the first axis of the
            resultant buffer.

        Returns: one row from `Shader.mixer`, `MixerOutput` and `MixerExtraT`.
        """
        keeps: Bool[Array, "height"]
        depths: Num[Array, "height"]
        extras: MixedExtraT
        # vmap over axis 1 (height) of the buffers. Axis 0 (width) is `i`.
        (keeps, depths), extras = jax.vmap(_per_pixel)(
            lax.concatenate(  # pyright: ignore[reportUnknownMemberType]
                (
                    lax.full((batch_size, 1), i),  # pyright: ignore
                    lax.broadcasted_iota(int, (batch_size, 1), 0),  # pyright: ignore
                ),
                1,
            )
        )
        assert isinstance(keeps, Bool[Array, "height"])
        assert isinstance(depths, Num[Array, "height"])
        assert isinstance(extras, tuple)

        return MixerOutput(keep=keeps, zbuffer=depths), extras

    # END OF `_per_row`

    @jaxtyped
    @partial(jit, donate_argnums=(1,), inline=True)
    @add_tracing_name
    def merge_buffers(
        mixer_outputs: Tuple[MixerOutput, MixedExtraT],
        old_buffers: Buffers[T],
    ) -> Buffers[T]:
        """Merge the rendered row into the buffers.

        Parameters:
          - mixer_outputs: the output from `Shader.mixer`, `MixerOutput` and
            `MixerExtraT`.
          - old_buffers: the buffers to be updated.

        Returns: the updated buffers.
        """
        keeps: CanvasMask = mixer_outputs[0].keep
        depths: ZBuffer = mixer_outputs[0].zbuffer
        extras: MixedExtraT = mixer_outputs[1]

        S = TypeVar("S")

        @partial(jit, donate_argnums=(2,), inline=True)
        def _merge_first_axis(_mask: Bool[Array, "_"], _new: S, _old: S) -> S:
            @partial(jit, donate_argnums=(2,), inline=True)
            def _merge_second_axis(__mask: BoolV, __new: S, __old: S) -> S:
                return cast(
                    S,
                    lax.cond(  # pyright: ignore[reportUnknownMemberType]
                        __mask, lambda: __new, lambda: __old
                    ),
                )

            return jax.vmap(_merge_second_axis)(_mask, _new, _old)

        new_buffers: Buffers[T] = tree_map(
            lambda new, old: jax.vmap(_merge_first_axis)(keeps, new, old),
            Buffers(zbuffer=depths, targets=tuple(extras)),
            old_buffers,
        )
        assert isinstance(new_buffers, Buffers)

        return new_buffers

    # END OF `merge_buffers`

    # iterate over axis 0 (width) of the buffers
    # (multiple row at a time, according to `row_indices``)
    # Not using vmap due to memory constraints
    # TODO: using map for readability when map supports unroll.
    # Reference: https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/control_flow/loops.html#map
    mixer_outputs = lax.scan(
        lambda _, x: ((), _per_row(x)),
        init=(),
        xs=row_indices,
        unroll=loop_unroll,
    )[1]

    buffers = merge_buffers(mixer_outputs, buffers)
    assert isinstance(buffers, Buffers)

    return buffers


@jaxtyped
@partial(
    jit,
    static_argnames=("shader", "loop_unroll"),
    donate_argnums=(2,),
    inline=True,
)
@add_tracing_name
def render(
    camera: Camera,
    shader: type[Shader[ShaderExtraInputT, VaryingT, MixedExtraT]],
    buffers: Buffers[T],
    face_indices: FaceIndices,
    extra: ShaderExtraInputT,
    loop_unroll: int = 1,
) -> Buffers[T]:
    """Render a scene with a shader.

    Parameters:
      - loop_unroll: the number of rows to be rendered in one loop. This may
        help improve the performance at the cost of increasing compilation time.
        Default: 1
    """
    vertices_count: int
    gl_InstanceID: ID
    with jax.ensure_compile_time_eval():
        assert len(extra) > 0
        assert isinstance(extra[0], Num[Array, "*"])
        vertices: Num[Array, "*"] = cast(Num[Array, "*"], extra[0])
        vertices_count: int = vertices.shape[0]

        gl_InstanceID = jnp.array(0, dtype=int)  # pyright: ignore
        assert isinstance(vertices_count, int)
        assert isinstance(gl_InstanceID, ID)

    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def vertex_processing(gl_VertexID: IntV) -> Tuple[PerVertex, VaryingT]:
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
        lax.iota(int, vertices_count),  # gl_VertexID # pyright: ignore
    )

    # everything after vertex processing, will directly update buffers
    buffers = _postprocessing(
        shader=shader,
        buffers=buffers,
        per_primitive=jax.vmap(PerPrimitive.create)(
            tree_map(
                lambda field: field[face_indices],
                per_vertices,
            )
        ),
        varyings=tree_map(lambda field: field[face_indices], varyings),
        extra=extra,
        viewport=camera.viewport,
        loop_unroll=loop_unroll,
    )
    assert isinstance(buffers, Buffers)

    return buffers
