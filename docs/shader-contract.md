# Shader inputs, fragment outputs and mixing

`Shader[ExtraT, VaryingT, TargetsT]` separates material inputs, interpolated
vertex data and render targets. The stages remain batched within one rendering
call; there is no separate transparent rendering pass.

The stage input and output types are defined in `renderer.shader_types`.
`renderer.shader` continues to re-export those names for existing imports.

Each stage method receives one frozen, keyword-only pytree dataclass. The record
keeps related stage values together while preserving their GLSL names.

| Method | Input record and fields | Output |
| --- | --- | --- |
| `vertex` | `VertexInput[ExtraT]`: `gl_VertexID`, `gl_InstanceID`, `camera`, `extra` | `PerVertex` and `VaryingT` |
| `primitive_chooser` | `PrimitiveInput[VaryingT]`: `gl_FragCoord`, `gl_FrontFacing`, `gl_PointCoord`, `keeps`, `values`, `barycentric_screen`, `barycentric_clip`, `double_sided`, `collection` | The selected candidate fields in their existing tuple form |
| `interpolate` | `InterpolationInput[VaryingT]`: `values`, `barycentric_screen`, `barycentric_clip` | `VaryingT` |
| `fragment` | `FragmentInput[ExtraT, VaryingT]`: `gl_FragCoord`, `gl_FrontFacing`, `gl_PointCoord`, `varying`, `extra` | `PerFragment` and `Fragment[TargetsT]` |
| `mix` | `MixInput[TargetsT]`: `gl_FragDepth`, `keeps`, `extra: Fragment[TargetsT]`, `destination: PixelState[TargetsT]` | `MixerOutput` and `TargetsT` |

The scalar or per-candidate `double_sided` value defaults to false. The
`collection` mode defaults to `FragmentCollection.NEAREST`; it is static JAX
metadata because it determines whether the chooser returns one candidate slot
or the full statically sized batch. Changing the mode selects a separate JIT
trace. Array and other dynamic values remain pytree leaves and can be traced.
Use `dataclasses.replace` to make a modified copy of an input record; it
preserves the concrete record type. Pyright does not fully type-check values
passed as `replace` field updates.
Register each concrete subclass of an input record with
`jax.tree_util.register_dataclass` before passing it to JAX.

The pipeline maps stage work over vertices, primitives and pixels while
keeping shared inputs such as `camera` and `extra` outside those mapped axes.
When applying `jax.vmap` to a stage yourself, use a matching `in_axes` pytree
and set shared fields to `None`; map only the per-vertex, per-candidate or
per-pixel fields. Fragment processing remains batched: `NEAREST` reduces the
candidate set to one slot, while `ALL` preserves every candidate slot and
passes its eligibility through `keeps`. The opaque mixer still selects the
nearest kept fragment in that batch.

For example, a subclass can change chooser input while delegating the actual
selection to its inherited implementation:

```python
from dataclasses import replace
from typing import override

from renderer.shader_types import PrimitiveInput
from renderer.types import TRUE_ARRAY


class DoubleSidedShader(BaseShader):
    @staticmethod
    @override
    def primitive_chooser(inputs: PrimitiveInput[VaryingT]):
        return BaseShader.primitive_chooser(
            replace(inputs, double_sided=TRUE_ARRAY)
        )
```

Here `BaseShader` is an existing shader subclass whose chooser is inherited.
`replace` retains its static `collection` value, so chooser mode stays fixed
for JAX compilation. This only changes chooser eligibility; fragment-stage
discard rules still apply.

`Fragment.targets` contains the values intended for the render targets.
`Fragment.alpha` carries opacity separately and defaults to a floating-point one.
`PerFragment` retains the keep flag and fragment depth state. The pipeline
resolves default fragment depth and combines the geometry and fragment keep
flags before calling `mix`.

`TargetsT` describes one target layout throughout fragment output, mixing,
destination pixels and `Buffers.targets`. A per-pixel target has no image axes;
the corresponding framebuffer arrays have leading width and height axes.
Python typing describes the container and leaf relationships, while array
dimensions remain a separate runtime contract.

## Reusing defaults

Implement `fragment` when converting interpolated data to a different target
layout. Built-in material shaders already perform shading there and construct
their canvas or depth-only target output at the same stage.

For an unchanged target layout, inherit `PassthroughShader[InputT, VaryingT]`.
It specialises the target type to `VaryingT` and supplies a default fragment that
wraps the interpolated data in `Fragment`. Both forms inherit the same default
opaque mixer; a shader does not need a mix override merely to package a colour.

## Collecting candidates

`Shader.fragment_collection` is inherited with the shader's mixing behaviour:

- `FragmentCollection.NEAREST` is the default. The chooser filters by raster
  coverage and face eligibility, then retains the closest geometry candidate.
- `FragmentCollection.ALL` retains all candidate slots and their eligibility
  mask, allowing fragment processing and mixing to consider multiple layers.
  The candidate axis stays statically sized for JAX; inactive slots are masked.

The collection mode is carried by `PrimitiveInput.collection`. Custom chooser
overrides should preserve it when forwarding modified records, along with any
material-specific visibility rules such as double-sided geometry.

Selection happens before fragment processing. In nearest mode, a later discard
or modified fragment depth cannot bring back a candidate already removed by
the chooser. Use all-candidate collection when mixing needs those candidates.
All-candidate collection does not sort them: custom mixing uses the resolved
fragment depths and final keep flags.

## Mixing and writing targets

The mixing signature is:

```python
def mix(inputs: MixInput[TargetsT]):
    ...
```

`inputs.extra` is a batched `Fragment[TargetsT]`. `inputs.destination` is a
`PixelState[TargetsT]` containing the current pixel's `targets` and `zbuffer`.
The mixer returns target values and a `MixerOutput` containing:

- `keep`: whether to write the target values;
- `zbuffer`: the resolved depth value;
- `write_depth`: whether to write that depth, independently of target writes.

The default opaque mixer selects the nearest retained fragment whose keep flag
is true and enables both writes for it. It ignores alpha. An empty batch or a
batch with no kept fragments leaves the destination unchanged. As before,
the opaque default does not test a new fragment against a previous rendering
call's stored depth; a custom mixer can use the destination depth explicitly.

The buffer commit stage applies the two write masks. It does not perform another
colour composition or infer blending semantics for arbitrary target fields.

## Custom mixing and future transparency

Keep custom behaviour in `Shader.mix`, and reuse it through shader inheritance
or a shared mixin. A multi-layer mixer must also inherit or declare
`FragmentCollection.ALL`. This keeps collection and mixing coherent without
requiring an independent render-selected mixer.

The contract can carry opacity and destination values, but this change does not
implement alpha blending. A future transparent mixer can sort valid fragments
by resolved depth, combine their contributions within the existing pixel batch,
and include the remaining background contribution once before the final write.
It must define its colour representation and depth-write policy explicitly.

Zero alpha is distinct from discard. It contributes no source colour in ordinary
alpha composition, but masking the entire calculation with `alpha > 0` can
remove the alpha derivative at zero. See [numerical behaviour](numerical-behaviour.md)
for the existing limits around visibility changes and discontinuities.

## Migrating a custom shader

1. Update all five stage overrides and call sites to accept their corresponding
   input record and read stage fields from `inputs`.
2. Return `Fragment(targets=...)` with the declared target layout from `fragment`.
3. Remove a mix override that only selects a fragment and converts its layout.
4. For a genuine custom mixer, accept `MixInput[TargetsT]`, consume
   `inputs.extra.targets` and `inputs.destination`, and return explicit target
   and depth write decisions.
5. Give `Buffers.targets` the same layout, such as the shader's public output
   NamedTuple, rather than an unrelated plain tuple.
6. Preserve `PrimitiveInput.collection` when forwarding records from custom
   chooser overrides. Choose `ALL` when the mixing algorithm needs more than the
   nearest geometry candidate.

The high-level `Renderer` API continues to accept and return its single-canvas
tuple; it adapts that layout at the material-specific pipeline boundary.
