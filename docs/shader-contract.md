# Fragment outputs and mixing

`Shader[InputT, VaryingT, TargetsT]` separates material inputs, interpolated
vertex data and render targets. The stages remain batched within one rendering
call; there is no separate transparent rendering pass.

| Stage | Output |
| --- | --- |
| `vertex` and `interpolate` | `VaryingT`, such as UVs and normals |
| `fragment` | `PerFragment` and `Fragment[TargetsT]` |
| `mix` | `MixerOutput` and `TargetsT` |

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

The collection mode is passed to `primitive_chooser` through its keyword-only
`collection` argument. Custom chooser overrides must forward it while preserving
their material-specific visibility rules, including double-sided geometry.

Selection happens before fragment processing. In nearest mode, a later discard
or modified fragment depth cannot bring back a candidate already removed by
the chooser. Use all-candidate collection when mixing needs those candidates.
All-candidate collection does not sort them: custom mixing uses the resolved
fragment depths and final keep flags.

## Mixing and writing targets

The mixing signature is:

```python
def mix(gl_FragDepth, keeps, extra, *, destination):
    ...
```

`extra` is a batched `Fragment[TargetsT]`. `destination` is a
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

1. Return `Fragment(targets=...)` with the declared target layout from `fragment`.
2. Remove a mix override that only selects a fragment and converts its layout.
3. For a genuine custom mixer, accept `destination`, consume `extra.targets`, and
   return explicit target and depth write decisions.
4. Give `Buffers.targets` the same layout, such as the shader's public output
   NamedTuple, rather than an unrelated plain tuple.
5. Forward `collection` from custom chooser overrides. Choose `ALL` when the
   mixing algorithm needs more than the nearest geometry candidate.

The high-level `Renderer` API continues to accept and return its single-canvas
tuple; it adapts that layout at the material-specific pipeline boundary.
