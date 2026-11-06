# Renderer inputs and buffer ownership

The high-level `Renderer` provides the built-in textured Phong renderer, with
optional shadows. Its input records are defined in `renderer.renderer_types`
and exported from `renderer` and `renderer.renderer`.

These signatures belong to the development API after `0.5.0`. To use the
examples on `master`, install the matching source:

```bash
pip install --upgrade "jaxrenderer @ git+https://github.com/JoeyTeng/jaxrenderer.git@master"
```

For the published `0.5.0` API, use its
[tagged documentation](https://github.com/JoeyTeng/jaxrenderer/blob/v0.5.0/README.md).

| Entry point | Inputs | Result |
| --- | --- | --- |
| `Renderer.render(inputs, buffers)` | `RenderInput`: merged model, light, prepared camera, optional shadow parameters and loop unroll | Updated buffers |
| `Renderer.get_camera_image(inputs)` | `CameraImageInput`: objects, light, camera or camera parameters, framebuffer dimensions, background values, optional shadow parameters and loop unroll | Colour canvas |

Both records are frozen, keyword-only JAX pytrees, with documentation on each
field. Use `dataclasses.replace` to change selected fields while retaining the
concrete input type. This makes a shallow copy: unchanged arrays are shared.
`CameraParameters`, `LightParameters` and `ShadowParameters` remain NamedTuples;
use their existing `_replace` method to change their fields.

For example, an override can adjust the light and delegate rendering:

```python
from dataclasses import replace
from typing import override

from renderer import Buffers, Renderer, RenderInput
from renderer.types import Canvas


class DimRenderer(Renderer):
    @classmethod
    @override
    def render(
        cls, inputs: RenderInput, buffers: Buffers[tuple[Canvas]]
    ) -> Buffers[tuple[Canvas]]:
        light = inputs.light._replace(ambient=inputs.light.ambient * 0.5)
        return super().render(replace(inputs, light=light), buffers)
```

`get_camera_image` continues to call the class's camera and buffer factories
and its `render` method, so these overrides also apply to that entry point.
An empty object collection returns the configured background canvas before
camera creation, model merging or shading.

## JAX compilation

`CameraImageInput.width`, `CameraImageInput.height` and both records'
`loop_unroll` fields are static pytree metadata. Changing one of these values
selects a separate JIT trace. Other fields remain dynamic pytrees, including
the camera and lighting values and the background colour and depth values.
Array shapes, dtypes and object collection structure also affect compilation.
Register concrete dataclass subclasses with `jax.tree_util.register_dataclass`
before passing them to JAX.

The framebuffer dimensions specify the output array size. The separate
`CameraParameters.viewWidth` and `viewHeight` specify the camera's viewport
mapping; they may differ from the framebuffer dimensions. Array axes remain
width, height, channels, as before.

## Buffer donation

`RenderInput` deliberately excludes buffers. The separate `buffers` argument
lets `Renderer.render` declare `donate_argnames=("buffers",)` without donating
the model, camera or light arrays. The caller must treat the supplied buffers
as consumed and use the returned buffers for subsequent work:

```python
buffers = Renderer.render(inputs, buffers=buffers)
buffers = Renderer.render(replace(inputs, light=new_light), buffers=buffers)
```

Do not keep using an alias to the old buffers after the call. Constructing a new
input record with `replace` does not copy buffers or make donated arrays valid
again. Copy any arrays that must be retained before donating them.

If an application adds an outer JIT boundary, declare donation there as well:

```python
import jax

render_frame = jax.jit(Renderer.render, donate_argnames=("buffers",))
buffers = render_frame(inputs, buffers=buffers)
```

An inner donation declaration does not establish ownership at an outer JIT
boundary. Donation permits memory reuse; the compiler decides which compatible
buffers it can reuse. See [JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html).

`get_camera_image` allocates its own buffers, so callers receive a canvas
without supplying or transferring ownership of an existing framebuffer.

## Migration

Wrap the former `get_camera_image` arguments in `CameraImageInput`. For
`render`, wrap every argument except `buffers` in `RenderInput`, then pass
`buffers` separately. Custom overrides should accept the same new signatures.
Their output types, the existing parameter NamedTuples and `create_buffers`
remain unchanged. The low-level `renderer.render` pipeline and the
[Shader stage contract](shader-contract.md) retain their existing interfaces.
