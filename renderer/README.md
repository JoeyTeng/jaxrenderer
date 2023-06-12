# Key Implementations of the Renderer

This document describes the key implementations of the renderer. For the implementations of the shaders, refer to [shaders/README.md](shaders/README.md).

## Pipeline Design

The design follows the [OpenGL rendering pipeline](https://www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview) largely. Only relevant parts are included here.

To use the default pipeline, call function `render` with an implementation of `Camera`, `Shader`, `Buffers` `FaceIndices` and `VertexShaderExtraInputT`.

Some example implementations of shaders are provided in `renderer/shaders`.

### Vertex Specification

This can be passed as `VertexShaderExtraInputT` to `render`, and they will be passed directly to the `Shader.vertex` method. The input should be stored as several batched buffers (`jnp.array`) with the batch axis as the axis 0, to be used for downstream `vmap` operations. See example implementations for more details.

### Vertex Processing

Implemented by `Shader.vertex`. An implementation **must** be provided. This function takes in the vertex specification and outputs the vertex position in the clip space, as well as other user-defined per-vertex attributes such as colour, normal, etc.

References:

- [Vertex Shader/Defined Inputs](https://www.khronos.org/opengl/wiki/Vertex_Shader/Defined_Inputs)
- [Vertex Shader#Outputs](https://www.khronos.org/opengl/wiki/Vertex_Shader#Outputs)

### Vertex Post-processing

#### Face Culling

This is not implemented in the pipeline, but can be achieved by discard fragment in the fragment shader according to `gl_FrontFacing` variable, and return `keeps=False` in first return value `PerFragment`. See example implementation `renderer/shaders/gouraud.py:GouraudShader.fragment`.

### Rasterization

Implemented in pipeline and cannot be changed.

The fragments are generated for each position in the zbuffer, and if the screen coordinate is in one primitive, the relevant fragment is passed to the fragment shader for computation.

The processing is done by iterating along the first axis of the buffer, and `vmap` along the second axis, and `vmap` along the primitives. Thus, all fragments at the same position are generated together, then mixed, then written to the buffer.

Currently no anti-aliasing strategy is supported.

#### Optional Early Depth Test

This is an additional stage in this pipeline which may be analogical to the early depth test in OpenGL. It is implemented in `Shader.primitive_chooser`. The default implementation is provided, which assumes that the depth is just the interpolated `z` value in the eye space. It just picks the values of the single primitive that is closest to the camera and is not discarded in the previous pipeline.

Custom implementations can overload to change this behaviour to achieve special effects like occlusion, transparency, etc. Note that the number of returned primitives must be static, i.e., same for all fragments, as required by `jax.jit`.

#### Interpolation of Attributes

Interpolation of all attributes for each fragment is defined by `Shader.interpolate`. The default implementation is provided, which simply linearly interpolates the attributes in the clip space according to the barycentric coordinates of the fragment. This behaviour is wrapped as a convenient function `interpolate` and mode `Interpolation.SMOOTH`. The interpolated values are then passed to the fragment shader. Currently only `Interpolation.SMOOTH` and `Interpolation.FLAT` are supported.

Reference:

- For interpolation modes, see [Interpolation Qualifiers](https://www.khronos.org/opengl/wiki/Type_Qualifier_(GLSL)#Interpolation_qualifiers)

### Fragment Processing

This is done by `Shader.fragment`. A default implementation is provided, which simply returns the `z` component of the fragment position in the screen space as the depth, and directly returns all extra attributes.

Reference:

- [Fragment Shader/Defined Inputs](https://www.khronos.org/opengl/wiki/Fragment_Shader/Defined_Inputs)
- [Fragment Shader#Outputs](https://www.khronos.org/opengl/wiki/Fragment_Shader#Outputs)

### Per-Sample Operations

#### Depth Test

This is not explicitly done in the pipeline, but is implicitly done by the default implementation of `Shader.mix`, which only takes the fragment with the largest depth value.

#### Colour Blending

This can be achieved by changing `Shader.mix` function as well. The value from previous buffer is not available though.

## Implementation Details

### Benchmarks

- Effect of back-face culling (very significant), see [Colab](https://colab.research.google.com/drive/1KOjeemLDPxMf-8H0ZpLgpd1DSXuFyNSK?usp=sharing)
- Effect of batched vs non-batched rendering on triangles. Non-batched pure `fori_loop`-based implementation is consistently faster, see [Colab](https://colab.research.google.com/drive/13p3US19TrVOTtLFkGKgg08KYzMb4747w?usp=sharing). This may be due to the extra GPU memory allocation required in `vmap`-ed implementation
- Effect of memory donation to suggest memory reuse, see [Colab](https://colab.research.google.com/drive/1VT7nvHV7au2oncMUjbZVNkPjZm0cN1wM?usp=sharing). The improvement is marginal.

### Why use `NamedTuple` but not `dataclasses`, etc

See experiment code [here in Colab](https://colab.research.google.com/drive/19b4VpAevvTVj_Ry9tEj88Q91ECXsku6z?usp=sharing). Basically tuples are well supported (and thus `NamedTuple`) by JAX as a [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) out of box, but `dataclass` is not.

Notice that [it is impossible to inherit from subclass of `NamedTuple`](https://github.com/python/typing/issues/427).
