# JAX Renderer

## Key Difference from [erwincoumans/tinyrenderer](https://github.com/erwincoumans/tinyrenderer)

- Native JAX implementation, supports `jit`, `vmap`, etc.
- Lighting is computed in main camera's eye space; while in PyTinyrenderer it is computed in world space.
- Texture specification is different: in PyTinyrenderer, the texture is specified in a flattened array, while in JAX Renderer, the texture is specified in a shape of (width, height, colour channels). A simple way to transform old specification to new specification is to use the convenient method `build_texture_from_PyTinyrenderer`.
- Rendering pipeline is different. PyTinyrenderer renders one object at a time, and share zbuffer and framebuffer across one pass. This renderer first merges all objects into one big mesh in world space, then process all vertices together, then interpolates and rasterise and render. For fragment shading, this is done by sweeping each row in a for loop, and batch compute all pixels together. For computing a pixel, all fragments for that pixels are batch compute together, then mixed. This is more memory efficient and allows `vmap` batching as far as possible.
- Fix bugs
  - Specular lighting was wrong, where it forgets to reverse the light direction vector.

## Known Issues

- The texture behaviour for cubes is not the same as the PyTinyrenderer. This is a bug and will be fixed soon.

## Roadmap

- [ ] Correctly implement a proper clipping algorithm
- [ ] Profile and accelerate implementation
- [ ] Differentiable rendering
- [ ] Build a ray tracer as well
