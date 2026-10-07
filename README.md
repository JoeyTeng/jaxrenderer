# JAX Renderer: Differentiable Rendering in Batch on Accelerators

[![PyPI Version](https://img.shields.io/pypi/v/jaxrenderer?logo=pypi)](https://pypi.org/project/jaxrenderer)
[![Python Versions](https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-blue?logo=python)](#installation)
[![License](https://img.shields.io/github/license/JoeyTeng/jaxrenderer)](https://github.com/JoeyTeng/jaxrenderer/blob/master/LICENSE)
[![Build & Publish](https://github.com/JoeyTeng/jaxrenderer/actions/workflows/pypi.yml/badge.svg)](https://github.com/JoeyTeng/jaxrenderer/actions/workflows/pypi.yml)
[![Lint & Test](https://github.com/JoeyTeng/jaxrenderer/actions/workflows/checks.yml/badge.svg)](https://github.com/JoeyTeng/jaxrenderer/actions/workflows/checks.yml)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-Ruff-D7FF64.svg)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/packaging-uv-blue)](https://docs.astral.sh/uv/)
[![Open in Colab](https://img.shields.io/badge/%7F-Open_demo_in_Colab-blue.svg?logo=googlecolab)](https://colab.research.google.com/github/JoeyTeng/jaxrenderer/blob/master/notebooks/Demo.ipynb)

JaxRenderer is a differentiable renderer implemented in [JAX](https://github.com/google/jax), which supports differentiable rendering and batch rendering on accelerators (e.g. GPU, TPU) using simple function transformations provided by JAX. It is designed to replace by [erwincoumans/tinyrenderer](https://github.com/erwincoumans/tinyrenderer) in [BRAX](https://github.com/google/brax) to support visualising simulation results through fast rendering on accelerators with no external dependencies (other than JAX).

You may find the [slides](https://github.com/JoeyTeng/jaxrenderer/blob/master/docs/final%20presentation%20slides.pdf) of my final year project presentation useful, where I gave a brief introduction to the renderer and the implementation details, including the design of the pipeline and comparing it with the OpenGL's.

## Installation

This project is distributed in [PyPI](https://pypi.org/project/jaxrenderer), and can be installed simply using `pip`:

```bash
pip install jaxrenderer
```

Python versions `3.12` to `3.14` are supported, with minimum versions of NumPy `2.1.3` and JAX `0.11.2`. You may need to install `jaxlib` separately if you are using GPU or TPU; by default, the CPU version of jaxlib is installed. Please refer to [JAX's installation guide](https://github.com/google/jax#installation) for more details.

### Google Colab TPU

For a TPU runtime, install JAX with its TPU support using [JAX's TPU installation instructions](https://docs.jax.dev/en/latest/installation.html#pip-installation-google-cloud-tpu). The renderer `0.4.0` TPU check used NumPy `2.1.3` and JAX `0.11.2` on a free Colab TPU v5e-1:

```python
%pip install "numpy==2.1.3" "jax[tpu]" jaxrenderer
```

Restart the runtime after installation. Before importing JAX or compiling renderer work, set the TPU matmul precision:

```python
%env JAX_DEFAULT_MATMUL_PRECISION=highest
```

Alternatively, in Python, set `os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"` before importing JAX, or call `jax.config.update("jax_default_matmul_precision", "highest")` before camera setup and render compilation. TPU's default float32 matrix multiplication can use reduced precision; `highest` brings the render regression in line with the CPU references. It can trade speed for accuracy. If your notebook uses Numba, keep a version compatible with NumPy `2.1.3` when installing these packages.

### Development

Development dependencies are managed with [uv](https://docs.astral.sh/uv/). The default development environment uses Python `3.14`. From the repository root, install the locked dependencies and run commands with:

```bash
uv sync --locked --all-groups
uv run pytest tests/ --import-mode importlib
```

For CI workflows and render regression guidance, see [CI documentation](docs/ci/).

## Usage

> Please note that the package is imported with name `renderer` rather than the PyPI package name `jaxrenderer`. This may change in the future though.

Some example scripts are provided in [examples](examples) folder. You may find the [demo notebook](notebooks/Demo.ipynb) useful as well. In the demo, there is batch rendering and differentiable rendering examples.

The following is a simple example of rendering a cube with a texture map:

```python
import jax.numpy as jnp
import renderer


ImageWidth: int = 640
ImageHeight: int = 480

# Create a cube with texture map of pure blue
cube = renderer.create_cube(
    half_extents=jnp.ones(3, dtype=jnp.single),
    texture_scaling=jnp.ones(2, dtype=jnp.single),
    # pure blue texture map
    diffuse_map=jnp.zeros((2, 2, 3), dtype=jnp.single).at[..., 2].set(1),
    specular_map=jnp.ones((2, 2), dtype=jnp.single) * 2.0,
)

# Render the cube
image = renderer.Renderer.get_camera_image(
    objects=[renderer.ModelObject(model=cube)],
    # Simply use defaults
    camera=renderer.CameraParameters(
        viewWidth=ImageWidth,
        viewHeight=ImageHeight,
        position=jnp.array([2.0, 4.0, 1.0], dtype=jnp.single),
    ),
    # Simply use defaults
    light=renderer.LightParameters(),
    width=ImageWidth,
    height=ImageHeight,
)
```

You may refer to [demo](https://colab.research.google.com/github/JoeyTeng/jaxrenderer/blob/master/notebooks/Demo.ipynb) for more complex examples, including differentiable rendering and batch rendering.

### Supported Shaders

#### Built-in Shaders

See [`renderer/shaders`](renderer/shaders) for more details.

| Shader Name | Description | Light Direction |
| ----------- | ----------- | --------------- |
| depth | Depth Shader, outputs only screen-space depth value | N.A. |
| gouraud | Gouraud Shading, interpolates vertex colour and outputs it as fragment colour | In model space |
| gouraud_texture | Gouraud Shading with Texture, interpolates vertex colour and samples texture map in fragment shader | In model space |
| phong | Phong Shading, interpolates vertex normal and computes light direction in fragment shader | In eye space, like "head light" |
| phong_darboux | Phong Shading with Normal Map in Tangent Space, interpolates vertex normal and computes light direction in fragment shader, and samples normal map in tangent space | In eye space, like "head light" |
| phong_reflection | Phong Shading with Phong Reflection Approximation, interpolates vertex normal and computes light direction in fragment shader, and samples texture map and specular map in fragment shader | In eye space |
| phong_reflection_shadow | Phong Shading with Phong Reflection Approximation and Shadow, interpolates vertex normal and computes light direction in fragment shader, samples texture map and specular map in fragment shader, and tests shadow in fragment shader | In eye space |

#### Custom Shaders

You may implement your own shaders by inheriting from `Shader` and implement the following methods:

- `vertex`: this is like vertex shader in OpenGL; it must be overridden.
- `primitive_chooser`: at this stage the visibility at each pixel level is tested, it works like pre-z test in OpenGL, makes the pipeline works like a deferred shading pipeline. Noted that you may override and return more than one primitive for each pixel to support transparency. The default implementation simply chooses the primitive with minimum z value (depth).
- `interpolate`: this controls how attributes associated with a fragment is interpolated from the vertices; the default implementation interpolates everything using perspective interpolation.
- `fragment`: this is like fragment shader in OpenGL; a default implementation is provided if you do not need to process any data in the fragment shader.
- `mix`: this is like blending stage in OpenGL; the default implementation simple uses the data from the fragment with minimum screen-space z value (depth).

## Gallery

![Batch Rendering Example, 30 Ants inference on A100 GPU with 90 iterations, rendered onto 84x84 canvas in 5.26s](docs/assets/84x84%2030ants%2090f%2030fps.gif)
> Batch Rendering Example, 30 Ants inference on A100 GPU with 90 iterations, rendered onto 84x84 canvas in 5.26s.

![Phong Reflection Model + Hard Shadow, 30 frames 1920x1080, 2492 triangles in 9.25s](docs/assets/head.gif)
> Phong Reflection Model + Hard Shadow, 30 frames 1920x1080, 2492 triangles in 9.25s.

![Differentiable Rendering Toy Example, deduce light colour parameters](docs/assets/differentiable%20rendering.gif)
> Differentiable Rendering Toy Example, deduce light colour parameters.

## Key Difference from [erwincoumans/tinyrenderer](https://github.com/erwincoumans/tinyrenderer)

- Native JAX implementation, supports `jit`, `vmap`, `grad`, etc.
- Lighting is computed in main camera's eye space; while in PyTinyrenderer it is computed in world space.
- Texture specification is different: in PyTinyrenderer, the texture is specified in a flattened array, while in JAX Renderer, the texture is specified in a shape of (width, height, colour channels). A simple way to transform old specification to new specification is to use the convenient method `build_texture_from_PyTinyrenderer`.
- Rendering pipeline is different. PyTinyrenderer renders one object at a time, and share zbuffer and framebuffer across one pass. This renderer first merges all objects into one big mesh in world space, then process all vertices together, then interpolates and rasterise and render. For fragment shading, this is done by sweeping each row in a for loop, and batch compute all pixels together. For computing a pixel, all fragments for that pixels are batch compute together, then mixed. This is more memory efficient and allows `vmap` batching as far as possible.
- Shadowing within the same object / mesh is allowed. This is not possible in PyTinyrenderer, as it deliberately checks if the shadow comes from the same object; if so, it will not consider to draw a shadow there.
- Quaternion (for specifying rotation/orientation) is in the form of `(w, x, y, z)` instead of `(x, y, z, w)` in PyTinyrenderer. This is for consistency with `BRAX`.
- No clipping is performed. To ensure correct rendering of objects with vertices at or behind camera plane, homogeneous interpolation (Olano and Greer, 1997)[^1] is used to avoid the need of homogeneous division.
- Fix bugs
  - Specular lighting was wrong, where it forgets to reverse the light direction vector.

[^1]: Marc Olano and Trey Greer. 1997. Triangle Scan Conversion Using 2D Homogeneous Coordinates. In _Proceedings of the ACM SIGGRAPH/EUROGRAPHICS Workshop on Graphics Hardware (HWWS ’97)_. ACM, New York, NY, USA, 89–95.

## Roadmap

- [ ] Support double-sided objects
- [ ] Profile and accelerate implementation
- [ ] Build a ray tracer as well
- [ ] Differentiable rendering with respect to mesh
- [x] Differentiable rendering with respect to light parameters
- [x] Differentiable rendering with respect to camera parameters _(not tested)_
- [ ] <s>Correctly implement a proper clipping algorithm</s>
