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

## Continuous Integration and Render Regression

The GitHub Actions workflow tests Python 3.12–3.14 on Linux using the locked dependencies. A separate Python 3.13 job installs NumPy `2.1.3`, checks the built wheel's dependency metadata, and runs the full test suite and CPU render and gradient regressions; its environment is isolated from the uv lockfile. On Python 3.14, Ruff checks import sorting and formatting, then compares the pull request head (or pushed commit) with its exact base revision and rejects newly introduced E4, E7, E9 and F lint violations. The comparison checks the paths supplied with `--paths`; CI passes `assets`, `renderer`, `examples`, `test_resources`, `tests` and `tools`. Existing violations are tolerated while new ones are blocked. `F722` and `F821` are ignored because jaxtyping shape annotations can trigger false positives. A separate `macos-latest` job uses Python 3.14 and CPU-only JAX to render the cube and a 30-frame head animation. The job also checks the numerical gradient of the light direction's x component against a central finite difference. The camera-gradient smoke check still runs, but its current output includes non-finite leaves and is not used as a numerical gate.

Run `uv run ruff check assets renderer examples test_resources tests tools` to inspect the existing lint diagnostics locally.
To compare a different path set locally, pass it after `--paths` to `tools/check_ruff_lint.py`; the rule selection stays fixed, so changing Ruff configuration alone cannot suppress newly introduced violations.

The Linux job runs strict Pyright from the uv lockfile on Python 3.14. Its diagnostics remain visible, but the check is advisory until the existing JAX typing issues are resolved. A dedicated `linux-render-regression` job also runs the CPU render and gradient regressions on Python 3.14 with the locked dependencies.

### Manual GPU checks before merging

The `Manual GPU regression` workflow runs the same test suite and render and
gradient regressions on a Modal T4 GPU. A maintainer dispatches it from `master`
once before merging, specifying a pull request number. The workflow freezes its
head and base revisions and records `accelerator/gpu` on the PR head. New commits
need new checks; results from a changed head or base are rejected.

GPU and TPU confirmation run automatically before PyPI publication. TPU is not
a PR merge requirement, allowing Kaggle's free quota and queue to be used less
frequently.

Provider controller dependencies are pinned in the `ci-modal` and `ci-kaggle`
groups in `uv.lock`. Each controller installs only its own group. Standard CI
installs all locked groups, including both controller SDKs; remote rendering
tests install the development and test groups. GPU tests
use Python 3.14, while TPU tests use Python 3.13 in an isolated environment.

The tests require the requested JAX device and reject CPU fallback. TPU tests use
`JAX_DEFAULT_MATMUL_PRECISION=highest`, as described in the Colab instructions.
Images, numerical metrics, runtime versions and diagnostics are retained as
GitHub Actions artefacts for 14 days for GPU checks and 30 days for TPU release
confirmation. Existing numerical tolerances apply to all backends; the
camera-gradient smoke test remains advisory about numerical values.

#### Account and environment setup

Create these environments under **Settings → Environments** in the GitHub
repository. Allow `master` for manual checks and add a separate deployment tag
rule for `v*` to both provider environments. The `PyPI` environment must also
allow the release tags. Environment rules use the triggering ref, including
when a release calls a reusable workflow. Provider credentials are used by the
trusted controller and are not passed to the remote test process.

| Environment | Secrets | Variables |
| --- | --- | --- |
| `modal-gpu` | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | None |
| `kaggle-tpu` | `KAGGLE_API_TOKEN` | `KAGGLE_USERNAME` |
| `PyPI` | `PYPI_API_TOKEN` | None |

- Register at [Modal](https://modal.com/) and use a Starter workspace. Create an
  API token in the workspace settings. Before running tests, set the workspace
  usage budget no higher than the available free credits and the net spend limit
  to `$0` under **Usage & Billing**. Starter includes `$30` of monthly compute
  credits, but the plan can charge for excess usage unless limits are configured;
  see [pricing](https://modal.com/pricing) and
  [budgets](https://modal.com/docs/guide/budgets). The workflow uses an on-demand
  T4 function, without a persistent deployment or automatic retries.
- Register at [Kaggle](https://www.kaggle.com/), complete the required account
  verification, and confirm access to a TPU notebook. Generate an API token under
  **Account Settings → API tokens**; see the [API documentation](https://www.kaggle.com/docs/api).
  The controller submits a private notebook with
  `TpuV5E8` and retrieves that run's output. Availability and quota depend on the
  account; exhaustion, queue timeout or missing TPU devices fail the check.
  Kaggle's CLI has no documented command to cancel an active kernel session;
  the submission sets a 30-minute execution timeout, while an Actions timeout
  stops waiting and must not be treated as proof that the remote run stopped.

#### Enabling the GPU merge requirement

First merge the workflow and configure the environments. GitHub requires a
`workflow_dispatch` workflow to exist on the default branch before it can run.
Run the GPU workflow against an open PR and inspect its device and numerical
reports before adding `accelerator/gpu` as a required status check in the
`master` ruleset. Require the PR branch to be up to date, and choose
GitHub Actions as the expected status source. Missing or unsuccessful checks must
block merging; do not replace a failed provider check with a skipped job.

```sh
gh workflow run accelerators.yml --ref master -f pr=25
```

#### Publishing a release

Publishing a GitHub Release starts `Build, validate and publish`. The workflow
freezes the tag's full commit SHA, requires that commit to be in `master`
history, and checks that the tag matches the package version. It reuses all
seven ordinary CI jobs: Ruff, Python 3.12–3.14 tests, Linux/macOS render and
gradient regressions, and the isolated NumPy `2.1.3` compatibility check. Ruff
compares the candidate with its frozen first parent; Pyright remains advisory.

The package is built once and its installed wheel is smoke-tested. Only after
CPU CI and the build pass do Modal GPU and Kaggle TPU confirmation run in
parallel. Each validates the same frozen SHA and workflow attempt, genuine
devices and the existing numerical tolerances. Failure, cancellation, timeout,
missing reports or insufficient free quota block PyPI publication; there is no
paid fallback or automatic retry.

The final `PyPI` job requires every gate to pass, rechecks the tag and downloads
the validated distributions from the same workflow attempt without rebuilding.
Its token is only provided to the upload step. Artefacts are scoped by attempt,
so use **Re-run all jobs** after a failure to obtain a complete new confirmation.
Neither release accelerator check writes a PR status.

To rehearse the publishing gates without creating a release, see
[Rehearsing the publishing workflow](docs/ci/release-rehearsal.md).

#### Manually confirming a release candidate

After the release preparation changes have landed, copy the full commit SHA
that the release tag will point to. Dispatch `TPU release confirmation`
from `master` with that SHA. The candidate must belong to this repository and
be on `master` or in its history. The controller freezes the SHA and workflow
attempt, submits one private Kaggle notebook and validates its bound result.

```sh
gh workflow run release-tpu.yml --ref master -f commit=FULL_COMMIT_SHA
```

The matching `GPU release confirmation` workflow also accepts a full candidate
SHA from `master`:

```sh
gh workflow run release-gpu.yml --ref master -f commit=FULL_COMMIT_SHA
```

These manual entries help diagnose account and device problems before a
release. Inspect their device reports, numerical metrics and test logs. They do
not publish packages or replace the automatic checks in the publishing run.
A changed candidate SHA requires a new confirmation. No `accelerator/tpu` PR
check is created.

Kaggle documents TPU selection via its CLI, but an
[open upstream issue](https://github.com/Kaggle/kaggle-cli/issues/1197) reports
submissions that receive the wrong runtime. A real TPU run is required to
confirm a release. Account configuration and mocked controller tests alone do
not establish accelerator compatibility.

The initial provider validation passed on a Modal Tesla T4 with Python 3.14.7,
JAX 0.11.2 and NumPy 2.5.3, including the full test suite and all four render and
gradient regressions. The Kaggle submission installed its dependencies but
failed TPU initialisation with `No jellyfish device found`; its tests did not
start. A successful live Kaggle run proving TPU allocation and passing the
regressions remains outstanding.

The render regression compares the cube and head frames 0 and 15 with the checked-in images in `tests/references/`. It allows small renderer differences while requiring all of these bounds:

- Foreground intersection-over-union (IoU) of at least `0.90`.
- Mean absolute RGB error over the union of foreground pixels of at most `5/255`.
- 95th-percentile per-pixel maximum-channel error of at most `20/255`.
- Each animation frame has a foreground ratio from `0.15` to `0.27`; adjacent frames have mean absolute RGB error from `0.5/255` to `5/255`.

The cube foreground mask includes pixels more than 20 intensity counts away from white; the head mask includes pixels with any channel above 8. This keeps background differences from dominating the image comparison. The job uploads the numerical report, keyframes, amplified difference images, and APNG for seven days, including when a test fails.

To reproduce the render checks locally from the repository root, install the locked dependency groups with uv, then run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg JAXRENDERER_ARTIFACT_DIR=render-artifacts \
  uv run python -m pytest -q tests/render_regression.py tests/test_smoke_grad.py
```

When an intentional rendering change updates the expected output, inspect the uploaded keyframes and numeric report first. Then render the same scenes on macOS with the current settings and replace only the affected PNGs in `tests/references/`. Run the render regression command against the proposed images and review the full-frame and animation checks before committing them. Do not change tolerances solely to make a new golden pass; change them only when a measured cross-platform or renderer variation justifies the new bounds.

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
