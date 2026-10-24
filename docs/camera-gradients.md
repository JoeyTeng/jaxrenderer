# Camera gradient checks

The camera regression in [`test_grad_over_camera`](../tests/test_smoke_grad.py)
checks derivatives with respect to selected physical camera parameters. It
rebuilds the camera matrices from those parameters, renders a small stable
scene, and compares autodiff with a central finite difference.

## Differentiate the inputs that define the camera

`Camera` stores both its input matrices and several matrices derived from them.
The earlier smoke check differentiated the whole `Camera` tree, then only
asserted that the resulting tree had leaves. That proved the tree was
non-empty, but did not check whether its gradients were finite, non-zero, or
numerically correct. A probe of that check found non-finite leaves; it did not
establish their cause.

The current regression follows the physical inputs through camera construction
and rendering. For parameters $\theta$, let $C(\theta)$ be the camera returned
by `Renderer.create_camera_from_parameters`, and let $Z(C)$ be the rendered
depth buffer for camera $C$. The sampled window is $W$. The test evaluates

$$
L(\theta) = \sum_{(x,y)\in W} Z(C(\theta))_{x,y}.
$$

Autodiff therefore applies the chain rule through both the camera construction
and render path:

$$
\frac{dL}{d\theta}
= \frac{\partial L}{\partial C}\frac{dC}{d\theta}.
$$

The test differentiates two inputs separately: the camera's x position and
the perspective far-clip distance. The scene and camera use these values:

| Input | Value |
| --- | --- |
| Viewport | 84 × 84, depth 1 |
| Background depth | 0 |
| Position, target, up | `(0, 0, 2)`, `(0, 0, 0)`, `(0, 1, 0)` |
| Perspective projection | `hfov=90°`, `vfov=90°`, `near=0.1`, `far=10` |
| Triangle vertices | `(-0.6, -0.6, 0)`, `(0.7, -0.5, 0)`, `(0, 0.7, 0)` |
| Loss window | `depth[36:46, 31:41]`, a 10 × 10 sum |

The depth array's first axis is x and its second is y. All 100 pixels in the
window must have positive depth at the base parameters and at both positive
and negative finite-difference perturbations for each input. Since background
depth is 0, this keeps the loss away from background; losing coverage at any
sampled pixel fails the fixture's assertion.

Changing position x while keeping the target fixed also changes the camera's
view direction, so this derivative includes the resulting view rotation; it
is not just a camera translation. The off-centre window gives that sensitivity
a non-zero value.

## Compare autodiff with a central difference

For a scalar parameter $p$, the test compares the autodiff value with

$$
D_h L(p) = \frac{L(p+h)-L(p-h)}{2h}.
$$

The steps are `0.01` for position x and `0.3` for the far distance. The
depth-buffer reductions use float32; smaller steps amplified rounding in the
measured finite differences. On the CPU reference run, these steps produced
relative errors below 0.7%. The regression requires finite autodiff and
finite-difference values, a non-zero sensitivity, and `allclose` with
`rtol=0.03` and `atol=1e-4`.

One CPU run recorded these values:

| Parameter | Autodiff | Central difference | Step | Relative error |
| --- | ---: | ---: | ---: | ---: |
| Position x | 0.0901874304 | 0.0896453857 | 0.01 | 0.6047% |
| Far distance | −0.0969289765 | −0.0969695970 | 0.3 | 0.0419% |

These numbers illustrate a run; they are not golden values or guarantees for
every backend. The assertions validate the relationship between the two
derivatives on the active backend.

The far-distance derivative also has a simple analytical check. With the camera
on the z axis, the triangle has eye-space coordinate $z=-d=-2$, where $d=2$ is
the positive eye distance, throughout the window.
The perspective matrix maps it to NDC depth

$$
z_{\mathrm{ndc}} = \frac{f+n}{f-n} - \frac{2fn}{d(f-n)}.
$$

For viewport depth 1, the viewport maps this to window depth

$$
z_{\mathrm{window}} = \frac{z_{\mathrm{ndc}}+1}{2}
= \frac{f}{f-n}\left(1-\frac{n}{d}\right).
$$

All 100 samples have this depth, so $L(f)=100z_{\mathrm{window}}$. With
$n=0.1$ and $d=2$,

$$
\frac{dL}{df} = -\frac{100n(1-n/d)}{(f-n)^2};
\qquad
\left.\frac{dL}{df}\right|_{f=10} \approx -0.0969288848.
$$

This agrees with the CPU autodiff value in the table. The expression clarifies
that the reported derivative is with respect to the perspective far distance,
not NDC or window depth.

## Run and inspect the check

Install the locked dependencies and run the same test used by the render and
gradient regression jobs:

```sh
uv sync --locked --all-groups
JAX_PLATFORMS=cpu MPLBACKEND=Agg \
  JAXRENDERER_ARTIFACT_DIR=render-artifacts \
  uv run pytest -q tests/test_smoke_grad.py::test_grad_over_camera --import-mode importlib
uv run python -m json.tool render-artifacts/numeric-report.json
```

The report's `camera_gradient` object contains `position_x_*` and `far_*`
fields for autodiff, central difference, absolute and relative error, and the
step size. CI runs the same numerical assertions and retains the generated
report as an artefact.

## Limits

The check covers these two parameters in this one stable scene. It does not
promise finite derivatives for every `Camera` tree leaf or every scene.
Triangle-edge coverage changes and visibility boundaries are discontinuous;
clipping, degenerate projected triangles, and singular or ill-conditioned
camera matrices can still produce undefined or non-finite gradients. The
non-finite leaves observed in the earlier whole-tree check remain
unclassified, so this regression makes no claim about their root cause.
