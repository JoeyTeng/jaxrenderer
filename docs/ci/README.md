# Continuous Integration and Render Regression

The GitHub Actions workflow tests Python 3.12–3.14 on Linux using the locked dependencies. A separate Python 3.13 job installs NumPy `2.1.3`, checks the built wheel's dependency metadata, and runs the full test suite and CPU render and gradient regressions; its environment is isolated from the uv lockfile. On Python 3.14, Ruff checks import sorting and formatting, then compares the pull request head (or pushed commit) with its exact base revision and rejects newly introduced E4, E7, E9 and F lint violations. The comparison checks the paths supplied with `--paths`; CI passes `assets`, `renderer`, `examples`, `test_resources`, `tests` and `tools`. Existing violations are tolerated while new ones are blocked. `F722` and `F821` are ignored because jaxtyping shape annotations can trigger false positives. A separate `macos-latest` job uses Python 3.14 and CPU-only JAX to render the cube and a 30-frame head animation. The job also checks the numerical gradient of the light direction's x component against a central finite difference. The camera-gradient smoke check still runs, but its current output includes non-finite leaves and is not used as a numerical gate.

Run `uv run ruff check assets renderer examples test_resources tests tools` to inspect the existing lint diagnostics locally. To compare a different path set locally, pass it after `--paths` to `tools/check_ruff_lint.py`; the rule selection stays fixed, so changing Ruff configuration alone cannot suppress newly introduced violations.

The Linux job runs strict Pyright from the uv lockfile on Python 3.14. Its diagnostics remain visible, but the check is advisory until the existing JAX typing issues are resolved. A dedicated `linux-render-regression` job also runs the CPU render and gradient regressions on Python 3.14 with the locked dependencies.

## Provider checks

The `Manual GPU regression` workflow runs the same test suite and render and gradient regressions on a Modal T4 GPU. A maintainer dispatches it from `master` once before merging, specifying a pull request number. The workflow freezes its head and base revisions and records `accelerator/gpu` on the PR head. New commits need new checks; results from a changed head or base are rejected.

GPU and TPU confirmation run automatically before PyPI publication. TPU is not a PR merge requirement, allowing Kaggle's free quota and queue to be used less frequently.

Provider controller dependencies are pinned in the `ci-modal` and `ci-kaggle` groups in `uv.lock`. Each controller installs only its own group. Standard CI installs all locked groups, including both controller SDKs; remote rendering tests install the development and test groups. GPU tests use Python 3.14, while TPU tests use Python 3.13 in an isolated environment.

The tests require the requested JAX device and reject CPU fallback. TPU tests use `JAX_DEFAULT_MATMUL_PRECISION=highest`, as described in the [Colab TPU instructions](../../README.md#google-colab-tpu). Images, numerical metrics, runtime versions and diagnostics are retained as GitHub Actions artefacts for 14 days for GPU checks and 30 days for TPU release confirmation. Existing numerical tolerances apply to all backends; the camera-gradient smoke test remains advisory about numerical values.

### Account and environment setup

Create these environments under **Settings → Environments** in the GitHub repository. Allow `master` for manual checks and add a separate deployment tag rule for `v*` to both provider environments. The `PyPI` environment must also allow the release tags. Environment rules use the triggering ref, including when a release calls a reusable workflow. Provider credentials are used by the trusted controller and are not passed to the remote test process.

| Environment | Secrets | Variables |
| --- | --- | --- |
| `modal-gpu` | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | None |
| `kaggle-tpu` | `KAGGLE_API_TOKEN` | `KAGGLE_USERNAME` |
| `PyPI` | `PYPI_API_TOKEN` | None |

- Register at [Modal](https://modal.com/) and use a Starter workspace. Create an API token in the workspace settings. Before running tests, set the workspace usage budget no higher than the available free credits and the net spend limit to `$0` under **Usage & Billing**. Starter includes `$30` of monthly compute credits, but the plan can charge for excess usage unless limits are configured; see [pricing](https://modal.com/pricing) and [budgets](https://modal.com/docs/guide/budgets). The workflow uses an on-demand T4 function, without a persistent deployment or automatic retries.
- Register at [Kaggle](https://www.kaggle.com/), complete the required account verification, and confirm access to a TPU notebook. Generate an API token under **Account Settings → API tokens**; see the [API documentation](https://www.kaggle.com/docs/api). The controller submits a private notebook with `TpuV5E8` and retrieves that run's output. Availability and quota depend on the account; exhaustion, queue timeout or missing TPU devices fail the check. Kaggle's CLI has no documented command to cancel an active kernel session; the submission sets a 30-minute execution timeout, while an Actions timeout stops waiting and must not be treated as proof that the remote run stopped.

### Enabling the GPU merge requirement

First merge the workflow and configure the environments. GitHub requires a `workflow_dispatch` workflow to exist on the default branch before it can run. Run the GPU workflow against an open PR and inspect its device and numerical reports before adding `accelerator/gpu` as a required status check in the `master` ruleset. Require the PR branch to be up to date, and choose GitHub Actions as the expected status source. Missing or unsuccessful checks must block merging; do not replace a failed provider check with a skipped job.

```sh
gh workflow run accelerators.yml --ref master -f pr=PR_NUMBER
```

### Publishing a release

Publishing a GitHub Release starts `Build, validate and publish`. The workflow freezes the tag's full commit SHA, requires that commit to be in `master` history, and checks that the tag matches the package version. It reuses all seven ordinary CI jobs: Ruff, Python 3.12–3.14 tests, Linux/macOS render and gradient regressions, and the isolated NumPy `2.1.3` compatibility check. Ruff compares the candidate with its frozen first parent; Pyright remains advisory.

The package is built once and its installed wheel is smoke-tested. Only after CPU CI and the build pass do Modal GPU and Kaggle TPU confirmation run in parallel. Each validates the same frozen SHA and workflow attempt, genuine devices and the existing numerical tolerances. Failure, cancellation, timeout, missing reports or insufficient free quota block PyPI publication; there is no paid fallback or automatic retry.

The final `PyPI` job requires every gate to pass, rechecks the tag and downloads the validated distributions from the same workflow attempt without rebuilding. Its token is only provided to the upload step. Artefacts are scoped by attempt, so use **Re-run all jobs** after a failure to obtain a complete new confirmation. Neither release accelerator check writes a PR status.

To rehearse the publishing gates without creating a release, see [Rehearsing the publishing workflow](release-rehearsal.md).

### Manually confirming a release candidate

After the release preparation changes have landed, copy the full commit SHA that the release tag will point to. Dispatch `TPU release confirmation` from `master` with that SHA. The candidate must belong to this repository and be on `master` or in its history. The controller freezes the SHA and workflow attempt, submits one private Kaggle notebook and validates its bound result.

```sh
gh workflow run release-tpu.yml --ref master -f commit=FULL_COMMIT_SHA
```

The matching `GPU release confirmation` workflow also accepts a full candidate SHA from `master`:

```sh
gh workflow run release-gpu.yml --ref master -f commit=FULL_COMMIT_SHA
```

These manual entries help diagnose account and device problems before a release. Inspect their device reports, numerical metrics and test logs. They do not publish packages or replace the automatic checks in the publishing run. A changed candidate SHA requires a new confirmation. No `accelerator/tpu` PR check is created.

Kaggle documents TPU selection via its CLI, but an [open upstream issue](https://github.com/Kaggle/kaggle-cli/issues/1197) reports submissions that receive the wrong runtime. A real TPU run is required to confirm a release. Account configuration and mocked controller tests alone do not establish accelerator compatibility.

The initial provider validation passed on a Modal Tesla T4 with Python 3.14.7, JAX 0.11.2 and NumPy 2.5.3, including the full test suite and all four render and gradient regressions. The Kaggle submission installed its dependencies but failed TPU initialisation with `No jellyfish device found`; its tests did not start. A successful live Kaggle run proving TPU allocation and passing the regressions remains outstanding.

## Render regression

The render regression compares the cube and head frames 0 and 15 with the checked-in images in `tests/references/`. It allows small renderer differences while requiring all of these bounds:

- Foreground intersection-over-union (IoU) of at least `0.90`.
- Mean absolute RGB error over the union of foreground pixels of at most `5/255`.
- 95th-percentile per-pixel maximum-channel error of at most `20/255`.
- Each animation frame has a foreground ratio from `0.15` to `0.27`; adjacent frames have mean absolute RGB error from `0.5/255` to `5/255`.

The cube foreground mask includes pixels more than 20 intensity counts away from white; the head mask includes pixels with any channel above 8. This keeps background differences from dominating the image comparison. The job uploads the numerical report, keyframes, amplified difference images, and APNG for seven days, including when a test fails.

### Reproducing render checks locally

To reproduce the render checks locally from the repository root, install the locked dependency groups with uv, then run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg JAXRENDERER_ARTIFACT_DIR=render-artifacts \
  uv run python -m pytest -q tests/render_regression.py tests/test_smoke_grad.py
```

When an intentional rendering change updates the expected output, inspect the uploaded keyframes and numeric report first. Then render the same scenes on macOS with the current settings and replace only the affected PNGs in `tests/references/`. Run the render regression command against the proposed images and review the full-frame and animation checks before committing them. Do not change tolerances solely to make a new golden pass; change them only when a measured cross-platform or renderer variation justifies the new bounds.
