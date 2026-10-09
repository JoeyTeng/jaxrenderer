# Continuous Integration and Render Regression

The GitHub Actions workflow tests Python 3.12–3.14 on Linux using the locked dependencies. A separate Python 3.13 job installs NumPy `2.1.3`, checks the built wheel's dependency metadata, and runs the full test suite and CPU render and gradient regressions; its environment is isolated from the uv lockfile. On Python 3.14, Ruff checks `.py` and `.pyi` files under `assets`, `renderer`, `examples`, `test_resources`, `tests` and `tools`. The lint gate requires zero E4, E7, E9, F and I violations on the frozen candidate; it does not compare against a base revision. `F722` and `F821` remain ignored because jaxtyping shape annotations can trigger false positives. A separate `macos-latest` job uses Python 3.14 and CPU-only JAX to render the cube and a 30-frame head animation. The job also checks the numerical gradient of the light direction's x component against a central finite difference. The camera-gradient smoke check still runs, but its current output includes non-finite leaves and is not used as a numerical gate.

Run `uv run ruff check assets renderer examples test_resources tests tools` to run the same zero-violation `.py` and `.pyi` lint gate locally. Ruff uses the rules and ignores in `pyproject.toml`; `typings/` remains out of scope.

The required Python 3.14 lint job also runs strict Pyright with warnings enabled on `renderer/types.py`; this core module must have no errors or warnings. The full-repository strict Pyright run remains advisory, with its diagnostics visible while broader JAX typing issues are addressed. A dedicated `linux-render-regression` job also runs the CPU render and gradient regressions on Python 3.14 with the locked dependencies.

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
- Register at [Kaggle](https://www.kaggle.com/), complete the required account verification, and confirm access to a TPU notebook. Generate an API token under **Account Settings → API tokens**; see the [API documentation](https://www.kaggle.com/docs/api). The controller submits a private notebook with `TpuV5E8` and retrieves that run's output. Availability and quota depend on the account; insufficient quota, a queue timeout or missing TPU devices fails the check.

The Kaggle wait has independent budgets: up to four hours to leave the queued state, then up to 45 minutes from the first active status for execution. Controller phase, status and elapsed times are persisted, with status changes written to the live workflow log. A timeout before terminal status leaves the remote state unknown: the CI gate fails closed, but this does not prove the regression failed or the notebook stopped. Kaggle's CLI has no documented command to cancel an active kernel session, so do not automatically resubmit after a timeout.

### Enabling the GPU merge requirement

First merge the workflow and configure the environments. GitHub requires a `workflow_dispatch` workflow to exist on the default branch before it can run. Run the GPU workflow against an open PR and inspect its device and numerical reports before adding `accelerator/gpu` as a required status check in the `master` ruleset. Require the PR branch to be up to date, and choose GitHub Actions as the expected status source. Missing or unsuccessful checks must block merging; do not replace a failed provider check with a skipped job.

```sh
gh workflow run accelerators.yml --ref master -f pr=PR_NUMBER
```

### Publishing a release

Publishing a GitHub Release starts `Build, validate and publish`. The workflow freezes the tag's full commit SHA, requires that commit to be in `master` history, and checks that the tag matches the package version. It reuses the ordinary CI gates: Ruff, strict Pyright on `renderer/types.py`, Python 3.12–3.14 tests, Linux/macOS render and gradient regressions, and the isolated NumPy `2.1.3` compatibility check. Ruff and the core type gate check the frozen candidate directly; full-repository Pyright remains advisory.

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

### Collecting delayed TPU results

When a publishing rehearsal passed its CPU, build and GPU gates but its TPU wait timed out, [Collect delayed TPU results](release-rehearsal.md#collecting-delayed-tpu-results) can retrieve the saved Kaggle result. A manual dispatch from `master` accepts the original workflow run ID, TPU source attempt and full candidate SHA. After this workflow is on `master`, a schedule checks eligible source runs every 30 minutes. GitHub Actions may delay scheduled runs, so the interval is not a guaranteed completion time. It scans only the previous seven days of `pypi.yml` runs and considers each run's latest attempt, after verifying the CPU, build and GPU gates passed and the TPU wait timed out. The manual collector remains available while original source artefacts are retained for 30 days.

A pending Kaggle run is non-terminal: it produces evidence but no terminal marker, and a later scheduled scan can check it again; this does not fail the periodic job. A verified success or explicit remote failure is terminal and recorded so later scans skip that source attempt. An explicit remote failure fails that collection job once; API, authentication and validation errors also fail without creating a terminal marker. Neither path submits a notebook, consumes additional TPU quota, changes the original failed Actions attempt, satisfies a later release's TPU gate or publishes a package. The GitHub token is used only for GitHub evidence and is never placed in the Kaggle notebook.

Kaggle documents TPU selection via its CLI, but an [open upstream issue](https://github.com/Kaggle/kaggle-cli/issues/1197) reports submissions that receive the wrong runtime. A real TPU run is required to confirm a release. Account configuration and mocked controller tests alone do not establish accelerator compatibility.

In [publishing rehearsal run 37848608473, attempt 1](https://github.com/JoeyTeng/jaxrenderer/actions/runs/37848608473/attempts/1), the CPU, build and Modal T4 gates passed, but the Actions TPU queue wait timed out. The original workflow attempt remains failed and no package was published. The saved Kaggle run was later collected and verified by [periodic workflow run 37985738198](https://github.com/JoeyTeng/jaxrenderer/actions/runs/37985738198): it confirmed the same candidate commit `785c4a55f1e7d810a4f56a2153b16311465111d9` on a genuine v5e-8 TPU. All 149 tests and four render and gradient regressions passed, cube and head image errors were zero, and the gradient relative error was `0.00012450704850647037` (163.5 seconds on the remote run). This delayed result does not change the original failed Actions attempt or publish a package. A later release still requires its own fresh gates.

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
