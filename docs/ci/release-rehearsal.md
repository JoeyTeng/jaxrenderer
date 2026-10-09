# Rehearsing the publishing workflow

To run the complete CPU, build, GPU and TPU gates from the current `master`
commit without creating a release, dispatch the workflow with no candidate
inputs:

```sh
gh workflow run pypi.yml --ref master
```

The rehearsal freezes `github.sha`, builds and smoke-tests the package, then
runs both provider tests afresh. It uses the configured Modal and Kaggle
environments, so it consumes Modal compute credits and Kaggle accelerator quota;
check the free-account limits in the
[CI account and environment setup](README.md#account-and-environment-setup)
before dispatching. The rehearsal does not create a release or version bump,
enter the `PyPI` environment, require a PyPI credential or approval, or upload
a package. It cannot verify that a release tag exists or points to this commit,
or test an actual PyPI upload.

The reusable GPU and TPU calls inherit the caller's available secret context as
a workaround described in [runner issue 4453](https://github.com/actions/runner/issues/4453),
where environment secrets were reported empty without inheritance. This also
makes repository and organisation secrets available to the called workflows;
only the named Modal or Kaggle credentials are assigned to each controller
step's environment. Those jobs retain their `modal-gpu` and `kaggle-tpu`
environments. Only the release-only upload step assigns the PyPI token to its
execution environment.

If the TPU wait times out while queued or running, the remote state is unknown.
The workflow records controller state and does not automatically resubmit. To
inspect the original notebook, download `tpu-release-binding-<attempt>` and
`release-tpu-<sha>-<attempt>` from the same workflow run. With Kaggle credentials
set in your local environment, make one bounded status-only query using those
original files and a separate output directory:

```sh
python -u -m tools.kaggle_ci \
  --binding tpu-release-binding.json \
  --resume-state kaggle-controller-state.json \
  --output-dir kaggle-status-check
```

This writes `kaggle-resume-report.json`; `version_verified` and `success` remain
false because the query does not verify the remote version or collect results.
It never resubmits. It cannot make the old failed workflow attempt pass or
satisfy a new release attempt's TPU gate. Before another submission, confirm the
previous notebook has ended: an Actions timeout only stops waiting and does not
prove the remote run stopped.

## Collecting delayed TPU results

For a rehearsal whose CPU, build and GPU gates passed but whose TPU wait timed out, use the original publishing run ID and attempt with the full commit SHA to ask the manual collector to retrieve the saved Kaggle state and result:

```sh
gh workflow run collect-tpu.yml --ref master \
  -f source_run_id=SOURCE_RUN_ID \
  -f source_attempt=SOURCE_ATTEMPT \
  -f commit=FULL_COMMIT_SHA
```

The collector revalidates the original run and its bound source artefacts, then makes a bounded status and result query. It does not push a notebook or consume new TPU submission quota. A pending result fails the collection job; dispatch it again later to check for the delayed result. The receipt is evidence for the original commit only: it does not change the original run's failed conclusion, complete a later release attempt or trigger publication. A later release still has to pass its own fresh gates.

After `collect-tpu-periodic.yml` has reached `master`, it also checks eligible source runs every 30 minutes. GitHub Actions may delay scheduled runs, so the interval is not a guaranteed completion time. It scans only `pypi.yml` runs created in the previous seven days and examines each run's latest attempt. The original CPU, build and GPU gates must have passed, and the TPU result must be a proven timeout. The seven-day window limits automatic discovery; manual collection can still use a source run while its original binding and controller artefacts remain available (30 days).

Recognised legacy controller artefacts contain no timeout status or outcome, so periodic discovery skips them after validating their binding and kernel identity. Unknown schemas or mismatched identities remain errors.

Pending remote work is normal, remains non-terminal and does not fail the periodic job, so a later scan can check it again. A verified success or explicit remote failure creates a terminal marker and is skipped by later scans. An explicit remote failure makes that collection job fail once; API, authentication or validation errors fail without creating a terminal marker. Discovery artefacts and each collection's full evidence are retained for 30 days. The workflow only checks an existing Kaggle run: it does not submit a notebook, spend new TPU quota, alter the original failed Actions attempt, satisfy a later release's gate or publish a package. The GitHub token is not placed in the notebook.
