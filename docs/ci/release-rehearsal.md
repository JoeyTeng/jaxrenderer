# Rehearsing the publishing workflow

To run the CPU, build, GPU and TPU submission stages from the current `master`
commit without creating a release, dispatch the workflow with no candidate
inputs:

```sh
gh workflow run pypi.yml --ref master
```

Leave `release_tag` empty for a rehearsal. Supplying an existing published tag
instead selects [publication recovery](release-recovery.md). The workflow
freezes `github.sha`, builds and smoke-tests the package, runs the Modal GPU
check, and submits one Kaggle TPU run. The configured Modal and Kaggle accounts
may consume compute credits and accelerator quota; check the free-account
[limits](README.md#account-and-environment-setup) before dispatching.

The publishing run records a successful TPU submission as pending validation.
It does not claim the TPU regression passed. The hourly collector later checks
the saved Kaggle run. A verified result starts a completion workflow: rehearsals
record the completion without publishing, while releases request fresh approval
in the `PyPI` environment. Completion reuses the original distributions and
does not rebuild them.

The reusable GPU and TPU submission calls inherit the caller's available secret
context as a workaround described in
[runner issue 4453](https://github.com/actions/runner/issues/4453), where
environment secrets were reported empty without inheritance. Only the named
Modal and Kaggle credentials are assigned to their controller steps. The
completion workflow provides the PyPI credential only to the upload step after
the environment approval.

If the workflow stops before observing a terminal TPU status, the remote state
is unknown. The submission job retains its binding and saved controller state
for a bounded status-only query. This query does not resubmit or validate a
completed result, and it cannot make the original workflow pass. Before another
submission, confirm that the previous notebook has ended.

## Collecting delayed TPU results

The periodic workflow runs hourly at minute 17. GitHub Actions can delay
scheduled runs, so this is not a guaranteed completion time. It scans eligible
`pypi.yml` runs from the previous seven days and considers each run's latest
attempt. CPU, build and GPU gates and the original TPU submission must have
passed. Original bindings, distributions, collection and completion artefacts
are retained for 30 days.

To manually collect a result, use the original publishing run ID and attempt,
candidate SHA, and the periodic collector run ID and attempt:

```sh
gh workflow run complete-release.yml --ref master \
  -f source_run_id=SOURCE_RUN_ID \
  -f source_attempt=SOURCE_ATTEMPT \
  -f commit=FULL_COMMIT_SHA \
  -f collector_run_id=COLLECTOR_RUN_ID \
  -f collector_attempt=COLLECTOR_ATTEMPT
```

This replay reads the saved collection evidence; it does not submit a notebook
or spend additional TPU quota. Pending or failed results are not ready for
completion. The completion workflow verifies the original candidate and tag,
source distribution artefact identity and file hashes before it can publish.
The original PyPI tag must still be absent from PyPI. A release requires fresh
approval in the `PyPI` environment; a rehearsal only records the verified
completion. Existing-tag recovery runs are excluded from periodic discovery.

Pending remote work is non-terminal and can be checked again by a later scan. A
verified success or explicit remote failure creates a terminal marker. API,
authentication or validation errors fail without creating one. Collection
checks an existing Kaggle run only; it never submits a notebook, alters the
original publishing attempt or places the GitHub token in the notebook.
