# Recovering an existing release

When a published GitHub Release has failed before its PyPI upload because the
controller needs a fix, merge that fix into protected `master` and dispatch:

```sh
gh workflow run pypi.yml --ref master -f release_tag=v0.5.0
```

Before dispatching, check **Settings → Environments → PyPI**. If deployment refs
are restricted, allow the `master` branch as well as the release tags matching
`v*`; retain the required reviewer approval. Recovery uses the triggering
`master` ref for environment access, even though it tests the original tag.
See [account and environment setup](README.md#account-and-environment-setup).

The non-empty `release_tag` selects publication recovery. Leaving it empty still
runs a [rehearsal](release-rehearsal.md) without publishing. Recovery accepts only
an already published release whose tag resolves to a commit in `master` history,
whose package version matches the tag, and whose version is absent from PyPI.
An unavailable or ambiguous check blocks the workflow. It never moves the tag,
creates another release or changes the source version.

The `Prepare release validation` workflow and controller come from the dispatched `master` commit; the build,
CPU checks and accelerator tests use the original tag's frozen commit. A separate
candidate binding records both commits, the tag object, release identity, version,
repository and workflow attempt. The ordinary accelerator bindings retain the
candidate commit and the same run and attempt.

Each recovery builds once and reruns CPU, installed-wheel and Modal GPU gates,
then submits a Kaggle TPU run. A successful submission is recorded as pending
validation. Delayed collection supports ordinary release and rehearsal runs;
existing-tag recovery runs are excluded because their controller and candidate
commits differ. Publication still requires the `PyPI` environment's fresh
approval and credential. There is no paid fallback or automatic retry. A
successful GitHub Release alone does not establish that the version has
reached PyPI.

## A delayed TPU run

An Actions timeout does not prove that the remote notebook has stopped. Inspect
the saved binding and controller state with the
[bounded status-only query](release-rehearsal.md#rehearsing-the-publishing-workflow)
before any new submission. This query neither submits a notebook nor validates a
completed result.

Within a new recovery attempt, TPU confirmation submits once and saves its
controller state. It does not wait for the regression or claim a validation
pass. Recovery does not resume the original failed Actions run.

The delayed-result collectors require matching controller and candidate
commits. Recovery runs record a `recovery-source-binding-<attempt>` marker and
are excluded from periodic discovery. Neither manual nor periodic collection
supports recovery runs. A collected receipt cannot complete a later
publication attempt's TPU validation.

## Kaggle quota output

Kaggle 2.2.4 authenticates during module import, before parsing `--no-warn`.
Initialisation can therefore print notices before a quota command emits its
JSON result. The controller's CLI adapter sends import-time output to standard
error, keeping it separate from the command result. Both streams retain the
existing size and time limits. Missing, unavailable or malformed quota data
still blocks notebook submission; no quota value is inferred from notices.
