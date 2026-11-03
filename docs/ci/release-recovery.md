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

The workflow and controller come from the dispatched `master` commit; the build,
CPU checks and accelerator tests use the original tag's frozen commit. A separate
candidate binding records both commits, the tag object, release identity, version,
repository and workflow attempt. The ordinary accelerator bindings retain the
candidate commit and the same run and attempt.

Each recovery builds once and reruns all CPU, installed-wheel, Modal GPU and
Kaggle TPU gates. Before upload it rechecks the candidate binding and tag, then
downloads that attempt's validated distributions without rebuilding. Publication
still requires the `PyPI` environment's approval and credential. There is no paid
fallback or automatic retry. A successful GitHub Release alone does not establish
that the version has reached PyPI.

## A delayed TPU run

An Actions timeout does not prove that the remote notebook has stopped. Inspect
the saved binding and controller state with the
[bounded status-only query](release-rehearsal.md#rehearsing-the-publishing-workflow)
before any new submission. This query neither submits a notebook nor validates a
completed result.

Within a new recovery attempt, TPU confirmation uses the same
[four-job relay](README.md#account-and-environment-setup) as ordinary releases.
It submits once and continues that notebook within the attempt. It does not
resume the original failed Actions run or reuse its notebook's result in a new
attempt.

The legacy delayed-result collectors require matching controller and candidate
commits. Recovery runs record a `recovery-source-binding-<attempt>` marker and
are excluded from periodic discovery. Neither manual nor periodic collection
currently supports recovery runs. A collected receipt cannot complete a later
publication attempt's TPU gate.

## Kaggle quota output

Kaggle 2.2.4 authenticates during module import, before parsing `--no-warn`.
Initialisation can therefore print notices before a quota command emits its
JSON result. The controller's CLI adapter sends import-time output to standard
error, keeping it separate from the command result. Both streams retain the
existing size and time limits. Missing, unavailable or malformed quota data
still blocks notebook submission; no quota value is inferred from notices.
