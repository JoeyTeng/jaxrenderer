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
