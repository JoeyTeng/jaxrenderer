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

Before dispatching again after a timeout, confirm that the previous Kaggle
notebook has ended: an Actions timeout only stops waiting and does not prove the
remote run stopped.
