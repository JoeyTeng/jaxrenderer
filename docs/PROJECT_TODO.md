# Rendering follow-ups

- [ ] **Define empty-scene behaviour.** Decide whether an empty object sequence
  returns the configured background buffers or raises a clear `ValueError`, then
  cover the chosen behaviour through the public renderer API.
- [ ] **Verify perspective-correct interpolation numerically.** Extend the
  existing perspective smoke test with a small analytical example whose expected
  interpolated value differs from affine interpolation.
- [ ] **Support double-sided objects.** Propagate `ModelObject.double_sided`
  through rendering and verify reversed winding and mixed single- and double-sided
  objects in the same scene.
- [ ] **Validate camera gradients.** Locate the non-finite leaves in the current
  camera-gradient smoke check, differentiate the original camera parameters in a
  stable scene, and compare a finite gradient with a central finite difference.
