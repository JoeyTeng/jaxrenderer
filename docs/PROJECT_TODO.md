# Rendering follow-ups

- [x] Restore the identity quaternion used by `ModelObject.replace_with_orientation()`
  when neither rotation argument is supplied ([#1](https://github.com/JoeyTeng/jaxrenderer/issues/1)).
- [x] Separate interpolated values from typed fragment targets, and retain
  inherited opaque mixing without the output union or caller-side cast.
- [ ] Measure the padding overhead in texture-map merging and investigate a more
  compact layout where the memory or runtime savings justify the change.
- [ ] Define and implement point primitives, including `gl_PointSize` and
  `gl_PointCoord`, throughout the rendering pipeline.
- [ ] Investigate tile-based deferred rendering (TBDR) for TPU performance
  ([#14](https://github.com/JoeyTeng/jaxrenderer/issues/14)).
