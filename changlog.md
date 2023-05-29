# Changelog

## 0.1.0

1. For built-in fragment shader outputs, a new field `use_default_depth` is added (default `False`) instead of relying on checking if `gl_FragDepth` is `NaN`.
2. Change `ZBuffer` value meaning to OpenGL convention: all values are non-negative, and the smaller the value, the closer the point is to the camera.
3. Change default `Shader.mixer` behaviour: now it picks the fragment with minimum depth value `gl_FragDepth`.
4. Change `shadow.py::Shadow.render_shadow_map` according to the z-value definition change, to add into the shadow map instead of subtracting so to move shadow maps further away from the "camera" (light).
5. Change `shaders/phong_reflection_shadow` according to the z-value definition change.
6. Transform shadow coordinates in VS (world => NDC) and FS (NDC => screen) instead of transforming in FS (model's screen => world => screen) to avoid precision loss in inverse matrix computation.
7. Fix issue of perspective-correction barycentric interpolation in `pipeline`.
8. Fix `shaders/phong*` so normals are correctly transformed into pre-projection eye coordinates, rather than being projected.
9. Rename `ModelView` => `View`, `model_view` => `view`, `model_view_matrix` => `view_matrix` as the matrix is actually view matrix that transforms from world to eye space, not model view matrix (model to eye space).
