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

## 0.1.1

1. Change the default behaviour of `renderer/utils.py::transpose_for_display` which will flip vertically as well by default, so the origin of the resultant matrix will be (height, width, channels) and with the origin located at the top-left corner. The previous behaviour can be achieved by setting `flip_vertical=False`.
2. `Scene.add_cube` now accepts one number for `texture_scaling` to scale texture map equally in both x and y directions.
3. Fix some assert message issues (in `Scene.add_cube`).
4. `CameraParameters` now accepts `position`, `target` and `up` in Python's tuples of floats as well, along with `jnp.array`.
5. `Scene.set_object_orientation` and `Scene.set_object_local_scaling` supports tuple of floats as well as inputs, additional to `jnp.array`.
6. `Model` now has a convenient method `create` to create a Model with same face indices shared by `faces`, `faces_norm` and `faces_uv`, and a default `specular_map`. This is useful for creating a mesh where all vertices has its own normal and uv coordinate specified, under same order (thus same face indices).
7. Correctly support Python Sequence for `utils.build_texture_from_PyTinyrenderer` as texture.
8. `quaternion` function to create an orientation from axis and angle, and `quaternion_mul` to composite quaternion.
9. `rotation_matrix` function to create a rotation matrix from axis and angle. Also allows `Scene` to set object orientation directly using rotation matrix.
10. Move `Renderer.merge_objects` into `geometry.py`, and expose in `__init__.py`.
11. `batch_models` and `Renderer.create_buffers` convenient method to facilitate batch rendering of multiple models.

## 0.1.2

1. Change the ordering of quaternions (in `geometry.py`) to `(w, x, y, z)` instead of `(x, y, z, w)` to be consistent with the convention used in `pytinyrenderer` and `BRAX`. Reference: [brax/math.py](https://github.com/google/brax/blob/aebd8b8cb34430f6eaf6f914293f901e3c8d9a22/brax/math.py).
2. Fix: remove unnecessary `@staticmethod` decorator in `merge_objects`.
3. Changed the way that `Camera` is created in `Renderer.create_camera_from_parameters` to force convert parameters into `float` weak type.
4. Force convert `LightParameters` to JAX arrays in `Renderer.get_camera_image` to avoid downstream errors.
5. Downgrade minimum Python version to `3.9`, `numpy` version to `1.22.0`, `jax` and `jaxlib` version to `0.4.4`.

## 0.1.3

1. Correctly force convert `LightParameters` to JAX arrays in `Renderer.get_camera_image` to avoid downstream errors.
2. Fix `geometry.py::transform_matrix_from_rotation`. Also, change the order of quaternion to `(w, x, y, z)` instead of `(x, y, z, w)` for consistency.
3. Force convert `ShadowParameters` to JAX arrays in `Renderer.get_camera_image` to avoid downstream errors.

## 0.2.0

1. Instead of clipping (planned to be implemented), now the rasteriser interpolates in homogeneous space directly. `Shader.interpolate` will not receive valid `barycentric_screen` values for now. Setting `Interpolation.SMOOTH` and `Interpolation.NOPERSPECTIVE` will result in same results, perspective-correct interpolations.
2. Reorganise example files and rename them.

## 0.2.1

1. Refactor `Scene.set_object_*` methods to be a simple wrapper of `self._replace` and `ModelObject.replace_with_*`, to expose APIs of `ModelObject`s and allows manipulation and rendering without `Scene`.
2. Expose `create_capsule` and `create_cube` APIs.

## 0.3.0

1. Fix `gl_FrontFacing` computation in pipeline so it is consistent to comment: `True` if not back-facing (i.e. front-facing & side facing).
2. Add an extra stage `Shader.primitive_chooser` to choose which primitive to be rendered for each fragment. The default implementation is provided, which assumes that the depth is just the interpolated `z` value in the eye space. It just picks the values of the single primitive that is closest to the camera and is not discarded in the previous pipeline.
3. Expose `loop_unroll` static option to allow unrolling several operations (row rendering) within a single iteration of the out-most loop (iterating along first axis of the canvas). This may be useful in some cases for performance improvement, but careful benchmarking is needed to determine the optimal value. The default value is `1` (no unrolling) as it is the most general case in larger canvases (benchmarked on `960x540` using [GPU T4 in Colab](https://colab.research.google.com/drive/1xhkYNz5WjvUCjQWpp72CLf9SIy3i5PnN)).
4. Bump the minimum Python version to Python 3.10
5. Lower the minimum jax & jaxlib version to 0.3.25.

## 0.3.1

1. Lower minimum Python version to 3.8
2. Introducing `type_extensions` package and improved typing annotations.

## 0.3.2

1. Bump minimum `jax` and `jaxlib` version to 0.4.0 as `jaxtyping` does not support `jax` 0.3.25.
2. Bug fix: add `static_argnames` for utility function `transpose_for_display`.
3. Change to [isort](https://github.com/PyCQA/isort) + [black](https://github.com/psf/black) code style.
4. Migrate full codebase to be type-checked with [pyright](https://github.com/microsoft/pyright).
5. Add smoke tests, and use GitHub Action as CI to run them.
