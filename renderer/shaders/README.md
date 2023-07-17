# Shaders

All implementations have back-face culling enabled, which discards all fragments that belongs to a triangle that is not facing the camera.

## Depth Shader `depth.py`

This is a simple implementation of depth shader, which simply outputs the depth of the fragment to the z-buffer.

## Gouraud Shading `gouraud.py`

This is a simple implementation of Gouraud shading, which interpolates the vertex colour and then directly outputs it as the fragment colour.

The light direction is computed in model's space, and the normals are in model's space as well.

## Gouraud Shading with Texture `gouraud_texture.py`

This is a simple implementation of Gouraud shading with texture, which interpolates the vertex colour. The texture is then sampled in the clip space in the Fragment Shader.

The light direction is computed in model's space, and the normals are in model's space as well.

## Phong Shading `phong.py`

This is a simple implementation of Phong shading, which interpolates the vertex normal and then computes the light direction in the Fragment Shader. Textures are sampled in clip space in the Fragment Shader.

Noticed that the light direction is not transformed and thus the light direction is used as if it is in the eye-space. This shows a "headlight" effect.

## Phong Shading with Normal Map in Tangent Space `phong_darboux.py`

This is a simple implementation of Phong shading with normal map in tangent space, which interpolates the vertex normal and then computes the light direction in the Fragment Shader. Normals are computed with model's given normal and sampled from normal map in tangent space. Textures are sampled in clip space in the Fragment Shader.

Noticed that the light direction is not transformed and thus the light direction is used as if it is in the eye-space. This shows a "headlight" effect.

## Phong Shading with Phong Reflection Approximation `phong_reflection.py`

This is a simple implementation of Phong shading with Phong reflection approximation, which interpolates the vertex normal and then computes the light direction in the Fragment Shader. Textures are sampled in clip space in the Fragment Shader.

Phong's Approximation is used to support `specular`, `ambient` and `diffuse` lighting. The specular lighting is computed with shininess factor given in a `SpecularMap`.

The light direction needs to be given in the pre-projection view/eye space via `light_dir_eye`. The normals are transformed into eye space as well for light computation.

## Phong Shading with Shadow and Phong Reflection Approximation `phong_reflection_shadow.py`

This is a simple implementation of Phong shading with Phong reflection approximation and shadow, which interpolates the vertex normal and then computes the light direction in the Fragment Shader. Textures are sampled in clip space in the Fragment Shader.

Phong's Approximation is used to support `specular`, `ambient` and `diffuse` lighting. The specular lighting is computed with shininess factor given in a `SpecularMap`.

The light direction needs to be given in the pre-projection view/eye space via `light_dir_eye`. The normals are transformed into eye space as well for light computation.

Shadows are simply tested in fragment shadow against a shadow map given by `Shadow`. Shadow coordinates is computed by first transform world coordinates of the vertices in the vertex shader into the clip space of the light, interpolated in clip space (`SMOOTH` by default) in the `Shader.interpolate`, then  projected into screen coordinates in the fragment shader, and finally compared against the shadow map. It is safe to interpolate in the light's clip space using barycentric coordinates obtained in the model/main camera's clip space, because there is a simple linear transformation (by a invertible matrix) between the two pre-projection clip spaces. CHECK: Thus after projection, the barycentric coordinates computed are the same. Reference of this method is here [Tutorial 16 : Shadow mapping/OpenGL-tutorial](http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/#basic-shader).

A common method of computing shadow to transform fragment coordinates first back to world space, then transform to light's screen space. This method does not work due to the precision lost. Even with handcrafted inverse matrices, the precision is still not enough. Using `jax.lax.linalg.triangular_solve` does not help either. This issue is not only about the depth (z-component), but also about the x- and y-components, leading to a failure of retrieving the shadow value in the shadow map.
