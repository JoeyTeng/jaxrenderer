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

Phong's Approximation is used to support `specular`, `ambient` and `diffuse` lighting. The spcular lighting is computed with shininess factor given in a `SpecularMap`.

The light direction needs to be given in the view/eye space.

## Phong Shading with Shadow and Phong Reflection Approximation `phong_reflection_shadow.py`

This is a simple implementation of Phong shading with Phong reflection approximation and shadow, which interpolates the vertex normal and then computes the light direction in the Fragment Shader. Textures are sampled in clip space in the Fragment Shader.

Phong's Approximation is used to support `specular`, `ambient` and `diffuse` lighting. The spcular lighting is computed with shininess factor given in a `SpecularMap`.

Shadows are simply tested in fragment shadow against a shadow map given by `Shadow`.
