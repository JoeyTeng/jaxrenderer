import jax.numpy as jnp

import renderer

ImageWidth: int = 640
ImageHeight: int = 480

# Create a cube with texture map of pure blue
cube = renderer.create_cube(
    half_extents=jnp.ones(  # pyright: ignore[reportUnknownMemberType]
        3, dtype=jnp.single
    ),
    texture_scaling=jnp.ones(  # pyright: ignore[reportUnknownMemberType]
        2, dtype=jnp.single
    ),
    diffuse_map=jnp.zeros(  # pyright: ignore[reportUnknownMemberType]
        (2, 2, 3), dtype=jnp.single
    )
    .at[..., 2]
    .set(1),
    specular_map=jnp.ones(  # pyright: ignore[reportUnknownMemberType]
        (2, 2), dtype=jnp.single
    )
    * 2.0,
)

# Render the cube
image = renderer.Renderer.get_camera_image(
    objects=[renderer.ModelObject(model=cube)],
    # Simply use defaults
    camera=renderer.CameraParameters(
        viewWidth=ImageWidth,
        viewHeight=ImageHeight,
        position=jnp.array(  # pyright: ignore[reportUnknownMemberType]
            [2.0, 4.0, 1.0], dtype=jnp.single
        ),
    ),
    # Simply use defaults
    light=renderer.LightParameters(),
    width=ImageWidth,
    height=ImageHeight,
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # pyright: ignore

ax.imshow(  # pyright: ignore[reportUnknownMemberType]
    renderer.transpose_for_display(image)
)

plt.show()  # pyright: ignore[reportUnknownMemberType]
