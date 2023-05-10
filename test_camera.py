import jax
import jax.lax as lax
import jax.numpy as jnp

from renderer.geometry import Camera

eye = jnp.array((1., 1, 3))
center = jnp.zeros(3, dtype=jnp.single)
up = jnp.array((0, 1., 0))

width = height = 800
viewport_dimension = jnp.array((width, height))
lowerbound = jnp.array((width / 8, height / 8))
depth = jnp.array(255.)

camera = Camera.create(eye, center, up, lowerbound, viewport_dimension, depth)

print("camera:")
for key, value in camera._asdict().items():
    print(f"{key}:")
    print(value)

