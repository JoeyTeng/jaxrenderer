import jax
import jax.numpy as jnp

from renderer import Camera
from renderer.geometry import normalise_homogeneous, to_homogeneous

eye = jnp.array((1, 3, -2))
centre = jnp.array((0, 0, 0))
up = jnp.array((0, 0, 1))

print(
    Camera.view_matrix(eye, centre, up) @ Camera.view_matrix_inv(
        eye, centre, up))

lowerbound = jnp.array((3, 2))
dimension = jnp.array((201, 3415))
depth = jnp.array((10))

viewport = Camera.viewport_matrix(lowerbound, dimension, depth)
viewport_inv = Camera.viewport_matrix_inv(viewport)
print(viewport @ viewport_inv)

proj = Camera.perspective_projection_matrix(45., 16. / 9., 1e-3, 1e3)
print(proj @ Camera.perspective_projection_matrix_inv(proj))

key = jax.random.PRNGKey(20230528)
key, subkey = jax.random.split(key)
verts = jax.random.uniform(subkey, shape=(101, 3), minval=-1., maxval=1.)
verts = to_homogeneous(verts)

camera: Camera = Camera.create(
    view=Camera.view_matrix(eye, centre, up),
    projection=proj,
    viewport=viewport,
    view_inv=Camera.view_matrix_inv(eye, centre, up),
)

assert isinstance(camera, Camera), f"{camera}"

w = jnp.array((
    (15., 23., 2., 1.),
    (0., 0., 1., 1.),
))[0]
e = camera.apply(w, camera.view)
print("e<=>w", w, camera.view_inv @ e)
c = camera.apply(e, camera.projection)
shuffle = jax.lax.cond(
    camera.projection[3, 3] == 0,
    # perspective projection
    lambda: jnp.array([0, 1, 3, 2]),
    # orthographic projection
    lambda: jnp.array([0, 1, 2, 3]),
)
print(camera.projection[..., shuffle])
e_comp = jax.lax.linalg.triangular_solve(camera.projection[..., shuffle],
                                         c[..., shuffle])[..., shuffle]
e_via_inv = Camera.perspective_projection_matrix_inv(camera.projection) @ c
print(
    "c<=>e",
    normalise_homogeneous(e),
    normalise_homogeneous(e_comp),
    normalise_homogeneous(e_via_inv),
    sep='\n',
)
s = camera.apply(c, camera.viewport)
c_comp = jax.lax.linalg.triangular_solve(camera.viewport, s)
c_via_inv = Camera.viewport_matrix_inv(camera.viewport) @ s
print(
    "s<=>c",
    normalise_homogeneous(c),
    normalise_homogeneous(c_comp),
    normalise_homogeneous(c_via_inv),
    sep='\n',
)
print(
    f"s {s / s[..., 3]} c {c / c[..., 3]} e {e / e[..., 3]} w {w / w[..., 3]}")
print(camera.to_screen_inv(s))
print(normalise_homogeneous(camera.screen_to_world @ s))
print(
    normalise_homogeneous(camera.view_inv @ normalise_homogeneous(
        Camera.perspective_projection_matrix_inv(camera.projection)
        @ normalise_homogeneous(
            Camera.viewport_matrix_inv(camera.viewport) @ s))))

# exit(0)

screen = camera.to_screen(verts)
# computed_w = camera.to_screen_inv(normalise_homogeneous(screen))
computed_w = normalise_homogeneous(
    camera.apply(normalise_homogeneous(screen), camera.screen_to_world))
mask = jnp.isclose(computed_w, verts, rtol=1e-2)
try:
    assert mask.all(), f"expected {verts[~mask]}, got {computed_w[~mask]}"
except AssertionError as e:
    print("inverse is not close to expected for many.")
