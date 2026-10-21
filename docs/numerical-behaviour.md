# Numerical behaviour

## Normals at zero scale

Normals are transformed by the inverse transpose of the object's local scale
and then normalised. At zero scale, the renderer uses the direction obtained
by replacing every zero scale component with the same positive `epsilon` and
taking `epsilon` to zero, with the source normal and non-zero scale components
held fixed.

For example, source normal `(1, 0, 1)` at scale `(0, 0, 1)` tends to `+X`.
Source normal `(0, 1, 1)` at scale `(0, 2, 3)` has no component on the zero
axis, so its direction is the normalised `(0, 1/2, 1/3)`. If the limiting
direction depends on which side a scale approaches zero, this convention picks
the positive side. The same rule applies when all scale components are zero.

> [!WARNING]
> A finite fallback does not make the transform smooth through zero. The
> result can depend on the sign or path of approach; a zero source normal has
> no defined direction. Very small scales, large dynamic ranges, floating
> point precision, and singular or ill-conditioned transforms can also make
> calculations unreliable. Finite output alone is not evidence that every
> render value or gradient is safe. At an ambiguous singular point, autodiff
> follows the fallback branch and need not match the derivative along a
> one-sided path.

## Checking inputs

`ModelObject.value_checks` is an opt-in diagnostic. It checks that `rcond` is
finite and between zero and one, scales are finite and non-degenerate relative
to their largest magnitude, the transform is finite and affine with a
well-conditioned linear part, and source normals are finite and non-zero. Negative scales and
uniformly tiny scales can pass when they are representable. The default
`rcond=1e-6` rejects flattened or relatively ill-conditioned inputs; it does
not establish that all rendered values or derivatives are bounded.

Call `err.throw()` in eager code, or call it after a jitted check returns its
error value. Jitting the unbound method keeps the object as a runtime argument:

```python
import jax
from renderer import ModelObject

err, _ = obj.value_checks(rcond=1e-6)
err.throw()

compiled_checks = jax.jit(ModelObject.value_checks)
err, _ = compiled_checks(obj, rcond=1e-6)
err.throw()
```

This uses JAX's [checkify error handling](https://docs.jax.dev/en/latest/debugging/checkify_guide.html).
Checks are not run automatically during rendering, and invalid inputs do not
produce Python runtime warnings by default.

## Debugging derivatives

A model snapshot contains no previous state or time interval, so it cannot
provide a physical velocity. A finite-difference `delta_scale / delta_t`
requires the previous and current scale plus their elapsed time. If a caller
supplies a time-dependent scale function, its time derivative can be evaluated
with a [JVP](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html):

```python
scale, scale_velocity = jax.jvp(scaling_function, (time,), (1.0,))
```

The reverse-mode gradient of a loss with respect to scale is a sensitivity,
not a physical-time velocity. A JVP evaluated at a singular scale does not
define a derivative that is undefined there. A tangent alone may also fail to
specify the limiting direction when multiple axes vanish at different orders;
the caller must supply the path or parameterisation.

JAX's [`jax_debug_nans`](https://docs.jax.dev/en/latest/debugging/flags.html)
and `jax_debug_infs` flags can help locate non-finite values in a targeted
computation or gradient. They are development aids, not a guarantee for every
render path; intentional infinities such as z-buffer sentinels can also trigger
infinity debugging.
