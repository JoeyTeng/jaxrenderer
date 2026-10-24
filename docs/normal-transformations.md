# Stable normal transformations

A normal must stay perpendicular to each transformed tangent. If a tangent $t$ is
transformed by $A$, it becomes $At$. The corresponding normal is transformed by
$A^{-T}$, because

$$
(A^{-T}n)^T(At)=n^T A^{-1}At=n^Tt=0.
$$

For local scale $s=(s_x,s_y,s_z)$, inverse scaling gives $n_i/s_i$ component by
component. The object transform then applies its inverse transpose. The helpers
below keep inverse scaling and row normalisation within a useful numeric range.

## Inverse scaling

`_inverse_scale_normals` handles each row of an `(N, 3)` normal array separately.

For finite non-zero values, `frexp` represents a value as a mantissa times a
power of two:

$$
n_i=m_{n,i}2^{e_{n,i}},\qquad s_i=m_{s,i}2^{e_{s,i}}.
$$

Thus each inverse-scaled component is

$$
\frac{n_i}{s_i}=\frac{m_{n,i}}{m_{s,i}}2^{d_i},\qquad
d_i=e_{n,i}-e_{s,i}.
$$

The helper chooses one exponent for each row,

$$
E=\max_{i\in I}d_i,
$$

where $I$ contains only components whose normal and scale are both non-zero. It
represents the row as

$$
v_i=\frac{m_{n,i}}{m_{s,i}}2^{d_i-E}.
$$

This is the original inverse-scaled row times the same positive factor $2^{-E}$.
The factor cancels during row normalisation, while the largest non-zero ratio no
longer needs to be constructed at its raw magnitude. A much smaller component can
still underflow after rescaling if it falls below the dtype's representable range.

Zero normal components are excluded from $I$. The exponent returned by
`frexp(0)` does not describe a normal-to-scale ratio; including it could let a
small scale on a zero component set $E$ and suppress genuine non-zero components.
The separate zero-scale selection helper also owns the limit or fallback when a
scale component is zero. This inverse-scaling helper is not an inverse for a
singular transform.
The returned `has_ratio` mask records whether $I$ is non-empty; rows without an
eligible component use $E=0$ while awaiting zero-scale selection.

For example, $n=(1,1,0)$ and $s=(2,1,1)$ give inverse-scaled direction
$(1/2,1,0)$. After normalisation the row is
$(1/\sqrt{5},2/\sqrt{5},0)$.

## Bounded powers

Each relative exponent $r=d_i-E$ is split into five integer exponents. Let
$q=\lfloor r/5\rfloor$ and $u=r-5q$. The helper uses $u$ factors of $2^{q+1}$ and
$5-u$ factors of $2^q$, whose product is $2^r$. In float32 and float64 these
factors remain within the dtype's exponent range; their full product may not.
They are applied sequentially to the mantissa ratio. A zero component multiplied
by finite factors stays zero, whereas forming an overflowing product first can
produce `0 * inf`.
Building the powers on ones and multiplying them into the data also preserves
the zero-component gradient behaviour needed for JAX 0.11.2. This measure does
not guarantee that every intermediate or every derivative is finite. When the
power-over-scale value for a zero normal component is non-finite, a separate guard
stops the scale-mantissa gradient at that component; it does not make every
possible gradient finite. When normals and scales have different floating-point
dtypes, the helper uses their promoted result dtype so a zero component is not
multiplied by a power that overflowed only because the normal used a narrower
dtype. Homogeneous low-precision inputs still have that dtype's limited range.

## Row normalisation

`_normalise_normal_rows` turns each non-zero row into a unit vector. For a finite
row $x$, it computes

```math
m=\max_i|x_i|,\qquad
\tilde m=\begin{cases}
m,&m>0,\\
1,&m=0,
\end{cases}\qquad
y=x/\tilde m,
```

It then computes $q=\sum_i y_i^2$ and uses
$\ell=\sqrt{q}$ when $q>0$, or $\ell=1$ when $q=0$, returning $y/\ell$. For a
non-zero row, $1\leq q\leq3$. This prevents overflow or underflow from squaring
the original magnitudes. In float32, directly squaring components near $10^{30}$
can overflow, while squaring components near $10^{-30}$ can underflow to zero.
For both $x=(10^{30},10^{30},0)$ and $x=(10^{-30},10^{-30},0)$,
max-component scaling allows the row to normalise to $(1/\sqrt{2},1/\sqrt{2},0)$.

The max-absolute-value factor is passed through `stop_gradient`. For a non-zero
row, in exact arithmetic,

$$
\frac{x/m}{\lVert x/m\rVert_2}=\frac{x}{\lVert x\rVert_2}.
$$

Since $m$ cancels, freezing its derivative leaves the exact derivative of the
normalised result unchanged. It also avoids differentiating through the max
scaling factor's reciprocal-square path, which can overflow in automatic
differentiation. Floating-point rounding and underflow still apply. At an all-zero
row, unit-vector normalisation is undefined; returning zero is a computational
convention, not a unique mathematical derivative.

After this step, the model applies the object's inverse-transpose transform to
each normal row independently. A common batch normalisation would cancel from
each direction in exact arithmetic, but its shared derivative can couple
otherwise unrelated rows and amplify floating-point cancellation. The per-row
transform keeps one normal's direction and scale gradient independent of other
normals in the mesh.

## Limits and diagnostics

These helpers stabilise inverse scaling and row-length calculation for finite
inputs. They do not guarantee that every intermediate is finite: the following
inverse-transpose matrix operation may still be singular or ill-conditioned, and
non-finite inputs are outside these guarantees. `ModelObject.value_checks()` is
an opt-in diagnostic that checks scale and transform conditioning and source
normal validity. Passing it does not certify every rendered value or gradient
as finite. See [numerical behaviour](numerical-behaviour.md) for the zero-scale
convention and [input checks](numerical-behaviour.md#checking-inputs).
