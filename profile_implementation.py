import timeit
from typing import Any, Callable, NewType, Optional

import jax
import jax.numpy as jnp

from renderer.original_implementation import (Canvas, Colour, Vec2i, line)
from renderer.renderer import line as line_vectorized

jax.config.update('jax_array', True)

OutputT = NewType("OutputT", jax.Array)


def print_format_timeit(
    raw_timings: list[float],
    number: int,
    repeat: int,
    time_unit: Optional[str] = None,
):
    """Adopted from timeit.main."""
    units = {"nsec": 1e-9, "usec": 1e-6, "msec": 1e-3, "sec": 1.0}
    precision = 3

    def format_time(dt):
        unit = time_unit

        if unit is not None:
            scale = units[unit]
        else:
            scales = [(scale, unit) for unit, scale in units.items()]
            scales.sort(reverse=True)
            for scale, unit in scales:
                if dt >= scale:
                    break

        return "%.*g %s" % (precision, dt / scale, unit)

    timings = [dt / number for dt in raw_timings]

    best = min(timings)
    print("%d loop%s, best of %d: %s per loop" %
          (number, 's' if number != 1 else '', repeat, format_time(best)))

    best = min(timings)
    worst = max(timings)
    if worst >= best * 4:
        import warnings
        warnings.warn_explicit(
            "The test results are likely unreliable. "
            "The worst time (%s) was more than four times "
            "slower than the best time (%s)." %
            (format_time(worst), format_time(best)), UserWarning, '', 0)


def profile(
    f: Callable[[Any], OutputT],
    *args,
    times: int = 100,
    repeat: int = 5,
    **kwargs,
) -> tuple[OutputT, int]:
    """Execute f once with *args and **kwargs. Block the returned JAX container
      from f, then return output. Using timeit to measure and print time cost.
    """
    output = f(*args, **kwargs)
    output.block_until_ready()

    raw_timings = timeit.repeat(
        'f(*args, **kwargs).block_until_ready()',
        repeat=repeat,
        number=times,
        globals=locals(),
    )
    print_format_timeit(raw_timings, times, repeat)

    return output


LineDrawer = Callable[[Vec2i, Vec2i, Canvas, Colour], Canvas]


def test_line(line_drawer: LineDrawer) -> Canvas:
    canvas: Canvas = jnp.zeros((100, 100, 3))

    red: Colour = jnp.array((1., 0, 0))
    white: Colour = jnp.array((1., 1., 1.))
    t0: Vec2i = jnp.array((13, 20))
    t1: Vec2i = jnp.array((80, 40))
    t2: Vec2i = jnp.array((20, 13))
    t3: Vec2i = jnp.array((40, 80))
    canvasA: Canvas = line_drawer(t0, t1, canvas, white)
    canvasB: Canvas = line_drawer(t2, t3, canvasA, red)
    canvasC: Canvas = line_drawer(t1, t0, canvasB, red)

    return canvasC


def profile_line():
    """Typical result from Colab Standard GPU instance:

    100 loops, best of 5: 16.9 msec per loop
    100 loops, best of 5: 3.2 msec per loop
    """
    profile(test_line, line)
    profile(test_line, line_vectorized)


if __name__ == '__main__':
    profile_line()
