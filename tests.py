import jax
import jax.numpy as jnp
import matplotlib.image as mpimg

from renderer.renderer import Canvas, Colour, line

jax.config.update('jax_array', True)


def test_line():
    canvas: Canvas = jnp.zeros((100, 100, 3))

    red: Colour = jnp.array((1., 0, 0))
    white: Colour = jnp.array((1., 1., 1.))
    canvasA: Canvas = line(13, 20, 80, 40, canvas, white)
    canvasB: Canvas = line(20, 13, 40, 80, canvasA, red)
    canvasC: Canvas = line(80, 40, 13, 20, canvasB, red)

    mpimg.imsave("test_line.png", canvasC, origin='lower')


if __name__ == '__main__':
    test_line()
