# pyright: basic
"""Display the blue cube render used by the baseline regression checks."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.render_baselines import render_cube

image = render_cube()

fig, ax = plt.subplots()  # pyright: ignore
ax.imshow(image)  # pyright: ignore[reportUnknownMemberType]
plt.show()  # pyright: ignore[reportUnknownMemberType]
