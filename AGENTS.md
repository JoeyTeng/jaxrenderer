# Repository guidance

- Write new human-facing prose intended for this repository, commits, pull requests, or other remote destinations in British English. Preserve established public API names and spellings.
- Keep documentation and pull request text concise and consistent with the surrounding project conventions.
- The default development environment uses Python 3.14; supported Python versions are 3.12–3.14. Install locked dependencies with `uv sync --locked --all-groups`.
- Run the default test suite with `uv run pytest tests/ --import-mode importlib`.
- On macOS, run the render and gradient regression checks with `JAX_PLATFORMS=cpu MPLBACKEND=Agg JAXRENDERER_ARTIFACT_DIR=render-artifacts uv run python -m pytest -q tests/render_regression.py tests/test_smoke_grad.py`.
- Update reference images only for intentional rendering changes. Inspect the generated images and numerical metrics, replace only affected files in `tests/references/`, and review the regression results before committing. Do not adjust tolerances solely to make a reference update pass.
