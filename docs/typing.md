# Typing compatibility

The package supports Python 3.12–3.14 and uses standard-library typing features available across those versions. It no longer re-exports generic aliases such as `List`, `Tuple`, `Type`, `DictT`, `NamedTuple`, `ParamSpec`, `Sequence` or `TypeAlias` from `renderer`.

Import typing constructs from their standard-library modules instead. For example, use `list[str]` and `dict[str, int]` for built-in containers, `collections.abc.Sequence` for sequence interfaces, and `typing.NamedTuple` or `typing.ParamSpec` where needed. The `replace_dict` helper has also been removed; use `base | updates` to create a merged dictionary with updates taking precedence.

The renderer-specific `JaxFloating` and `JaxInteger` aliases remain available from `renderer`.

Renderer modules use native annotation behaviour without `from __future__ import annotations`. Self-references remain explicitly quoted for Python 3.12 and 3.13. Annotations are evaluated eagerly on those versions and lazily on Python 3.14; use `typing.get_type_hints` for resolved runtime annotations rather than relying on string values in `__annotations__`.
