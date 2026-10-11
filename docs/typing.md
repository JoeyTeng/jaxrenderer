# Typing compatibility

The package supports Python 3.12–3.14 and uses standard-library typing features available across those versions. It no longer re-exports generic aliases such as `List`, `Tuple`, `Type`, `DictT`, `NamedTuple`, `ParamSpec`, `Sequence` or `TypeAlias` from `renderer`.

Import typing constructs from their standard-library modules instead. For example, use `list[str]` and `dict[str, int]` for built-in containers, `collections.abc.Sequence` for sequence interfaces, and `typing.NamedTuple` or `typing.ParamSpec` where needed. The `replace_dict` helper has also been removed; use `base | updates` to create a merged dictionary with updates taking precedence.

The renderer-specific `JaxFloating` and `JaxInteger` aliases remain available from `renderer`.
