"""Export backward compatible bindings to replace new features used in later
    Python versions to support Python 3.8+.
"""

import sys
from typing import Any, TypeVar

import jax.numpy as jnp

if sys.version_info < (3, 11):
    from typing_extensions import NamedTuple
else:
    from typing import NamedTuple

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec, TypeAlias
else:
    from typing import ParamSpec, TypeAlias

if sys.version_info < (3, 9):
    """Type subscription requires python >= 3.9."""
    from typing import List, Sequence, Tuple, Type
else:
    from builtins import list as List
    from builtins import tuple as Tuple
    from builtins import type as Type
    from collections.abc import Sequence

if sys.version_info < (3, 9):
    """Type subscription requires python >= 3.9."""
    JaxFloating: TypeAlias = jnp.floating
    JaxInteger: TypeAlias = jnp.integer
else:
    JaxFloating: TypeAlias = jnp.floating[Any]
    JaxInteger: TypeAlias = jnp.integer[Any]

K = TypeVar("K")
V = TypeVar("V")
if sys.version_info < (3, 9):
    from typing import Dict

    DictT: TypeAlias = Dict[K, V]
else:
    DictT: TypeAlias = dict[K, V]


__all__ = [
    "JaxFloating",
    "JaxInteger",
    "List",
    "NamedTuple",
    "ParamSpec",
    "replace_dict",
    "Sequence",
    "Tuple",
    "Type",
    "TypeAlias",
]


def replace_dict(base: DictT[K, V], new: DictT[K, V]) -> DictT[K, V]:
    """Replace items in the old dictionary with new values in new dict."""
    if sys.version_info < (3, 9):
        old = base.copy()
        old.update(new)

        return old

    return base | new
