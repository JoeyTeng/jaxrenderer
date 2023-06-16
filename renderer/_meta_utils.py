import functools
import inspect
import sys
from typing import Callable, TypeVar

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec

import jax

ArgT = ParamSpec("ArgT")
RetT = TypeVar("RetT")


def add_tracing_name(func: Callable[ArgT, RetT]) -> Callable[ArgT, RetT]:
    """Add tracing name to function."""

    members: dict[str, str]
    members = dict(inspect.getmembers(func, lambda v: isinstance(v, str)))
    annotation: str = (f"{members.get('__module__', '')}"
                       f":{members.get('__qualname__', '')}")

    @functools.wraps(func)
    def wrapper(*args: ArgT.args, **kwargs: ArgT.kwargs) -> RetT:
        with jax.named_scope(annotation):
            with jax.profiler.TraceAnnotation(annotation):
                return func(*args, **kwargs)

    return wrapper
