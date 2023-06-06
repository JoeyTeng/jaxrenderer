import functools
import inspect
from typing import Callable, ParamSpec, TypeVar

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
