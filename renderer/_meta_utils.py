import functools
import inspect
import sys
from typing import Any, Callable, Protocol, TypeVar, cast

import jax

from ._backport import DictT, ParamSpec

__all__ = ["add_tracing_name", "export", "typed_jit"]

ArgT = ParamSpec("ArgT")
RetT = TypeVar("RetT")


def add_tracing_name(func: Callable[ArgT, RetT]) -> Callable[ArgT, RetT]:
    """Add tracing name to function."""

    members: DictT[str, str]
    members = dict(inspect.getmembers(func, lambda v: isinstance(v, str)))
    annotation: str = (
        f"{members.get('__module__', '')}" f":{members.get('__qualname__', '')}"
    )

    @functools.wraps(func)
    def wrapper(*args: ArgT.args, **kwargs: ArgT.kwargs) -> RetT:
        with jax.named_scope(annotation):
            with jax.profiler.TraceAnnotation(annotation):
                return func(*args, **kwargs)

    return wrapper


T = TypeVar("T", Callable[..., Any], type)


def export(fn: T) -> T:
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]  # pyright: ignore[reportGeneralTypeIssues]

    return fn


T_co = TypeVar("T_co", covariant=True)


class Wrapped(Protocol[ArgT, T_co]):
    def __call__(self, *args: ArgT.args, **kwargs: ArgT.kwargs) -> T_co:
        ...

    def lower(self, *args: ArgT.args, **kwargs: ArgT.kwargs) -> jax.stages.Lowered:
        ...


def typed_jit(
    f: Callable[ArgT, RetT],
    *args: Any,
    **kwargs: Any,
) -> Callable[ArgT, RetT]:
    """Typed version of jax.jit.

    This is a temporary solution until type information can be deduced well under
    `jax.jit` and `partial`.

    See: https://github.com/google/jax/issues/10311
    """
    jitted: Wrapped[ArgT, RetT] = cast(
        Wrapped[ArgT, RetT],
        functools.update_wrapper(
            jax.jit(f, *args, **kwargs),  # pyright: ignore[reportUnknownMemberType]
            f,
        ),
    )

    return jitted
