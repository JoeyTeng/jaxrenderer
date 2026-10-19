from collections.abc import Callable
import functools
import inspect
import sys
from typing import Any, ParamSpec, Protocol, TypeVar, cast, overload

import jax
from jax import Array
import jax.numpy as jnp
import jax.typing
import jaxtyping

__all__ = [
    "add_tracing_name",
    "export",
    "is_array_type",
    "typed_array",
    "typed_asarray",
    "typed_full",
    "typed_grad",
    "typed_jit",
    "typed_jaxtyped",
    "typed_modf",
    "typed_ones",
    "typed_zeros",
]

ArgT = ParamSpec("ArgT")
RetT = TypeVar("RetT")


def is_array_type(value: object, array_type: object) -> bool:
    """Check a jaxtyping specification in the active shape context."""
    # This cast bridges Pyright's static type-form limitation; the runtime
    # specification passed to isinstance is unchanged.
    return isinstance(value, cast(type[Array], array_type))


def add_tracing_name(func: Callable[ArgT, RetT]) -> Callable[ArgT, RetT]:
    """Add tracing name to function."""

    members: dict[str, str]
    members = dict(inspect.getmembers(func, lambda v: isinstance(v, str)))
    annotation: str = (
        f"{members.get('__module__', '')}:{members.get('__qualname__', '')}"
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
        setattr(mod, "__all__", [fn.__name__])

    return fn


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
    return cast(
        Callable[ArgT, RetT],
        functools.update_wrapper(
            jax.jit(f, *args, **kwargs),  # pyright: ignore[reportUnknownMemberType]
            f,
        ),
    )


# These manually maintained aliases describe the JAX subset used here. The casts
# do not validate upstream signatures, and Array does not prove shape or dtype.
# typed_grad supports one Array argument, a scalar-Array loss and default options;
# its type does not guarantee a particular shape. Check this subset against JAX
# when upgrading the dependency.
type _Shape = int | tuple[int, ...]


class _ArrayFactory(Protocol):
    def __call__(
        self, value: object, /, dtype: jax.typing.DTypeLike | None = None
    ) -> jax.Array: ...


class _ShapeFactory(Protocol):
    def __call__(
        self, shape: _Shape, dtype: jax.typing.DTypeLike | None = None
    ) -> jax.Array: ...


class _FullFactory(Protocol):
    def __call__(
        self,
        shape: _Shape,
        fill_value: jax.typing.ArrayLike,
        dtype: jax.typing.DTypeLike | None = None,
    ) -> jax.Array: ...


class _Modf(Protocol):
    def __call__(self, x: jax.Array, /) -> tuple[jax.Array, jax.Array]: ...


class _Grad(Protocol):
    def __call__(
        self, fun: Callable[[jax.Array], jax.Array], /
    ) -> Callable[[jax.Array], jax.Array]: ...


class _JaxAPI(Protocol):
    grad: _Grad


class _JaxNumpyAPI(Protocol):
    array: _ArrayFactory
    asarray: _ArrayFactory
    ones: _ShapeFactory
    zeros: _ShapeFactory
    full: _FullFactory
    modf: _Modf


# These casts declare the subset used by this project; they do not wrap JAX.
_jax_api = cast(_JaxAPI, jax)
_jnp_api = cast(_JaxNumpyAPI, jnp)


_DecoratedTarget = TypeVar("_DecoratedTarget", bound=Callable[..., object])


class _JaxtypedIdentityDecorator(Protocol):
    def __call__(self, target: _DecoratedTarget, /) -> _DecoratedTarget: ...


class _JaxtypedFactory(Protocol):
    @overload
    def __call__(self, target: _DecoratedTarget, /) -> _DecoratedTarget: ...

    @overload
    def __call__(self, *, typechecker: None) -> _JaxtypedIdentityDecorator: ...


class _JaxtypingAPI(Protocol):
    jaxtyped: _JaxtypedFactory


# This manually maintained alias covers bare function/class decoration and the
# typechecker=None factory form. It does not wrap jaxtyping; check this subset
# against jaxtyping when upgrading the dependency.
typed_jaxtyped = cast(_JaxtypingAPI, jaxtyping).jaxtyped

typed_array = _jnp_api.array
typed_asarray = _jnp_api.asarray
typed_ones = _jnp_api.ones
typed_zeros = _jnp_api.zeros
typed_full = _jnp_api.full
typed_modf = _jnp_api.modf
typed_grad = _jax_api.grad
