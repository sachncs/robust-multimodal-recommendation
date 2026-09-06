"""Name-to-factory registries for pluggable pipeline components.

Every stage of the pipeline has a ``kind`` in the configuration. A registry
turns that string into the component it names, which is what lets the
configuration select an implementation and lets new implementations be added
from outside this package.

Registration lives with the components themselves — ``morel.codebook``
registers codebooks, ``morel.recommend`` registers recommenders — so this
module holds only the mechanism and stays free of dependencies on the layers
above it.

Example:
    Adding an implementation without modifying morel::

        from morel.codebook import CODEBOOKS

        @CODEBOOKS.register("my-codebook")
        def build_mine(*, dim, size, router, seed=None):
            return MyCodebook(dim, size)

    ``Pipeline`` then builds it for any config with
    ``codebook.kind = "my-codebook"``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar, overload

from morel.core.errors import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a component name to a factory that builds it.

    Args:
        name: Human-readable name of the thing being registered, used in
            error messages (for example ``"codebook"``).
    """

    def __init__(self, name: str) -> None:
        """Create an empty registry labelled ``name``."""
        self.name = name
        self.factories: dict[str, Callable[..., T]] = {}

    @overload
    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]: ...

    @overload
    def register(self, key: str, factory: Callable[..., T]) -> Callable[..., T]: ...

    def register(
        self,
        key: str,
        factory: Callable[..., T] | None = None,
        *,
        replace: bool = False,
    ) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
        """Register ``factory`` under ``key``.

        Usable directly or as a decorator::

            registry.register("top", build_top)

            @registry.register("top")
            def build_top(...): ...

        Args:
            key: Name the configuration will use to select this component.
            factory: Callable that builds the component. Omit to use the
                decorator form.
            replace: Allow overwriting an existing entry. Off by default so
                that two components accidentally claiming the same name is a
                loud error rather than one silently shadowing the other.

        Returns
        -------
            The factory, so the decorator form leaves it bound to its
            original name.

        Raises
        ------
            ConfigError: If ``key`` is already registered and ``replace`` is
                False, or if ``key`` is empty.
        """
        if not key:
            raise ConfigError(f"{self.name} name must be a non-empty string")

        def do_register(target: Callable[..., T]) -> Callable[..., T]:
            if key in self.factories and not replace:
                raise ConfigError(
                    f"{self.name} {key!r} is already registered; "
                    f"pass replace=True to override it deliberately"
                )
            self.factories[key] = target
            return target

        if factory is None:
            return do_register
        return do_register(factory)

    def get(self, key: str) -> Callable[..., T]:
        """Return the factory registered under ``key``.

        Raises
        ------
            ConfigError: If nothing is registered under ``key``. The message
                lists the names that are available, because an unknown kind
                is almost always a typo in a config file.
        """
        try:
            return self.factories[key]
        except KeyError:
            raise ConfigError(
                f"unknown {self.name} {key!r}; available: {', '.join(self.available()) or '(none)'}"
            ) from None

    def create(self, key: str, **kwargs: object) -> T:
        """Build the component registered under ``key``."""
        return self.get(key)(**kwargs)

    def unregister(self, key: str) -> None:
        """Remove the component registered under ``key``.

        Mainly useful for tests and for tearing down a temporary
        registration; ordinary use should prefer ``replace=True`` on
        :meth:`register`.

        Raises
        ------
            ConfigError: If nothing is registered under ``key``.
        """
        if key not in self.factories:
            raise ConfigError(f"cannot unregister {self.name} {key!r}: it is not registered")
        del self.factories[key]

    def available(self) -> tuple[str, ...]:
        """Return the registered names, sorted."""
        return tuple(sorted(self.factories))

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` names a registered component."""
        return key in self.factories

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered names, sorted."""
        return iter(self.available())

    def __len__(self) -> int:
        """Return how many components are registered."""
        return len(self.factories)

    def __repr__(self) -> str:
        """Return a debug representation listing the registered names."""
        return f"Registry({self.name!r}, {list(self.available())!r})"


__all__ = ["Registry"]
