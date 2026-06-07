"""tool-result-cache - content-addressable cache for agent tool calls.

When an agent calls `search_web(q="anthropic prompt cache")` three times in
a row, it's wasting tokens, money, and time. `ToolCache` is a small LRU
cache keyed on a stable hash of (tool_name, args). Optional TTL per entry,
optional max size, zero dependencies.

    from tool_result_cache import ToolCache

    cache = ToolCache(max_size=128, ttl_s=300)

    # use directly
    result = cache.get_or_set(
        tool_name="search_web",
        args={"q": "anthropic prompt cache"},
        compute=lambda: expensive_search(q="anthropic prompt cache"),
    )

    # or decorate
    @cache.wrap(tool_name="search_web")
    def search_web(q: str) -> list[str]:
        return expensive_search(q)

    cache.hits, cache.misses, cache.evictions
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

__version__ = "0.1.0"
__all__ = [
    "ToolCache",
    "CacheStats",
    "make_key",
]


T = TypeVar("T")


# ---- key helper ------------------------------------------------------------


def make_key(tool_name: str, args: Any) -> str:
    """Stable key from tool name + canonical-JSON args. SHA-256 hex digest."""
    if args is None:
        canon = "null"
    else:
        try:
            canon = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            canon = repr(args)
    h = hashlib.sha256()
    h.update(tool_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(canon.encode("utf-8"))
    return h.hexdigest()


# ---- stats -----------------------------------------------------------------


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0


# ---- entry -----------------------------------------------------------------


@dataclass
class _Entry:
    value: Any
    expires_at: float | None


# ---- main class ------------------------------------------------------------


class ToolCache:
    """LRU cache for tool results, keyed by (tool_name, args).

    Args:
        max_size: max entries; 0 or negative disables capacity-based eviction.
        ttl_s: optional default TTL in seconds. None means entries never expire
            by time (only by LRU eviction). Per-call `ttl_s` overrides this.
        clock: optional callable returning monotonic seconds; used for tests.
    """

    def __init__(
        self,
        max_size: int = 1024,
        ttl_s: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_size = int(max_size)
        self._default_ttl = ttl_s
        self._clock = clock
        self._data: "OrderedDict[str, _Entry]" = OrderedDict()
        self._stats = CacheStats()

    # ---- introspection ------------------------------------------------

    @property
    def hits(self) -> int:
        return self._stats.hits

    @property
    def misses(self) -> int:
        return self._stats.misses

    @property
    def evictions(self) -> int:
        return self._stats.evictions

    @property
    def expirations(self) -> int:
        return self._stats.expirations

    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            expirations=self._stats.expirations,
        )

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._touch(key) is not None

    # ---- core -------------------------------------------------------

    def get(self, tool_name: str, args: Any) -> Any | None:
        """Return cached value or None. Counts as hit/miss."""
        key = make_key(tool_name, args)
        entry = self._touch(key)
        if entry is None:
            self._stats.misses += 1
            return None
        self._stats.hits += 1
        return entry.value

    def set(
        self,
        tool_name: str,
        args: Any,
        value: Any,
        *,
        ttl_s: float | None = None,
    ) -> None:
        """Insert or replace a cached value."""
        key = make_key(tool_name, args)
        expires_at = self._compute_expiry(ttl_s)
        self._data[key] = _Entry(value=value, expires_at=expires_at)
        self._data.move_to_end(key)
        self._evict_if_needed()

    def get_or_set(
        self,
        tool_name: str,
        args: Any,
        compute: Callable[[], T],
        *,
        ttl_s: float | None = None,
    ) -> T:
        """Return cached value if present, else call `compute` and cache it."""
        key = make_key(tool_name, args)
        entry = self._touch(key)
        if entry is not None:
            self._stats.hits += 1
            return entry.value
        self._stats.misses += 1
        value = compute()
        expires_at = self._compute_expiry(ttl_s)
        self._data[key] = _Entry(value=value, expires_at=expires_at)
        self._data.move_to_end(key)
        self._evict_if_needed()
        return value

    def invalidate(self, tool_name: str, args: Any) -> bool:
        """Drop a single entry. Returns True if it was present."""
        key = make_key(tool_name, args)
        return self._data.pop(key, None) is not None

    def clear(self) -> None:
        """Drop all entries (and reset stats)."""
        self._data.clear()
        self._stats = CacheStats()

    # ---- decorator helper -------------------------------------------

    def wrap(
        self,
        *,
        tool_name: str | None = None,
        ttl_s: float | None = None,
    ):
        """Decorator: wrap a function so its results are cached.

        Args:
            tool_name: cache namespace; defaults to the wrapped function's name.
            ttl_s: per-call TTL; defaults to the cache's default.
        """

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            resolved = tool_name or fn.__name__

            @wraps(fn)
            def inner(*args: Any, **kwargs: Any) -> T:
                key_args = {"args": list(args), "kwargs": kwargs}
                return self.get_or_set(
                    tool_name=resolved,
                    args=key_args,
                    compute=lambda: fn(*args, **kwargs),
                    ttl_s=ttl_s,
                )

            return inner

        return decorator

    # ---- internals --------------------------------------------------

    def _touch(self, key: str) -> _Entry | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and self._clock() >= entry.expires_at:
            del self._data[key]
            self._stats.expirations += 1
            return None
        self._data.move_to_end(key)
        return entry

    def _compute_expiry(self, ttl_s: float | None) -> float | None:
        effective = ttl_s if ttl_s is not None else self._default_ttl
        if effective is None:
            return None
        return self._clock() + float(effective)

    def _evict_if_needed(self) -> None:
        if self._max_size <= 0:
            return
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)
            self._stats.evictions += 1
