# tool-result-cache

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tool-result-cache.svg)](https://pypi.org/project/tool-result-cache/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Content-addressable cache for LLM agent tool calls.** Same tool, same args ⇒ same answer, returned from memory. LRU eviction. Optional TTL. Zero deps.

```python
from tool_result_cache import ToolCache

cache = ToolCache(max_size=128, ttl_s=300)

# direct API
result = cache.get_or_set(
    tool_name="search_web",
    args={"q": "anthropic prompt cache"},
    compute=lambda: expensive_search(q="anthropic prompt cache"),
)

# decorator API
@cache.wrap()
def search_web(q: str) -> list[str]:
    return expensive_search(q)

search_web("anthropic prompt cache")   # cache MISS, computes
search_web("anthropic prompt cache")   # cache HIT

print(cache.hits, cache.misses, cache.evictions, cache.expirations)
```

## Why

Agents repeat themselves. `search_web("anthropic prompt cache")`. Two minutes later, after a tool that returned something confusing: `search_web("anthropic prompt cache")`. There's no upstream rate limiter to save you. There's just a bill that keeps growing.

`tool-result-cache` is an OrderedDict-based LRU plus an optional TTL plus a tiny decorator. It's content-addressable on `(tool_name, args)` so the cache survives unrelated calls between repeats. JSON-canonical arg keys mean `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` hit the same entry.

For loop *detection* (raise on repeats), pair with [`tool-loop-guard`](https://github.com/MukundaKatta/tool-loop-guard). For idempotency *keys* (no caching, just hash), see [`llm-message-hash-py`](https://github.com/MukundaKatta/llm-message-hash-py).

## Install

```bash
pip install tool-result-cache
```

## API

```python
cache = ToolCache(
    max_size=1024,    # 0 disables LRU eviction
    ttl_s=None,       # default per-entry TTL in seconds
    clock=time.monotonic,
)

# direct
cache.set(tool_name, args, value, ttl_s=...)
value = cache.get(tool_name, args)              # None on miss
value = cache.get_or_set(tool_name, args, compute, ttl_s=...)
cache.invalidate(tool_name, args)               # bool: was present?
cache.clear()

# decorator
@cache.wrap(tool_name=None, ttl_s=None)
def my_tool(*args, **kwargs):
    ...

# observability
cache.hits, cache.misses, cache.evictions, cache.expirations
cache.stats                                     # CacheStats snapshot

# helpers
make_key(tool_name, args) -> "<sha256 hex>"
```

## Companion libraries

- [`tool-loop-guard`](https://github.com/MukundaKatta/tool-loop-guard) — raises on repeated calls (catches a stuck agent).
- [`llm-message-hash-py`](https://github.com/MukundaKatta/llm-message-hash-py) — canonical hash for LLM-request idempotency.
- [`cachebench`](https://github.com/MukundaKatta/cachebench) — measure hit ratios over time.

## License

MIT
