"""Tests for tool_result_cache.ToolCache."""

from __future__ import annotations

import pytest

from tool_result_cache import CacheStats, ToolCache, make_key


# ---- make_key --------------------------------------------------------------


def test_make_key_canonicalizes_arg_order():
    a = make_key("t", {"a": 1, "b": 2})
    b = make_key("t", {"b": 2, "a": 1})
    assert a == b


def test_make_key_differs_by_tool_name():
    assert make_key("search", {"q": "x"}) != make_key("fetch", {"q": "x"})


def test_make_key_is_hex_64():
    k = make_key("t", None)
    assert len(k) == 64
    int(k, 16)  # parses as hex


def test_make_key_handles_none_args():
    a = make_key("t", None)
    b = make_key("t", None)
    assert a == b


def test_make_key_falls_back_to_repr_on_unserializable():
    class _N:
        def __repr__(self):
            return "<N>"

    # the default=str path in json.dumps handles it; we just check stable
    a = make_key("t", _N())
    b = make_key("t", _N())
    assert a == b


# ---- get / set / get_or_set ----------------------------------------------


def test_set_then_get_round_trip():
    c = ToolCache()
    c.set("search", {"q": "x"}, ["result-a"])
    assert c.get("search", {"q": "x"}) == ["result-a"]


def test_get_miss_returns_none_and_counts():
    c = ToolCache()
    assert c.get("search", {"q": "x"}) is None
    assert c.misses == 1
    assert c.hits == 0


def test_get_hit_counts():
    c = ToolCache()
    c.set("search", {"q": "x"}, "ok")
    assert c.get("search", {"q": "x"}) == "ok"
    assert c.hits == 1
    assert c.misses == 0


def test_get_or_set_computes_on_miss():
    calls = []

    def compute():
        calls.append(1)
        return 42

    c = ToolCache()
    assert c.get_or_set("t", {"k": 1}, compute) == 42
    assert c.get_or_set("t", {"k": 1}, compute) == 42
    assert len(calls) == 1
    assert c.hits == 1
    assert c.misses == 1


def test_get_or_set_canonicalizes_args():
    c = ToolCache()
    c.get_or_set("t", {"a": 1, "b": 2}, lambda: "x")
    # different key order, same key — must hit
    assert c.get_or_set("t", {"b": 2, "a": 1}, lambda: "BAD") == "x"
    assert c.hits == 1


# ---- LRU eviction ---------------------------------------------------------


def test_lru_evicts_oldest_when_over_capacity():
    c = ToolCache(max_size=2)
    c.set("t", {"k": 1}, "a")
    c.set("t", {"k": 2}, "b")
    c.set("t", {"k": 3}, "c")  # evicts {k:1}
    assert c.get("t", {"k": 1}) is None
    assert c.get("t", {"k": 2}) == "b"
    assert c.evictions == 1


def test_access_promotes_entry_in_lru():
    c = ToolCache(max_size=2)
    c.set("t", {"k": 1}, "a")
    c.set("t", {"k": 2}, "b")
    # touch {k:1} so {k:2} becomes oldest
    c.get("t", {"k": 1})
    c.set("t", {"k": 3}, "c")  # evicts {k:2}
    assert c.get("t", {"k": 2}) is None
    assert c.get("t", {"k": 1}) == "a"


def test_max_size_zero_disables_eviction():
    c = ToolCache(max_size=0)
    for i in range(100):
        c.set("t", {"k": i}, i)
    assert c.evictions == 0
    assert len(c) == 100


# ---- TTL ------------------------------------------------------------------


def test_ttl_expires_entry():
    # set() reads clock once for expires_at; get() reads clock once in _touch
    clock = iter([0.0, 6.0])

    c = ToolCache(ttl_s=5.0, clock=lambda: next(clock))
    c.set("t", {"k": 1}, "ok")
    assert c.get("t", {"k": 1}) is None
    assert c.expirations == 1
    assert c.misses == 1


def test_per_call_ttl_overrides_default():
    clock = iter([0.0, 3.0])

    c = ToolCache(ttl_s=1.0, clock=lambda: next(clock))
    c.set("t", {"k": 1}, "ok", ttl_s=10.0)
    assert c.get("t", {"k": 1}) == "ok"


def test_no_ttl_means_no_time_expiration():
    big_clock = iter([0.0, 1e9])
    c = ToolCache(ttl_s=None, clock=lambda: next(big_clock))
    c.set("t", {"k": 1}, "ok")
    assert c.get("t", {"k": 1}) == "ok"


# ---- invalidate / clear ---------------------------------------------------


def test_invalidate_drops_entry_and_reports():
    c = ToolCache()
    c.set("t", {"k": 1}, "ok")
    assert c.invalidate("t", {"k": 1}) is True
    assert c.invalidate("t", {"k": 1}) is False
    assert c.get("t", {"k": 1}) is None


def test_clear_resets_data_and_stats():
    c = ToolCache()
    c.set("t", {"k": 1}, "ok")
    c.get("t", {"k": 1})
    c.clear()
    assert len(c) == 0
    assert c.hits == 0 and c.misses == 0


# ---- contains ------------------------------------------------------------


def test_contains_returns_true_for_present_key_string():
    c = ToolCache()
    c.set("t", {"k": 1}, "ok")
    key = make_key("t", {"k": 1})
    assert key in c
    assert "totally-not-a-key" not in c


def test_contains_rejects_non_string():
    c = ToolCache()
    c.set("t", {"k": 1}, "ok")
    assert 42 not in c
    assert None not in c


# ---- decorator -----------------------------------------------------------


def test_wrap_caches_by_function_name_by_default():
    c = ToolCache()
    calls = []

    @c.wrap()
    def search(q: str) -> str:
        calls.append(q)
        return f"result:{q}"

    assert search("x") == "result:x"
    assert search("x") == "result:x"
    assert calls == ["x"]


def test_wrap_caches_kwargs_too():
    c = ToolCache()
    calls = []

    @c.wrap()
    def search(q, *, limit=10):
        calls.append((q, limit))
        return (q, limit)

    search("x", limit=5)
    search("x", limit=5)
    assert calls == [("x", 5)]


def test_wrap_distinguishes_positional_and_keyword():
    c = ToolCache()
    calls = []

    @c.wrap()
    def f(a, b=2):
        calls.append((a, b))
        return (a, b)

    f(1, 2)
    f(1, b=2)  # different call signature → different key
    assert len(calls) == 2


def test_wrap_with_explicit_tool_name_namespace():
    c = ToolCache()

    @c.wrap(tool_name="custom-search")
    def search(q):
        return f"got:{q}"

    search("a")
    # Hitting via direct API with matching key_args should hit too:
    direct = c.get_or_set(
        "custom-search",
        {"args": ["a"], "kwargs": {}},
        lambda: "BAD",
    )
    assert direct == "got:a"
    assert c.hits == 1


# ---- stats snapshot -------------------------------------------------------


def test_stats_returns_a_copy():
    c = ToolCache()
    c.set("t", {"k": 1}, "ok")
    c.get("t", {"k": 1})
    s1 = c.stats
    c.get("t", {"k": 1})
    s2 = c.stats
    assert isinstance(s1, CacheStats)
    assert s1.hits == 1
    assert s2.hits == 2
    # snapshot didn't mutate
    assert s1.hits == 1


# ---- set replace / eviction accounting -----------------------------------


def test_set_replacing_existing_key_does_not_grow_or_evict():
    c = ToolCache(max_size=2)
    c.set("t", {"k": 1}, "a")
    c.set("t", {"k": 2}, "b")
    c.set("t", {"k": 1}, "a2")  # replace existing key, not a new entry
    assert len(c) == 2
    assert c.evictions == 0
    assert c.get("t", {"k": 1}) == "a2"


def test_set_replacing_promotes_to_most_recent():
    c = ToolCache(max_size=2)
    c.set("t", {"k": 1}, "a")
    c.set("t", {"k": 2}, "b")
    c.set("t", {"k": 1}, "a2")  # re-set {k:1} → it becomes most-recent
    c.set("t", {"k": 3}, "c")  # should evict {k:2}, the oldest
    assert c.get("t", {"k": 2}) is None
    assert c.get("t", {"k": 1}) == "a2"


# ---- get_or_set failure semantics ----------------------------------------


def test_get_or_set_does_not_cache_when_compute_raises():
    c = ToolCache()

    def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        c.get_or_set("t", {"k": 1}, boom)
    assert len(c) == 0
    assert c.misses == 1
    # a later successful compute must run (not return a stale/poisoned value)
    assert c.get_or_set("t", {"k": 1}, lambda: "ok") == "ok"


# ---- expirations stat -----------------------------------------------------


def test_expirations_stat_and_snapshot():
    clock = iter([0.0, 6.0])
    c = ToolCache(ttl_s=5.0, clock=lambda: next(clock))
    c.set("t", {"k": 1}, "ok")
    assert c.get("t", {"k": 1}) is None
    assert c.expirations == 1
    assert c.stats.expirations == 1


# ---- decorator per-call TTL ----------------------------------------------


def test_wrap_respects_ttl_expiry():
    # clock calls: set-expiry(0.0), get_or_set hit-check(2.0 → expired) then
    # recompute-expiry(2.0); second call hit-check before expiry not needed.
    clock = iter([0.0, 2.0, 2.0])
    c = ToolCache(clock=lambda: next(clock))
    calls = []

    @c.wrap(ttl_s=1.0)
    def search(q):
        calls.append(q)
        return f"r:{q}"

    assert search("x") == "r:x"  # miss → compute, expiry at 1.0
    assert search("x") == "r:x"  # clock=2.0 ≥ 1.0 → expired → recompute
    assert calls == ["x", "x"]
    assert c.expirations == 1
