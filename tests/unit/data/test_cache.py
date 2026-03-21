"""Tests for data cache module."""

import threading
import time

from ashare_analyzer.data.cache import TTLCache


class TestTTLCache:
    """Test TTLCache class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        cache = TTLCache()
        assert cache._default_ttl == 600
        assert cache._cache.maxsize == 1000

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        cache = TTLCache(default_ttl=300, max_size=500)
        assert cache._default_ttl == 300

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = TTLCache(default_ttl=60)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_get_nonexistent_key(self):
        """Test getting a key that doesn't exist."""
        cache = TTLCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_delete_key(self):
        """Test deleting a key."""
        cache = TTLCache()

        cache.set("key1", "value1")
        cache.delete("key1")

        assert cache.get("key1") is None

    def test_clear_cache(self):
        """Test clearing all cache entries."""
        cache = TTLCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_custom_ttl(self):
        """Test setting a custom TTL for an entry."""
        cache = TTLCache(default_ttl=60)

        cache.set("key1", "value1", ttl=1)  # 1 second TTL

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_is_valid_existing_key(self):
        """Test is_valid for existing key."""
        cache = TTLCache()

        cache.set("key1", "value1")

        assert cache.is_valid("key1") is True

    def test_is_valid_nonexistent_key(self):
        """Test is_valid for nonexistent key."""
        cache = TTLCache()

        assert cache.is_valid("nonexistent") is False

    def test_is_valid_expired_key(self):
        """Test is_valid for expired key."""
        cache = TTLCache()

        cache.set("key1", "value1", ttl=1)
        time.sleep(1.1)

        assert cache.is_valid("key1") is False


class TestTTLCachePatternInvalidation:
    """Test pattern-based cache invalidation."""

    def test_invalidate_all(self):
        """Test invalidating all entries."""
        cache = TTLCache()

        cache.set("stock:600519", "data1")
        cache.set("stock:000001", "data2")
        cache.set("news:600519", "news1")

        cache.invalidate()

        assert cache.get("stock:600519") is None
        assert cache.get("stock:000001") is None
        assert cache.get("news:600519") is None

    def test_invalidate_by_pattern(self):
        """Test invalidating entries by pattern."""
        cache = TTLCache()

        cache.set("stock:600519", "data1")
        cache.set("stock:000001", "data2")
        cache.set("news:600519", "news1")

        cache.invalidate("stock:*")

        assert cache.get("stock:600519") is None
        assert cache.get("stock:000001") is None
        assert cache.get("news:600519") == "news1"

    def test_invalidate_specific_pattern(self):
        """Test invalidating specific pattern."""
        cache = TTLCache()

        cache.set("stock:600519:daily", "daily")
        cache.set("stock:600519:realtime", "realtime")
        cache.set("stock:000001:daily", "daily2")

        cache.invalidate("stock:600519:*")

        assert cache.get("stock:600519:daily") is None
        assert cache.get("stock:600519:realtime") is None
        assert cache.get("stock:000001:daily") == "daily2"


class TestTTLCacheThreadSafety:
    """Test thread safety of TTLCache."""

    def test_concurrent_set_and_get(self):
        """Test concurrent set and get operations."""
        cache = TTLCache()
        errors = []
        iterations = 100

        def writer(start: int):
            try:
                for i in range(iterations):
                    cache.set(f"key_{start}_{i}", f"value_{start}_{i}")
            except Exception as e:
                errors.append(e)

        def reader(start: int):
            try:
                for i in range(iterations):
                    cache.get(f"key_{start}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_invalidate(self):
        """Test concurrent invalidation and access."""
        cache = TTLCache()
        errors = []

        def invalidate():
            try:
                for _ in range(50):
                    cache.invalidate("test:*")
            except Exception as e:
                errors.append(e)

        def access():
            try:
                for i in range(50):
                    cache.set(f"test_{i}", f"value_{i}")
                    cache.get(f"test_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=invalidate),
            threading.Thread(target=access),
            threading.Thread(target=access),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestTTLCacheCachedDecorator:
    """Test the cached decorator."""

    def test_cached_decorator_basic(self):
        """Test basic cached decorator usage."""
        cache = TTLCache()
        call_count = 0

        @cache.cached(key_func=lambda x: f"test:{x}")
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

    def test_cached_decorator_different_args(self):
        """Test cached decorator with different arguments."""
        cache = TTLCache()
        call_count = 0

        @cache.cached(key_func=lambda x: f"test:{x}")
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(10)

        assert result1 == 10
        assert result2 == 20
        assert call_count == 2  # Both executed

    def test_cached_decorator_with_ttl(self):
        """Test cached decorator with custom TTL."""
        cache = TTLCache()
        call_count = 0

        @cache.cached(key_func=lambda x: f"test:{x}", ttl=1)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        time.sleep(1.1)

        # Should re-execute after TTL expires
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 2

    def test_cached_decorator_no_key_func(self):
        """Test cached decorator without key_func."""
        cache = TTLCache()

        @cache.cached()
        def simple_function(x: int) -> int:
            return x * 2

        result = simple_function(5)
        assert result == 10

        # Should work but key will be auto-generated
        result2 = simple_function(5)
        assert result2 == 10

    def test_cached_decorator_none_result(self):
        """Test cached decorator with None result (not cached)."""
        cache = TTLCache()
        call_count = 0

        @cache.cached(key_func=lambda: "test")
        def returns_none() -> None:
            nonlocal call_count
            call_count += 1
            return None

        # Both calls should execute since None is not cached
        returns_none()
        returns_none()

        assert call_count == 2


class TestTTLCacheEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_overwrite_existing_key(self):
        """Test overwriting an existing key."""
        cache = TTLCache()

        cache.set("key1", "value1")
        cache.set("key1", "value2")

        assert cache.get("key1") == "value2"

    def test_complex_value_types(self):
        """Test caching complex value types."""
        cache = TTLCache()

        # Dictionary
        cache.set("dict_key", {"a": 1, "b": 2})
        assert cache.get("dict_key") == {"a": 1, "b": 2}

        # List
        cache.set("list_key", [1, 2, 3])
        assert cache.get("list_key") == [1, 2, 3]

        # Nested structure
        cache.set("nested", {"data": [1, 2, {"inner": "value"}]})
        assert cache.get("nested") == {"data": [1, 2, {"inner": "value"}]}

    def test_empty_string_key(self):
        """Test using empty string as key."""
        cache = TTLCache()

        cache.set("", "empty_key_value")
        assert cache.get("") == "empty_key_value"

    def test_special_characters_in_key(self):
        """Test keys with special characters."""
        cache = TTLCache()

        cache.set("key:with:colons", "value1")
        cache.set("key-with-dashes", "value2")
        cache.set("key.with.dots", "value3")
        cache.set("key/with/slashes", "value4")

        assert cache.get("key:with:colons") == "value1"
        assert cache.get("key-with-dashes") == "value2"
        assert cache.get("key.with.dots") == "value3"
        assert cache.get("key/with/slashes") == "value4"

    def test_unicode_keys_and_values(self):
        """Test Unicode keys and values."""
        cache = TTLCache()

        cache.set("股票:600519", "贵州茅台")
        cache.set("emoji_key_🚀", "rocket_value")

        assert cache.get("股票:600519") == "贵州茅台"
        assert cache.get("emoji_key_🚀") == "rocket_value"

    def test_large_value(self):
        """Test caching large values."""
        cache = TTLCache()

        large_value = {"data": list(range(10000))}
        cache.set("large_key", large_value)

        result = cache.get("large_key")
        assert result is not None
        assert len(result["data"]) == 10000

    def test_delete_nonexistent_key(self):
        """Test deleting a key that doesn't exist."""
        cache = TTLCache()

        # Should not raise an error
        cache.delete("nonexistent_key")

    def test_zero_ttl(self):
        """Test setting with zero TTL."""
        cache = TTLCache()

        # Zero TTL should expire immediately
        cache.set("key1", "value1", ttl=0)

        # Might already be expired
        cache.get("key1")  # Result depends on timing, just verify no error
        # The behavior depends on implementation
        # Either it's expired or it exists briefly

    def test_negative_ttl(self):
        """Test setting with negative TTL."""
        cache = TTLCache()

        # Negative TTL is technically allowed but creates already-expired entry
        cache.set("key1", "value1", ttl=-1)

        # Entry should be expired immediately
        result = cache.get("key1")
        assert result is None


class TestTTLCacheExpiration:
    """Test TTL expiration behavior."""

    def test_default_ttl_expiration(self):
        """Test that default TTL is applied."""
        cache = TTLCache(default_ttl=1)

        cache.set("key1", "value1")

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_multiple_entries_different_ttls(self):
        """Test entries with different TTLs."""
        cache = TTLCache(default_ttl=10)

        cache.set("short", "value1", ttl=1)
        cache.set("long", "value2", ttl=5)

        time.sleep(1.1)

        # Short should be expired
        assert cache.get("short") is None
        # Long should still exist
        assert cache.get("long") == "value2"

    def test_is_valid_after_expiration(self):
        """Test is_valid returns False after expiration."""
        cache = TTLCache()

        cache.set("key1", "value1", ttl=1)

        assert cache.is_valid("key1") is True

        time.sleep(1.1)

        assert cache.is_valid("key1") is False
