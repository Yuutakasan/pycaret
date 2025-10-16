"""
Unit Tests for Cache Manager
=============================

Tests caching functionality including memory cache, disk cache,
TTL expiration, and cache statistics.

Author: Testing & QA Agent
Created: 2025-10-08
"""

import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.dashboard.orchestrator import CacheManager, CacheEntry, PipelineConfig


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating cache entry."""
        data = {"key": "value"}
        entry = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            hash_key="test_key",
            ttl=60
        )

        assert entry.data == data
        assert entry.hash_key == "test_key"
        assert entry.ttl == 60

    def test_cache_entry_not_expired(self):
        """Test cache entry within TTL."""
        entry = CacheEntry(
            data="test",
            timestamp=datetime.now(),
            hash_key="key",
            ttl=60
        )

        assert not entry.is_expired()

    def test_cache_entry_expired(self):
        """Test cache entry beyond TTL."""
        old_timestamp = datetime.now() - timedelta(seconds=3700)
        entry = CacheEntry(
            data="test",
            timestamp=old_timestamp,
            hash_key="key",
            ttl=3600
        )

        assert entry.is_expired()

    def test_cache_entry_edge_case(self):
        """Test cache entry at exact TTL boundary."""
        timestamp = datetime.now() - timedelta(seconds=60)
        entry = CacheEntry(
            data="test",
            timestamp=timestamp,
            hash_key="key",
            ttl=60
        )

        # At exact boundary, should be expired
        assert entry.is_expired()


class TestCacheManager:
    """Test CacheManager class."""

    def test_initialization(self, pipeline_config):
        """Test cache manager initialization."""
        cache_manager = CacheManager(pipeline_config)

        assert cache_manager.config == pipeline_config
        assert cache_manager.cache_dir.exists()
        assert isinstance(cache_manager.memory_cache, dict)
        assert len(cache_manager.memory_cache) == 0

    def test_generate_key(self, cache_manager):
        """Test cache key generation."""
        key1 = cache_manager._generate_key("arg1", "arg2", param1="value1")
        key2 = cache_manager._generate_key("arg1", "arg2", param1="value1")
        key3 = cache_manager._generate_key("arg1", "arg3", param1="value1")

        assert key1 == key2  # Same inputs = same key
        assert key1 != key3  # Different inputs = different keys
        assert len(key1) == 32  # MD5 hash length

    def test_set_and_get_memory_cache(self, cache_manager):
        """Test setting and getting from memory cache."""
        test_data = {"result": "test_value"}
        cache_manager.set("test_key", test_data)

        retrieved = cache_manager.get("test_key")
        assert retrieved == test_data

    def test_set_and_get_disk_cache(self, cache_manager):
        """Test setting and getting from disk cache."""
        test_data = {"result": "test_value"}
        cache_manager.set("test_key", test_data)

        # Clear memory cache to force disk read
        cache_manager.memory_cache.clear()

        retrieved = cache_manager.get("test_key")
        assert retrieved == test_data

    def test_cache_miss(self, cache_manager):
        """Test cache miss returns None."""
        result = cache_manager.get("nonexistent_key")
        assert result is None

    def test_cache_disabled(self, disabled_cache_config):
        """Test cache operations when disabled."""
        cache_manager = CacheManager(disabled_cache_config)

        cache_manager.set("key", "value")
        result = cache_manager.get("key")

        assert result is None  # Should not cache when disabled

    def test_cache_expiration(self, cache_manager):
        """Test cache entry expiration."""
        # Create entry with 1 second TTL
        cache_manager.set("test_key", "test_value", ttl=1)

        # Should be available immediately
        assert cache_manager.get("test_key") == "test_value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache_manager.get("test_key") is None

    def test_invalidate_all(self, cache_manager):
        """Test invalidating all cache entries."""
        # Add multiple entries
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")

        count = cache_manager.invalidate()

        assert count >= 3  # At least 3 entries invalidated
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
        assert cache_manager.get("key3") is None

    def test_invalidate_pattern(self, cache_manager):
        """Test invalidating cache entries by pattern."""
        cache_manager.set("sales_key1", "value1")
        cache_manager.set("sales_key2", "value2")
        cache_manager.set("customers_key", "value3")

        count = cache_manager.invalidate(pattern="sales")

        assert count == 2
        assert cache_manager.get("sales_key1") is None
        assert cache_manager.get("sales_key2") is None
        assert cache_manager.get("customers_key") == "value3"

    def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")

        stats = cache_manager.get_stats()

        assert "memory_entries" in stats
        assert "disk_entries" in stats
        assert "total_size_mb" in stats
        assert "cache_dir" in stats

        assert stats["memory_entries"] >= 2
        assert stats["disk_entries"] >= 2

    def test_large_data_caching(self, cache_manager):
        """Test caching large data structures."""
        import numpy as np

        large_data = {
            "array": np.random.rand(1000, 1000),
            "list": list(range(10000)),
            "dict": {f"key_{i}": f"value_{i}" for i in range(1000)}
        }

        cache_manager.set("large_key", large_data)
        retrieved = cache_manager.get("large_key")

        assert retrieved is not None
        assert isinstance(retrieved["array"], np.ndarray)
        assert len(retrieved["list"]) == 10000
        assert len(retrieved["dict"]) == 1000

    def test_cache_corruption_handling(self, cache_manager, temp_dir):
        """Test handling of corrupted cache files."""
        # Create corrupted cache file
        corrupted_file = cache_manager.cache_dir / "corrupted_key.pkl"
        corrupted_file.write_text("This is not valid pickle data")

        # Should handle gracefully
        result = cache_manager.get("corrupted_key")
        assert result is None
        assert not corrupted_file.exists()  # Should be cleaned up

    def test_concurrent_cache_access(self, cache_manager):
        """Test thread-safe cache access."""
        from concurrent.futures import ThreadPoolExecutor

        def set_and_get(index):
            key = f"key_{index}"
            value = f"value_{index}"
            cache_manager.set(key, value)
            return cache_manager.get(key)

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(set_and_get, range(50)))

        assert len(results) == 50
        assert all(r is not None for r in results)

    def test_custom_ttl(self, cache_manager):
        """Test custom TTL values."""
        cache_manager.set("short_ttl", "value1", ttl=1)
        cache_manager.set("long_ttl", "value2", ttl=3600)

        time.sleep(1.1)

        assert cache_manager.get("short_ttl") is None
        assert cache_manager.get("long_ttl") == "value2"

    def test_cache_key_collision_resistance(self, cache_manager):
        """Test that different inputs produce different keys."""
        keys = set()

        for i in range(100):
            key = cache_manager._generate_key(f"arg_{i}", param=i)
            assert key not in keys
            keys.add(key)

        assert len(keys) == 100

    def test_memory_cache_priority(self, cache_manager):
        """Test that memory cache is checked before disk cache."""
        # Set value in cache
        cache_manager.set("test_key", "original_value")

        # Modify memory cache directly
        for entry in cache_manager.memory_cache.values():
            if entry.hash_key == "test_key":
                entry.data = "modified_value"
                break

        # Should get modified value from memory
        result = cache_manager.get("test_key")
        assert result == "modified_value"


class TestCachedDecorator:
    """Test @cached decorator functionality."""

    def test_decorator_caches_result(self, pipeline_config):
        """Test that decorator caches function results."""
        from src.dashboard.orchestrator import cached

        call_count = {"count": 0}

        class TestClass:
            def __init__(self):
                self.cache_manager = CacheManager(pipeline_config)

            @cached(ttl=60)
            def expensive_operation(self, x):
                call_count["count"] += 1
                return x * 2

        obj = TestClass()

        # First call
        result1 = obj.expensive_operation(5)
        assert result1 == 10
        assert call_count["count"] == 1

        # Second call - should use cache
        result2 = obj.expensive_operation(5)
        assert result2 == 10
        assert call_count["count"] == 1  # Not incremented

    def test_decorator_different_args(self, pipeline_config):
        """Test decorator with different arguments."""
        from src.dashboard.orchestrator import cached

        class TestClass:
            def __init__(self):
                self.cache_manager = CacheManager(pipeline_config)

            @cached()
            def operation(self, x, y):
                return x + y

        obj = TestClass()

        assert obj.operation(1, 2) == 3
        assert obj.operation(2, 3) == 5
        assert obj.operation(1, 2) == 3  # Cached

    def test_decorator_without_cache_manager(self):
        """Test decorator on object without cache_manager."""
        from src.dashboard.orchestrator import cached

        call_count = {"count": 0}

        class TestClass:
            @cached()
            def operation(self, x):
                call_count["count"] += 1
                return x * 2

        obj = TestClass()

        result1 = obj.operation(5)
        result2 = obj.operation(5)

        # Should call function both times (no caching)
        assert result1 == 10
        assert result2 == 10
        assert call_count["count"] == 2
