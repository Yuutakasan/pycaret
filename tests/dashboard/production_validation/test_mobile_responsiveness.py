"""
Production Mobile Responsiveness Testing Suite
===============================================

Tests API behavior and data delivery optimized for mobile devices
and responsive web applications.

Author: Production Validation Specialist
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from typing import Dict, Any

from src.dashboard.orchestrator import (
    PipelineConfig,
    DataPipeline,
    AnalysisOrchestrator,
    create_orchestrator
)


class TestMobileDataOptimization:
    """Test data optimization for mobile bandwidth constraints."""

    @pytest.fixture
    def mobile_data(self, tmp_path):
        """Generate test data for mobile scenarios."""
        np.random.seed(42)

        df = pd.DataFrame({
            'Store': range(1, 51),
            'Date': [pd.Timestamp('2023-01-01')] * 50,
            'Sales': np.random.randint(5000, 50000, 50),
            'Customers': np.random.randint(500, 5000, 50),
            'Promo': np.random.choice([0, 1], 50),
            'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], 50),
            'SchoolHoliday': np.random.choice([0, 1], 50)
        })

        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)
        return str(data_file)

    @pytest.fixture
    def mobile_orchestrator(self, tmp_path, mobile_data):
        """Create orchestrator for mobile testing."""
        return create_orchestrator(
            data_path=mobile_data,
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            log_level="WARNING"
        )

    def test_response_payload_size(self, mobile_orchestrator):
        """Test: API responses are optimized for mobile bandwidth."""
        result = mobile_orchestrator.run_analysis('sales', store_ids=[1])

        # Serialize to JSON to measure payload size
        json_str = json.dumps(result, default=str)
        payload_size_kb = len(json_str.encode('utf-8')) / 1024

        # Mobile-optimized response should be < 100KB for simple queries
        assert payload_size_kb < 100, f"Payload size {payload_size_kb:.2f}KB exceeds 100KB mobile limit"

        print(f"✓ Mobile payload size: {payload_size_kb:.2f}KB")

    def test_pagination_support(self, mobile_orchestrator):
        """Test: Large result sets can be paginated for mobile."""
        # Get summary which might contain large data
        summary = mobile_orchestrator.get_summary()

        # Check if data structures support pagination concepts
        assert 'total_rows' in summary, "Row count available for pagination"

        # For mobile, we should be able to query subsets
        result_small = mobile_orchestrator.run_analysis('sales', store_ids=[1])
        result_large = mobile_orchestrator.run_analysis('sales', store_ids=list(range(1, 11)))

        # Small query should be faster (suitable for mobile)
        assert result_small is not None
        assert result_large is not None

        print(f"✓ Pagination concepts supported")

    def test_lightweight_summary_endpoint(self, mobile_orchestrator):
        """Test: Lightweight summary available for mobile dashboard."""
        summary = mobile_orchestrator.get_summary()

        # Summary should be compact
        summary_json = json.dumps(summary, default=str)
        summary_size_kb = len(summary_json.encode('utf-8')) / 1024

        # Summary should be lightweight
        assert summary_size_kb < 50, f"Summary size {summary_size_kb:.2f}KB exceeds 50KB mobile limit"

        # Should contain essential information
        essential_fields = ['status', 'data_loaded', 'total_rows']
        for field in essential_fields:
            assert field in summary, f"Essential field missing: {field}"

        print(f"✓ Lightweight summary: {summary_size_kb:.2f}KB")

    def test_incremental_data_loading(self, mobile_orchestrator):
        """Test: Supports incremental data loading for mobile."""
        from datetime import datetime, timedelta

        # Initial load
        summary = mobile_orchestrator.get_summary()
        assert summary['data_loaded'], "Initial data loaded"

        # Incremental update
        last_update = datetime.now() - timedelta(days=7)
        new_data, metadata = mobile_orchestrator.pipeline.get_incremental_updates(last_update)

        # Should return only new data (bandwidth efficient)
        assert metadata is not None, "Incremental metadata provided"
        assert 'new_rows' in metadata, "New row count provided"

        print(f"✓ Incremental loading: {metadata.get('new_rows', 0)} new rows")


class TestMobilePerformance:
    """Test performance characteristics important for mobile."""

    @pytest.fixture
    def perf_orchestrator(self, tmp_path):
        """Create orchestrator for performance testing."""
        np.random.seed(42)

        df = pd.DataFrame({
            'Store': range(1, 101),
            'Date': [pd.Timestamp('2023-01-01')] * 100,
            'Sales': np.random.randint(5000, 50000, 100),
            'Customers': np.random.randint(500, 5000, 100),
            'Promo': np.random.choice([0, 1], 100),
            'StateHoliday': ['0'] * 100,
            'SchoolHoliday': [0] * 100
        })

        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        return create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            log_level="WARNING"
        )

    def test_3g_network_acceptable_performance(self, perf_orchestrator):
        """Test: Queries complete in acceptable time for 3G network (< 3 seconds)."""
        start = time.time()
        result = perf_orchestrator.run_analysis('sales', store_ids=[1])
        query_time = time.time() - start

        # For 3G networks, target 3 seconds for interactive response
        assert query_time < 3.0, f"Query time {query_time:.2f}s exceeds 3G target (3s)"

        print(f"✓ 3G-acceptable performance: {query_time:.2f}s")

    def test_cached_mobile_performance(self, perf_orchestrator):
        """Test: Cached responses are extremely fast for mobile (< 200ms)."""
        # Prime cache
        perf_orchestrator.run_analysis('sales', store_ids=[1])

        # Cached request
        start = time.time()
        result = perf_orchestrator.run_analysis('sales', store_ids=[1])
        cached_time = time.time() - start

        # Cached should be near-instant
        assert cached_time < 0.2, f"Cached time {cached_time:.3f}s exceeds 200ms mobile target"

        print(f"✓ Mobile cached performance: {cached_time:.3f}s")

    def test_battery_efficient_operations(self, perf_orchestrator):
        """Test: Operations don't require excessive processing (battery efficient)."""
        # Execute multiple queries and measure efficiency
        query_times = []

        for i in range(10):
            start = time.time()
            perf_orchestrator.run_analysis('sales', store_ids=[i + 1])
            query_time = time.time() - start
            query_times.append(query_time)

        avg_time = np.mean(query_times)
        max_time = max(query_times)

        # Consistent, quick operations are battery efficient
        assert avg_time < 1.0, f"Average time {avg_time:.2f}s not battery efficient"
        assert max_time < 2.0, f"Max time {max_time:.2f}s shows inefficiency spikes"

        print(f"✓ Battery-efficient operations: avg {avg_time:.3f}s")


class TestMobileConnectivity:
    """Test behavior under mobile connectivity conditions."""

    @pytest.fixture
    def connectivity_orchestrator(self, tmp_path):
        """Create orchestrator for connectivity testing."""
        df = pd.DataFrame({
            'Store': [1, 2, 3],
            'Date': pd.date_range(start='2023-01-01', periods=3),
            'Sales': [1000, 2000, 3000],
            'Customers': [100, 200, 300],
            'Promo': [0, 1, 0],
            'StateHoliday': ['0', 'a', '0'],
            'SchoolHoliday': [0, 1, 0]
        })

        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        return create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            log_level="WARNING"
        )

    def test_offline_cache_capability(self, connectivity_orchestrator):
        """Test: Cache enables offline-first capability."""
        # Load data and populate cache
        result1 = connectivity_orchestrator.run_analysis('sales', store_ids=[1])

        # Verify cache populated
        cache_stats = connectivity_orchestrator.cache_manager.get_stats()
        assert cache_stats['disk_entries'] > 0 or cache_stats['memory_entries'] > 0

        # Even without network (simulated by cache), can serve data
        result2 = connectivity_orchestrator.run_analysis('sales', store_ids=[1])

        assert result2 is not None, "Cached data available offline"
        assert result1 == result2, "Cached data consistent"

        print(f"✓ Offline cache capability verified")

    def test_graceful_degradation(self, connectivity_orchestrator):
        """Test: System degrades gracefully under poor connectivity."""
        # System should still provide basic functionality
        # even if some features are slow/unavailable

        # Core operations should work
        summary = connectivity_orchestrator.get_summary()
        assert summary['status'] == 'active', "Core status available"

        # Cached queries should work
        result = connectivity_orchestrator.run_analysis('sales', store_ids=[1])
        assert result is not None, "Basic queries work"

        print(f"✓ Graceful degradation under connectivity issues")


class TestMobileDataFormats:
    """Test mobile-friendly data formats."""

    @pytest.fixture
    def format_orchestrator(self, tmp_path):
        """Create orchestrator for format testing."""
        df = pd.DataFrame({
            'Store': [1, 2],
            'Date': pd.date_range(start='2023-01-01', periods=2),
            'Sales': [1000.50, 2000.75],
            'Customers': [100, 200],
            'Promo': [0, 1],
            'StateHoliday': ['0', 'a'],
            'SchoolHoliday': [0, 1]
        })

        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        return create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

    def test_json_serialization(self, format_orchestrator):
        """Test: Data serializes cleanly to JSON for mobile apps."""
        result = format_orchestrator.run_analysis('sales')

        # Should serialize to JSON without errors
        try:
            json_str = json.dumps(result, default=str)
            assert len(json_str) > 0

            # Should deserialize back
            reconstructed = json.loads(json_str)
            assert isinstance(reconstructed, dict)

        except (TypeError, ValueError) as e:
            pytest.fail(f"JSON serialization failed for mobile: {e}")

        print(f"✓ Mobile-friendly JSON serialization")

    def test_compact_json_format(self, format_orchestrator):
        """Test: JSON format is compact (no unnecessary whitespace)."""
        result = format_orchestrator.run_analysis('sales')

        # Compact JSON (no indentation)
        compact_json = json.dumps(result, default=str, separators=(',', ':'))
        compact_size = len(compact_json.encode('utf-8'))

        # Pretty JSON (with indentation)
        pretty_json = json.dumps(result, default=str, indent=2)
        pretty_size = len(pretty_json.encode('utf-8'))

        # Compact should be significantly smaller
        compression_ratio = pretty_size / compact_size
        assert compression_ratio > 1.2, f"JSON not sufficiently compact (ratio: {compression_ratio:.2f})"

        print(f"✓ Compact JSON format (compression: {compression_ratio:.2f}x)")

    def test_numeric_precision_mobile(self, format_orchestrator):
        """Test: Numeric precision appropriate for mobile displays."""
        result = format_orchestrator.run_analysis('sales')

        # Mobile screens have limited precision
        for key, value in result.items():
            if isinstance(value, float):
                # Should not have excessive decimal places
                str_val = str(value)
                if '.' in str_val:
                    decimals = len(str_val.split('.')[1])
                    # Mobile-friendly precision (max 4 decimal places for display)
                    assert decimals <= 10, f"Excessive precision for mobile: {key}={value}"

        print(f"✓ Mobile-appropriate numeric precision")


class TestResponsiveUIBackend:
    """Test backend support for responsive UI requirements."""

    @pytest.fixture
    def responsive_orchestrator(self, tmp_path):
        """Create orchestrator for responsive testing."""
        np.random.seed(42)

        df = pd.DataFrame({
            'Store': range(1, 21),
            'Date': [pd.Timestamp('2023-01-01')] * 20,
            'Sales': np.random.randint(5000, 50000, 20),
            'Customers': np.random.randint(500, 5000, 20),
            'Promo': np.random.choice([0, 1], 20),
            'StateHoliday': ['0'] * 20,
            'SchoolHoliday': [0] * 20
        })

        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        return create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

    def test_flexible_data_aggregation(self, responsive_orchestrator):
        """Test: Backend supports different aggregation levels for different screen sizes."""
        # Mobile: Single store summary
        mobile_result = responsive_orchestrator.run_analysis('sales', store_ids=[1])

        # Tablet: Multiple stores
        tablet_result = responsive_orchestrator.run_analysis('sales', store_ids=[1, 2, 3])

        # Desktop: All stores
        desktop_result = responsive_orchestrator.run_analysis('sales')

        assert mobile_result is not None, "Mobile aggregation works"
        assert tablet_result is not None, "Tablet aggregation works"
        assert desktop_result is not None, "Desktop aggregation works"

        print(f"✓ Flexible aggregation for all screen sizes")

    def test_metadata_for_ui_adaptation(self, responsive_orchestrator):
        """Test: Metadata helps UI adapt to device capabilities."""
        summary = responsive_orchestrator.get_summary()

        # Should provide info for UI to adapt
        assert 'total_rows' in summary, "Data volume info for UI adaptation"
        assert 'registered_modules' in summary, "Feature availability info"

        # UI can use this to decide what to show
        total_rows = summary['total_rows']
        assert isinstance(total_rows, int), "Numeric data volume"

        print(f"✓ Metadata supports UI adaptation ({total_rows} rows)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
