"""
Production Performance Testing Suite
=====================================

Tests system performance with realistic data volumes and production scenarios.
Validates response times, throughput, and resource utilization.

Author: Production Validation Specialist
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

from src.dashboard.orchestrator import (
    PipelineConfig,
    DataPipeline,
    AnalysisOrchestrator,
    create_orchestrator
)


class TestProductionPerformance:
    """Performance tests with production-realistic data volumes."""

    @pytest.fixture
    def production_data(self, tmp_path):
        """Generate production-sized dataset (100 stores × 4 years)."""
        np.random.seed(42)

        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        stores = range(1, 101)  # 100 stores

        records = []
        for store in stores:
            for date in dates:
                records.append({
                    'Store': store,
                    'Date': date,
                    'Sales': np.random.randint(5000, 50000),
                    'Customers': np.random.randint(500, 5000),
                    'Promo': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], p=[0.9, 0.03, 0.04, 0.03]),
                    'SchoolHoliday': np.random.choice([0, 1], p=[0.8, 0.2])
                })

        df = pd.DataFrame(records)
        data_file = tmp_path / "production_data.csv"
        df.to_csv(data_file, index=False)

        return str(data_file), len(df)

    @pytest.fixture
    def production_config(self, tmp_path, production_data):
        """Production-like configuration."""
        data_file, _ = production_data
        return PipelineConfig(
            data_path=data_file,
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            cache_ttl=3600,
            incremental_enabled=True,
            max_workers=4,
            chunk_size=10000,
            log_level="WARNING"
        )

    def test_initial_load_performance(self, production_config, production_data):
        """Test: Initial data load completes within 10 seconds."""
        _, expected_rows = production_data

        start_time = time.time()
        pipeline = DataPipeline(production_config)
        df = pipeline.load_data()
        load_time = time.time() - start_time

        assert len(df) == expected_rows, "All records loaded"
        assert load_time < 10.0, f"Load time {load_time:.2f}s exceeds 10s limit"

        print(f"✓ Loaded {expected_rows:,} records in {load_time:.2f}s")

    def test_preprocessing_performance(self, production_config, production_data):
        """Test: Data preprocessing completes within 15 seconds."""
        pipeline = DataPipeline(production_config)
        df = pipeline.load_data()

        start_time = time.time()
        processed_df = pipeline.preprocess_data(df)
        preprocess_time = time.time() - start_time

        assert preprocess_time < 15.0, f"Preprocessing time {preprocess_time:.2f}s exceeds 15s limit"
        assert 'Year' in processed_df.columns, "Derived features created"
        assert processed_df.isnull().sum().sum() == 0, "No null values remain"

        print(f"✓ Preprocessed {len(processed_df):,} records in {preprocess_time:.2f}s")

    def test_query_response_time(self, production_config):
        """Test: Filtered queries return within 2 seconds."""
        orchestrator = AnalysisOrchestrator(production_config)
        orchestrator.initialize()

        # Register test analysis module
        def test_analysis(data, **kwargs):
            return {
                'total_sales': data['Sales'].sum(),
                'avg_sales': data['Sales'].mean(),
                'store_count': data['Store'].nunique()
            }

        orchestrator.register_module('test', test_analysis)

        # Test various query patterns
        test_cases = [
            {'store_ids': [1, 2, 3], 'start_date': '2023-01-01', 'end_date': '2023-12-31'},
            {'store_ids': list(range(1, 51)), 'start_date': '2023-06-01', 'end_date': '2023-12-31'},
            {'start_date': '2023-01-01', 'end_date': '2023-03-31'},
        ]

        for query in test_cases:
            start_time = time.time()
            result = orchestrator.run_analysis('test', **query)
            query_time = time.time() - start_time

            assert query_time < 2.0, f"Query time {query_time:.2f}s exceeds 2s limit"
            assert 'total_sales' in result, "Analysis completed"

            print(f"✓ Query completed in {query_time:.3f}s: {query}")

    def test_cache_performance(self, production_config):
        """Test: Cached queries return in <100ms."""
        orchestrator = AnalysisOrchestrator(production_config)
        orchestrator.initialize()

        def test_analysis(data, **kwargs):
            return {'total_sales': data['Sales'].sum()}

        orchestrator.register_module('test', test_analysis)

        # First query (uncached)
        params = {'store_ids': [1, 2, 3], 'start_date': '2023-01-01'}
        first_time = time.time()
        orchestrator.run_analysis('test', **params)
        first_duration = time.time() - first_time

        # Second query (cached)
        second_time = time.time()
        orchestrator.run_analysis('test', **params)
        second_duration = time.time() - second_time

        assert second_duration < 0.1, f"Cached query {second_duration:.3f}s exceeds 100ms"
        assert second_duration < first_duration * 0.1, "Cache provides 10x speedup"

        print(f"✓ First query: {first_duration:.3f}s, Cached: {second_duration:.3f}s")

    def test_memory_usage(self, production_config, production_data):
        """Test: Memory usage stays under 500MB for production dataset."""
        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Load and process data
        orchestrator = AnalysisOrchestrator(production_config)
        orchestrator.initialize()

        # Run multiple analyses
        def test_analysis(data, **kwargs):
            return {
                'sales_stats': data['Sales'].describe().to_dict(),
                'customer_stats': data['Customers'].describe().to_dict()
            }

        orchestrator.register_module('test', test_analysis)

        for store_id in range(1, 11):
            orchestrator.run_analysis('test', store_ids=[store_id])

        # Measure peak memory
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = peak_memory - baseline_memory

        assert memory_increase < 500, f"Memory increase {memory_increase:.1f}MB exceeds 500MB limit"

        print(f"✓ Memory increase: {memory_increase:.1f}MB (baseline: {baseline_memory:.1f}MB)")

    def test_sustained_load_performance(self, production_config):
        """Test: System maintains performance under sustained load (100 queries)."""
        orchestrator = AnalysisOrchestrator(production_config)
        orchestrator.initialize()

        def test_analysis(data, **kwargs):
            return {'total_sales': data['Sales'].sum()}

        orchestrator.register_module('test', test_analysis)

        query_times = []

        for i in range(100):
            store_id = (i % 100) + 1
            start_time = time.time()
            orchestrator.run_analysis('test', store_ids=[store_id])
            query_time = time.time() - start_time
            query_times.append(query_time)

        avg_time = np.mean(query_times)
        p95_time = np.percentile(query_times, 95)
        p99_time = np.percentile(query_times, 99)

        assert avg_time < 0.5, f"Average query time {avg_time:.3f}s exceeds 500ms"
        assert p95_time < 1.0, f"P95 query time {p95_time:.3f}s exceeds 1s"
        assert p99_time < 2.0, f"P99 query time {p99_time:.3f}s exceeds 2s"

        print(f"✓ 100 queries - Avg: {avg_time:.3f}s, P95: {p95_time:.3f}s, P99: {p99_time:.3f}s")

    def test_large_result_set_performance(self, production_config):
        """Test: Queries returning large result sets complete efficiently."""
        orchestrator = AnalysisOrchestrator(production_config)
        orchestrator.initialize()

        def large_result_analysis(data, **kwargs):
            # Return aggregations for all stores and dates
            return {
                'daily_sales': data.groupby(['Date', 'Store'])['Sales'].sum().to_dict(),
                'store_totals': data.groupby('Store')['Sales'].sum().to_dict(),
                'date_totals': data.groupby('Date')['Sales'].sum().to_dict()
            }

        orchestrator.register_module('large', large_result_analysis)

        start_time = time.time()
        result = orchestrator.run_analysis('large')
        execution_time = time.time() - start_time

        assert execution_time < 20.0, f"Large query time {execution_time:.2f}s exceeds 20s limit"
        assert len(result['store_totals']) == 100, "All stores included"

        print(f"✓ Large result set query: {execution_time:.2f}s")

    def test_incremental_update_performance(self, production_config):
        """Test: Incremental updates are processed efficiently."""
        pipeline = DataPipeline(production_config)
        df = pipeline.load_data()
        processed = pipeline.preprocess_data(df)

        # Simulate incremental update
        last_update = datetime.now() - timedelta(days=30)

        start_time = time.time()
        new_data, metadata = pipeline.get_incremental_updates(last_update)
        update_time = time.time() - start_time

        assert update_time < 1.0, f"Incremental update time {update_time:.3f}s exceeds 1s"
        assert 'new_rows' in metadata, "Update metadata provided"

        print(f"✓ Incremental update: {metadata['new_rows']} rows in {update_time:.3f}s")


class TestPerformanceBenchmarks:
    """Benchmark tests for production optimization."""

    def test_throughput_benchmark(self, production_config):
        """Benchmark: Measure queries per second."""
        orchestrator = AnalysisOrchestrator(production_config)
        orchestrator.initialize()

        def quick_analysis(data, **kwargs):
            return {'count': len(data)}

        orchestrator.register_module('quick', quick_analysis)

        # Run for 10 seconds
        start_time = time.time()
        query_count = 0

        while time.time() - start_time < 10:
            store_id = (query_count % 100) + 1
            orchestrator.run_analysis('quick', store_ids=[store_id])
            query_count += 1

        duration = time.time() - start_time
        qps = query_count / duration

        print(f"✓ Throughput: {qps:.1f} queries/second ({query_count} queries in {duration:.1f}s)")
        assert qps > 10, f"Throughput {qps:.1f} QPS is below minimum 10 QPS"

    def test_cache_hit_rate(self, production_config):
        """Benchmark: Measure cache effectiveness."""
        orchestrator = AnalysisOrchestrator(production_config)
        orchestrator.initialize()

        def test_analysis(data, **kwargs):
            return {'sales': data['Sales'].sum()}

        orchestrator.register_module('test', test_analysis)

        # Run queries with repetition
        queries = [{'store_ids': [i]} for i in range(1, 11)] * 5  # Each query repeated 5 times

        query_times = []
        for query in queries:
            start = time.time()
            orchestrator.run_analysis('test', **query)
            query_times.append(time.time() - start)

        # First 10 should be slower (cache misses)
        # Remaining 40 should be faster (cache hits)
        avg_uncached = np.mean(query_times[:10])
        avg_cached = np.mean(query_times[10:])

        speedup = avg_uncached / avg_cached

        print(f"✓ Cache speedup: {speedup:.1f}x (uncached: {avg_uncached:.3f}s, cached: {avg_cached:.3f}s)")
        assert speedup > 5, f"Cache speedup {speedup:.1f}x is below expected 5x"


@pytest.mark.slow
class TestScalabilityLimits:
    """Test system limits and scalability boundaries."""

    def test_maximum_dataset_size(self, tmp_path):
        """Test: System handles maximum realistic dataset (200 stores × 5 years)."""
        np.random.seed(42)

        dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='D')
        stores = range(1, 201)  # 200 stores

        records = []
        for store in stores:
            for date in dates:
                records.append({
                    'Store': store,
                    'Date': date,
                    'Sales': np.random.randint(5000, 50000),
                    'Customers': np.random.randint(500, 5000),
                    'Promo': np.random.choice([0, 1]),
                    'StateHoliday': '0',
                    'SchoolHoliday': 0
                })

        df = pd.DataFrame(records)
        data_file = tmp_path / "max_data.csv"
        df.to_csv(data_file, index=False)

        config = PipelineConfig(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

        start_time = time.time()
        orchestrator = AnalysisOrchestrator(config)
        orchestrator.initialize()
        init_time = time.time() - start_time

        assert init_time < 30.0, f"Initialization time {init_time:.2f}s exceeds 30s limit"
        assert orchestrator.pipeline.processed_data is not None

        print(f"✓ Initialized {len(records):,} records in {init_time:.2f}s")

    def test_concurrent_query_limit(self, production_config):
        """Test: System handles 10 concurrent queries."""
        orchestrator = AnalysisOrchestrator(production_config)
        orchestrator.initialize()

        def test_analysis(data, **kwargs):
            # Simulate some processing
            time.sleep(0.1)
            return {'sales': data['Sales'].sum()}

        orchestrator.register_module('test', test_analysis)

        # Execute 10 concurrent queries
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                future = executor.submit(
                    orchestrator.run_analysis,
                    'test',
                    store_ids=[i + 1]
                )
                futures.append(future)

            results = [f.result() for f in as_completed(futures)]

        total_time = time.time() - start_time

        assert len(results) == 10, "All queries completed"
        assert total_time < 5.0, f"Concurrent execution time {total_time:.2f}s exceeds 5s"

        print(f"✓ 10 concurrent queries completed in {total_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
