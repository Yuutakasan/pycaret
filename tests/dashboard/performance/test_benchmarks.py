"""
Performance Benchmark Tests
===========================

Tests performance characteristics, benchmarks,
and resource usage of dashboard components.

Author: Testing & QA Agent
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.dashboard.orchestrator import (
    create_orchestrator,
    DataPipeline,
    PipelineConfig,
    CacheManager
)


@pytest.mark.performance
class TestDataLoadingPerformance:
    """Benchmark data loading operations."""

    def test_load_time_small_dataset(self, temp_data_file, temp_dir, performance_timer):
        """Benchmark loading small dataset."""
        config = PipelineConfig(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache"),
            cache_enabled=False
        )
        pipeline = DataPipeline(config)

        with performance_timer() as timer:
            df = pipeline.load_data()

        assert timer.elapsed_ms < 1000  # Should load in <1s
        assert len(df) > 0

    @pytest.mark.slow
    def test_load_time_large_dataset(self, temp_dir, large_dataset, performance_timer):
        """Benchmark loading large dataset."""
        data_file = temp_dir / "large.csv"
        large_dataset.to_csv(data_file, index=False)

        config = PipelineConfig(
            data_path=str(data_file),
            cache_dir=str(temp_dir / "cache"),
            cache_enabled=False
        )
        pipeline = DataPipeline(config)

        with performance_timer() as timer:
            df = pipeline.load_data()

        # Should load within reasonable time
        assert timer.elapsed_seconds < 10
        assert len(df) > 100000

    def test_preprocess_time(self, data_pipeline, sample_data, performance_timer):
        """Benchmark preprocessing time."""
        with performance_timer() as timer:
            df = data_pipeline.preprocess_data(sample_data)

        assert timer.elapsed_ms < 500
        assert len(df) > 0

    @pytest.mark.slow
    def test_preprocess_large_dataset(self, temp_dir, large_dataset, performance_timer):
        """Benchmark preprocessing large dataset."""
        data_file = temp_dir / "large.csv"
        large_dataset.to_csv(data_file, index=False)

        config = PipelineConfig(
            data_path=str(data_file),
            cache_dir=str(temp_dir / "cache")
        )
        pipeline = DataPipeline(config)
        pipeline.load_data()

        with performance_timer() as timer:
            df = pipeline.preprocess_data(pipeline.raw_data)

        assert timer.elapsed_seconds < 15
        assert len(df) > 100000


@pytest.mark.performance
class TestCachePerformance:
    """Benchmark caching operations."""

    def test_cache_write_performance(self, cache_manager, performance_timer):
        """Benchmark cache write performance."""
        test_data = {"value": list(range(10000))}

        with performance_timer() as timer:
            for i in range(100):
                cache_manager.set(f"key_{i}", test_data)

        assert timer.elapsed_ms < 1000  # 100 writes in <1s

    def test_cache_read_performance(self, cache_manager, performance_timer):
        """Benchmark cache read performance."""
        # Populate cache
        test_data = {"value": list(range(10000))}
        for i in range(100):
            cache_manager.set(f"key_{i}", test_data)

        with performance_timer() as timer:
            for i in range(100):
                cache_manager.get(f"key_{i}")

        assert timer.elapsed_ms < 100  # 100 reads in <100ms

    def test_cache_hit_vs_miss(self, cache_manager, performance_timer):
        """Compare cache hit vs miss performance."""
        test_data = {"value": list(range(10000))}
        cache_manager.set("test_key", test_data)

        # Measure cache hit
        with performance_timer() as hit_timer:
            for _ in range(100):
                cache_manager.get("test_key")
        hit_time = hit_timer.elapsed_ms

        # Measure cache miss
        with performance_timer() as miss_timer:
            for _ in range(100):
                cache_manager.get("nonexistent_key")
        miss_time = miss_timer.elapsed_ms

        # Hits should be faster than misses
        assert hit_time < miss_time

    def test_memory_vs_disk_cache(self, cache_manager, performance_timer):
        """Compare memory vs disk cache performance."""
        test_data = {"value": list(range(10000))}
        cache_manager.set("test_key", test_data)

        # Memory cache (first read)
        with performance_timer() as mem_timer:
            cache_manager.get("test_key")
        mem_time = mem_timer.elapsed_ms

        # Clear memory, force disk read
        cache_manager.memory_cache.clear()

        with performance_timer() as disk_timer:
            cache_manager.get("test_key")
        disk_time = disk_timer.elapsed_ms

        # Memory should be faster
        assert mem_time < disk_time


@pytest.mark.performance
class TestAnalysisPerformance:
    """Benchmark analysis operations."""

    def test_single_analysis_time(self, initialized_orchestrator, performance_timer):
        """Benchmark single analysis execution."""
        with performance_timer() as timer:
            result = initialized_orchestrator.run_analysis("sales")

        assert timer.elapsed_ms < 500
        assert result is not None

    def test_batch_analysis_time(self, initialized_orchestrator, performance_timer):
        """Benchmark batch analysis execution."""
        with performance_timer() as timer:
            results = initialized_orchestrator.run_all_analyses()

        assert timer.elapsed_seconds < 2
        assert len(results) >= 3

    def test_filtered_analysis_performance(self, initialized_orchestrator, performance_timer):
        """Benchmark analysis with filters."""
        with performance_timer() as timer:
            result = initialized_orchestrator.run_analysis(
                "sales",
                store_ids=[1, 2, 3],
                start_date="2023-06-01",
                end_date="2023-06-30"
            )

        assert timer.elapsed_ms < 1000
        assert result is not None

    def test_repeated_analysis_caching(self, initialized_orchestrator, performance_timer):
        """Test that repeated analyses benefit from caching."""
        # First run
        with performance_timer() as timer1:
            result1 = initialized_orchestrator.run_analysis("sales")
        time1 = timer1.elapsed_ms

        # Second run (cached)
        with performance_timer() as timer2:
            result2 = initialized_orchestrator.run_analysis("sales")
        time2 = timer2.elapsed_ms

        # Cached should be significantly faster
        assert time2 < time1 * 0.5
        assert result1 == result2


@pytest.mark.performance
class TestMemoryUsage:
    """Benchmark memory usage."""

    def test_data_loading_memory(self, data_pipeline, memory_profiler):
        """Measure memory usage during data loading."""
        with memory_profiler() as profiler:
            df = data_pipeline.load_data()

        # Should not use excessive memory
        assert profiler.memory_increase_mb < 100

    @pytest.mark.slow
    def test_large_dataset_memory(self, temp_dir, large_dataset, memory_profiler):
        """Measure memory usage with large dataset."""
        data_file = temp_dir / "large.csv"
        large_dataset.to_csv(data_file, index=False)

        config = PipelineConfig(
            data_path=str(data_file),
            cache_dir=str(temp_dir / "cache")
        )

        with memory_profiler() as profiler:
            pipeline = DataPipeline(config)
            df = pipeline.load_data()
            pipeline.preprocess_data(df)

        # Should handle large data efficiently
        assert profiler.memory_increase_mb < 500

    def test_cache_memory_overhead(self, cache_manager, memory_profiler):
        """Measure cache memory overhead."""
        test_data = {"value": list(range(10000))}

        with memory_profiler() as profiler:
            for i in range(100):
                cache_manager.set(f"key_{i}", test_data)

        # Cache overhead should be reasonable
        assert profiler.memory_increase_mb < 50

    def test_memory_cleanup(self, cache_manager, memory_profiler):
        """Test that memory is properly cleaned up."""
        test_data = {"value": list(range(10000))}

        # Populate cache
        for i in range(100):
            cache_manager.set(f"key_{i}", test_data)

        with memory_profiler() as profiler:
            cache_manager.invalidate()

        # Memory should be freed
        assert profiler.memory_increase_mb < 1


@pytest.mark.performance
class TestConcurrentPerformance:
    """Benchmark concurrent operations."""

    def test_concurrent_analysis_throughput(self, initialized_orchestrator, performance_timer):
        """Test throughput with concurrent analyses."""
        def run_analysis(name):
            return initialized_orchestrator.run_analysis(name)

        modules = ["sales", "customers", "promotions"] * 10

        with performance_timer() as timer:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(run_analysis, modules))

        # Should complete reasonably fast
        assert timer.elapsed_seconds < 5
        assert len(results) == 30

    def test_concurrent_cache_access(self, cache_manager, performance_timer):
        """Test concurrent cache access performance."""
        test_data = {"value": list(range(1000))}

        # Populate cache
        for i in range(100):
            cache_manager.set(f"key_{i}", test_data)

        def access_cache(key_index):
            return cache_manager.get(f"key_{key_index}")

        with performance_timer() as timer:
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(access_cache, range(100)))

        assert timer.elapsed_ms < 1000
        assert all(r is not None for r in results)

    def test_concurrent_data_filtering(self, initialized_orchestrator, performance_timer):
        """Test concurrent filtering operations."""
        df = initialized_orchestrator.pipeline.processed_data

        def filter_data(store_id):
            return initialized_orchestrator.pipeline.filter_by_store(df, [store_id])

        with performance_timer() as timer:
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(filter_data, range(1, 11)))

        assert timer.elapsed_seconds < 2
        assert all(len(r) > 0 for r in results)


@pytest.mark.performance
class TestScalability:
    """Test scalability characteristics."""

    @pytest.mark.slow
    def test_data_size_scaling(self, temp_dir):
        """Test performance scaling with data size."""
        sizes = [1000, 10000, 100000]
        times = []

        for size in sizes:
            # Generate data
            df = pd.DataFrame({
                'Store': np.random.randint(1, 11, size),
                'Date': pd.date_range('2023-01-01', periods=size, freq='H'),
                'Sales': np.random.randint(1000, 10000, size),
                'Customers': np.random.randint(100, 1000, size),
                'Promo': np.random.choice([0, 1], size),
                'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], size),
                'SchoolHoliday': np.random.choice([0, 1], size)
            })

            data_file = temp_dir / f"data_{size}.csv"
            df.to_csv(data_file, index=False)

            # Measure time
            config = PipelineConfig(
                data_path=str(data_file),
                cache_dir=str(temp_dir / f"cache_{size}"),
                cache_enabled=False
            )
            pipeline = DataPipeline(config)

            start = time.time()
            pipeline.load_data()
            pipeline.preprocess_data(pipeline.raw_data)
            elapsed = time.time() - start

            times.append(elapsed)

        # Should scale sub-linearly (benefit from optimizations)
        assert times[1] < times[0] * 15  # 10x data should be <15x time
        assert times[2] < times[1] * 15

    def test_store_count_scaling(self, initialized_orchestrator, performance_timer):
        """Test scaling with number of stores."""
        store_counts = [5, 10, 20]
        times = []

        for count in store_counts:
            store_ids = list(range(1, count + 1))

            with performance_timer() as timer:
                result = initialized_orchestrator.run_analysis(
                    "sales",
                    store_ids=store_ids
                )

            times.append(timer.elapsed_ms)

        # Should scale reasonably
        assert all(t < 2000 for t in times)  # All under 2s


@pytest.mark.performance
class TestResourceEfficiency:
    """Test resource efficiency."""

    def test_cpu_utilization(self, initialized_orchestrator):
        """Test CPU utilization during operations."""
        process = psutil.Process(os.getpid())

        # Get baseline CPU
        baseline = process.cpu_percent(interval=0.1)

        # Run intensive operation
        for _ in range(10):
            initialized_orchestrator.run_all_analyses()

        # CPU should be utilized but not maxed out
        cpu_percent = process.cpu_percent(interval=0.1)
        assert cpu_percent > baseline
        assert cpu_percent < 90  # Should not saturate CPU

    def test_io_efficiency(self, temp_data_file, temp_dir, performance_timer):
        """Test I/O efficiency."""
        orchestrator = create_orchestrator(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache")
        )

        # First load (disk I/O)
        with performance_timer() as timer1:
            orchestrator.pipeline.load_data()
        time1 = timer1.elapsed_ms

        # Second load (should use cache, minimal I/O)
        with performance_timer() as timer2:
            orchestrator.pipeline.load_data()
        time2 = timer2.elapsed_ms

        # Cached load should be much faster
        assert time2 < time1 * 0.1

    def test_cache_size_efficiency(self, cache_manager):
        """Test cache storage efficiency."""
        # Add diverse data
        for i in range(100):
            cache_manager.set(f"key_{i}", {"data": list(range(1000))})

        stats = cache_manager.get_stats()

        # Cache size should be reasonable
        assert stats['total_size_mb'] < 50  # Under 50MB for test data
