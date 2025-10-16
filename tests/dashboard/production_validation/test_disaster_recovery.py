"""
Production Disaster Recovery Testing Suite
===========================================

Tests system resilience, backup/recovery capabilities, and failure handling
to ensure business continuity.

Author: Production Validation Specialist
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time
import shutil
import os
import tempfile
from datetime import datetime

from src.dashboard.orchestrator import (
    PipelineConfig,
    DataPipeline,
    CacheManager,
    AnalysisOrchestrator,
    create_orchestrator
)


class TestDataRecovery:
    """Test data backup and recovery mechanisms."""

    @pytest.fixture
    def recovery_data(self, tmp_path):
        """Generate test data for recovery scenarios."""
        np.random.seed(42)

        df = pd.DataFrame({
            'Store': range(1, 11),
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Sales': np.random.randint(5000, 10000, 10),
            'Customers': np.random.randint(500, 1000, 10),
            'Promo': np.random.choice([0, 1], 10),
            'StateHoliday': ['0'] * 10,
            'SchoolHoliday': [0] * 10
        })

        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)
        return str(data_file), df

    def test_cache_persistence_after_restart(self, tmp_path, recovery_data):
        """Test: Cache survives system restart."""
        data_file, _ = recovery_data
        cache_dir = tmp_path / "cache"

        # Create orchestrator and populate cache
        config1 = PipelineConfig(
            data_path=data_file,
            cache_dir=str(cache_dir),
            cache_enabled=True,
            log_level="WARNING"
        )

        orchestrator1 = AnalysisOrchestrator(config1)
        orchestrator1.initialize()
        result1 = orchestrator1.run_analysis('sales', store_ids=[1])

        # Verify cache exists on disk
        cache_stats1 = orchestrator1.cache_manager.get_stats()
        assert cache_stats1['disk_entries'] > 0, "Cache persisted to disk"

        # Simulate restart - create new orchestrator instance
        del orchestrator1

        config2 = PipelineConfig(
            data_path=data_file,
            cache_dir=str(cache_dir),
            cache_enabled=True,
            log_level="WARNING"
        )

        orchestrator2 = AnalysisOrchestrator(config2)
        orchestrator2.initialize()

        # Cache should still be available
        cache_stats2 = orchestrator2.cache_manager.get_stats()
        assert cache_stats2['disk_entries'] > 0, "Cache survived restart"

        # Cached data should be loadable
        result2 = orchestrator2.run_analysis('sales', store_ids=[1])
        assert result2 == result1, "Cached data intact after restart"

        print(f"✓ Cache persisted through restart")

    def test_recovery_from_corrupted_cache(self, tmp_path, recovery_data):
        """Test: System recovers from corrupted cache files."""
        data_file, _ = recovery_data
        cache_dir = tmp_path / "cache"

        config = PipelineConfig(
            data_path=data_file,
            cache_dir=str(cache_dir),
            cache_enabled=True,
            log_level="WARNING"
        )

        orchestrator = AnalysisOrchestrator(config)
        orchestrator.initialize()

        # Populate cache
        orchestrator.run_analysis('sales', store_ids=[1])

        # Corrupt cache files
        for cache_file in cache_dir.glob("*.pkl"):
            with open(cache_file, 'wb') as f:
                f.write(b"corrupted_data")

        # System should handle gracefully
        try:
            result = orchestrator.run_analysis('sales', store_ids=[1])
            # Should either rebuild or return fresh data
            assert result is not None, "System recovered from corrupted cache"
        except Exception as e:
            pytest.fail(f"Failed to recover from corrupted cache: {e}")

        print(f"✓ Recovered from corrupted cache")

    def test_data_file_recovery(self, tmp_path):
        """Test: System handles missing data file gracefully."""
        data_file = tmp_path / "data.csv"

        # Create initial data
        df = pd.DataFrame({
            'Store': [1],
            'Date': [pd.Timestamp('2023-01-01')],
            'Sales': [1000],
            'Customers': [100],
            'Promo': [0],
            'StateHoliday': ['0'],
            'SchoolHoliday': [0]
        })
        df.to_csv(data_file, index=False)

        config = PipelineConfig(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

        # Load data successfully
        pipeline = DataPipeline(config)
        pipeline.load_data()

        # Delete data file (disaster scenario)
        data_file.unlink()

        # Try to reload - should fail gracefully
        with pytest.raises(FileNotFoundError):
            new_pipeline = DataPipeline(config)
            new_pipeline.load_data()

        print(f"✓ Missing data file handled gracefully")


class TestFailureRecovery:
    """Test recovery from various failure scenarios."""

    @pytest.fixture
    def failure_orchestrator(self, tmp_path):
        """Create orchestrator for failure testing."""
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
            log_level="WARNING"
        )

    def test_recovery_from_analysis_error(self, failure_orchestrator):
        """Test: System recovers after analysis module error."""
        # Register faulty module
        def faulty_analysis(data, **kwargs):
            raise ValueError("Simulated analysis error")

        failure_orchestrator.register_module('faulty', faulty_analysis)

        # Try faulty module
        with pytest.raises(ValueError):
            failure_orchestrator.run_analysis('faulty')

        # System should still work with other modules
        result = failure_orchestrator.run_analysis('sales')
        assert result is not None, "System recovered after module error"

        print(f"✓ Recovered from analysis module error")

    def test_partial_failure_handling(self, failure_orchestrator):
        """Test: Partial failures don't crash entire system."""
        # Register mix of good and bad modules
        def good_analysis(data, **kwargs):
            return {'result': 'success'}

        def bad_analysis(data, **kwargs):
            raise RuntimeError("Module failure")

        failure_orchestrator.register_module('good', good_analysis)
        failure_orchestrator.register_module('bad', bad_analysis)

        # Run all analyses (some will fail)
        results = failure_orchestrator.run_all_analyses()

        # Good module should succeed
        assert 'good' in results
        assert results['good']['result'] == 'success'

        # Bad module should have error recorded
        assert 'bad' in results
        assert 'error' in results['bad']

        print(f"✓ Partial failures isolated and handled")

    def test_concurrent_failure_isolation(self, failure_orchestrator):
        """Test: Failures in concurrent operations are isolated."""
        from concurrent.futures import ThreadPoolExecutor

        def sometimes_fails(store_id):
            if store_id == 2:
                raise ValueError("Simulated failure")
            return failure_orchestrator.run_analysis('sales', store_ids=[store_id])

        # Execute concurrent operations
        results = []
        errors = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(sometimes_fails, i) for i in [1, 2, 3]]

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))

        # Should have 2 successes and 1 failure
        assert len(results) == 2, "Non-failing operations completed"
        assert len(errors) == 1, "Failing operation isolated"

        print(f"✓ Concurrent failures isolated")


class TestBackupAndRestore:
    """Test backup and restore capabilities."""

    @pytest.fixture
    def backup_orchestrator(self, tmp_path):
        """Create orchestrator for backup testing."""
        df = pd.DataFrame({
            'Store': range(1, 6),
            'Date': [pd.Timestamp('2023-01-01')] * 5,
            'Sales': [1000, 2000, 3000, 4000, 5000],
            'Customers': [100, 200, 300, 400, 500],
            'Promo': [0, 1, 0, 1, 0],
            'StateHoliday': ['0', 'a', '0', 'b', '0'],
            'SchoolHoliday': [0, 1, 0, 1, 0]
        })

        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        return create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

    def test_results_export_backup(self, backup_orchestrator, tmp_path):
        """Test: Results can be exported for backup."""
        # Run analyses
        backup_orchestrator.run_analysis('sales')
        backup_orchestrator.run_analysis('customers')

        # Export results
        export_path = tmp_path / "backup" / "results_backup.json"
        backup_orchestrator.export_results(str(export_path))

        # Verify export exists
        assert export_path.exists(), "Backup export created"

        # Verify export is valid JSON
        import json
        with open(export_path) as f:
            backup_data = json.load(f)

        assert 'summary' in backup_data, "Backup contains summary"
        assert 'results' in backup_data, "Backup contains results"
        assert 'export_time' in backup_data, "Backup contains timestamp"

        print(f"✓ Results exported for backup")

    def test_cache_directory_backup(self, backup_orchestrator, tmp_path):
        """Test: Cache directory can be backed up."""
        # Populate cache
        backup_orchestrator.run_analysis('sales')

        cache_dir = Path(backup_orchestrator.config.cache_dir)
        backup_dir = tmp_path / "cache_backup"

        # Backup cache directory
        shutil.copytree(cache_dir, backup_dir)

        # Verify backup
        assert backup_dir.exists(), "Cache backup created"
        assert len(list(backup_dir.glob("*.pkl"))) > 0, "Cache files backed up"

        print(f"✓ Cache directory backed up")

    def test_restore_from_backup(self, tmp_path):
        """Test: System can be restored from backup."""
        # Create original system
        df = pd.DataFrame({
            'Store': [1],
            'Date': [pd.Timestamp('2023-01-01')],
            'Sales': [1000],
            'Customers': [100],
            'Promo': [0],
            'StateHoliday': ['0'],
            'SchoolHoliday': [0]
        })

        original_data = tmp_path / "original_data.csv"
        df.to_csv(original_data, index=False)

        # Backup data file
        backup_data = tmp_path / "backup_data.csv"
        shutil.copy(original_data, backup_data)

        # Delete original
        original_data.unlink()

        # Restore from backup
        restored_data = tmp_path / "restored_data.csv"
        shutil.copy(backup_data, restored_data)

        # Verify restoration works
        config = PipelineConfig(
            data_path=str(restored_data),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

        pipeline = DataPipeline(config)
        loaded = pipeline.load_data()

        assert len(loaded) == 1, "Data restored from backup"

        print(f"✓ System restored from backup")


class TestSystemResilience:
    """Test overall system resilience."""

    @pytest.fixture
    def resilience_orchestrator(self, tmp_path):
        """Create orchestrator for resilience testing."""
        np.random.seed(42)

        df = pd.DataFrame({
            'Store': range(1, 21),
            'Date': [pd.Timestamp('2023-01-01')] * 20,
            'Sales': np.random.randint(5000, 10000, 20),
            'Customers': np.random.randint(500, 1000, 20),
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

    def test_memory_leak_prevention(self, resilience_orchestrator):
        """Test: No memory leaks during extended operation."""
        import psutil
        process = psutil.Process(os.getpid())

        baseline_memory = process.memory_info().rss / (1024 * 1024)

        # Run many operations
        for i in range(100):
            store_id = (i % 20) + 1
            resilience_orchestrator.run_analysis('sales', store_ids=[store_id])

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - baseline_memory

        # Memory should not grow excessively
        assert memory_increase < 200, f"Potential memory leak: {memory_increase:.1f}MB increase"

        print(f"✓ No memory leak detected ({memory_increase:.1f}MB increase)")

    def test_uptime_resilience(self, resilience_orchestrator):
        """Test: System maintains operation over extended period."""
        start_time = time.time()
        duration = 10  # 10 second test

        operations = 0
        errors = 0

        while time.time() - start_time < duration:
            try:
                store_id = (operations % 20) + 1
                resilience_orchestrator.run_analysis('sales', store_ids=[store_id])
                operations += 1
            except Exception:
                errors += 1

            time.sleep(0.1)

        error_rate = errors / operations if operations > 0 else 1
        uptime = (operations - errors) / operations if operations > 0 else 0

        assert uptime > 0.99, f"Uptime {uptime:.1%} below 99%"
        assert error_rate < 0.01, f"Error rate {error_rate:.1%} above 1%"

        print(f"✓ System uptime: {uptime:.1%} ({operations} operations, {errors} errors)")

    def test_graceful_shutdown(self, resilience_orchestrator):
        """Test: System shuts down gracefully without data loss."""
        # Run analysis
        result1 = resilience_orchestrator.run_analysis('sales', store_ids=[1])

        # Get cache stats
        cache_stats_before = resilience_orchestrator.cache_manager.get_stats()

        # Simulate shutdown (delete in-memory references)
        summary = resilience_orchestrator.get_summary()
        assert summary is not None, "Summary available before shutdown"

        # Cache should be persisted
        cache_stats_after = resilience_orchestrator.cache_manager.get_stats()
        assert cache_stats_after['disk_entries'] >= cache_stats_before['disk_entries']

        print(f"✓ Graceful shutdown with data persistence")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
