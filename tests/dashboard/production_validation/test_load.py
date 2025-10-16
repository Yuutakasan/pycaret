"""
Production Load Testing Suite
==============================

Tests system behavior under concurrent user load and stress conditions.
Simulates multiple store managers accessing the dashboard simultaneously.

Author: Production Validation Specialist
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import psutil
import os

from src.dashboard.orchestrator import (
    PipelineConfig,
    DataPipeline,
    AnalysisOrchestrator,
    create_orchestrator
)


class UserSimulator:
    """Simulates a store manager using the dashboard."""

    def __init__(self, user_id: int, orchestrator: AnalysisOrchestrator):
        self.user_id = user_id
        self.orchestrator = orchestrator
        self.queries_executed = 0
        self.total_time = 0
        self.errors = []

    def execute_workflow(self, store_id: int) -> Dict[str, Any]:
        """Simulate typical user workflow."""
        try:
            results = {}

            # Query 1: Sales overview
            start = time.time()
            result = self.orchestrator.run_analysis(
                'sales',
                store_ids=[store_id],
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            results['sales'] = result
            self.queries_executed += 1

            # Query 2: Customer analysis
            result = self.orchestrator.run_analysis(
                'customers',
                store_ids=[store_id],
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            results['customers'] = result
            self.queries_executed += 1

            # Query 3: Promotion effectiveness
            result = self.orchestrator.run_analysis(
                'promotions',
                store_ids=[store_id]
            )
            results['promotions'] = result
            self.queries_executed += 1

            self.total_time += time.time() - start
            return results

        except Exception as e:
            self.errors.append(str(e))
            raise


class TestConcurrentUserLoad:
    """Test system under concurrent user load."""

    @pytest.fixture
    def production_data(self, tmp_path):
        """Generate production dataset."""
        np.random.seed(42)

        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        stores = range(1, 51)  # 50 stores

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
        data_file = tmp_path / "production_data.csv"
        df.to_csv(data_file, index=False)
        return str(data_file)

    @pytest.fixture
    def shared_orchestrator(self, tmp_path, production_data):
        """Create shared orchestrator for concurrent access."""
        config = PipelineConfig(
            data_path=production_data,
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            cache_ttl=3600,
            log_level="WARNING"
        )
        orchestrator = create_orchestrator(
            data_path=production_data,
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            cache_ttl=3600,
            log_level="WARNING"
        )
        return orchestrator

    def test_5_concurrent_users(self, shared_orchestrator):
        """Test: 5 users accessing dashboard simultaneously."""
        num_users = 5
        users = [UserSimulator(i, shared_orchestrator) for i in range(num_users)]

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            for user in users:
                store_id = user.user_id + 1
                future = executor.submit(user.execute_workflow, store_id)
                futures.append(future)

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"User workflow failed: {e}")

        total_time = time.time() - start_time

        assert len(results) == num_users, "All users completed workflows"
        assert total_time < 15.0, f"Total time {total_time:.2f}s exceeds 15s for 5 users"

        # Check all users executed successfully
        total_queries = sum(u.queries_executed for u in users)
        total_errors = sum(len(u.errors) for u in users)

        assert total_errors == 0, f"Encountered {total_errors} errors"
        assert total_queries == num_users * 3, "All queries executed"

        avg_response = np.mean([u.total_time / u.queries_executed for u in users])
        print(f"✓ 5 users completed in {total_time:.2f}s, avg response: {avg_response:.3f}s")

    def test_10_concurrent_users(self, shared_orchestrator):
        """Test: 10 users accessing dashboard simultaneously."""
        num_users = 10
        users = [UserSimulator(i, shared_orchestrator) for i in range(num_users)]

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            for user in users:
                store_id = (user.user_id % 50) + 1
                future = executor.submit(user.execute_workflow, store_id)
                futures.append(future)

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"User workflow failed: {e}")

        total_time = time.time() - start_time

        assert len(results) == num_users, "All users completed workflows"
        assert total_time < 30.0, f"Total time {total_time:.2f}s exceeds 30s for 10 users"

        total_errors = sum(len(u.errors) for u in users)
        assert total_errors == 0, f"Encountered {total_errors} errors"

        avg_response = np.mean([u.total_time / u.queries_executed for u in users])
        print(f"✓ 10 users completed in {total_time:.2f}s, avg response: {avg_response:.3f}s")

    def test_20_concurrent_users(self, shared_orchestrator):
        """Test: 20 users accessing dashboard simultaneously (stress test)."""
        num_users = 20
        users = [UserSimulator(i, shared_orchestrator) for i in range(num_users)]

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            for user in users:
                store_id = (user.user_id % 50) + 1
                future = executor.submit(user.execute_workflow, store_id)
                futures.append(future)

            results = []
            errors = 0
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                except Exception as e:
                    errors += 1

        total_time = time.time() - start_time

        success_rate = len(results) / num_users
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} below 95%"
        assert total_time < 60.0, f"Total time {total_time:.2f}s exceeds 60s for 20 users"

        print(f"✓ 20 users: {len(results)} succeeded in {total_time:.2f}s ({success_rate:.1%})")

    def test_sustained_concurrent_load(self, shared_orchestrator):
        """Test: Sustained load with 5 users for 2 minutes."""
        num_users = 5
        duration_seconds = 120

        users = [UserSimulator(i, shared_orchestrator) for i in range(num_users)]
        results_queue = queue.Queue()
        stop_flag = threading.Event()

        def user_loop(user: UserSimulator):
            """Continuously execute workflows until stopped."""
            while not stop_flag.is_set():
                try:
                    store_id = (user.user_id % 50) + 1
                    result = user.execute_workflow(store_id)
                    results_queue.put(('success', user.user_id))
                except Exception as e:
                    results_queue.put(('error', str(e)))
                time.sleep(np.random.uniform(1, 3))  # Think time

        # Start user threads
        threads = []
        for user in users:
            thread = threading.Thread(target=user_loop, args=(user,))
            thread.start()
            threads.append(thread)

        # Run for specified duration
        time.sleep(duration_seconds)
        stop_flag.set()

        # Wait for threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Collect results
        successes = 0
        errors = 0
        while not results_queue.empty():
            status, _ = results_queue.get()
            if status == 'success':
                successes += 1
            else:
                errors += 1

        total_queries = sum(u.queries_executed for u in users)
        error_rate = errors / total_queries if total_queries > 0 else 0

        assert error_rate < 0.05, f"Error rate {error_rate:.1%} exceeds 5%"
        assert total_queries > num_users * 10, "Sufficient queries executed"

        qpm = (total_queries / duration_seconds) * 60  # Queries per minute
        print(f"✓ Sustained load: {total_queries} queries in {duration_seconds}s ({qpm:.1f} QPM)")


class TestResourceUtilization:
    """Test resource utilization under load."""

    @pytest.fixture
    def production_orchestrator(self, tmp_path):
        """Create production orchestrator."""
        np.random.seed(42)

        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        stores = range(1, 51)

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
        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        return create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            log_level="WARNING"
        )

    def test_memory_under_load(self, production_orchestrator):
        """Test: Memory usage remains stable under load."""
        process = psutil.Process(os.getpid())

        baseline_memory = process.memory_info().rss / (1024 * 1024)
        memory_readings = []

        # Simulate 50 concurrent queries
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(50):
                store_id = (i % 50) + 1
                future = executor.submit(
                    production_orchestrator.run_analysis,
                    'sales',
                    store_ids=[store_id]
                )
                futures.append(future)

                # Take memory reading
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_readings.append(current_memory - baseline_memory)

            # Wait for completion
            for future in as_completed(futures):
                future.result()

        peak_memory = max(memory_readings)
        avg_memory = np.mean(memory_readings)

        assert peak_memory < 800, f"Peak memory {peak_memory:.1f}MB exceeds 800MB limit"
        assert avg_memory < 400, f"Avg memory {avg_memory:.1f}MB exceeds 400MB limit"

        print(f"✓ Memory - Peak: {peak_memory:.1f}MB, Avg: {avg_memory:.1f}MB")

    def test_cpu_utilization(self, production_orchestrator):
        """Test: CPU utilization is reasonable under load."""
        cpu_readings = []

        def measure_cpu():
            """Measure CPU in background."""
            for _ in range(20):
                cpu_readings.append(psutil.cpu_percent(interval=0.5))

        # Start CPU monitoring
        cpu_thread = threading.Thread(target=measure_cpu)
        cpu_thread.start()

        # Execute load
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(50):
                store_id = (i % 50) + 1
                future = executor.submit(
                    production_orchestrator.run_analysis,
                    'sales',
                    store_ids=[store_id]
                )
                futures.append(future)

            for future in as_completed(futures):
                future.result()

        cpu_thread.join()

        avg_cpu = np.mean(cpu_readings)
        max_cpu = max(cpu_readings)

        # CPU should be utilized but not maxed out
        assert avg_cpu < 80, f"Avg CPU {avg_cpu:.1f}% exceeds 80%"
        assert max_cpu < 95, f"Max CPU {max_cpu:.1f}% exceeds 95%"

        print(f"✓ CPU - Avg: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")


class TestDataConsistency:
    """Test data consistency under concurrent access."""

    def test_concurrent_read_consistency(self, tmp_path):
        """Test: Concurrent reads return consistent results."""
        np.random.seed(42)

        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100)
        records = []
        for date in dates:
            records.append({
                'Store': 1,
                'Date': date,
                'Sales': 10000,
                'Customers': 1000,
                'Promo': 0,
                'StateHoliday': '0',
                'SchoolHoliday': 0
            })

        df = pd.DataFrame(records)
        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        orchestrator = create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

        # Execute 20 concurrent reads
        results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(orchestrator.run_analysis, 'sales', store_ids=[1])
                for _ in range(20)
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # All results should be identical
        expected_total = 10000 * 100
        for result in results:
            assert result['total_sales'] == expected_total, "Inconsistent read result"

        print(f"✓ 20 concurrent reads returned consistent results")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
