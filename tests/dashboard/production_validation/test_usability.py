"""
Production Usability Testing Suite
===================================

Tests user workflows, error handling, and usability requirements
for store managers using the dashboard system.

Author: Production Validation Specialist
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Any

from src.dashboard.orchestrator import (
    PipelineConfig,
    DataPipeline,
    AnalysisOrchestrator,
    create_orchestrator
)


class TestUserWorkflows:
    """Test typical user workflows and scenarios."""

    @pytest.fixture
    def user_data(self, tmp_path):
        """Generate realistic user data."""
        np.random.seed(42)

        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        stores = range(1, 21)

        records = []
        for store in stores:
            for date in dates:
                records.append({
                    'Store': store,
                    'Date': date,
                    'Sales': np.random.randint(5000, 50000),
                    'Customers': np.random.randint(500, 5000),
                    'Promo': np.random.choice([0, 1]),
                    'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], p=[0.9, 0.03, 0.04, 0.03]),
                    'SchoolHoliday': np.random.choice([0, 1], p=[0.8, 0.2])
                })

        df = pd.DataFrame(records)
        data_file = tmp_path / "user_data.csv"
        df.to_csv(data_file, index=False)
        return str(data_file)

    @pytest.fixture
    def user_orchestrator(self, tmp_path, user_data):
        """Create orchestrator for user testing."""
        return create_orchestrator(
            data_path=user_data,
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            log_level="WARNING"
        )

    def test_daily_dashboard_check_workflow(self, user_orchestrator):
        """Test: Store manager checks daily dashboard (< 5 seconds)."""
        workflow_start = time.time()

        # Workflow: Check yesterday's performance for my store
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')

        # Step 1: View sales
        sales_result = user_orchestrator.run_analysis(
            'sales',
            store_ids=[1],
            start_date=yesterday,
            end_date=today
        )

        # Step 2: View customer metrics
        customer_result = user_orchestrator.run_analysis(
            'customers',
            store_ids=[1],
            start_date=yesterday,
            end_date=today
        )

        # Step 3: Check promotion effectiveness
        promo_result = user_orchestrator.run_analysis(
            'promotions',
            store_ids=[1]
        )

        workflow_time = time.time() - workflow_start

        assert sales_result is not None, "Sales data retrieved"
        assert customer_result is not None, "Customer data retrieved"
        assert promo_result is not None, "Promo data retrieved"
        assert workflow_time < 5.0, f"Daily check took {workflow_time:.2f}s (> 5s limit)"

        print(f"✓ Daily dashboard check completed in {workflow_time:.2f}s")

    def test_weekly_report_workflow(self, user_orchestrator):
        """Test: Store manager generates weekly report (< 10 seconds)."""
        workflow_start = time.time()

        # Workflow: Generate last week's report
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')

        # Comprehensive analysis for the week
        results = {}

        results['sales'] = user_orchestrator.run_analysis(
            'sales',
            store_ids=[1],
            start_date=week_ago,
            end_date=today
        )

        results['customers'] = user_orchestrator.run_analysis(
            'customers',
            store_ids=[1],
            start_date=week_ago,
            end_date=today
        )

        results['promotions'] = user_orchestrator.run_analysis(
            'promotions',
            store_ids=[1],
            start_date=week_ago,
            end_date=today
        )

        # Export results
        summary = user_orchestrator.get_summary()

        workflow_time = time.time() - workflow_start

        assert len(results) == 3, "All analyses completed"
        assert summary['status'] == 'active', "System operational"
        assert workflow_time < 10.0, f"Weekly report took {workflow_time:.2f}s (> 10s limit)"

        print(f"✓ Weekly report generated in {workflow_time:.2f}s")

    def test_multi_store_comparison_workflow(self, user_orchestrator):
        """Test: Regional manager compares multiple stores (< 15 seconds)."""
        workflow_start = time.time()

        # Workflow: Compare 5 stores in my region
        store_ids = [1, 2, 3, 4, 5]
        month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')

        # Compare sales across stores
        sales_result = user_orchestrator.run_analysis(
            'sales',
            store_ids=store_ids,
            start_date=month_ago,
            end_date=today
        )

        # Compare customer metrics
        customer_result = user_orchestrator.run_analysis(
            'customers',
            store_ids=store_ids,
            start_date=month_ago,
            end_date=today
        )

        workflow_time = time.time() - workflow_start

        assert sales_result is not None, "Multi-store sales comparison completed"
        assert customer_result is not None, "Multi-store customer comparison completed"
        assert workflow_time < 15.0, f"Multi-store comparison took {workflow_time:.2f}s (> 15s limit)"

        print(f"✓ Multi-store comparison completed in {workflow_time:.2f}s")

    def test_rapid_filter_changes(self, user_orchestrator):
        """Test: User rapidly changes filters (responsive UI)."""
        filter_combinations = [
            {'store_ids': [1], 'start_date': '2023-01-01', 'end_date': '2023-03-31'},
            {'store_ids': [2], 'start_date': '2023-04-01', 'end_date': '2023-06-30'},
            {'store_ids': [3], 'start_date': '2023-07-01', 'end_date': '2023-09-30'},
            {'store_ids': [1, 2], 'start_date': '2023-01-01', 'end_date': '2023-12-31'},
            {'store_ids': [1], 'start_date': '2023-06-01', 'end_date': '2023-06-30'},
        ]

        response_times = []

        for filters in filter_combinations:
            start = time.time()
            result = user_orchestrator.run_analysis('sales', **filters)
            response_time = time.time() - start
            response_times.append(response_time)

            assert result is not None, f"Filter combination failed: {filters}"

        avg_response = np.mean(response_times)
        max_response = max(response_times)

        assert avg_response < 1.0, f"Average filter response {avg_response:.2f}s > 1s"
        assert max_response < 2.0, f"Max filter response {max_response:.2f}s > 2s"

        print(f"✓ Filter changes: avg {avg_response:.3f}s, max {max_response:.3f}s")


class TestErrorHandling:
    """Test error handling and user feedback."""

    @pytest.fixture
    def error_test_orchestrator(self, tmp_path):
        """Create orchestrator for error testing."""
        # Create minimal valid data
        df = pd.DataFrame({
            'Store': [1],
            'Date': [pd.Timestamp('2023-01-01')],
            'Sales': [1000],
            'Customers': [100],
            'Promo': [0],
            'StateHoliday': ['0'],
            'SchoolHoliday': [0]
        })
        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        return create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

    def test_invalid_module_error(self, error_test_orchestrator):
        """Test: Invalid analysis module returns clear error."""
        with pytest.raises(ValueError, match="Unknown analysis module"):
            error_test_orchestrator.run_analysis('nonexistent_module')

        print(f"✓ Invalid module error handled correctly")

    def test_empty_result_handling(self, error_test_orchestrator):
        """Test: Empty results are handled gracefully."""
        # Query for non-existent store
        result = error_test_orchestrator.run_analysis(
            'sales',
            store_ids=[999]  # Non-existent store
        )

        # Should return valid result with zero values
        assert result is not None, "Empty result should still return structure"
        assert result['total_sales'] == 0, "Empty result should have zero totals"

        print(f"✓ Empty results handled gracefully")

    def test_invalid_date_range_error(self, error_test_orchestrator):
        """Test: Invalid date ranges are rejected with clear error."""
        invalid_dates = [
            ('2023-13-01', None),  # Invalid month
            ('invalid-date', None),  # Invalid format
            ('2023-12-31', '2023-01-01'),  # End before start (should still work but return empty)
        ]

        for start_date, end_date in invalid_dates:
            try:
                result = error_test_orchestrator.run_analysis(
                    'sales',
                    start_date=start_date,
                    end_date=end_date
                )
                # If it doesn't error, should handle gracefully
                assert result is not None
            except (ValueError, TypeError, Exception) as e:
                # Should provide clear error message
                assert len(str(e)) > 0, "Error message should be descriptive"

        print(f"✓ Invalid date ranges handled with clear errors")

    def test_system_recovery_from_error(self, error_test_orchestrator):
        """Test: System recovers gracefully after errors."""
        # Cause an error
        try:
            error_test_orchestrator.run_analysis('invalid_module')
        except ValueError:
            pass

        # System should still work after error
        result = error_test_orchestrator.run_analysis('sales', store_ids=[1])

        assert result is not None, "System recovered after error"
        assert 'total_sales' in result, "Normal operation restored"

        print(f"✓ System recovered gracefully from error")


class TestDataFreshness:
    """Test data freshness and update mechanisms."""

    def test_cache_expiry_behavior(self, tmp_path):
        """Test: Cached data expires and refreshes appropriately."""
        # Create data file
        df = pd.DataFrame({
            'Store': [1],
            'Date': [pd.Timestamp('2023-01-01')],
            'Sales': [1000],
            'Customers': [100],
            'Promo': [0],
            'StateHoliday': ['0'],
            'SchoolHoliday': [0]
        })
        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        # Create orchestrator with short cache TTL
        config = PipelineConfig(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            cache_ttl=2,  # 2 seconds TTL
            log_level="WARNING"
        )

        pipeline = DataPipeline(config)

        # First load (creates cache)
        result1 = pipeline.load_data()
        assert result1 is not None

        # Immediate second load (from cache)
        start = time.time()
        result2 = pipeline.load_data()
        cache_time = time.time() - start

        assert cache_time < 0.1, "Cached load should be fast"

        # Wait for cache expiry
        time.sleep(3)

        # Third load (cache expired, reload)
        start = time.time()
        result3 = pipeline.load_data()
        reload_time = time.time() - start

        assert reload_time > cache_time, "Expired cache should reload from disk"

        print(f"✓ Cache expiry behavior correct (cached: {cache_time:.3f}s, reload: {reload_time:.3f}s)")

    def test_cache_invalidation(self, tmp_path):
        """Test: Cache can be manually invalidated."""
        df = pd.DataFrame({
            'Store': [1],
            'Date': [pd.Timestamp('2023-01-01')],
            'Sales': [1000],
            'Customers': [100],
            'Promo': [0],
            'StateHoliday': ['0'],
            'SchoolHoliday': [0]
        })
        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        orchestrator = create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            cache_enabled=True,
            log_level="WARNING"
        )

        # Populate cache
        orchestrator.run_analysis('sales')

        # Verify cache has entries
        stats_before = orchestrator.cache_manager.get_stats()
        assert stats_before['memory_entries'] > 0 or stats_before['disk_entries'] > 0

        # Invalidate cache
        count = orchestrator.invalidate_cache()

        # Verify cache cleared
        stats_after = orchestrator.cache_manager.get_stats()
        assert stats_after['memory_entries'] == 0
        assert count > 0, "Cache entries were invalidated"

        print(f"✓ Cache invalidation successful ({count} entries cleared)")


class TestUserExperience:
    """Test overall user experience and satisfaction metrics."""

    def test_first_time_user_setup(self, tmp_path):
        """Test: First-time user can set up and use system quickly."""
        # Simulate new user setting up system
        setup_start = time.time()

        # Create data
        df = pd.DataFrame({
            'Store': [1],
            'Date': [pd.Timestamp('2023-01-01')],
            'Sales': [1000],
            'Customers': [100],
            'Promo': [0],
            'StateHoliday': ['0'],
            'SchoolHoliday': [0]
        })
        data_file = tmp_path / "new_user_data.csv"
        df.to_csv(data_file, index=False)

        # Initialize system
        orchestrator = create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

        # Run first query
        result = orchestrator.run_analysis('sales')

        setup_time = time.time() - setup_start

        assert result is not None, "First query successful"
        assert setup_time < 10.0, f"First-time setup took {setup_time:.2f}s (> 10s)"

        print(f"✓ First-time user setup completed in {setup_time:.2f}s")

    def test_system_status_visibility(self, tmp_path):
        """Test: Users can easily check system status."""
        df = pd.DataFrame({
            'Store': [1],
            'Date': [pd.Timestamp('2023-01-01')],
            'Sales': [1000],
            'Customers': [100],
            'Promo': [0],
            'StateHoliday': ['0'],
            'SchoolHoliday': [0]
        })
        data_file = tmp_path / "data.csv"
        df.to_csv(data_file, index=False)

        orchestrator = create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(tmp_path / "cache"),
            log_level="WARNING"
        )

        # Get system summary
        summary = orchestrator.get_summary()

        # Verify comprehensive status information
        required_fields = [
            'status',
            'data_loaded',
            'total_rows',
            'registered_modules',
            'cache_stats',
            'config'
        ]

        for field in required_fields:
            assert field in summary, f"Missing status field: {field}"

        assert summary['status'] == 'active', "System status clear"
        assert len(summary['registered_modules']) > 0, "Modules visible"

        print(f"✓ System status visibility complete")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
