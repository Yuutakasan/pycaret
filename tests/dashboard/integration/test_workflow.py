"""
Integration Tests for Data Pipeline Workflow
============================================

Tests end-to-end workflows, data flow through pipeline,
and integration between components.

Author: Testing & QA Agent
Created: 2025-10-08
"""

import pytest
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

from src.dashboard.orchestrator import (
    create_orchestrator,
    AnalysisOrchestrator,
    DataPipeline,
    PipelineConfig
)


@pytest.mark.integration
class TestCompleteWorkflow:
    """Test complete analysis workflow from start to finish."""

    def test_end_to_end_pipeline(self, temp_data_file, temp_dir):
        """Test complete pipeline from load to export."""
        orchestrator = create_orchestrator(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache")
        )

        # Verify initialization
        assert orchestrator.pipeline.processed_data is not None

        # Run all analyses
        results = orchestrator.run_all_analyses(
            store_ids=[1, 2, 3],
            start_date="2023-06-01",
            end_date="2023-06-30"
        )

        # Verify results
        assert len(results) >= 3
        assert "sales" in results
        assert "customers" in results
        assert "promotions" in results

        # Export results
        output_path = temp_dir / "results.json"
        orchestrator.export_results(str(output_path))

        assert output_path.exists()

        # Verify exported data
        with open(output_path) as f:
            exported = json.load(f)

        assert "summary" in exported
        assert "results" in exported

    def test_incremental_update_workflow(self, temp_data_file, temp_dir, sample_data):
        """Test incremental update workflow."""
        orchestrator = create_orchestrator(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache"),
            incremental_enabled=True
        )

        # Get initial state
        summary1 = orchestrator.get_summary()
        initial_rows = summary1['total_rows']

        # Simulate incremental update
        last_update = pd.to_datetime("2023-06-01")
        new_data, metadata = orchestrator.pipeline.get_incremental_updates(last_update)

        assert len(new_data) > 0
        assert metadata['new_rows'] > 0
        assert metadata['total_rows'] == initial_rows

    def test_cache_persistence_workflow(self, temp_data_file, temp_dir):
        """Test that cache persists across orchestrator instances."""
        # Create first orchestrator and run analysis
        orchestrator1 = create_orchestrator(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache")
        )

        result1 = orchestrator1.run_analysis("sales")

        # Create second orchestrator with same cache
        orchestrator2 = create_orchestrator(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache")
        )

        # Should use cached data
        stats = orchestrator2.cache_manager.get_stats()
        assert stats['disk_entries'] > 0

    def test_multi_filter_workflow(self, temp_data_file, temp_dir):
        """Test workflow with multiple filters applied."""
        orchestrator = create_orchestrator(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache")
        )

        # Run analysis with multiple filters
        results = orchestrator.run_all_analyses(
            store_ids=[1, 2, 3, 4, 5],
            start_date="2023-03-01",
            end_date="2023-09-30"
        )

        # Verify all analyses completed
        assert all('error' not in r for r in results.values())

        # Verify filters were applied
        for module_name, result_data in orchestrator.results.items():
            filters = result_data['filters']
            assert filters['store_ids'] == [1, 2, 3, 4, 5]
            assert filters['start_date'] == "2023-03-01"
            assert filters['end_date'] == "2023-09-30"


@pytest.mark.integration
class TestDataIntegrity:
    """Test data integrity through the pipeline."""

    def test_data_consistency_through_pipeline(self, data_pipeline, sample_data):
        """Test that data remains consistent through processing."""
        # Load and preprocess
        df = data_pipeline.load_data()
        processed = data_pipeline.preprocess_data(df)

        # Verify no data loss for valid records
        assert len(processed) > 0
        assert all(col in processed.columns for col in data_pipeline.config.required_columns)

        # Verify derived features are consistent
        assert (processed['Year'] == processed['Date'].dt.year).all()
        assert (processed['Month'] == processed['Date'].dt.month).all()

    def test_filter_chain_integrity(self, initialized_orchestrator):
        """Test integrity of filter chain."""
        # Get base data
        df = initialized_orchestrator.pipeline.processed_data

        # Apply filters in sequence
        filtered1 = initialized_orchestrator.pipeline.filter_by_store(df, [1, 2, 3])
        filtered2 = initialized_orchestrator.pipeline.filter_by_date_range(
            filtered1,
            start_date="2023-06-01",
            end_date="2023-06-30"
        )

        # Verify constraints
        assert set(filtered2['Store'].unique()).issubset({1, 2, 3})
        assert filtered2['Date'].min() >= pd.to_datetime("2023-06-01")
        assert filtered2['Date'].max() <= pd.to_datetime("2023-06-30")

    def test_analysis_result_integrity(self, initialized_orchestrator):
        """Test that analysis results are accurate."""
        from src.dashboard.orchestrator import sales_analysis

        initialized_orchestrator.register_module("sales", sales_analysis)

        # Get test data
        test_data = initialized_orchestrator.pipeline.processed_data.copy()

        # Run analysis
        result = initialized_orchestrator.run_analysis("sales")

        # Verify accuracy
        assert result['total_sales'] == test_data['Sales'].sum()
        assert result['avg_sales'] == test_data['Sales'].mean()


@pytest.mark.integration
class TestComponentInteraction:
    """Test interaction between components."""

    def test_pipeline_orchestrator_integration(self, pipeline_config):
        """Test integration between pipeline and orchestrator."""
        orchestrator = AnalysisOrchestrator(pipeline_config)
        orchestrator.initialize()

        # Verify pipeline is accessible
        assert orchestrator.pipeline.processed_data is not None

        # Verify cache manager is shared
        assert orchestrator.cache_manager is orchestrator.pipeline.cache_manager

    def test_cache_manager_pipeline_integration(self, data_pipeline):
        """Test cache manager integration with pipeline."""
        # Load data (should cache)
        df1 = data_pipeline.load_data()

        # Clear memory cache only
        data_pipeline.cache_manager.memory_cache.clear()

        # Load again (should hit disk cache)
        df2 = data_pipeline.load_data()

        pd.testing.assert_frame_equal(df1, df2)

    def test_module_registration_integration(self, initialized_orchestrator):
        """Test module registration and execution integration."""
        execution_log = []

        def logging_module(df, **kwargs):
            execution_log.append({
                'rows': len(df),
                'timestamp': datetime.now().isoformat(),
                'kwargs': kwargs
            })
            return {'status': 'executed'}

        initialized_orchestrator.register_module("logger", logging_module)
        result = initialized_orchestrator.run_analysis("logger", param1="value1")

        assert len(execution_log) == 1
        assert execution_log[0]['kwargs']['param1'] == "value1"
        assert result['status'] == 'executed'


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across components."""

    def test_pipeline_error_propagation(self, temp_dir):
        """Test that pipeline errors are properly propagated."""
        config = PipelineConfig(
            data_path="/nonexistent/path.csv",
            cache_dir=str(temp_dir / "cache")
        )
        pipeline = DataPipeline(config)

        with pytest.raises(Exception):
            pipeline.load_data()

    def test_analysis_error_handling(self, initialized_orchestrator):
        """Test error handling in analysis execution."""
        def failing_module(df, **kwargs):
            raise ValueError("Analysis failed")

        initialized_orchestrator.register_module("failing", failing_module)

        with pytest.raises(ValueError, match="Analysis failed"):
            initialized_orchestrator.run_analysis("failing")

    def test_partial_failure_in_batch(self, initialized_orchestrator):
        """Test that batch execution continues after partial failure."""
        def success_module(df, **kwargs):
            return {"status": "success"}

        def failure_module(df, **kwargs):
            raise RuntimeError("Module error")

        initialized_orchestrator.register_module("success", success_module)
        initialized_orchestrator.register_module("failure", failure_module)

        results = initialized_orchestrator.run_all_analyses()

        assert "success" in results
        assert results["success"]["status"] == "success"
        assert "failure" in results
        assert "error" in results["failure"]


@pytest.mark.integration
class TestPerformance:
    """Test performance characteristics of integrated system."""

    def test_caching_improves_performance(self, temp_data_file, temp_dir, performance_timer):
        """Test that caching improves performance."""
        orchestrator = create_orchestrator(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache")
        )

        # First run (no cache)
        with performance_timer() as timer1:
            result1 = orchestrator.run_analysis("sales")
        time1 = timer1.elapsed_ms

        # Second run (with cache)
        with performance_timer() as timer2:
            result2 = orchestrator.run_analysis("sales")
        time2 = timer2.elapsed_ms

        # Cached run should be faster
        assert time2 < time1
        assert result1 == result2

    def test_filter_performance(self, initialized_orchestrator, performance_timer):
        """Test performance of filtering operations."""
        df = initialized_orchestrator.pipeline.processed_data

        with performance_timer() as timer:
            filtered = initialized_orchestrator.pipeline.filter_by_store(df, [1, 2, 3])
            filtered = initialized_orchestrator.pipeline.filter_by_date_range(
                filtered,
                start_date="2023-06-01",
                end_date="2023-06-30"
            )

        # Should complete quickly
        assert timer.elapsed_ms < 1000  # 1 second
        assert len(filtered) > 0

    @pytest.mark.slow
    def test_large_dataset_integration(self, temp_dir, large_dataset, performance_timer):
        """Test integration with large dataset."""
        # Save large dataset
        data_file = temp_dir / "large.csv"
        large_dataset.to_csv(data_file, index=False)

        # Create orchestrator
        orchestrator = create_orchestrator(
            data_path=str(data_file),
            cache_dir=str(temp_dir / "cache")
        )

        # Time full workflow
        with performance_timer() as timer:
            results = orchestrator.run_all_analyses(store_ids=list(range(1, 11)))

        # Should complete in reasonable time
        assert timer.elapsed_seconds < 60
        assert all(r is not None for r in results.values())


@pytest.mark.integration
class TestStateManagement:
    """Test state management across operations."""

    def test_orchestrator_state_consistency(self, initialized_orchestrator):
        """Test that orchestrator maintains consistent state."""
        # Get initial state
        summary1 = initialized_orchestrator.get_summary()

        # Run analysis
        initialized_orchestrator.run_analysis("sales")

        # Get updated state
        summary2 = initialized_orchestrator.get_summary()

        # State should be updated
        assert len(summary2['completed_analyses']) > len(summary1['completed_analyses'])
        assert summary2['total_rows'] == summary1['total_rows']  # Data unchanged

    def test_cache_state_across_operations(self, initialized_orchestrator):
        """Test cache state management."""
        # Initial cache stats
        stats1 = initialized_orchestrator.cache_manager.get_stats()

        # Perform operations
        initialized_orchestrator.run_analysis("sales")
        initialized_orchestrator.run_analysis("customers")

        # Updated cache stats
        stats2 = initialized_orchestrator.cache_manager.get_stats()

        assert stats2['memory_entries'] >= stats1['memory_entries']
        assert stats2['disk_entries'] >= stats1['disk_entries']

    def test_result_accumulation(self, initialized_orchestrator):
        """Test that results accumulate correctly."""
        modules = ["sales", "customers", "promotions"]

        for module in modules:
            initialized_orchestrator.run_analysis(module)

        assert len(initialized_orchestrator.results) == 3
        assert all(module in initialized_orchestrator.results for module in modules)

        # Each result should have metadata
        for module, result_data in initialized_orchestrator.results.items():
            assert 'result' in result_data
            assert 'timestamp' in result_data
            assert 'filters' in result_data
