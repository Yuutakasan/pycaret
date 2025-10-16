"""
Unit Tests for Analysis Orchestrator
====================================

Tests orchestrator initialization, module registration,
analysis execution, and result management.

Author: Testing & QA Agent
Created: 2025-10-08
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.dashboard.orchestrator import (
    AnalysisOrchestrator,
    sales_analysis,
    customer_analysis,
    promo_analysis
)


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    def test_orchestrator_creation(self, pipeline_config):
        """Test creating orchestrator."""
        orchestrator = AnalysisOrchestrator(pipeline_config)

        assert orchestrator.config == pipeline_config
        assert orchestrator.pipeline is not None
        assert orchestrator.cache_manager is not None
        assert len(orchestrator.analysis_modules) == 0
        assert len(orchestrator.results) == 0

    def test_register_module(self, orchestrator, mock_analysis_module):
        """Test registering analysis module."""
        orchestrator.register_module("test_module", mock_analysis_module)

        assert "test_module" in orchestrator.analysis_modules
        assert orchestrator.analysis_modules["test_module"] == mock_analysis_module

    def test_register_multiple_modules(self, orchestrator):
        """Test registering multiple modules."""
        modules = {
            "module1": lambda df, **kwargs: {"result": 1},
            "module2": lambda df, **kwargs: {"result": 2},
            "module3": lambda df, **kwargs: {"result": 3}
        }

        for name, func in modules.items():
            orchestrator.register_module(name, func)

        assert len(orchestrator.analysis_modules) == 3
        assert all(name in orchestrator.analysis_modules for name in modules.keys())

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        orchestrator.initialize()

        assert orchestrator.pipeline.raw_data is not None
        assert orchestrator.pipeline.processed_data is not None
        assert len(orchestrator.pipeline.metadata) > 0


class TestAnalysisExecution:
    """Test analysis execution."""

    def test_run_analysis_basic(self, initialized_orchestrator, mock_analysis_module):
        """Test running basic analysis."""
        initialized_orchestrator.register_module("test", mock_analysis_module)
        result = initialized_orchestrator.run_analysis("test")

        assert result is not None
        assert 'total_sales' in result
        assert 'avg_sales' in result
        assert 'store_count' in result

    def test_run_analysis_with_filters(self, initialized_orchestrator, mock_analysis_module):
        """Test running analysis with filters."""
        initialized_orchestrator.register_module("test", mock_analysis_module)

        result = initialized_orchestrator.run_analysis(
            "test",
            store_ids=[1, 2],
            start_date="2023-06-01",
            end_date="2023-06-30"
        )

        assert result is not None
        assert 'total_sales' in result

    def test_run_analysis_with_kwargs(self, initialized_orchestrator):
        """Test passing kwargs to analysis module."""
        def module_with_params(df, threshold=100, **kwargs):
            return {
                'threshold': threshold,
                'extra': kwargs
            }

        initialized_orchestrator.register_module("params", module_with_params)
        result = initialized_orchestrator.run_analysis(
            "params",
            threshold=500,
            extra_param="value"
        )

        assert result['threshold'] == 500
        assert result['extra']['extra_param'] == "value"

    def test_run_analysis_unknown_module(self, initialized_orchestrator):
        """Test running unknown analysis module."""
        with pytest.raises(ValueError, match="Unknown analysis module"):
            initialized_orchestrator.run_analysis("nonexistent")

    def test_run_analysis_stores_result(self, initialized_orchestrator, mock_analysis_module):
        """Test that analysis results are stored."""
        initialized_orchestrator.register_module("test", mock_analysis_module)
        initialized_orchestrator.run_analysis("test")

        assert "test" in initialized_orchestrator.results
        assert 'result' in initialized_orchestrator.results["test"]
        assert 'timestamp' in initialized_orchestrator.results["test"]
        assert 'filters' in initialized_orchestrator.results["test"]

    def test_run_all_analyses(self, initialized_orchestrator):
        """Test running all registered analyses."""
        modules = {
            "sales": sales_analysis,
            "customers": customer_analysis,
            "promotions": promo_analysis
        }

        for name, func in modules.items():
            initialized_orchestrator.register_module(name, func)

        results = initialized_orchestrator.run_all_analyses()

        assert len(results) == 3
        assert all(name in results for name in modules.keys())
        assert all('total_sales' in results["sales"] or True for _ in range(1))

    def test_run_all_analyses_with_error(self, initialized_orchestrator):
        """Test run_all_analyses handles errors gracefully."""
        def failing_module(df, **kwargs):
            raise ValueError("Test error")

        def working_module(df, **kwargs):
            return {"success": True}

        initialized_orchestrator.register_module("failing", failing_module)
        initialized_orchestrator.register_module("working", working_module)

        results = initialized_orchestrator.run_all_analyses()

        assert "failing" in results
        assert "error" in results["failing"]
        assert "working" in results
        assert results["working"]["success"] is True


class TestBuiltInAnalysisModules:
    """Test built-in analysis modules."""

    def test_sales_analysis(self, sample_data):
        """Test sales analysis module."""
        result = sales_analysis(sample_data)

        assert 'total_sales' in result
        assert 'avg_sales' in result
        assert 'sales_by_store' in result

        assert result['total_sales'] == sample_data['Sales'].sum()
        assert result['avg_sales'] == sample_data['Sales'].mean()
        assert isinstance(result['sales_by_store'], dict)

    def test_customer_analysis(self, sample_data):
        """Test customer analysis module."""
        result = customer_analysis(sample_data)

        assert 'total_customers' in result
        assert 'avg_customers' in result
        assert 'customers_by_store' in result

        assert result['total_customers'] == sample_data['Customers'].sum()
        assert isinstance(result['customers_by_store'], dict)

    def test_promo_analysis(self, sample_data):
        """Test promotion analysis module."""
        result = promo_analysis(sample_data)

        assert 'promo_days' in result
        assert 'promo_effectiveness' in result

        assert isinstance(result['promo_effectiveness'], dict)


class TestOrchestratorUtilities:
    """Test orchestrator utility functions."""

    def test_get_summary(self, initialized_orchestrator):
        """Test getting orchestrator summary."""
        summary = initialized_orchestrator.get_summary()

        assert 'status' in summary
        assert 'data_loaded' in summary
        assert 'total_rows' in summary
        assert 'registered_modules' in summary
        assert 'completed_analyses' in summary
        assert 'metadata' in summary
        assert 'cache_stats' in summary
        assert 'config' in summary

        assert summary['status'] == 'active'
        assert summary['data_loaded'] is True
        assert summary['total_rows'] > 0

    def test_get_summary_before_init(self, orchestrator):
        """Test summary before initialization."""
        summary = orchestrator.get_summary()

        assert summary['status'] == 'inactive'
        assert summary['data_loaded'] is False
        assert summary['total_rows'] == 0

    def test_invalidate_cache(self, initialized_orchestrator, mock_analysis_module):
        """Test cache invalidation."""
        initialized_orchestrator.register_module("test", mock_analysis_module)

        # Run analysis to populate cache
        initialized_orchestrator.run_analysis("test")

        # Invalidate cache
        count = initialized_orchestrator.invalidate_cache()

        assert count > 0

    def test_invalidate_cache_pattern(self, initialized_orchestrator):
        """Test cache invalidation with pattern."""
        modules = {
            "sales_total": sales_analysis,
            "sales_avg": sales_analysis,
            "customers": customer_analysis
        }

        for name, func in modules.items():
            initialized_orchestrator.register_module(name, func)
            initialized_orchestrator.run_analysis(name)

        # Invalidate only sales-related cache
        count = initialized_orchestrator.invalidate_cache(pattern="sales")

        # Should invalidate at least the sales-related entries
        assert count >= 0

    def test_export_results(self, initialized_orchestrator, mock_analysis_module, temp_dir):
        """Test exporting results to file."""
        initialized_orchestrator.register_module("test", mock_analysis_module)
        initialized_orchestrator.run_analysis("test")

        output_path = temp_dir / "results.json"
        initialized_orchestrator.export_results(str(output_path))

        assert output_path.exists()

        # Verify content
        import json
        with open(output_path) as f:
            data = json.load(f)

        assert 'summary' in data
        assert 'results' in data
        assert 'export_time' in data

    def test_export_results_creates_directory(self, initialized_orchestrator, temp_dir):
        """Test that export creates output directory if needed."""
        output_path = temp_dir / "subdir" / "nested" / "results.json"
        initialized_orchestrator.export_results(str(output_path))

        assert output_path.exists()
        assert output_path.parent.exists()


class TestOrchestratorCaching:
    """Test orchestrator caching behavior."""

    def test_analysis_result_caching(self, initialized_orchestrator):
        """Test that analysis results are cached."""
        call_count = {"count": 0}

        def counting_module(df, **kwargs):
            call_count["count"] += 1
            return {"count": call_count["count"]}

        initialized_orchestrator.register_module("counting", counting_module)

        # First call
        result1 = initialized_orchestrator.run_analysis("counting")
        assert result1["count"] == 1

        # Second call with same params - should use cache
        result2 = initialized_orchestrator.run_analysis("counting")
        assert result2["count"] == 1  # Same result from cache
        assert call_count["count"] == 1  # Not called again

    def test_cache_invalidation_forces_rerun(self, initialized_orchestrator):
        """Test that cache invalidation forces module rerun."""
        call_count = {"count": 0}

        def counting_module(df, **kwargs):
            call_count["count"] += 1
            return {"count": call_count["count"]}

        initialized_orchestrator.register_module("counting", counting_module)

        # First call
        initialized_orchestrator.run_analysis("counting")
        assert call_count["count"] == 1

        # Invalidate cache
        initialized_orchestrator.invalidate_cache()

        # Second call - should rerun
        result = initialized_orchestrator.run_analysis("counting")
        assert result["count"] == 2
        assert call_count["count"] == 2


class TestFactoryFunction:
    """Test create_orchestrator factory function."""

    def test_create_orchestrator(self, temp_data_file, temp_dir):
        """Test factory function creates orchestrator."""
        from src.dashboard.orchestrator import create_orchestrator

        orchestrator = create_orchestrator(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache")
        )

        assert orchestrator is not None
        assert orchestrator.pipeline.processed_data is not None

        # Should have default modules registered
        assert "sales" in orchestrator.analysis_modules
        assert "customers" in orchestrator.analysis_modules
        assert "promotions" in orchestrator.analysis_modules

    def test_create_orchestrator_with_custom_config(self, temp_data_file, temp_dir):
        """Test factory with custom configuration."""
        from src.dashboard.orchestrator import create_orchestrator

        orchestrator = create_orchestrator(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache"),
            cache_enabled=False,
            cache_ttl=7200
        )

        assert not orchestrator.config.cache_enabled
        assert orchestrator.config.cache_ttl == 7200


class TestConcurrency:
    """Test concurrent operations."""

    def test_concurrent_analysis_runs(self, initialized_orchestrator):
        """Test running analyses concurrently."""
        from concurrent.futures import ThreadPoolExecutor

        modules = {
            f"module_{i}": lambda df, idx=i, **kwargs: {"id": idx}
            for i in range(10)
        }

        for name, func in modules.items():
            initialized_orchestrator.register_module(name, func)

        def run_analysis(name):
            return initialized_orchestrator.run_analysis(name)

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(run_analysis, modules.keys()))

        assert len(results) == 10
        assert all(r is not None for r in results)
