"""
Test Configuration and Fixtures
================================

Provides shared fixtures, mock data generators, and test utilities
for dashboard test suite.

Author: Testing & QA Agent
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import shutil
from unittest.mock import MagicMock, Mock

from src.dashboard.orchestrator import (
    PipelineConfig,
    CacheManager,
    DataPipeline,
    AnalysisOrchestrator
)


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample retail data for testing."""
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    stores = range(1, 11)  # 10 stores

    data = []
    for store in stores:
        for date in dates:
            data.append({
                'Store': store,
                'Date': date,
                'Sales': np.random.randint(1000, 10000),
                'Customers': np.random.randint(100, 1000),
                'Promo': np.random.choice([0, 1]),
                'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], p=[0.9, 0.03, 0.04, 0.03]),
                'SchoolHoliday': np.random.choice([0, 1], p=[0.8, 0.2])
            })

    return pd.DataFrame(data)


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Generate large dataset for performance testing."""
    np.random.seed(42)

    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    stores = range(1, 101)  # 100 stores

    data = []
    for store in stores:
        for date in dates:
            data.append({
                'Store': store,
                'Date': date,
                'Sales': np.random.randint(1000, 10000),
                'Customers': np.random.randint(100, 1000),
                'Promo': np.random.choice([0, 1]),
                'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], p=[0.9, 0.03, 0.04, 0.03]),
                'SchoolHoliday': np.random.choice([0, 1], p=[0.8, 0.2])
            })

    return pd.DataFrame(data)


@pytest.fixture
def malformed_data() -> pd.DataFrame:
    """Generate data with missing values and anomalies."""
    np.random.seed(42)

    data = {
        'Store': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Date': pd.date_range(start='2023-01-01', periods=10),
        'Sales': [1000, np.nan, 2000, -500, 3000, None, 4000, 5000, 6000, 7000],
        'Customers': [100, 200, np.nan, 400, 500, 600, None, 800, 900, 1000],
        'Promo': [0, 1, 0, 1, np.nan, 0, 1, 0, 1, 0],
        'StateHoliday': ['0', 'a', None, 'b', '0', 'c', '0', 'a', 'b', None],
        'SchoolHoliday': [0, 1, 0, np.nan, 1, 0, 1, 0, None, 1]
    }

    return pd.DataFrame(data)


@pytest.fixture
def forecast_data() -> Dict[str, Any]:
    """Generate mock forecast data."""
    dates = pd.date_range(start='2024-01-01', periods=30)

    return {
        'forecast': pd.Series(
            data=np.random.randint(5000, 8000, size=30),
            index=dates,
            name='Sales'
        ),
        'lower_bound': pd.Series(
            data=np.random.randint(4000, 6000, size=30),
            index=dates,
            name='Lower'
        ),
        'upper_bound': pd.Series(
            data=np.random.randint(7000, 10000, size=30),
            index=dates,
            name='Upper'
        ),
        'confidence': 0.95,
        'model_type': 'ARIMA'
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test files."""
    test_dir = tmp_path / "test_dashboard"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def temp_data_file(temp_dir, sample_data):
    """Create temporary CSV file with sample data."""
    data_file = temp_dir / "test_data.csv"
    sample_data.to_csv(data_file, index=False)
    return str(data_file)


@pytest.fixture
def pipeline_config(temp_dir, temp_data_file):
    """Create pipeline configuration for testing."""
    return PipelineConfig(
        data_path=temp_data_file,
        cache_dir=str(temp_dir / "cache"),
        cache_enabled=True,
        cache_ttl=60,  # Short TTL for testing
        incremental_enabled=True,
        max_workers=2,
        chunk_size=1000,
        log_level="WARNING"
    )


@pytest.fixture
def disabled_cache_config(temp_dir, temp_data_file):
    """Create configuration with caching disabled."""
    return PipelineConfig(
        data_path=temp_data_file,
        cache_dir=str(temp_dir / "cache"),
        cache_enabled=False,
        incremental_enabled=False,
        log_level="WARNING"
    )


# ============================================================================
# Component Fixtures
# ============================================================================

@pytest.fixture
def cache_manager(pipeline_config):
    """Create cache manager instance."""
    return CacheManager(pipeline_config)


@pytest.fixture
def data_pipeline(pipeline_config):
    """Create data pipeline instance."""
    return DataPipeline(pipeline_config)


@pytest.fixture
def orchestrator(pipeline_config):
    """Create orchestrator instance."""
    return AnalysisOrchestrator(pipeline_config)


@pytest.fixture
def initialized_orchestrator(orchestrator):
    """Create and initialize orchestrator with data loaded."""
    orchestrator.initialize()
    return orchestrator


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_analysis_module():
    """Create mock analysis module."""
    def analysis_func(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        return {
            'total_sales': data['Sales'].sum(),
            'avg_sales': data['Sales'].mean(),
            'store_count': data['Store'].nunique(),
            'parameters': kwargs
        }
    return analysis_func


@pytest.fixture
def mock_visualization_data():
    """Generate mock visualization data."""
    return {
        'chart_type': 'line',
        'data': {
            'x': list(range(10)),
            'y': [np.random.randint(1000, 5000) for _ in range(10)]
        },
        'config': {
            'title': 'Test Chart',
            'xlabel': 'Time',
            'ylabel': 'Sales'
        }
    }


@pytest.fixture
def mock_alert_config():
    """Generate mock alert configuration."""
    return {
        'alert_id': 'test_alert_001',
        'metric': 'sales',
        'threshold': 5000,
        'condition': 'below',
        'severity': 'high',
        'enabled': True,
        'recipients': ['test@example.com']
    }


# ============================================================================
# Performance Testing Utilities
# ============================================================================

@pytest.fixture
def performance_timer():
    """Fixture for timing operations."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = datetime.now()
            return self

        def __exit__(self, *args):
            self.end_time = datetime.now()

        @property
        def elapsed_ms(self) -> float:
            """Get elapsed time in milliseconds."""
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds() * 1000
            return 0

        @property
        def elapsed_seconds(self) -> float:
            """Get elapsed time in seconds."""
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return 0

    return Timer


@pytest.fixture
def memory_profiler():
    """Fixture for tracking memory usage."""
    import psutil
    import os

    class MemoryProfiler:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = None
            self.final_memory = None

        def __enter__(self):
            self.initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            return self

        def __exit__(self, *args):
            self.final_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

        @property
        def memory_increase_mb(self) -> float:
            """Get memory increase in MB."""
            if self.initial_memory and self.final_memory:
                return self.final_memory - self.initial_memory
            return 0

    return MemoryProfiler


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_dataframe_equal_unordered(df1: pd.DataFrame, df2: pd.DataFrame):
    """Assert two dataframes are equal regardless of row order."""
    pd.testing.assert_frame_equal(
        df1.sort_values(by=list(df1.columns)).reset_index(drop=True),
        df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
    )


def assert_dict_contains_subset(subset: Dict, superset: Dict):
    """Assert that all keys in subset exist in superset with same values."""
    for key, value in subset.items():
        assert key in superset, f"Key '{key}' not found in superset"
        assert superset[key] == value, f"Value mismatch for key '{key}'"


@pytest.fixture
def assert_helpers():
    """Provide assertion helper functions."""
    return {
        'dataframe_equal_unordered': assert_dataframe_equal_unordered,
        'dict_contains_subset': assert_dict_contains_subset
    }


# ============================================================================
# Parametrization Fixtures
# ============================================================================

@pytest.fixture(params=[True, False])
def cache_enabled(request):
    """Parametrize tests with cache enabled/disabled."""
    return request.param


@pytest.fixture(params=['sales', 'customers', 'promotions'])
def analysis_module_name(request):
    """Parametrize tests with different analysis modules."""
    return request.param


@pytest.fixture(params=[1, 10, 100])
def batch_size(request):
    """Parametrize tests with different batch sizes."""
    return request.param


# ============================================================================
# Cleanup and Setup Hooks
# ============================================================================

@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    import logging
    logging.getLogger().handlers.clear()
    yield
    logging.getLogger().handlers.clear()


@pytest.fixture(autouse=True)
def reset_numpy_random():
    """Reset numpy random seed between tests."""
    np.random.seed(42)
    yield


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "visualization: marks tests as visualization tests"
    )
