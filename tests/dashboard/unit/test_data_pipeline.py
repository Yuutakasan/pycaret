"""
Unit Tests for Data Pipeline
=============================

Tests data loading, preprocessing, filtering, and incremental updates.

Author: Testing & QA Agent
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.dashboard.orchestrator import DataPipeline, PipelineConfig


class TestDataPipelineInitialization:
    """Test DataPipeline initialization."""

    def test_pipeline_creation(self, pipeline_config):
        """Test creating data pipeline."""
        pipeline = DataPipeline(pipeline_config)

        assert pipeline.config == pipeline_config
        assert pipeline.cache_manager is not None
        assert pipeline.raw_data is None
        assert pipeline.processed_data is None
        assert isinstance(pipeline.metadata, dict)

    def test_pipeline_with_custom_config(self, temp_data_file):
        """Test pipeline with custom configuration."""
        config = PipelineConfig(
            data_path=temp_data_file,
            cache_enabled=False,
            incremental_enabled=False,
            chunk_size=5000
        )

        pipeline = DataPipeline(config)
        assert not pipeline.config.cache_enabled
        assert not pipeline.config.incremental_enabled
        assert pipeline.config.chunk_size == 5000


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_data_success(self, data_pipeline):
        """Test successful data loading."""
        df = data_pipeline.load_data()

        assert df is not None
        assert len(df) > 0
        assert all(col in df.columns for col in data_pipeline.config.required_columns)
        assert data_pipeline.raw_data is not None

    def test_load_data_metadata(self, data_pipeline):
        """Test metadata generation during load."""
        df = data_pipeline.load_data()

        assert 'load_time' in data_pipeline.metadata
        assert 'rows' in data_pipeline.metadata
        assert 'columns' in data_pipeline.metadata
        assert 'memory_mb' in data_pipeline.metadata

        assert data_pipeline.metadata['rows'] == len(df)
        assert len(data_pipeline.metadata['columns']) == len(df.columns)

    def test_load_data_caching(self, data_pipeline):
        """Test that data loading uses cache."""
        df1 = data_pipeline.load_data()
        df2 = data_pipeline.load_data()

        # Should return same object from cache
        pd.testing.assert_frame_equal(df1, df2)

    def test_load_data_missing_file(self, temp_dir):
        """Test loading from nonexistent file."""
        config = PipelineConfig(
            data_path=str(temp_dir / "nonexistent.csv"),
            cache_dir=str(temp_dir / "cache")
        )
        pipeline = DataPipeline(config)

        with pytest.raises(Exception):
            pipeline.load_data()

    def test_load_data_missing_columns(self, temp_dir):
        """Test loading data with missing required columns."""
        # Create CSV without required columns
        df = pd.DataFrame({
            'Col1': [1, 2, 3],
            'Col2': [4, 5, 6]
        })
        data_file = temp_dir / "incomplete.csv"
        df.to_csv(data_file, index=False)

        config = PipelineConfig(
            data_path=str(data_file),
            cache_dir=str(temp_dir / "cache"),
            cache_enabled=False
        )
        pipeline = DataPipeline(config)

        with pytest.raises(ValueError, match="Missing required columns"):
            pipeline.load_data()

    def test_force_reload(self, data_pipeline):
        """Test force reload bypasses cache."""
        df1 = data_pipeline.load_data()

        # Modify cached data
        data_pipeline.raw_data.loc[0, 'Sales'] = 999999

        # Force reload should get fresh data
        df2 = data_pipeline.load_data(force_reload=True)

        assert df2.loc[0, 'Sales'] != 999999


class TestDataPreprocessing:
    """Test data preprocessing functionality."""

    def test_preprocess_basic(self, data_pipeline, sample_data):
        """Test basic preprocessing."""
        df = data_pipeline.preprocess_data(sample_data)

        assert df is not None
        assert len(df) > 0
        assert data_pipeline.processed_data is not None

    def test_preprocess_date_conversion(self, data_pipeline, sample_data):
        """Test date column conversion."""
        df = data_pipeline.preprocess_data(sample_data)

        assert pd.api.types.is_datetime64_any_dtype(df['Date'])

    def test_preprocess_derived_features(self, data_pipeline, sample_data):
        """Test derived feature creation."""
        df = data_pipeline.preprocess_data(sample_data)

        assert 'Year' in df.columns
        assert 'Month' in df.columns
        assert 'Quarter' in df.columns
        assert 'DayOfWeek' in df.columns
        assert 'WeekOfYear' in df.columns

        # Validate derived values
        assert df['Month'].min() >= 1
        assert df['Month'].max() <= 12
        assert df['Quarter'].min() >= 1
        assert df['Quarter'].max() <= 4
        assert df['DayOfWeek'].min() >= 0
        assert df['DayOfWeek'].max() <= 6

    def test_preprocess_missing_values(self, data_pipeline, malformed_data):
        """Test handling of missing values."""
        df = data_pipeline.preprocess_data(malformed_data)

        # Check numeric columns filled with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert df[numeric_cols].isna().sum().sum() == 0

        # Check categorical columns filled with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        assert df[categorical_cols].isna().sum().sum() == 0

    def test_preprocess_duplicate_removal(self, data_pipeline):
        """Test duplicate row removal."""
        # Create data with duplicates
        df = pd.DataFrame({
            'Store': [1, 1, 2, 2, 3],
            'Date': pd.date_range('2023-01-01', periods=5),
            'Sales': [1000, 1000, 2000, 2000, 3000],
            'Customers': [100, 100, 200, 200, 300],
            'Promo': [0, 0, 1, 1, 0],
            'StateHoliday': ['0', '0', 'a', 'a', '0'],
            'SchoolHoliday': [0, 0, 1, 1, 0]
        })

        # Create duplicates
        df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

        result = data_pipeline.preprocess_data(df)

        assert len(result) < len(df)

    def test_preprocess_preserves_original(self, data_pipeline, sample_data):
        """Test that preprocessing doesn't modify original data."""
        original = sample_data.copy()
        data_pipeline.preprocess_data(sample_data)

        pd.testing.assert_frame_equal(sample_data, original)

    def test_preprocess_error_handling(self, data_pipeline):
        """Test preprocessing error handling."""
        invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})

        with pytest.raises(Exception):
            data_pipeline.preprocess_data(invalid_df)


class TestDataFiltering:
    """Test data filtering functionality."""

    def test_filter_by_store_single(self, data_pipeline, sample_data):
        """Test filtering by single store."""
        df = data_pipeline.preprocess_data(sample_data)
        filtered = data_pipeline.filter_by_store(df, [1])

        assert len(filtered) > 0
        assert filtered['Store'].unique().tolist() == [1]

    def test_filter_by_store_multiple(self, data_pipeline, sample_data):
        """Test filtering by multiple stores."""
        df = data_pipeline.preprocess_data(sample_data)
        filtered = data_pipeline.filter_by_store(df, [1, 2, 3])

        assert len(filtered) > 0
        assert set(filtered['Store'].unique()) == {1, 2, 3}

    def test_filter_by_store_empty_list(self, data_pipeline, sample_data):
        """Test filtering with empty store list."""
        df = data_pipeline.preprocess_data(sample_data)
        filtered = data_pipeline.filter_by_store(df, [])

        # Empty list should return all data
        pd.testing.assert_frame_equal(filtered, df)

    def test_filter_by_store_nonexistent(self, data_pipeline, sample_data):
        """Test filtering by nonexistent store."""
        df = data_pipeline.preprocess_data(sample_data)
        filtered = data_pipeline.filter_by_store(df, [999])

        assert len(filtered) == 0

    def test_filter_by_date_range_both(self, data_pipeline, sample_data):
        """Test filtering by start and end date."""
        df = data_pipeline.preprocess_data(sample_data)
        filtered = data_pipeline.filter_by_date_range(
            df,
            start_date="2023-06-01",
            end_date="2023-06-30"
        )

        assert len(filtered) > 0
        assert filtered['Date'].min() >= pd.to_datetime("2023-06-01")
        assert filtered['Date'].max() <= pd.to_datetime("2023-06-30")

    def test_filter_by_date_range_start_only(self, data_pipeline, sample_data):
        """Test filtering by start date only."""
        df = data_pipeline.preprocess_data(sample_data)
        filtered = data_pipeline.filter_by_date_range(df, start_date="2023-07-01")

        assert len(filtered) > 0
        assert filtered['Date'].min() >= pd.to_datetime("2023-07-01")

    def test_filter_by_date_range_end_only(self, data_pipeline, sample_data):
        """Test filtering by end date only."""
        df = data_pipeline.preprocess_data(sample_data)
        filtered = data_pipeline.filter_by_date_range(df, end_date="2023-03-31")

        assert len(filtered) > 0
        assert filtered['Date'].max() <= pd.to_datetime("2023-03-31")

    def test_filter_by_date_range_none(self, data_pipeline, sample_data):
        """Test filtering with no date constraints."""
        df = data_pipeline.preprocess_data(sample_data)
        filtered = data_pipeline.filter_by_date_range(df)

        pd.testing.assert_frame_equal(filtered, df)

    def test_combined_filters(self, data_pipeline, sample_data):
        """Test combining store and date filters."""
        df = data_pipeline.preprocess_data(sample_data)

        # Filter by store first
        filtered1 = data_pipeline.filter_by_store(df, [1, 2])

        # Then filter by date
        filtered2 = data_pipeline.filter_by_date_range(
            filtered1,
            start_date="2023-06-01",
            end_date="2023-06-30"
        )

        assert len(filtered2) > 0
        assert set(filtered2['Store'].unique()).issubset({1, 2})
        assert filtered2['Date'].min() >= pd.to_datetime("2023-06-01")
        assert filtered2['Date'].max() <= pd.to_datetime("2023-06-30")


class TestIncrementalUpdates:
    """Test incremental update functionality."""

    def test_get_incremental_updates(self, data_pipeline, sample_data):
        """Test getting incremental updates."""
        df = data_pipeline.preprocess_data(sample_data)
        data_pipeline.processed_data = df

        last_update = pd.to_datetime("2023-06-01")
        new_data, metadata = data_pipeline.get_incremental_updates(last_update)

        assert len(new_data) > 0
        assert new_data['Date'].min() > last_update
        assert 'new_rows' in metadata
        assert 'total_rows' in metadata
        assert metadata['new_rows'] == len(new_data)

    def test_incremental_updates_no_new_data(self, data_pipeline, sample_data):
        """Test incremental updates when no new data."""
        df = data_pipeline.preprocess_data(sample_data)
        data_pipeline.processed_data = df

        # Use future date
        last_update = pd.to_datetime("2024-12-31")
        new_data, metadata = data_pipeline.get_incremental_updates(last_update)

        assert len(new_data) == 0
        assert metadata['new_rows'] == 0

    def test_incremental_disabled(self, temp_data_file, temp_dir):
        """Test incremental updates when disabled."""
        config = PipelineConfig(
            data_path=temp_data_file,
            cache_dir=str(temp_dir / "cache"),
            incremental_enabled=False
        )
        pipeline = DataPipeline(config)
        df = pipeline.load_data()
        pipeline.preprocess_data(df)

        last_update = pd.to_datetime("2023-06-01")
        new_data, metadata = pipeline.get_incremental_updates(last_update)

        # Should return all processed data
        pd.testing.assert_frame_equal(new_data, pipeline.processed_data)

    def test_incremental_updates_metadata(self, data_pipeline, sample_data):
        """Test incremental update metadata."""
        df = data_pipeline.preprocess_data(sample_data)
        data_pipeline.processed_data = df

        last_update = pd.to_datetime("2023-06-01")
        new_data, metadata = data_pipeline.get_incremental_updates(last_update)

        assert 'last_update' in metadata
        assert 'current_time' in metadata
        assert 'new_rows' in metadata
        assert 'total_rows' in metadata

        assert metadata['total_rows'] == len(df)
        assert metadata['new_rows'] <= metadata['total_rows']


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self, data_pipeline):
        """Test preprocessing empty dataframe."""
        empty_df = pd.DataFrame(columns=data_pipeline.config.required_columns)
        result = data_pipeline.preprocess_data(empty_df)

        assert len(result) == 0
        assert all(col in result.columns for col in data_pipeline.config.required_columns)

    def test_single_row_dataframe(self, data_pipeline):
        """Test processing single row."""
        single_row = pd.DataFrame({
            'Store': [1],
            'Date': [pd.to_datetime('2023-01-01')],
            'Sales': [1000],
            'Customers': [100],
            'Promo': [0],
            'StateHoliday': ['0'],
            'SchoolHoliday': [0]
        })

        result = data_pipeline.preprocess_data(single_row)
        assert len(result) == 1

    def test_large_dataset_performance(self, temp_dir, large_dataset):
        """Test performance with large dataset."""
        import time

        # Save large dataset
        data_file = temp_dir / "large_data.csv"
        large_dataset.to_csv(data_file, index=False)

        config = PipelineConfig(
            data_path=str(data_file),
            cache_dir=str(temp_dir / "cache"),
            log_level="ERROR"
        )
        pipeline = DataPipeline(config)

        # Time the load and preprocess
        start = time.time()
        df = pipeline.load_data()
        pipeline.preprocess_data(df)
        elapsed = time.time() - start

        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 30  # seconds
        assert len(pipeline.processed_data) > 100000
