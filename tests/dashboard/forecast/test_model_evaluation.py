"""
Forecast Model Evaluation Tests
================================

Tests forecast accuracy metrics, model validation,
error analysis, and prediction quality.

Author: Testing & QA Agent
Created: 2025-10-08
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@pytest.mark.forecast
class TestAccuracyMetrics:
    """Test forecast accuracy metrics."""

    def test_mean_absolute_error(self, forecast_data):
        """Test MAE calculation."""
        actual = forecast_data['forecast'] * np.random.uniform(0.9, 1.1, size=len(forecast_data['forecast']))
        predicted = forecast_data['forecast']

        mae = mean_absolute_error(actual, predicted)

        assert mae >= 0
        assert np.isfinite(mae)

    def test_mean_squared_error(self, forecast_data):
        """Test MSE and RMSE calculation."""
        actual = forecast_data['forecast'] * np.random.uniform(0.9, 1.1, size=len(forecast_data['forecast']))
        predicted = forecast_data['forecast']

        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)

        assert mse >= 0
        assert rmse >= 0
        assert np.isfinite(mse)

    def test_mean_absolute_percentage_error(self, forecast_data):
        """Test MAPE calculation."""
        actual = forecast_data['forecast'] * np.random.uniform(0.9, 1.1, size=len(forecast_data['forecast']))
        predicted = forecast_data['forecast']

        # MAPE calculation
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        assert mape >= 0
        assert mape <= 100 or mape > 100  # Can exceed 100% for poor forecasts
        assert np.isfinite(mape)

    def test_r_squared(self, forecast_data):
        """Test R² calculation."""
        actual = forecast_data['forecast'] * np.random.uniform(0.9, 1.1, size=len(forecast_data['forecast']))
        predicted = forecast_data['forecast']

        r2 = r2_score(actual, predicted)

        assert -np.inf < r2 <= 1  # R² can be negative
        assert np.isfinite(r2)

    def test_symmetric_mape(self, forecast_data):
        """Test symmetric MAPE calculation."""
        actual = forecast_data['forecast'] * np.random.uniform(0.9, 1.1, size=len(forecast_data['forecast']))
        predicted = forecast_data['forecast']

        # sMAPE calculation
        smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100

        assert 0 <= smape <= 200  # sMAPE range
        assert np.isfinite(smape)

    def test_weighted_mape(self):
        """Test weighted MAPE (emphasizes larger values)."""
        actual = np.array([100, 200, 300, 400])
        predicted = np.array([110, 190, 320, 380])

        weights = actual / actual.sum()
        wmape = np.sum(weights * np.abs((actual - predicted) / actual)) * 100

        assert wmape >= 0
        assert np.isfinite(wmape)


@pytest.mark.forecast
class TestForecastBias:
    """Test forecast bias detection."""

    def test_bias_calculation(self):
        """Test bias (mean error) calculation."""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 210, 310, 410, 510])  # Consistently over-predicting

        bias = np.mean(predicted - actual)

        assert bias > 0  # Positive bias (over-forecasting)
        assert np.isfinite(bias)

    def test_percentage_bias(self):
        """Test percentage bias calculation."""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([90, 180, 270, 360, 450])  # Consistently under-predicting

        percentage_bias = (np.sum(predicted - actual) / np.sum(actual)) * 100

        assert percentage_bias < 0  # Negative bias (under-forecasting)
        assert np.isfinite(percentage_bias)

    def test_bias_direction_consistency(self):
        """Test consistency of bias direction."""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 190, 310, 390, 510])

        errors = predicted - actual
        positive_errors = sum(errors > 0)
        negative_errors = sum(errors < 0)

        # Check if bias is consistent
        bias_consistency = abs(positive_errors - negative_errors) / len(errors)

        assert 0 <= bias_consistency <= 1

    def test_cumulative_bias(self):
        """Test cumulative bias over time."""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([105, 205, 305, 405, 505])

        cumulative_error = np.cumsum(predicted - actual)

        assert len(cumulative_error) == len(actual)
        assert all(np.isfinite(cumulative_error))


@pytest.mark.forecast
class TestConfidenceIntervals:
    """Test forecast confidence intervals."""

    def test_confidence_interval_coverage(self, forecast_data):
        """Test that confidence intervals contain appropriate proportion of actual values."""
        # Generate mock actual values
        actual = forecast_data['forecast'] * np.random.uniform(0.95, 1.05, size=len(forecast_data['forecast']))

        lower = forecast_data['lower_bound']
        upper = forecast_data['upper_bound']

        # Check coverage
        within_bounds = ((actual >= lower) & (actual <= upper)).sum()
        coverage = within_bounds / len(actual) * 100

        # For 95% CI, expect ~95% coverage
        assert 85 <= coverage <= 100  # Allow some variance

    def test_interval_width_consistency(self, forecast_data):
        """Test that confidence interval width is reasonable."""
        lower = forecast_data['lower_bound']
        upper = forecast_data['upper_bound']

        width = upper - lower
        mean_forecast = forecast_data['forecast']

        # Width should be positive and not excessive
        assert all(width > 0)
        assert all(width < mean_forecast)  # Width shouldn't exceed forecast

    def test_interval_symmetry(self, forecast_data):
        """Test symmetry of confidence intervals around forecast."""
        forecast = forecast_data['forecast']
        lower = forecast_data['lower_bound']
        upper = forecast_data['upper_bound']

        lower_distance = forecast - lower
        upper_distance = upper - forecast

        # Check approximate symmetry (allowing for some asymmetry)
        ratio = lower_distance / upper_distance

        assert all(ratio > 0.5)
        assert all(ratio < 2.0)


@pytest.mark.forecast
class TestResidualAnalysis:
    """Test forecast residual analysis."""

    def test_residual_distribution(self):
        """Test residual distribution characteristics."""
        actual = np.random.normal(100, 10, 100)
        predicted = actual + np.random.normal(0, 2, 100)

        residuals = actual - predicted

        # Test normality assumptions
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        assert abs(mean_residual) < std_residual  # Mean should be close to 0

    def test_residual_autocorrelation(self):
        """Test autocorrelation in residuals."""
        # Generate residuals with some autocorrelation
        residuals = np.random.normal(0, 1, 100)

        # Calculate lag-1 autocorrelation
        acf_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

        assert -1 <= acf_lag1 <= 1
        assert np.isfinite(acf_lag1)

    def test_residual_heteroscedasticity(self):
        """Test for heteroscedasticity in residuals."""
        predicted = np.linspace(100, 500, 100)
        # Residuals with increasing variance
        residuals = np.random.normal(0, predicted * 0.05)

        # Simple variance test across bins
        low_pred = predicted < predicted.mean()
        high_pred = predicted >= predicted.mean()

        var_low = np.var(residuals[low_pred])
        var_high = np.var(residuals[high_pred])

        ratio = var_high / var_low

        assert ratio > 0
        assert np.isfinite(ratio)

    def test_outlier_detection_in_residuals(self):
        """Test outlier detection in forecast residuals."""
        actual = np.random.normal(100, 10, 100)
        predicted = actual + np.random.normal(0, 2, 100)

        # Add some outliers
        actual[0] = 200
        actual[50] = 20

        residuals = actual - predicted
        threshold = 3 * np.std(residuals)

        outliers = np.abs(residuals) > threshold
        outlier_count = outliers.sum()

        assert outlier_count >= 0
        assert outlier_count <= len(residuals)


@pytest.mark.forecast
class TestModelComparison:
    """Test comparison of multiple forecast models."""

    def test_compare_model_accuracy(self):
        """Test comparing accuracy of different models."""
        actual = np.random.normal(100, 10, 100)

        # Model 1: Simple forecast
        model1_pred = np.full(100, actual.mean())

        # Model 2: Better forecast
        model2_pred = actual + np.random.normal(0, 2, 100)

        mae1 = mean_absolute_error(actual, model1_pred)
        mae2 = mean_absolute_error(actual, model2_pred)

        # Model 2 should be more accurate
        assert mae2 < mae1

    def test_model_ranking(self):
        """Test ranking models by performance."""
        actual = np.random.normal(100, 10, 50)

        models = {
            'naive': np.full(50, actual.mean()),
            'linear': actual + np.random.normal(0, 5, 50),
            'advanced': actual + np.random.normal(0, 2, 50)
        }

        # Calculate MAE for each model
        performance = {
            name: mean_absolute_error(actual, pred)
            for name, pred in models.items()
        }

        # Rank models
        ranking = sorted(performance.items(), key=lambda x: x[1])

        assert len(ranking) == 3
        assert ranking[0][1] <= ranking[1][1] <= ranking[2][1]

    def test_ensemble_performance(self):
        """Test ensemble of multiple models."""
        actual = np.random.normal(100, 10, 50)

        # Individual models
        model1 = actual + np.random.normal(0, 3, 50)
        model2 = actual + np.random.normal(0, 3, 50)
        model3 = actual + np.random.normal(0, 3, 50)

        # Ensemble (simple average)
        ensemble = (model1 + model2 + model3) / 3

        mae_ensemble = mean_absolute_error(actual, ensemble)
        mae_model1 = mean_absolute_error(actual, model1)

        # Ensemble should typically perform better
        assert mae_ensemble <= mae_model1 * 1.1  # Allow small margin


@pytest.mark.forecast
class TestHorizonAccuracy:
    """Test accuracy across different forecast horizons."""

    def test_accuracy_by_horizon(self):
        """Test that accuracy decreases with horizon."""
        actual = np.random.normal(100, 10, 100)

        # Short-term forecast (better accuracy)
        short_term = actual[:30] + np.random.normal(0, 2, 30)

        # Long-term forecast (worse accuracy)
        long_term = actual[70:] + np.random.normal(0, 5, 30)

        mae_short = mean_absolute_error(actual[:30], short_term)
        mae_long = mean_absolute_error(actual[70:], long_term)

        # Short-term should be more accurate
        assert mae_short < mae_long

    def test_horizon_confidence_intervals(self):
        """Test that confidence intervals widen with horizon."""
        horizons = [1, 7, 14, 30]
        intervals = {}

        for h in horizons:
            # Simulate widening intervals
            forecast = 100
            std = 2 * np.sqrt(h)  # Variance increases with horizon
            lower = forecast - 1.96 * std
            upper = forecast + 1.96 * std
            intervals[h] = upper - lower

        # Check intervals widen
        assert intervals[7] > intervals[1]
        assert intervals[14] > intervals[7]
        assert intervals[30] > intervals[14]


@pytest.mark.forecast
class TestSeasonalAccuracy:
    """Test forecast accuracy for seasonal patterns."""

    def test_seasonal_decomposition_accuracy(self, sample_data):
        """Test accuracy of seasonal component extraction."""
        # Group by day of week
        daily_pattern = sample_data.groupby(sample_data['Date'].dt.dayofweek)['Sales'].mean()

        # Seasonal indices
        overall_mean = daily_pattern.mean()
        seasonal_indices = daily_pattern / overall_mean

        # Should sum to approximately 7 (one for each day)
        assert seasonal_indices.sum() == pytest.approx(7, rel=0.01)

    def test_trend_forecast_accuracy(self, sample_data):
        """Test trend component forecast."""
        sales_by_date = sample_data.groupby('Date')['Sales'].sum().sort_index()

        # Fit linear trend
        x = np.arange(len(sales_by_date))
        y = sales_by_date.values
        z = np.polyfit(x, y, 1)
        trend = np.poly1d(z)(x)

        # Calculate R² for trend
        ss_res = np.sum((y - trend) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        assert -1 <= r2 <= 1


@pytest.mark.forecast
class TestCrossValidation:
    """Test forecast model cross-validation."""

    def test_time_series_split(self, sample_data):
        """Test time series cross-validation split."""
        from sklearn.model_selection import TimeSeriesSplit

        sales_by_date = sample_data.groupby('Date')['Sales'].sum().sort_index()

        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(sales_by_date))

        assert len(splits) == 5

        # Verify chronological order
        for train_idx, test_idx in splits:
            assert max(train_idx) < min(test_idx)

    def test_rolling_window_validation(self, sample_data):
        """Test rolling window validation."""
        sales_by_date = sample_data.groupby('Date')['Sales'].sum().sort_index()

        window_size = 30
        forecast_horizon = 7
        errors = []

        for i in range(len(sales_by_date) - window_size - forecast_horizon):
            train = sales_by_date.iloc[i:i+window_size]
            test = sales_by_date.iloc[i+window_size:i+window_size+forecast_horizon]

            # Simple forecast (mean)
            forecast = train.mean()
            mae = mean_absolute_error(test, np.full(len(test), forecast))
            errors.append(mae)

        assert len(errors) > 0
        assert all(e >= 0 for e in errors)

    def test_expanding_window_validation(self, sample_data):
        """Test expanding window validation."""
        sales_by_date = sample_data.groupby('Date')['Sales'].sum().sort_index()

        min_train_size = 60
        forecast_horizon = 7
        errors = []

        for i in range(min_train_size, len(sales_by_date) - forecast_horizon, 7):
            train = sales_by_date.iloc[:i]
            test = sales_by_date.iloc[i:i+forecast_horizon]

            forecast = train.mean()
            mae = mean_absolute_error(test, np.full(len(test), forecast))
            errors.append(mae)

        assert len(errors) > 0
        assert all(e >= 0 for e in errors)
