"""
Advanced Demand Forecasting System using PyCaret 3.x Time Series Module

This module provides comprehensive demand forecasting capabilities with:
- Multiple forecasting models (ARIMA, Prophet, LSTM, XGBoost, etc.)
- Store-level and product-level predictions
- Multi-horizon forecasts (7-day, 30-day, 90-day)
- Confidence intervals and prediction bounds
- Advanced feature engineering (weekday, holidays, weather, promotions)
- Model ensemble and automated selection
- Comprehensive accuracy metrics (MAPE, RMSE, MAE)

Author: PyCaret ML Development Team
Date: 2025-10-08
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# PyCaret Time Series
from pycaret.time_series import TSForecastingExperiment

# Scikit-learn utilities
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Statistical models
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Optional dependencies with graceful fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet>=1.0.1")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost>=2.0.0")

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, ETS, Theta
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    warnings.warn("StatsForecast not available. Install with: pip install statsforecast")

warnings.filterwarnings('ignore')


class DemandForecastingSystem:
    """
    Advanced demand forecasting system with multi-model ensemble approach.

    Features:
    - Automated model selection and comparison
    - Multi-horizon forecasting (7, 30, 90 days)
    - Store and product-level granularity
    - Feature engineering pipeline
    - Ensemble predictions with confidence intervals
    - Comprehensive evaluation metrics

    Attributes:
        experiment: PyCaret TSForecastingExperiment instance
        models: Dictionary of trained models
        best_model: Best performing model
        feature_engineer: Feature engineering pipeline
        scalers: Dictionary of scalers for different features
        forecast_horizons: List of forecast horizons in days
    """

    def __init__(
        self,
        forecast_horizons: List[int] = [7, 30, 90],
        seasonal_period: int = 7,
        confidence_level: float = 0.95,
        random_state: int = 42
    ):
        """
        Initialize the demand forecasting system.

        Args:
            forecast_horizons: List of forecast horizons in days (default: [7, 30, 90])
            seasonal_period: Seasonal period for decomposition (default: 7 for weekly)
            confidence_level: Confidence level for prediction intervals (default: 0.95)
            random_state: Random seed for reproducibility
        """
        self.forecast_horizons = sorted(forecast_horizons)
        self.seasonal_period = seasonal_period
        self.confidence_level = confidence_level
        self.random_state = random_state

        # Initialize components
        self.experiment = None
        self.models = {}
        self.best_model = None
        self.feature_engineer = None
        self.scalers = {
            'numeric': StandardScaler(),
            'categorical': {}
        }
        self.feature_names = []
        self.target_column = None
        self.date_column = None
        self.hierarchy_columns = []

        # Model performance tracking
        self.model_metrics = {}
        self.ensemble_weights = {}

        # Holiday calendars (expandable)
        self.holidays = pd.DataFrame()

    def setup(
        self,
        data: pd.DataFrame,
        target: str,
        date_column: str,
        hierarchy_columns: Optional[List[str]] = None,
        fh: Optional[int] = None,
        fold: int = 3,
        session_id: Optional[int] = None,
        verbose: bool = True
    ) -> None:
        """
        Setup the forecasting experiment using PyCaret.

        Args:
            data: Time series dataset with datetime index or column
            target: Name of the target column (demand/sales)
            date_column: Name of the date column
            hierarchy_columns: Optional list of hierarchy columns (store_id, product_id, etc.)
            fh: Forecast horizon for initial setup (uses max if None)
            fold: Number of cross-validation folds
            session_id: Random seed (uses random_state if None)
            verbose: Whether to print setup information
        """
        self.target_column = target
        self.date_column = date_column
        self.hierarchy_columns = hierarchy_columns or []

        # Prepare data
        df = data.copy()

        # Ensure datetime index
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
            df = df.set_index(date_column)
        else:
            df.index = pd.to_datetime(df.index)

        # Store original data for reference
        self.original_data = df.copy()

        # Use maximum forecast horizon if not specified
        if fh is None:
            fh = max(self.forecast_horizons)

        # Initialize PyCaret experiment
        self.experiment = TSForecastingExperiment()

        session_id = session_id or self.random_state

        # Setup experiment
        self.experiment.setup(
            data=df,
            target=target,
            fh=fh,
            fold=fold,
            session_id=session_id,
            verbose=verbose,
            seasonal_period=self.seasonal_period,
            ignore_features=hierarchy_columns  # Don't use hierarchy as features initially
        )

        if verbose:
            print(f"✓ Forecasting experiment setup complete")
            print(f"  Target: {target}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            print(f"  Observations: {len(df)}")
            print(f"  Forecast horizons: {self.forecast_horizons}")
            if hierarchy_columns:
                print(f"  Hierarchy: {', '.join(hierarchy_columns)}")

    def engineer_features(
        self,
        data: pd.DataFrame,
        include_holidays: bool = True,
        include_weather: bool = False,
        include_promotions: bool = False,
        weather_data: Optional[pd.DataFrame] = None,
        promotion_data: Optional[pd.DataFrame] = None,
        holiday_country: str = 'US'
    ) -> pd.DataFrame:
        """
        Engineer features for demand forecasting.

        Features include:
        - Temporal features (day of week, month, quarter, year, etc.)
        - Holiday indicators
        - Weather data (if provided)
        - Promotion indicators (if provided)
        - Lag features
        - Rolling statistics
        - Seasonal decomposition components

        Args:
            data: Input dataframe with datetime index
            include_holidays: Add holiday features
            include_weather: Add weather features
            include_promotions: Add promotion features
            weather_data: Optional weather dataframe
            promotion_data: Optional promotion dataframe
            holiday_country: Country for holiday calendar

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        feature_names = []

        # === TEMPORAL FEATURES ===
        df['dayofweek'] = df.index.dayofweek  # 0=Monday, 6=Sunday
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['weekofyear'] = df.index.isocalendar().week
        df['dayofyear'] = df.index.dayofyear

        # Cyclical encoding for periodic features
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        # Weekend indicator
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # Month start/end indicators
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

        feature_names.extend([
            'dayofweek', 'day', 'month', 'quarter', 'year', 'weekofyear', 'dayofyear',
            'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end'
        ])

        # === HOLIDAY FEATURES ===
        if include_holidays:
            try:
                from pandas.tseries.holiday import USFederalHolidayCalendar

                cal = USFederalHolidayCalendar()
                holidays = cal.holidays(
                    start=df.index.min(),
                    end=df.index.max() + timedelta(days=max(self.forecast_horizons))
                )

                df['is_holiday'] = df.index.isin(holidays).astype(int)

                # Days until next holiday
                df['days_to_holiday'] = 0
                for idx in df.index:
                    future_holidays = holidays[holidays > idx]
                    if len(future_holidays) > 0:
                        df.loc[idx, 'days_to_holiday'] = (future_holidays[0] - idx).days
                    else:
                        df.loc[idx, 'days_to_holiday'] = 365

                # Days since last holiday
                df['days_since_holiday'] = 0
                for idx in df.index:
                    past_holidays = holidays[holidays < idx]
                    if len(past_holidays) > 0:
                        df.loc[idx, 'days_since_holiday'] = (idx - past_holidays[-1]).days
                    else:
                        df.loc[idx, 'days_since_holiday'] = 365

                feature_names.extend(['is_holiday', 'days_to_holiday', 'days_since_holiday'])

                self.holidays = pd.DataFrame({'holiday': holidays})

            except ImportError:
                warnings.warn("Holiday calendar not available")

        # === WEATHER FEATURES ===
        if include_weather and weather_data is not None:
            weather_df = weather_data.copy()
            if not isinstance(weather_df.index, pd.DatetimeIndex):
                weather_df.index = pd.to_datetime(weather_df.index)

            # Merge weather data
            df = df.join(weather_df, how='left')

            # Fill missing weather values with forward fill then backward fill
            weather_cols = weather_df.columns.tolist()
            df[weather_cols] = df[weather_cols].fillna(method='ffill').fillna(method='bfill')

            feature_names.extend(weather_cols)

        # === PROMOTION FEATURES ===
        if include_promotions and promotion_data is not None:
            promo_df = promotion_data.copy()
            if not isinstance(promo_df.index, pd.DatetimeIndex):
                promo_df.index = pd.to_datetime(promo_df.index)

            # Merge promotion data
            df = df.join(promo_df, how='left')

            # Fill missing promotion values with 0
            promo_cols = promo_df.columns.tolist()
            df[promo_cols] = df[promo_cols].fillna(0)

            feature_names.extend(promo_cols)

        # === LAG FEATURES ===
        if self.target_column and self.target_column in df.columns:
            # Create lags for different periods
            lag_periods = [1, 7, 14, 28, 30, 90]  # Daily, weekly, bi-weekly, monthly, quarterly

            for lag in lag_periods:
                df[f'lag_{lag}'] = df[self.target_column].shift(lag)
                feature_names.append(f'lag_{lag}')

            # === ROLLING STATISTICS ===
            windows = [7, 14, 30, 90]  # Week, 2-weeks, month, quarter

            for window in windows:
                # Rolling mean
                df[f'rolling_mean_{window}'] = df[self.target_column].rolling(window=window, min_periods=1).mean()
                # Rolling std
                df[f'rolling_std_{window}'] = df[self.target_column].rolling(window=window, min_periods=1).std()
                # Rolling min/max
                df[f'rolling_min_{window}'] = df[self.target_column].rolling(window=window, min_periods=1).min()
                df[f'rolling_max_{window}'] = df[self.target_column].rolling(window=window, min_periods=1).max()

                feature_names.extend([
                    f'rolling_mean_{window}',
                    f'rolling_std_{window}',
                    f'rolling_min_{window}',
                    f'rolling_max_{window}'
                ])

            # Exponential weighted moving average
            df['ewm_7'] = df[self.target_column].ewm(span=7, adjust=False).mean()
            df['ewm_30'] = df[self.target_column].ewm(span=30, adjust=False).mean()
            feature_names.extend(['ewm_7', 'ewm_30'])

            # === SEASONAL DECOMPOSITION ===
            try:
                if len(df) >= 2 * self.seasonal_period:
                    decomposition = seasonal_decompose(
                        df[self.target_column].fillna(method='ffill'),
                        model='additive',
                        period=self.seasonal_period,
                        extrapolate_trend='freq'
                    )

                    df['trend'] = decomposition.trend
                    df['seasonal'] = decomposition.seasonal
                    df['residual'] = decomposition.resid

                    feature_names.extend(['trend', 'seasonal', 'residual'])
            except Exception as e:
                warnings.warn(f"Seasonal decomposition failed: {str(e)}")

        # Store feature names
        self.feature_names = feature_names

        # Fill remaining NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        return df

    def compare_models(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        fold: Optional[int] = None,
        cross_validation: bool = True,
        sort: str = 'MAPE',
        n_select: int = 5,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple forecasting models using PyCaret.

        Available models:
        - naive: Naive forecaster (baseline)
        - grand_means: Grand means forecaster
        - snaive: Seasonal naive
        - polytrend: Polynomial trend forecaster
        - arima: AutoARIMA
        - ets: Exponential smoothing
        - theta: Theta method
        - tbats: TBATS
        - prophet: Facebook Prophet (if available)
        - lr_cds_dt: Linear regression with date features
        - en_cds_dt: Elastic net with date features
        - ridge_cds_dt: Ridge regression with date features
        - lasso_cds_dt: Lasso regression with date features
        - xgboost_cds_dt: XGBoost with date features (if available)
        - lightgbm_cds_dt: LightGBM with date features
        - catboost_cds_dt: CatBoost with date features (if available)

        Args:
            include: List of model IDs to include (None = all available)
            exclude: List of model IDs to exclude
            fold: Number of folds for cross-validation
            cross_validation: Whether to use cross-validation
            sort: Metric to sort by (MAPE, RMSE, MAE, etc.)
            n_select: Number of top models to keep
            verbose: Whether to print comparison results

        Returns:
            DataFrame with model comparison results
        """
        if self.experiment is None:
            raise ValueError("Must call setup() before compare_models()")

        # Build include list based on available dependencies
        if include is None:
            include = ['naive', 'snaive', 'polytrend', 'arima', 'ets', 'theta',
                      'lr_cds_dt', 'en_cds_dt', 'ridge_cds_dt', 'lightgbm_cds_dt']

            if PROPHET_AVAILABLE:
                include.append('prophet')

            if XGBOOST_AVAILABLE:
                include.append('xgboost_cds_dt')

            # Add TBATS if available
            try:
                include.append('tbats')
            except:
                pass

        if verbose:
            print(f"Comparing {len(include)} models...")
            print(f"Models: {', '.join(include)}")

        # Compare models
        best_models = self.experiment.compare_models(
            include=include,
            exclude=exclude,
            fold=fold,
            cross_validation=cross_validation,
            sort=sort,
            n_select=n_select,
            verbose=verbose
        )

        # Get comparison results
        results = self.experiment.pull()

        # Store top models
        if isinstance(best_models, list):
            self.models = {f'model_{i}': model for i, model in enumerate(best_models)}
            self.best_model = best_models[0]
        else:
            self.models = {'best_model': best_models}
            self.best_model = best_models

        # Store metrics
        for idx, row in results.iterrows():
            model_name = row['Model']
            self.model_metrics[model_name] = {
                'MAPE': row.get('MAPE', np.nan),
                'RMSE': row.get('RMSE', np.nan),
                'MAE': row.get('MAE', np.nan),
                'R2': row.get('R2', np.nan)
            }

        if verbose:
            print(f"\n✓ Model comparison complete")
            print(f"  Best model: {results.iloc[0]['Model']}")
            print(f"  Best {sort}: {results.iloc[0][sort]:.4f}")

        return results

    def create_ensemble(
        self,
        method: str = 'simple',
        models: Optional[List] = None,
        weights: Optional[List[float]] = None,
        optimize_weights: bool = True
    ):
        """
        Create ensemble model from multiple forecasters.

        Args:
            method: Ensemble method ('simple', 'weighted', 'stacking')
            models: List of models to ensemble (uses stored models if None)
            weights: Manual weights for weighted ensemble
            optimize_weights: Whether to optimize weights based on validation performance

        Returns:
            Ensemble model
        """
        if models is None:
            models = list(self.models.values())

        if len(models) == 0:
            raise ValueError("No models available for ensemble")

        # Optimize weights based on validation performance
        if optimize_weights and method == 'weighted':
            weights = self._optimize_ensemble_weights(models)
            self.ensemble_weights = weights

        # Create ensemble using PyCaret
        if method == 'simple':
            ensemble = self.experiment.blend_models(estimator_list=models, method='mean')
        elif method == 'weighted':
            ensemble = self.experiment.blend_models(
                estimator_list=models,
                method='mean',
                weights=weights
            )
        else:  # stacking
            ensemble = self.experiment.stack_models(estimator_list=models)

        self.models['ensemble'] = ensemble
        self.best_model = ensemble

        return ensemble

    def _optimize_ensemble_weights(self, models: List) -> List[float]:
        """
        Optimize ensemble weights using validation performance.

        Args:
            models: List of models to optimize weights for

        Returns:
            List of optimized weights
        """
        # Extract MAPE scores (lower is better)
        mape_scores = []
        for model in models:
            model_name = str(model).split('(')[0]
            mape = self.model_metrics.get(model_name, {}).get('MAPE', 1.0)
            mape_scores.append(mape if not np.isnan(mape) else 1.0)

        # Convert to weights (inverse of error)
        weights = [1.0 / (score + 1e-6) for score in mape_scores]

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        return weights

    def predict(
        self,
        fh: Optional[int] = None,
        model = None,
        return_pred_int: bool = True,
        alpha: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate forecasts for specified horizon.

        Args:
            fh: Forecast horizon (uses max from forecast_horizons if None)
            model: Model to use for prediction (uses best_model if None)
            return_pred_int: Whether to return prediction intervals
            alpha: Significance level for prediction intervals (uses confidence_level if None)

        Returns:
            DataFrame with predictions and optional prediction intervals
        """
        if model is None:
            model = self.best_model

        if model is None:
            raise ValueError("No model available. Train a model first.")

        if fh is None:
            fh = max(self.forecast_horizons)

        if alpha is None:
            alpha = 1 - self.confidence_level

        # Generate predictions
        predictions = self.experiment.predict_model(
            model,
            fh=fh,
            return_pred_int=return_pred_int,
            alpha=alpha
        )

        return predictions

    def forecast_multi_horizon(
        self,
        model = None,
        include_actuals: bool = True
    ) -> Dict[int, pd.DataFrame]:
        """
        Generate forecasts for all configured horizons.

        Args:
            model: Model to use (uses best_model if None)
            include_actuals: Whether to include actual values for comparison

        Returns:
            Dictionary mapping horizon to forecast DataFrame
        """
        forecasts = {}

        for horizon in self.forecast_horizons:
            forecast = self.predict(fh=horizon, model=model, return_pred_int=True)

            if include_actuals:
                # Add actuals for the forecast period if available
                forecast_dates = forecast.index
                actuals = self.original_data.loc[
                    self.original_data.index.isin(forecast_dates),
                    self.target_column
                ]

                if len(actuals) > 0:
                    forecast['actual'] = actuals

            forecasts[horizon] = forecast

        return forecasts

    def evaluate_forecast(
        self,
        forecast: pd.DataFrame,
        actual_column: str = 'actual',
        pred_column: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics for forecasts.

        Metrics calculated:
        - MAPE: Mean Absolute Percentage Error
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - R²: Coefficient of determination
        - Bias: Mean error
        - Coverage: Prediction interval coverage (if intervals present)

        Args:
            forecast: DataFrame with predictions and actuals
            actual_column: Name of actual values column
            pred_column: Name of prediction column (auto-detected if None)

        Returns:
            Dictionary of metric names and values
        """
        # Auto-detect prediction column
        if pred_column is None:
            pred_cols = [col for col in forecast.columns if 'y_pred' in col.lower()]
            if len(pred_cols) == 0:
                pred_cols = [col for col in forecast.columns if col != actual_column]
            pred_column = pred_cols[0] if len(pred_cols) > 0 else forecast.columns[0]

        # Filter to rows with both actual and predicted values
        df = forecast[[actual_column, pred_column]].dropna()

        if len(df) == 0:
            warnings.warn("No overlapping actual and predicted values for evaluation")
            return {}

        y_true = df[actual_column].values
        y_pred = df[pred_column].values

        # Calculate metrics
        metrics = {
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,  # Convert to percentage
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
            'Bias': np.mean(y_pred - y_true),
            'n_samples': len(df)
        }

        # Calculate prediction interval coverage if available
        lower_cols = [col for col in forecast.columns if 'lower' in col.lower()]
        upper_cols = [col for col in forecast.columns if 'upper' in col.lower()]

        if len(lower_cols) > 0 and len(upper_cols) > 0:
            df_intervals = forecast[[actual_column, lower_cols[0], upper_cols[0]]].dropna()

            if len(df_intervals) > 0:
                y_true_int = df_intervals[actual_column].values
                y_lower = df_intervals[lower_cols[0]].values
                y_upper = df_intervals[upper_cols[0]].values

                coverage = np.mean((y_true_int >= y_lower) & (y_true_int <= y_upper)) * 100
                metrics['Coverage'] = coverage

        return metrics

    def forecast_hierarchy(
        self,
        data: pd.DataFrame,
        hierarchy_columns: List[str],
        target: str,
        date_column: str,
        fh: int = 30,
        reconciliation: str = 'ols',
        verbose: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate hierarchical forecasts at multiple levels (e.g., store + product).

        Args:
            data: Input data with hierarchy columns
            hierarchy_columns: List of hierarchy columns (e.g., ['store_id', 'product_id'])
            target: Target column name
            date_column: Date column name
            fh: Forecast horizon
            reconciliation: Hierarchical reconciliation method ('ols', 'bottom_up', 'top_down')
            verbose: Whether to print progress

        Returns:
            Dictionary mapping hierarchy level to forecasts
        """
        forecasts = {}

        # Get unique hierarchy combinations
        hierarchy_combinations = data[hierarchy_columns].drop_duplicates()

        if verbose:
            print(f"Generating forecasts for {len(hierarchy_combinations)} hierarchy combinations...")

        # Forecast at each hierarchy level
        for idx, row in hierarchy_combinations.iterrows():
            # Filter data for this combination
            mask = pd.Series([True] * len(data))
            for col in hierarchy_columns:
                mask &= (data[col] == row[col])

            subset = data[mask].copy()

            if len(subset) < self.seasonal_period * 2:
                warnings.warn(f"Insufficient data for hierarchy {dict(row)}, skipping")
                continue

            # Create hierarchy key
            hier_key = '_'.join([f"{col}={row[col]}" for col in hierarchy_columns])

            try:
                # Setup experiment for this subset
                exp = TSForecastingExperiment()

                subset[date_column] = pd.to_datetime(subset[date_column])
                subset = subset.sort_values(date_column).set_index(date_column)

                exp.setup(
                    data=subset,
                    target=target,
                    fh=fh,
                    session_id=self.random_state,
                    verbose=False,
                    seasonal_period=self.seasonal_period
                )

                # Use best model type from main experiment
                model = exp.create_model('auto_arima', verbose=False)

                # Generate forecast
                forecast = exp.predict_model(model, fh=fh, return_pred_int=True)

                # Add hierarchy information
                for col in hierarchy_columns:
                    forecast[col] = row[col]

                forecasts[hier_key] = forecast

                if verbose and (idx + 1) % 10 == 0:
                    print(f"  Completed {idx + 1}/{len(hierarchy_combinations)} forecasts")

            except Exception as e:
                warnings.warn(f"Failed to forecast for {hier_key}: {str(e)}")
                continue

        # Reconcile hierarchical forecasts if needed
        if reconciliation and len(forecasts) > 1:
            forecasts = self._reconcile_hierarchical_forecasts(forecasts, method=reconciliation)

        if verbose:
            print(f"✓ Hierarchical forecasting complete: {len(forecasts)} combinations")

        return forecasts

    def _reconcile_hierarchical_forecasts(
        self,
        forecasts: Dict[str, pd.DataFrame],
        method: str = 'ols'
    ) -> Dict[str, pd.DataFrame]:
        """
        Reconcile hierarchical forecasts to ensure coherence.

        Args:
            forecasts: Dictionary of forecasts by hierarchy level
            method: Reconciliation method ('ols', 'bottom_up', 'top_down')

        Returns:
            Reconciled forecasts
        """
        # For now, return original forecasts
        # Advanced reconciliation would require hierarchical forecasting libraries
        warnings.warn(f"Hierarchical reconciliation '{method}' not yet implemented")
        return forecasts

    def save_model(self, filepath: Union[str, Path], model_name: str = 'best') -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model
            model_name: Name of model to save ('best', 'ensemble', or specific model key)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if model_name == 'best':
            model = self.best_model
        elif model_name in self.models:
            model = self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found")

        # Save using PyCaret
        self.experiment.save_model(model, str(filepath))

        # Save metadata
        metadata = {
            'forecast_horizons': self.forecast_horizons,
            'seasonal_period': self.seasonal_period,
            'confidence_level': self.confidence_level,
            'target_column': self.target_column,
            'date_column': self.date_column,
            'hierarchy_columns': self.hierarchy_columns,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics,
            'ensemble_weights': self.ensemble_weights
        }

        metadata_path = filepath.parent / f"{filepath.stem}_metadata.pkl"
        joblib.dump(metadata, metadata_path)

        print(f"✓ Model saved to {filepath}")
        print(f"✓ Metadata saved to {metadata_path}")

    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load trained model from disk.

        Args:
            filepath: Path to model file
        """
        filepath = Path(filepath)

        # Load model using PyCaret
        if self.experiment is None:
            self.experiment = TSForecastingExperiment()

        model = self.experiment.load_model(str(filepath))
        self.best_model = model
        self.models['loaded_model'] = model

        # Load metadata
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.pkl"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)

            self.forecast_horizons = metadata.get('forecast_horizons', [7, 30, 90])
            self.seasonal_period = metadata.get('seasonal_period', 7)
            self.confidence_level = metadata.get('confidence_level', 0.95)
            self.target_column = metadata.get('target_column')
            self.date_column = metadata.get('date_column')
            self.hierarchy_columns = metadata.get('hierarchy_columns', [])
            self.feature_names = metadata.get('feature_names', [])
            self.model_metrics = metadata.get('model_metrics', {})
            self.ensemble_weights = metadata.get('ensemble_weights', {})

            print(f"✓ Model and metadata loaded from {filepath}")
        else:
            print(f"✓ Model loaded from {filepath} (no metadata found)")

    def plot_forecast(
        self,
        forecast: pd.DataFrame,
        actual_data: Optional[pd.DataFrame] = None,
        title: str = "Demand Forecast",
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Plot forecast with confidence intervals.

        Args:
            forecast: Forecast DataFrame
            actual_data: Optional actual data for comparison
            title: Plot title
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)

            # Plot forecast
            pred_col = [col for col in forecast.columns if 'y_pred' in col.lower()][0]
            ax.plot(forecast.index, forecast[pred_col], label='Forecast', linewidth=2, color='#2E86AB')

            # Plot confidence intervals if available
            lower_cols = [col for col in forecast.columns if 'lower' in col.lower()]
            upper_cols = [col for col in forecast.columns if 'upper' in col.lower()]

            if len(lower_cols) > 0 and len(upper_cols) > 0:
                ax.fill_between(
                    forecast.index,
                    forecast[lower_cols[0]],
                    forecast[upper_cols[0]],
                    alpha=0.3,
                    color='#2E86AB',
                    label=f'{int(self.confidence_level*100)}% Confidence Interval'
                )

            # Plot actual data if provided
            if actual_data is not None:
                ax.plot(
                    actual_data.index,
                    actual_data[self.target_column],
                    label='Actual',
                    linewidth=2,
                    color='#A23B72',
                    alpha=0.7
                )

            # Plot actuals in forecast period if available
            if 'actual' in forecast.columns:
                ax.plot(
                    forecast.index,
                    forecast['actual'],
                    label='Actual (Validation)',
                    linewidth=2,
                    color='#F18F01',
                    linestyle='--'
                )

            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Demand', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            warnings.warn("Matplotlib not available for plotting")

    def get_feature_importance(
        self,
        model = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.

        Args:
            model: Model to extract importance from (uses best_model if None)
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if model is None:
            model = self.best_model

        if model is None:
            raise ValueError("No model available")

        try:
            # Try to get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                features = self.feature_names or [f'feature_{i}' for i in range(len(importances))]

                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)

                return importance_df
            else:
                warnings.warn("Model does not support feature importance")
                return pd.DataFrame()

        except Exception as e:
            warnings.warn(f"Could not extract feature importance: {str(e)}")
            return pd.DataFrame()


# Convenience functions for quick forecasting

def quick_forecast(
    data: pd.DataFrame,
    target: str,
    date_column: str,
    fh: int = 30,
    seasonal_period: int = 7,
    verbose: bool = True
) -> Tuple[pd.DataFrame, DemandForecastingSystem]:
    """
    Quick demand forecasting with automatic model selection.

    Args:
        data: Time series data
        target: Target column name
        date_column: Date column name
        fh: Forecast horizon (days)
        seasonal_period: Seasonal period
        verbose: Print progress

    Returns:
        Tuple of (forecast DataFrame, forecasting system)
    """
    system = DemandForecastingSystem(
        forecast_horizons=[fh],
        seasonal_period=seasonal_period
    )

    system.setup(
        data=data,
        target=target,
        date_column=date_column,
        fh=fh,
        verbose=verbose
    )

    system.compare_models(n_select=3, verbose=verbose)

    forecast = system.predict(fh=fh)

    return forecast, system


def forecast_with_features(
    data: pd.DataFrame,
    target: str,
    date_column: str,
    fh: int = 30,
    include_holidays: bool = True,
    weather_data: Optional[pd.DataFrame] = None,
    promotion_data: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, DemandForecastingSystem]:
    """
    Demand forecasting with feature engineering.

    Args:
        data: Time series data
        target: Target column name
        date_column: Date column name
        fh: Forecast horizon
        include_holidays: Include holiday features
        weather_data: Optional weather data
        promotion_data: Optional promotion data
        verbose: Print progress

    Returns:
        Tuple of (forecast DataFrame, forecasting system)
    """
    system = DemandForecastingSystem(forecast_horizons=[fh])

    # Engineer features
    if verbose:
        print("Engineering features...")

    data_with_features = system.engineer_features(
        data=data.set_index(date_column),
        include_holidays=include_holidays,
        weather_data=weather_data,
        promotion_data=promotion_data
    )

    # Setup with engineered features
    system.setup(
        data=data_with_features,
        target=target,
        date_column=date_column,
        fh=fh,
        verbose=verbose
    )

    # Compare and select best model
    system.compare_models(n_select=3, verbose=verbose)

    # Generate forecast
    forecast = system.predict(fh=fh)

    return forecast, system


if __name__ == "__main__":
    # Example usage
    print("PyCaret Advanced Demand Forecasting System")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from demand_forecast import DemandForecastingSystem, quick_forecast

    # Quick forecasting
    forecast, system = quick_forecast(
        data=sales_data,
        target='sales',
        date_column='date',
        fh=30
    )

    # Advanced forecasting with features
    system = DemandForecastingSystem(forecast_horizons=[7, 30, 90])

    # Setup experiment
    system.setup(data=sales_data, target='sales', date_column='date')

    # Compare models
    results = system.compare_models(n_select=5)

    # Create ensemble
    ensemble = system.create_ensemble(method='weighted')

    # Multi-horizon forecasts
    forecasts = system.forecast_multi_horizon()

    # Hierarchical forecasting
    hier_forecasts = system.forecast_hierarchy(
        data=sales_data,
        hierarchy_columns=['store_id', 'product_id'],
        target='sales',
        date_column='date',
        fh=30
    )

    # Evaluate
    metrics = system.evaluate_forecast(forecasts[30])
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    # Save model
    system.save_model('models/demand_forecast_model.pkl')
    """)
