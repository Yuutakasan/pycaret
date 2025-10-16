"""
Dashboard Orchestration and Data Pipeline
==========================================

Coordinates data loading, preprocessing, analysis modules, caching,
and incremental updates for the retail analytics dashboard.

Author: Backend API Developer Agent
Created: 2025-10-08
"""

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from functools import wraps
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: datetime
    hash_key: str
    ttl: int = 3600  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)


@dataclass
class PipelineConfig:
    """Configuration for data pipeline."""
    data_path: str
    cache_dir: str = "./cache"
    cache_enabled: bool = True
    cache_ttl: int = 3600
    incremental_enabled: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    log_level: str = "INFO"
    store_column: str = "Store"
    date_column: str = "Date"
    required_columns: List[str] = field(default_factory=lambda: [
        "Store", "Date", "Sales", "Customers", "Promo",
        "StateHoliday", "SchoolHoliday"
    ])

    def __post_init__(self):
        """Validate configuration."""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logging.getLogger().setLevel(self.log_level)


class CacheManager:
    """Manages caching layer for performance optimization."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize cache manager.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, CacheEntry] = {}
        logger.info(f"Cache manager initialized with dir: {self.cache_dir}")

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        if not self.config.cache_enabled:
            return None

        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                logger.debug(f"Cache hit (memory): {key}")
                return entry.data
            else:
                del self.memory_cache[key]
                logger.debug(f"Cache expired (memory): {key}")

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                if not entry.is_expired():
                    self.memory_cache[key] = entry
                    logger.debug(f"Cache hit (disk): {key}")
                    return entry.data
                else:
                    cache_file.unlink()
                    logger.debug(f"Cache expired (disk): {key}")
            except Exception as e:
                logger.warning(f"Error loading cache {key}: {e}")
                cache_file.unlink(missing_ok=True)

        return None

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Store in cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds (uses config default if None)
        """
        if not self.config.cache_enabled:
            return

        ttl = ttl or self.config.cache_ttl
        entry = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            hash_key=key,
            ttl=ttl
        )

        # Store in memory cache
        self.memory_cache[key] = entry

        # Store in disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
            logger.debug(f"Cache set: {key}")
        except Exception as e:
            logger.warning(f"Error saving cache {key}: {e}")

    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            pattern: Pattern to match keys (None = invalidate all)

        Returns:
            Number of entries invalidated
        """
        count = 0

        # Invalidate memory cache
        if pattern:
            keys_to_delete = [k for k in self.memory_cache if pattern in k]
        else:
            keys_to_delete = list(self.memory_cache.keys())

        for key in keys_to_delete:
            del self.memory_cache[key]
            count += 1

        # Invalidate disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            if pattern is None or pattern in cache_file.stem:
                cache_file.unlink()
                count += 1

        logger.info(f"Invalidated {count} cache entries")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_entries = len(self.memory_cache)
        disk_entries = len(list(self.cache_dir.glob("*.pkl")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))

        return {
            "memory_entries": memory_entries,
            "disk_entries": disk_entries,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


def cached(ttl: Optional[int] = None):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'cache_manager'):
                return func(self, *args, **kwargs)

            cache_key = f"{func.__name__}_{self.cache_manager._generate_key(*args, **kwargs)}"

            # Try to get from cache
            result = self.cache_manager.get(cache_key)
            if result is not None:
                return result

            # Execute function
            result = func(self, *args, **kwargs)

            # Store in cache
            self.cache_manager.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator


class DataPipeline:
    """Manages data loading and preprocessing pipeline."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize data pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.cache_manager = CacheManager(config)
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        logger.info("Data pipeline initialized")

    @cached(ttl=7200)
    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load raw data from source.

        Args:
            force_reload: Force reload from disk

        Returns:
            Raw dataframe
        """
        try:
            logger.info(f"Loading data from: {self.config.data_path}")

            # Load data
            df = pd.read_csv(self.config.data_path)

            # Validate required columns
            missing_cols = set(self.config.required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Store metadata
            self.metadata['load_time'] = datetime.now().isoformat()
            self.metadata['rows'] = len(df)
            self.metadata['columns'] = list(df.columns)
            self.metadata['memory_mb'] = df.memory_usage(deep=True).sum() / (1024 * 1024)

            self.raw_data = df
            logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw data.

        Args:
            df: Raw dataframe

        Returns:
            Preprocessed dataframe
        """
        try:
            logger.info("Starting data preprocessing")
            df = df.copy()

            # Convert date column
            if self.config.date_column in df.columns:
                df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])

            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna('Unknown')

            # Add derived features
            if self.config.date_column in df.columns:
                df['Year'] = df[self.config.date_column].dt.year
                df['Month'] = df[self.config.date_column].dt.month
                df['Quarter'] = df[self.config.date_column].dt.quarter
                df['DayOfWeek'] = df[self.config.date_column].dt.dayofweek
                df['WeekOfYear'] = df[self.config.date_column].dt.isocalendar().week

            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            duplicates_removed = initial_rows - len(df)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate rows")

            self.processed_data = df
            logger.info(f"Data preprocessing completed: {len(df)} rows")

            return df

        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    @cached(ttl=3600)
    def filter_by_store(self, df: pd.DataFrame, store_ids: List[int]) -> pd.DataFrame:
        """
        Filter data by store IDs.

        Args:
            df: Input dataframe
            store_ids: List of store IDs to filter

        Returns:
            Filtered dataframe
        """
        if not store_ids:
            return df

        try:
            filtered = df[df[self.config.store_column].isin(store_ids)]
            logger.info(f"Filtered to {len(store_ids)} stores: {len(filtered)} rows")
            return filtered
        except Exception as e:
            logger.error(f"Error filtering by store: {e}")
            raise

    @cached(ttl=3600)
    def filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter data by date range.

        Args:
            df: Input dataframe
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            Filtered dataframe
        """
        try:
            if start_date:
                start = pd.to_datetime(start_date)
                df = df[df[self.config.date_column] >= start]

            if end_date:
                end = pd.to_datetime(end_date)
                df = df[df[self.config.date_column] <= end]

            logger.info(f"Date range filter applied: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error filtering by date range: {e}")
            raise

    def get_incremental_updates(
        self,
        last_update: datetime
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get incremental data updates since last update.

        Args:
            last_update: Timestamp of last update

        Returns:
            Tuple of (new data, update metadata)
        """
        if not self.config.incremental_enabled:
            return self.processed_data, {}

        try:
            if self.processed_data is None:
                return pd.DataFrame(), {}

            # Filter for new/updated records
            new_data = self.processed_data[
                self.processed_data[self.config.date_column] > last_update
            ]

            metadata = {
                'last_update': last_update.isoformat(),
                'current_time': datetime.now().isoformat(),
                'new_rows': len(new_data),
                'total_rows': len(self.processed_data)
            }

            logger.info(f"Incremental update: {len(new_data)} new rows")
            return new_data, metadata

        except Exception as e:
            logger.error(f"Error getting incremental updates: {e}")
            raise


class AnalysisOrchestrator:
    """Orchestrates analysis modules and coordinates execution."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize analysis orchestrator.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.pipeline = DataPipeline(config)
        self.cache_manager = self.pipeline.cache_manager
        self.analysis_modules: Dict[str, Callable] = {}
        self.results: Dict[str, Any] = {}
        logger.info("Analysis orchestrator initialized")

    def register_module(self, name: str, module: Callable) -> None:
        """
        Register an analysis module.

        Args:
            name: Module name
            module: Callable analysis module
        """
        self.analysis_modules[name] = module
        logger.info(f"Registered analysis module: {name}")

    def initialize(self) -> None:
        """Initialize orchestrator and load data."""
        try:
            # Load and preprocess data
            raw_data = self.pipeline.load_data()
            self.pipeline.preprocess_data(raw_data)

            logger.info("Orchestrator initialization completed")

        except Exception as e:
            logger.error(f"Error initializing orchestrator: {e}")
            raise

    @cached(ttl=1800)
    def run_analysis(
        self,
        module_name: str,
        store_ids: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run specific analysis module.

        Args:
            module_name: Name of analysis module
            store_ids: Store IDs to filter
            start_date: Start date filter
            end_date: End date filter
            **kwargs: Additional module arguments

        Returns:
            Analysis results
        """
        try:
            if module_name not in self.analysis_modules:
                raise ValueError(f"Unknown analysis module: {module_name}")

            logger.info(f"Running analysis: {module_name}")

            # Get filtered data
            data = self.pipeline.processed_data.copy()

            if store_ids:
                data = self.pipeline.filter_by_store(data, store_ids)

            if start_date or end_date:
                data = self.pipeline.filter_by_date_range(data, start_date, end_date)

            # Run analysis module
            module = self.analysis_modules[module_name]
            result = module(data, **kwargs)

            # Store result
            self.results[module_name] = {
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'filters': {
                    'store_ids': store_ids,
                    'start_date': start_date,
                    'end_date': end_date
                }
            }

            logger.info(f"Analysis completed: {module_name}")
            return result

        except Exception as e:
            logger.error(f"Error running analysis {module_name}: {e}")
            raise

    def run_all_analyses(
        self,
        store_ids: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run all registered analysis modules.

        Args:
            store_ids: Store IDs to filter
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dictionary of all analysis results
        """
        results = {}

        for module_name in self.analysis_modules:
            try:
                result = self.run_analysis(
                    module_name,
                    store_ids=store_ids,
                    start_date=start_date,
                    end_date=end_date
                )
                results[module_name] = result
            except Exception as e:
                logger.error(f"Error in module {module_name}: {e}")
                results[module_name] = {'error': str(e)}

        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get orchestrator summary and statistics.

        Returns:
            Summary dictionary
        """
        return {
            'status': 'active' if self.pipeline.processed_data is not None else 'inactive',
            'data_loaded': self.pipeline.processed_data is not None,
            'total_rows': len(self.pipeline.processed_data) if self.pipeline.processed_data is not None else 0,
            'registered_modules': list(self.analysis_modules.keys()),
            'completed_analyses': list(self.results.keys()),
            'metadata': self.pipeline.metadata,
            'cache_stats': self.cache_manager.get_stats(),
            'config': {
                'cache_enabled': self.config.cache_enabled,
                'incremental_enabled': self.config.incremental_enabled,
                'cache_ttl': self.config.cache_ttl
            }
        }

    def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            pattern: Pattern to match (None = all)

        Returns:
            Number of entries invalidated
        """
        return self.cache_manager.invalidate(pattern)

    def export_results(self, output_path: str) -> None:
        """
        Export analysis results to file.

        Args:
            output_path: Output file path
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                'summary': self.get_summary(),
                'results': self.results,
                'export_time': datetime.now().isoformat()
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Results exported to: {output_path}")

        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise


# Example analysis modules (to be replaced with actual implementations)
def sales_analysis(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Example sales analysis module."""
    return {
        'total_sales': data['Sales'].sum(),
        'avg_sales': data['Sales'].mean(),
        'sales_by_store': data.groupby('Store')['Sales'].sum().to_dict()
    }


def customer_analysis(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Example customer analysis module."""
    return {
        'total_customers': data['Customers'].sum(),
        'avg_customers': data['Customers'].mean(),
        'customers_by_store': data.groupby('Store')['Customers'].sum().to_dict()
    }


def promo_analysis(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Example promotion analysis module."""
    return {
        'promo_days': data['Promo'].sum(),
        'promo_effectiveness': data.groupby('Promo')['Sales'].mean().to_dict()
    }


# Factory function
def create_orchestrator(
    data_path: str,
    cache_dir: str = "./cache",
    **config_kwargs
) -> AnalysisOrchestrator:
    """
    Create and initialize analysis orchestrator.

    Args:
        data_path: Path to data file
        cache_dir: Cache directory path
        **config_kwargs: Additional configuration parameters

    Returns:
        Initialized AnalysisOrchestrator
    """
    config = PipelineConfig(
        data_path=data_path,
        cache_dir=cache_dir,
        **config_kwargs
    )

    orchestrator = AnalysisOrchestrator(config)

    # Register default analysis modules
    orchestrator.register_module('sales', sales_analysis)
    orchestrator.register_module('customers', customer_analysis)
    orchestrator.register_module('promotions', promo_analysis)

    # Initialize
    orchestrator.initialize()

    return orchestrator


if __name__ == "__main__":
    # Example usage
    try:
        # Create orchestrator
        orchestrator = create_orchestrator(
            data_path="data/train.csv",
            cache_dir="./cache",
            cache_enabled=True,
            cache_ttl=3600
        )

        # Run all analyses
        results = orchestrator.run_all_analyses(
            store_ids=[1, 2, 3],
            start_date="2023-01-01",
            end_date="2023-12-31"
        )

        # Print summary
        summary = orchestrator.get_summary()
        print(json.dumps(summary, indent=2, default=str))

        # Export results
        orchestrator.export_results("output/analysis_results.json")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
