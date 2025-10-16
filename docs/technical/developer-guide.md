# PyCaret Developer Guide

<div align="center">

![PyCaret Logo](../images/logo.png)

**Technical Documentation for Developers**
**Version 3.4.0**

</div>

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Development Setup](#development-setup)
4. [API Design](#api-design)
5. [Extension Points](#extension-points)
6. [Testing Strategy](#testing-strategy)
7. [Performance Optimization](#performance-optimization)
8. [Contributing Guidelines](#contributing-guidelines)

---

## Architecture Overview

### High-Level Architecture

PyCaret follows a modular architecture with clear separation of concerns:

```
pycaret/
├── classification/      # Classification module
├── regression/          # Regression module
├── clustering/          # Clustering module
├── anomaly/            # Anomaly detection module
├── time_series/        # Time series forecasting module
├── containers/         # Model and metric containers
├── internal/           # Core internal modules
│   ├── preprocess/    # Preprocessing pipeline
│   ├── pycaret_experiment/ # Experiment management
│   ├── display/       # Display components
│   ├── plots/         # Visualization utilities
│   └── parallel/      # Parallel processing
├── loggers/           # Integration with MLOps platforms
├── parallel/          # Distributed computing backends
└── utils/            # Utility functions
```

### Design Principles

1. **Low-Code Interface**: Minimize code required for common tasks
2. **Modularity**: Each module operates independently
3. **Extensibility**: Easy to add new models and metrics
4. **Automation**: Intelligent defaults with override capabilities
5. **Compatibility**: Wrapper around scikit-learn and other libraries

### Key Design Patterns

- **Factory Pattern**: For model and metric creation
- **Strategy Pattern**: For different preprocessing strategies
- **Observer Pattern**: For logging and monitoring
- **Pipeline Pattern**: For data transformation workflows

---

## Core Components

### 1. Experiment Classes

All modules inherit from base experiment classes:

```python
# Base hierarchy
PyCaretExperiment (base)
├── TabularExperiment
│   ├── SupervisedExperiment
│   │   ├── NonTSSupervisedExperiment
│   │   │   ├── Classification
│   │   │   └── Regression
│   │   └── TSSupervisedExperiment
│   │       └── TimeSeriesForecasting
│   └── UnsupervisedExperiment
│       ├── Clustering
│       └── AnomalyDetection
```

**Key Responsibilities:**
- Setup and configuration management
- Data preprocessing pipeline
- Model training and evaluation
- Hyperparameter tuning
- Model persistence

### 2. Container System

Containers encapsulate models and metrics:

```python
from pycaret.containers.models.base_model import ModelContainer
from pycaret.containers.metrics.base_metric import MetricContainer

# Example: Creating a custom model container
class CustomModelContainer(ModelContainer):
    def __init__(self, id, name, class_def, args, tunable):
        super().__init__(
            id=id,
            name=name,
            class_def=class_def,
            args=args,
            tunable=tunable
        )
```

### 3. Preprocessing Pipeline

The preprocessing system uses scikit-learn pipelines:

```python
from pycaret.internal.preprocess.preprocessor import Preprocessor

# Preprocessing steps
- Missing value imputation
- Categorical encoding
- Numeric feature scaling
- Feature engineering
- Outlier removal
- Multicollinearity handling
- Feature selection
```

### 4. Display System

Manages output display across different environments:

```python
from pycaret.internal.display import CommonDisplay

# Display backends
- Jupyter Notebook (IPython widgets)
- Terminal (text-based)
- Web dashboards
- Static HTML reports
```

---

## Development Setup

### Prerequisites

- Python 3.9, 3.10, 3.11, or 3.12
- Git
- Virtual environment tool (venv, conda)

### Clone and Setup

```bash
# Clone repository
git clone https://github.com/pycaret/pycaret.git
cd pycaret

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,full]"

# Verify installation
pytest tests/ -v
```

### Development Tools

```bash
# Code formatting
black pycaret/
isort pycaret/

# Linting
flake8 pycaret/

# Type checking
mypy pycaret/

# Run tests
pytest tests/ --cov=pycaret --cov-report=html
```

### Project Structure

```
pycaret/
├── pycaret/              # Source code
├── tests/                # Test suite
├── docs/                 # Documentation
├── examples/             # Example notebooks
├── pyproject.toml        # Project configuration
├── requirements.txt      # Core dependencies
├── requirements-optional.txt  # Optional dependencies
└── README.md
```

---

## API Design

### Functional API

The functional API provides module-level functions:

```python
from pycaret.classification import (
    setup,
    compare_models,
    create_model,
    tune_model,
    evaluate_model,
    predict_model,
    save_model,
    load_model
)

# Usage
s = setup(data, target='target_column')
best = compare_models()
tuned = tune_model(best)
predictions = predict_model(tuned, data=test_data)
```

### OOP API

The OOP API provides class-based interfaces:

```python
from pycaret.classification import ClassificationExperiment

# Usage
clf = ClassificationExperiment()
clf.setup(data, target='target_column')
best = clf.compare_models()
tuned = clf.tune_model(best)
predictions = clf.predict_model(tuned, data=test_data)
```

### API Consistency

All modules follow consistent naming:

| Operation | Function | Description |
|-----------|----------|-------------|
| `setup()` | Initialize experiment | Configure environment |
| `compare_models()` | Model comparison | Train and compare models |
| `create_model()` | Create specific model | Train individual model |
| `tune_model()` | Hyperparameter tuning | Optimize model |
| `evaluate_model()` | Model evaluation | Generate evaluation plots |
| `predict_model()` | Make predictions | Predict on new data |
| `save_model()` | Save model | Persist to disk |
| `load_model()` | Load model | Load from disk |

---

## Extension Points

### Adding Custom Models

```python
from pycaret.containers.models.classification import ClassificationContainer
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. Create your model class
class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, param1=1.0, param2='default'):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y):
        # Training logic
        return self

    def predict(self, X):
        # Prediction logic
        return predictions

# 2. Register with PyCaret
from pycaret.classification import setup, add_model

s = setup(data, target='target')
custom_model = add_model(
    CustomClassifier(),
    name='Custom Classifier',
    tunable={
        'param1': [0.1, 1.0, 10.0],
        'param2': ['option1', 'option2']
    }
)
```

### Adding Custom Metrics

```python
from pycaret.containers.metrics.classification import get_all_metrics
from sklearn.metrics import make_scorer

# 1. Define custom metric function
def custom_metric(y_true, y_pred):
    # Metric calculation logic
    return score

# 2. Create scorer
custom_scorer = make_scorer(
    custom_metric,
    greater_is_better=True,
    needs_proba=False
)

# 3. Add to PyCaret
from pycaret.classification import setup, add_metric

s = setup(data, target='target')
add_metric(
    'custom_metric',
    'Custom Metric',
    custom_scorer
)
```

### Custom Preprocessing

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1):
        self.param = param

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Transformation logic
        return X_transformed

# Use in setup
from pycaret.classification import setup

s = setup(
    data,
    target='target',
    custom_pipeline=CustomTransformer()
)
```

---

## Testing Strategy

### Test Structure

```
tests/
├── test_classification.py
├── test_regression.py
├── test_clustering.py
├── test_anomaly.py
├── test_time_series.py
├── test_containers.py
├── test_preprocessing.py
└── test_utils.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific module
pytest tests/test_classification.py

# Run with coverage
pytest --cov=pycaret --cov-report=html

# Run in parallel
pytest -n auto
```

### Writing Tests

```python
import pytest
from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models

class TestClassification:
    @pytest.fixture
    def data(self):
        return get_data('juice')

    def test_setup(self, data):
        s = setup(data, target='Purchase', session_id=123)
        assert s is not None

    def test_compare_models(self, data):
        s = setup(data, target='Purchase', session_id=123)
        best = compare_models(n_select=1)
        assert best is not None
```

### Integration Testing

```python
def test_full_workflow():
    # Load data
    data = get_data('juice')

    # Setup
    s = setup(data, target='Purchase', session_id=123)

    # Compare models
    best = compare_models(n_select=1)

    # Tune model
    tuned = tune_model(best)

    # Predict
    predictions = predict_model(tuned)

    # Assertions
    assert predictions is not None
    assert 'prediction_label' in predictions.columns
```

---

## Performance Optimization

### GPU Acceleration

```python
from pycaret.classification import setup, compare_models

# Enable GPU
s = setup(
    data,
    target='target',
    use_gpu=True  # Requires CUDA-compatible GPU
)

# Supported models: XGBoost, LightGBM, CatBoost, cuML models
```

### Intel Optimization

```python
# Install Intel extension
# pip install scikit-learn-intelex

from pycaret.classification import setup

s = setup(
    data,
    target='target',
    engine='sklearnex'  # Use Intel optimizations
)
```

### Parallel Processing

```python
# Using Fugue backend for distributed computing
from pycaret.parallel import FugueBackend
from pycaret.classification import setup, compare_models

# Dask backend
backend = FugueBackend(backend='dask')

s = setup(
    data,
    target='target',
    n_jobs=-1  # Use all cores
)

# Compare models in parallel
best = compare_models(parallel=backend)
```

### Memory Optimization

```python
# Reduce memory usage
s = setup(
    data,
    target='target',
    low_memory=True,  # Memory-efficient mode
    data_split_stratify=False,  # Disable stratification
    remove_multicollinearity=True  # Remove redundant features
)
```

---

## Contributing Guidelines

### Code Style

PyCaret follows PEP 8 with Black formatting:

```bash
# Format code
black pycaret/

# Sort imports
isort pycaret/

# Check style
flake8 pycaret/
```

### Commit Messages

Follow conventional commits:

```
feat: Add new classification model
fix: Resolve memory leak in preprocessing
docs: Update API documentation
test: Add unit tests for regression module
refactor: Improve performance of compare_models
```

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and commit
4. Add tests for new functionality
5. Run test suite: `pytest`
6. Push to your fork: `git push origin feature/my-feature`
7. Create pull request

### Documentation

- Update docstrings for all public functions
- Add examples to documentation
- Update CHANGELOG.md
- Add type hints

```python
def my_function(param1: str, param2: int = 10) -> pd.DataFrame:
    """
    Brief description of the function.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, default=10
        Description of param2

    Returns
    -------
    pd.DataFrame
        Description of return value

    Examples
    --------
    >>> result = my_function('value', 20)
    >>> print(result)
    """
    pass
```

---

## Resources

### Official Documentation
- API Reference: https://pycaret.gitbook.io/
- GitHub Repository: https://github.com/pycaret/pycaret
- Tutorials: https://pycaret.gitbook.io/docs/get-started/tutorials

### Community
- Slack: https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w
- GitHub Discussions: https://github.com/pycaret/pycaret/discussions
- YouTube: https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g

### Related Libraries
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- CatBoost: https://catboost.ai/

---

**© 2025 PyCaret. Licensed under MIT License.**
