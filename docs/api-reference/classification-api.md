# Classification API Reference

<div align="center">

![PyCaret Logo](../images/logo.png)

**Complete API Reference for Classification Module**
**Version 3.4.0**

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Setup Functions](#setup-functions)
3. [Model Training](#model-training)
4. [Model Tuning](#model-tuning)
5. [Model Analysis](#model-analysis)
6. [Prediction Functions](#prediction-functions)
7. [Model Management](#model-management)
8. [Utilities](#utilities)

---

## Overview

The classification module provides supervised learning for categorical target variables. It supports both Functional and OOP APIs.

### Import

```python
# Functional API
from pycaret.classification import *

# OOP API
from pycaret.classification import ClassificationExperiment
```

---

## Setup Functions

### setup()

Initialize the training environment and preprocessing pipeline.

**Signature:**
```python
setup(
    data: pd.DataFrame,
    target: str,
    train_size: float = 0.7,
    test_data: Optional[pd.DataFrame] = None,
    preprocess: bool = True,
    imputation_type: str = 'simple',
    numeric_imputation: str = 'mean',
    categorical_imputation: str = 'mode',
    categorical_features: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
    date_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    ignore_features: Optional[List[str]] = None,
    keep_features: Optional[List[str]] = None,
    ordinal_features: Optional[Dict[str, list]] = None,
    high_cardinality_features: Optional[List[str]] = None,
    numeric_features_encoding: str = 'passthrough',
    categorical_features_encoding: str = 'onehot',
    high_cardinality_method: str = 'frequency',
    max_encoding_ohe: int = 25,
    encoding_method: Optional[Any] = None,
    rare_to_value: Optional[float] = None,
    rare_value: str = 'rare',
    polynomial_features: bool = False,
    polynomial_degree: int = 2,
    low_variance_threshold: Optional[float] = None,
    group_features: Optional[Dict[str, List[str]]] = None,
    drop_groups: bool = False,
    remove_multicollinearity: bool = False,
    multicollinearity_threshold: float = 0.9,
    bin_numeric_features: Optional[List[str]] = None,
    remove_outliers: bool = False,
    outliers_method: str = 'iforest',
    outliers_threshold: float = 0.05,
    transformation: bool = False,
    transformation_method: str = 'yeo-johnson',
    normalize: bool = False,
    normalize_method: str = 'zscore',
    pca: bool = False,
    pca_method: str = 'linear',
    pca_components: Optional[Union[int, float]] = None,
    feature_selection: bool = False,
    feature_selection_method: str = 'classic',
    feature_selection_estimator: str = 'lightgbm',
    n_features_to_select: Union[int, float] = 0.2,
    custom_pipeline: Optional[Any] = None,
    custom_pipeline_position: int = -1,
    data_split_shuffle: bool = True,
    data_split_stratify: Union[bool, List[str]] = True,
    fold_strategy: Union[str, Any] = 'stratifiedkfold',
    fold: int = 10,
    fold_shuffle: bool = False,
    fold_groups: Optional[Union[str, pd.DataFrame]] = None,
    n_jobs: Optional[int] = -1,
    use_gpu: bool = False,
    html: bool = True,
    session_id: Optional[int] = None,
    system_log: Union[bool, str, logging.Logger] = True,
    log_experiment: Union[bool, str, BaseLogger, List[Union[str, BaseLogger]]] = False,
    experiment_name: Optional[str] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    log_plots: Union[bool, list] = False,
    log_profile: bool = False,
    log_data: bool = False,
    verbose: bool = True,
    memory: Union[bool, str, Memory] = True,
    profile: bool = False,
    profile_kwargs: Optional[Dict[str, Any]] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | pd.DataFrame | Required | Training dataset |
| `target` | str | Required | Name of target column |
| `train_size` | float | 0.7 | Proportion of dataset for training (0.0-1.0) |
| `test_data` | pd.DataFrame | None | Separate test dataset |
| `preprocess` | bool | True | Enable preprocessing |
| `session_id` | int | None | Random seed for reproducibility |
| `normalize` | bool | False | Normalize numeric features |
| `transformation` | bool | False | Apply power transformation |
| `pca` | bool | False | Apply principal component analysis |
| `feature_selection` | bool | False | Enable feature selection |
| `remove_outliers` | bool | False | Remove outliers from dataset |
| `fold` | int | 10 | Number of cross-validation folds |
| `n_jobs` | int | -1 | Number of parallel jobs (-1 = all cores) |
| `use_gpu` | bool | False | Enable GPU acceleration |
| `verbose` | bool | True | Print progress information |

**Returns:**
- `ClassificationExperiment`: Configured experiment object

**Example:**
```python
from pycaret.datasets import get_data
from pycaret.classification import setup

# Load data
data = get_data('juice')

# Basic setup
s = setup(
    data=data,
    target='Purchase',
    session_id=123
)

# Advanced setup with preprocessing
s = setup(
    data=data,
    target='Purchase',
    session_id=123,
    normalize=True,
    transformation=True,
    remove_outliers=True,
    remove_multicollinearity=True,
    polynomial_features=True,
    feature_selection=True,
    fold=5
)
```

---

## Model Training

### compare_models()

Train and compare all available models.

**Signature:**
```python
compare_models(
    include: Optional[List[Union[str, Any]]] = None,
    exclude: Optional[List[str]] = None,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    sort: str = 'Accuracy',
    n_select: int = 1,
    budget_time: Optional[float] = None,
    turbo: bool = True,
    errors: str = 'ignore',
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    verbose: bool = True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include` | List[str] | None | Models to include |
| `exclude` | List[str] | None | Models to exclude |
| `fold` | int | None | Cross-validation folds (uses setup value if None) |
| `sort` | str | 'Accuracy' | Metric to sort by |
| `n_select` | int | 1 | Number of top models to return |
| `budget_time` | float | None | Maximum time budget in minutes |
| `turbo` | bool | True | Fast mode (less fold, fewer models) |
| `errors` | str | 'ignore' | How to handle errors ('ignore' or 'raise') |

**Returns:**
- `estimator` or `list`: Best model(s)

**Example:**
```python
# Compare all models
best_model = compare_models()

# Compare top 3 models
top3 = compare_models(n_select=3)

# Compare specific models
best_rf = compare_models(include=['rf', 'xgboost', 'lightgbm'])

# Time-constrained comparison
best_fast = compare_models(budget_time=0.5)  # 30 seconds
```

### create_model()

Train a specific model.

**Signature:**
```python
create_model(
    estimator: Union[str, Any],
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    verbose: bool = True,
    **kwargs
)
```

**Available Models:**

| ID | Model Name | Library |
|----|-----------|---------|
| `'lr'` | Logistic Regression | scikit-learn |
| `'knn'` | K Neighbors Classifier | scikit-learn |
| `'nb'` | Naive Bayes | scikit-learn |
| `'dt'` | Decision Tree | scikit-learn |
| `'svm'` | SVM - Linear Kernel | scikit-learn |
| `'rbfsvm'` | SVM - Radial Kernel | scikit-learn |
| `'gpc'` | Gaussian Process | scikit-learn |
| `'mlp'` | MLP Classifier | scikit-learn |
| `'ridge'` | Ridge Classifier | scikit-learn |
| `'rf'` | Random Forest | scikit-learn |
| `'qda'` | Quadratic Discriminant | scikit-learn |
| `'ada'` | AdaBoost Classifier | scikit-learn |
| `'gbc'` | Gradient Boosting | scikit-learn |
| `'lda'` | Linear Discriminant | scikit-learn |
| `'et'` | Extra Trees | scikit-learn |
| `'xgboost'` | Extreme Gradient Boosting | xgboost |
| `'lightgbm'` | Light Gradient Boosting | lightgbm |
| `'catboost'` | CatBoost Classifier | catboost |

**Example:**
```python
# Create Random Forest
rf = create_model('rf')

# Create with custom parameters
xgb = create_model('xgboost', n_estimators=500, learning_rate=0.05)

# Create with specific fold
lgbm = create_model('lightgbm', fold=3)
```

---

## Model Tuning

### tune_model()

Hyperparameter tuning for a trained model.

**Signature:**
```python
tune_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    n_iter: int = 10,
    custom_grid: Optional[Union[Dict[str, list], Any]] = None,
    optimize: str = 'Accuracy',
    custom_scorer: Optional[Any] = None,
    search_library: str = 'scikit-learn',
    search_algorithm: Optional[str] = None,
    early_stopping: Any = False,
    early_stopping_max_iters: int = 10,
    choose_better: bool = True,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    return_tuner: bool = False,
    verbose: bool = True,
    tuner_verbose: Union[int, bool] = True,
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | estimator | Required | Trained model object |
| `n_iter` | int | 10 | Number of iterations |
| `optimize` | str | 'Accuracy' | Metric to optimize |
| `search_library` | str | 'scikit-learn' | Library for tuning ('scikit-learn', 'optuna', 'tune-sklearn', 'scikit-optimize') |
| `search_algorithm` | str | None | Search algorithm ('random', 'grid', 'bayesian') |
| `early_stopping` | bool | False | Enable early stopping |
| `choose_better` | bool | True | Return better model |

**Example:**
```python
# Basic tuning
tuned_rf = tune_model(rf)

# Optuna tuning
tuned_optuna = tune_model(rf, search_library='optuna', n_iter=100)

# Custom grid
custom_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}
tuned_custom = tune_model(rf, custom_grid=custom_grid)

# Optimize for specific metric
tuned_f1 = tune_model(rf, optimize='F1')
```

### ensemble_model()

Create ensemble of models.

**Signature:**
```python
ensemble_model(
    estimator,
    method: str = 'Bagging',
    fold: Optional[Union[int, Any]] = None,
    n_estimators: int = 10,
    round: int = 4,
    choose_better: bool = True,
    optimize: str = 'Accuracy',
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    **kwargs
)
```

**Example:**
```python
# Bagging
bagged_rf = ensemble_model(rf, method='Bagging', n_estimators=10)

# Boosting
boosted_dt = ensemble_model(dt, method='Boosting', n_estimators=50)
```

### blend_models()

Create blender (voting ensemble).

**Signature:**
```python
blend_models(
    estimator_list: list,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    choose_better: bool = True,
    optimize: str = 'Accuracy',
    method: str = 'auto',
    weights: Optional[List[float]] = None,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True
)
```

**Example:**
```python
# Create models
rf = create_model('rf')
xgb = create_model('xgboost')
lgbm = create_model('lightgbm')

# Blend models
blender = blend_models([rf, xgb, lgbm])

# Weighted blend
blender_weighted = blend_models(
    [rf, xgb, lgbm],
    weights=[0.5, 0.3, 0.2]
)
```

### stack_models()

Create meta-model (stacking ensemble).

**Signature:**
```python
stack_models(
    estimator_list: list,
    meta_model=None,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    method: str = 'auto',
    restack: bool = True,
    choose_better: bool = True,
    optimize: str = 'Accuracy',
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True
)
```

**Example:**
```python
# Stack with default meta-model (Logistic Regression)
stacker = stack_models([rf, xgb, lgbm])

# Stack with custom meta-model
meta = create_model('gbc')
stacker_custom = stack_models([rf, xgb, lgbm], meta_model=meta)
```

---

## Model Analysis

### evaluate_model()

Generate interactive evaluation plots.

**Signature:**
```python
evaluate_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    fit_kwargs: Optional[dict] = None,
    plot_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None
)
```

**Available Plots:**
- AUC-ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Class Prediction Error
- Classification Report
- Feature Importance
- Learning Curve
- Manifold Learning
- Calibration Curve
- Validation Curve
- Dimension Learning
- Decision Boundary

**Example:**
```python
# Interactive evaluation (Jupyter only)
evaluate_model(tuned_rf)
```

### plot_model()

Generate specific plots.

**Signature:**
```python
plot_model(
    estimator,
    plot: str = 'auc',
    scale: float = 1,
    save: Union[str, bool] = False,
    fold: Optional[Union[int, Any]] = None,
    fit_kwargs: Optional[dict] = None,
    plot_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display_format: Optional[str] = None
)
```

**Plot Types:**

| Plot | Description |
|------|-------------|
| `'auc'` | AUC-ROC Curve |
| `'pr'` | Precision-Recall Curve |
| `'confusion_matrix'` | Confusion Matrix |
| `'error'` | Class Prediction Error |
| `'class_report'` | Classification Report |
| `'boundary'` | Decision Boundary |
| `'rfe'` | Recursive Feature Selection |
| `'learning'` | Learning Curve |
| `'manifold'` | Manifold Learning |
| `'calibration'` | Calibration Curve |
| `'vc'` | Validation Curve |
| `'dimension'` | Dimension Learning |
| `'feature'` | Feature Importance |
| `'feature_all'` | Feature Importance (all) |
| `'parameter'` | Model Hyperparameter |
| `'lift'` | Lift Chart |
| `'gain'` | Gain Chart |
| `'tree'` | Decision Tree |

**Example:**
```python
# AUC curve
plot_model(tuned_rf, plot='auc')

# Confusion matrix
plot_model(tuned_rf, plot='confusion_matrix')

# Feature importance
plot_model(tuned_rf, plot='feature')

# Save plot
plot_model(tuned_rf, plot='auc', save=True)
```

### interpret_model()

Generate SHAP values and interpretations.

**Signature:**
```python
interpret_model(
    estimator,
    plot: str = 'summary',
    feature: Optional[str] = None,
    observation: Optional[int] = None,
    use_train_data: bool = False,
    X_new_sample: Optional[pd.DataFrame] = None,
    y_new_sample: Optional[pd.DataFrame] = None,
    save: Union[str, bool] = False,
    **kwargs
)
```

**Example:**
```python
# SHAP summary plot
interpret_model(tuned_rf)

# SHAP for specific observation
interpret_model(tuned_rf, plot='correlation', observation=0)
```

---

## Prediction Functions

### predict_model()

Make predictions on new data.

**Signature:**
```python
predict_model(
    estimator,
    data: Optional[pd.DataFrame] = None,
    probability_threshold: Optional[float] = None,
    encoded_labels: bool = False,
    drift_report: bool = False,
    raw_score: bool = False,
    round: int = 4,
    verbose: bool = True
) -> pd.DataFrame
```

**Returns:**
- `pd.DataFrame`: Original data with prediction columns

**Example:**
```python
# Predict on hold-out set
holdout_pred = predict_model(tuned_rf)

# Predict on new data
new_data = pd.read_csv('new_customers.csv')
predictions = predict_model(tuned_rf, data=new_data)

# Get probability scores
predictions_proba = predict_model(tuned_rf, raw_score=True)

# Custom threshold
predictions_custom = predict_model(
    tuned_rf,
    probability_threshold=0.7
)
```

### finalize_model()

Train model on full dataset (train + test).

**Signature:**
```python
finalize_model(
    estimator,
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    model_only: bool = True,
    experiment_custom_tags: Optional[Dict[str, Any]] = None
)
```

**Example:**
```python
# Finalize before deployment
final_model = finalize_model(tuned_rf)
```

---

## Model Management

### save_model()

Save model to pickle file.

**Signature:**
```python
save_model(
    model,
    model_name: str,
    model_only: bool = False,
    verbose: bool = True,
    **kwargs
)
```

**Example:**
```python
# Save model
save_model(final_model, 'my_classification_model')

# Model saved to: my_classification_model.pkl
```

### load_model()

Load model from pickle file.

**Signature:**
```python
load_model(
    model_name: str,
    platform: Optional[str] = None,
    authentication: Optional[Dict[str, str]] = None,
    verbose: bool = True
)
```

**Example:**
```python
# Load model
loaded_model = load_model('my_classification_model')

# Use loaded model
predictions = predict_model(loaded_model, data=new_data)
```

### deploy_model()

Deploy model to cloud.

**Signature:**
```python
deploy_model(
    model,
    model_name: str,
    authentication: Dict[str, str],
    platform: str = 'aws'
)
```

**Platforms:**
- `'aws'`: Amazon Web Services
- `'gcp'`: Google Cloud Platform
- `'azure'`: Microsoft Azure

**Example:**
```python
# Deploy to AWS
deploy_model(
    model=final_model,
    model_name='production_classifier',
    platform='aws',
    authentication={'bucket': 'my-bucket'}
)
```

---

## Utilities

### get_config()

Get configuration parameters.

**Signature:**
```python
get_config(variable: str = None)
```

**Example:**
```python
# Get all config
config = get_config()

# Get specific parameter
seed = get_config('seed')
```

### set_config()

Set configuration parameters.

**Signature:**
```python
set_config(variable: str, value)
```

**Example:**
```python
# Set GPU usage
set_config('use_gpu', True)

# Set number of jobs
set_config('n_jobs', 4)
```

### get_metrics()

Get all available metrics.

**Signature:**
```python
get_metrics(reset: bool = False) -> pd.DataFrame
```

**Example:**
```python
# Get metrics
metrics = get_metrics()
print(metrics)
```

### add_metric()

Add custom metric.

**Signature:**
```python
add_metric(
    id: str,
    name: str,
    score_func: type,
    target: str = 'pred',
    greater_is_better: bool = True,
    multiclass: bool = True,
    **kwargs
)
```

**Example:**
```python
from sklearn.metrics import make_scorer, cohen_kappa_score

# Add custom metric
kappa_scorer = make_scorer(cohen_kappa_score)
add_metric('kappa', 'Cohen Kappa', kappa_scorer)
```

### remove_metric()

Remove metric.

**Signature:**
```python
remove_metric(name_or_id: str)
```

**Example:**
```python
# Remove metric
remove_metric('kappa')
```

### get_logs()

Get experiment logs.

**Signature:**
```python
get_logs(
    experiment_name: Optional[str] = None,
    save: bool = False
) -> pd.DataFrame
```

**Example:**
```python
# Get logs
logs = get_logs()
print(logs)
```

---

## Complete Workflow Example

```python
# 1. Import and load data
from pycaret.datasets import get_data
from pycaret.classification import *

data = get_data('credit')

# 2. Initialize setup
s = setup(
    data=data,
    target='default',
    session_id=123,
    normalize=True,
    transformation=True,
    ignore_low_variance=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95,
    fix_imbalance=True
)

# 3. Compare models
top3 = compare_models(n_select=3)

# 4. Create specific models
xgb = create_model('xgboost')
lgbm = create_model('lightgbm')
rf = create_model('rf')

# 5. Tune best models
tuned_xgb = tune_model(xgb, n_iter=50, optimize='AUC')
tuned_lgbm = tune_model(lgbm, n_iter=50, optimize='AUC')

# 6. Ensemble models
bagged_xgb = ensemble_model(tuned_xgb)

# 7. Blend models
blender = blend_models([tuned_xgb, tuned_lgbm, rf])

# 8. Stack models
stacker = stack_models([tuned_xgb, tuned_lgbm, bagged_xgb])

# 9. Evaluate
plot_model(stacker, plot='auc')
plot_model(stacker, plot='confusion_matrix')
plot_model(stacker, plot='feature')
interpret_model(stacker)

# 10. Predict on holdout
holdout_pred = predict_model(stacker)

# 11. Finalize model
final_model = finalize_model(stacker)

# 12. Save model
save_model(final_model, 'final_credit_model')

# 13. Load and predict
loaded = load_model('final_credit_model')
new_predictions = predict_model(loaded, data=new_data)
```

---

**© 2025 PyCaret. Licensed under MIT License.**
