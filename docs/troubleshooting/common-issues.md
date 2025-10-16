# PyCaret Troubleshooting Guide

<div align="center">

![PyCaret Logo](../images/logo.png)

**Common Issues and Solutions**
**Version 3.4.0**

</div>

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Setup Issues](#setup-issues)
3. [Model Training Issues](#model-training-issues)
4. [Memory Issues](#memory-issues)
5. [GPU Issues](#gpu-issues)
6. [Performance Issues](#performance-issues)
7. [Prediction Issues](#prediction-issues)
8. [MLOps Integration Issues](#mlops-integration-issues)
9. [Data Issues](#data-issues)
10. [Error Messages](#error-messages)

---

## Installation Issues

### Issue 1.1: Dependency Conflicts

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all packages
```

**Cause:** Conflicting package versions in environment

**Solutions:**

**Option 1: Fresh Environment (Recommended)**
```bash
# Create new environment
python -m venv fresh_pycaret
source fresh_pycaret/bin/activate  # Windows: fresh_pycaret\Scripts\activate

# Install PyCaret first
pip install pycaret[full]
```

**Option 2: Force Reinstall**
```bash
pip install --upgrade --force-reinstall pycaret
```

**Option 3: Use Conda**
```bash
conda create -n pycaret_env python=3.10
conda activate pycaret_env
pip install pycaret[full]
```

### Issue 1.2: NumPy Version Conflict

**Symptom:**
```
RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf
```

**Solution:**
```bash
# Uninstall NumPy
pip uninstall numpy -y

# Install compatible version
pip install "numpy>=1.21,<1.27"

# Reinstall PyCaret
pip install --upgrade --no-cache-dir pycaret
```

### Issue 1.3: Scikit-learn Compatibility

**Symptom:**
```
ImportError: cannot import name 'check_is_fitted' from 'sklearn.utils.validation'
```

**Solution:**
```bash
# Install compatible scikit-learn
pip install "scikit-learn<1.5"
```

### Issue 1.4: LightGBM Installation Fails

**Windows:**
```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Then install LightGBM
pip install lightgbm
```

**macOS:**
```bash
# Install libomp
brew install libomp

# Install LightGBM
pip install lightgbm
```

**Linux:**
```bash
# Install dependencies
sudo apt-get install cmake build-essential

# Install LightGBM
pip install lightgbm
```

---

## Setup Issues

### Issue 2.1: Target Column Not Found

**Symptom:**
```
ValueError: Target column 'target_name' not found in data
```

**Solutions:**

**Check column names:**
```python
# Print all column names
print(data.columns.tolist())

# Check for spaces or special characters
print([f"'{col}'" for col in data.columns])
```

**Fix column names:**
```python
# Strip whitespace
data.columns = data.columns.str.strip()

# Use correct name
s = setup(data, target='correct_column_name')
```

### Issue 2.2: Categorical Features Not Recognized

**Symptom:**
Numeric features treated as categorical or vice versa

**Solution:**
```python
# Explicitly define feature types
s = setup(
    data=data,
    target='target',
    categorical_features=['cat1', 'cat2', 'cat3'],
    numeric_features=['num1', 'num2', 'num3'],
    date_features=['date1']
)
```

### Issue 2.3: High Cardinality Warning

**Symptom:**
```
Warning: Categorical feature 'X' has high cardinality (>25 unique values)
```

**Solutions:**

**Option 1: Use frequency encoding**
```python
s = setup(
    data=data,
    target='target',
    high_cardinality_features=['feature_name'],
    high_cardinality_method='frequency'
)
```

**Option 2: Ignore feature**
```python
s = setup(
    data=data,
    target='target',
    ignore_features=['high_cardinality_feature']
)
```

### Issue 2.4: Imbalanced Dataset

**Symptom:**
```
Warning: Target is imbalanced (90% vs 10%)
```

**Solution:**
```python
s = setup(
    data=data,
    target='target',
    fix_imbalance=True,
    fix_imbalance_method='smote'  # or 'smote', 'adasyn'
)
```

---

## Model Training Issues

### Issue 3.1: compare_models() Takes Too Long

**Solutions:**

**Option 1: Use turbo mode**
```python
best = compare_models(turbo=True)
```

**Option 2: Limit time**
```python
best = compare_models(budget_time=0.5)  # 30 seconds
```

**Option 3: Select specific models**
```python
best = compare_models(include=['lr', 'rf', 'xgboost'])
```

**Option 4: Reduce folds**
```python
s = setup(data, target='target', fold=3)  # Default is 10
best = compare_models()
```

### Issue 3.2: Model Training Fails

**Symptom:**
```
TypeError: fit() got an unexpected keyword argument 'X'
```

**Solution:**
```python
# Check model compatibility
from pycaret.classification import models
available = models()
print(available)

# Exclude problematic models
best = compare_models(exclude=['problematic_model'])
```

### Issue 3.3: All Models Return Same Score

**Cause:** Data leakage or incorrect setup

**Solution:**
```python
# Check for data leakage
s = setup(
    data=data,
    target='target',
    ignore_features=['id', 'timestamp'],  # Remove ID columns
    remove_multicollinearity=True
)
```

### Issue 3.4: tune_model() Not Improving

**Solutions:**

**Increase iterations:**
```python
tuned = tune_model(model, n_iter=100)  # Default is 10
```

**Use better search algorithm:**
```python
tuned = tune_model(
    model,
    search_library='optuna',
    n_iter=100,
    optimize='F1'
)
```

**Custom parameter grid:**
```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.01, 0.05, 0.1]
}
tuned = tune_model(model, custom_grid=param_grid)
```

---

## Memory Issues

### Issue 4.1: MemoryError During Setup

**Symptom:**
```
MemoryError: Unable to allocate ... GB for an array
```

**Solutions:**

**Option 1: Low memory mode**
```python
s = setup(
    data=data,
    target='target',
    low_memory=True,
    data_split_stratify=False
)
```

**Option 2: Sample data**
```python
# Use subset of data
sample_data = data.sample(frac=0.5, random_state=123)
s = setup(sample_data, target='target')
```

**Option 3: Reduce preprocessing**
```python
s = setup(
    data=data,
    target='target',
    polynomial_features=False,
    remove_multicollinearity=False,
    pca=False
)
```

### Issue 4.2: Kernel Crashes in Jupyter

**Solutions:**

**Increase memory limit:**
```python
# In Jupyter config
c.NotebookApp.iopub_data_rate_limit = 1e10
```

**Reduce output verbosity:**
```python
s = setup(data, target='target', verbose=False, html=False)
```

**Clear memory:**
```python
import gc

# After each operation
del model
gc.collect()
```

---

## GPU Issues

### Issue 5.1: GPU Not Detected

**Symptom:**
```
use_gpu=True but no GPU detected
```

**Solutions:**

**Check CUDA:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

**Install GPU libraries:**
```bash
# XGBoost GPU
pip install xgboost

# LightGBM GPU
pip install lightgbm --install-option=--gpu

# Check installation
python -c "import xgboost as xgb; print(xgb.__version__)"
```

### Issue 5.2: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

**Reduce batch size:**
```python
s = setup(
    data=data,
    target='target',
    use_gpu=True,
    # Reduce data size
    train_size=0.6
)
```

**Use CPU for some models:**
```python
# Train on GPU
gpu_model = create_model('xgboost')

# Train on CPU
s.use_gpu = False
cpu_model = create_model('rf')
```

---

## Performance Issues

### Issue 6.1: Training Too Slow

**Solutions:**

**Use parallel processing:**
```python
s = setup(data, target='target', n_jobs=-1)  # Use all cores
```

**Enable GPU:**
```python
s = setup(data, target='target', use_gpu=True)
```

**Use Intel optimization:**
```python
s = setup(data, target='target', engine='sklearnex')
```

**Reduce cross-validation:**
```python
s = setup(data, target='target', fold=3)
```

### Issue 6.2: Predictions Too Slow

**Solutions:**

**Use lighter model:**
```python
# Instead of ensemble/stack, use single model
fast_model = create_model('lightgbm')
```

**Reduce model complexity:**
```python
simple_model = create_model('lr')  # Logistic Regression
```

**Batch predictions:**
```python
# Predict in batches
batch_size = 1000
predictions = []
for i in range(0, len(data), batch_size):
    batch = data.iloc[i:i+batch_size]
    pred = predict_model(model, data=batch)
    predictions.append(pred)
```

---

## Prediction Issues

### Issue 7.1: Predictions Return NaN

**Cause:** Missing values in new data

**Solution:**
```python
# Check for missing values
print(new_data.isnull().sum())

# Fill missing values before prediction
new_data_filled = new_data.fillna(new_data.median())
predictions = predict_model(model, data=new_data_filled)
```

### Issue 7.2: Prediction Columns Missing

**Symptom:**
Prediction output doesn't have expected columns

**Solution:**
```python
# For classification - get probabilities
predictions = predict_model(model, raw_score=True)

# Columns will include:
# - prediction_label (predicted class)
# - prediction_score (probability)
```

### Issue 7.3: Different Results Each Run

**Cause:** Random seed not set

**Solution:**
```python
# Set seed in setup
s = setup(data, target='target', session_id=123)

# All subsequent operations will be reproducible
```

---

## MLOps Integration Issues

### Issue 8.1: MLflow Connection Failed

**Symptom:**
```
ConnectionError: Could not connect to MLflow server
```

**Solution:**
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri('http://localhost:5000')

# Or file-based
mlflow.set_tracking_uri('file:///path/to/mlruns')

# Then setup
s = setup(
    data=data,
    target='target',
    log_experiment=True,
    experiment_name='my_experiment'
)
```

### Issue 8.2: Model Deployment Fails

**AWS Deployment:**
```python
# Ensure credentials are set
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret'

# Deploy
deploy_model(
    model=final_model,
    model_name='model_name',
    platform='aws',
    authentication={'bucket': 'your-bucket'}
)
```

---

## Data Issues

### Issue 9.1: DateTime Not Recognized

**Solution:**
```python
# Convert to datetime
data['date_column'] = pd.to_datetime(data['date_column'])

# Setup with date features
s = setup(
    data=data,
    target='target',
    date_features=['date_column']
)
```

### Issue 9.2: Text Features Not Handled

**Solution:**
```python
# Identify text features
s = setup(
    data=data,
    target='target',
    text_features=['description', 'comments']
)
```

### Issue 9.3: Duplicate Rows

**Solution:**
```python
# Remove duplicates before setup
data = data.drop_duplicates()

# Or identify and handle
print(f"Duplicates: {data.duplicated().sum()}")
data = data.drop_duplicates(keep='first')
```

---

## Error Messages

### Error: "No module named 'pycaret'"

**Solution:**
```bash
# Check installation
pip show pycaret

# Reinstall
pip install --upgrade pycaret

# Check Python environment
which python  # Linux/Mac
where python  # Windows
```

### Error: "Target variable is constant"

**Solution:**
```python
# Check target distribution
print(data['target'].value_counts())

# Ensure at least 2 classes
data = data[data['target'].notna()]
```

### Error: "Sample size too small"

**Solution:**
```python
# Reduce folds
s = setup(data, target='target', fold=3)

# Or reduce train_size
s = setup(data, target='target', train_size=0.8)
```

### Error: "Invalid parameter value"

**Solution:**
```python
# Check parameter types and ranges
# Example: normalize_method must be in ['zscore', 'minmax', 'maxabs', 'robust']

s = setup(
    data=data,
    target='target',
    normalize=True,
    normalize_method='zscore'  # Valid value
)
```

---

## Diagnostic Commands

### Check PyCaret Version

```python
import pycaret
print(f"PyCaret version: {pycaret.__version__}")
```

### Check Environment

```python
from pycaret.utils import check_metric
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")
```

### Check Dependencies

```python
import pkg_resources

packages = [
    'pandas', 'numpy', 'scikit-learn', 'xgboost',
    'lightgbm', 'catboost', 'optuna', 'mlflow'
]

for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except:
        print(f"{package}: NOT INSTALLED")
```

### Enable Debug Logging

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# PyCaret will print detailed logs
s = setup(data, target='target', verbose=True)
```

---

## Getting Help

### Before Asking for Help

1. **Search existing issues**: https://github.com/pycaret/pycaret/issues
2. **Check documentation**: https://pycaret.gitbook.io/
3. **Review this troubleshooting guide**

### When Asking for Help

Include:

1. **System information:**
```python
import pycaret
import platform
import sys

print(f"PyCaret: {pycaret.__version__}")
print(f"Python: {sys.version}")
print(f"OS: {platform.platform()}")
```

2. **Complete error message** (full traceback)

3. **Minimal reproducible example:**
```python
from pycaret.datasets import get_data
from pycaret.classification import *

# Your code that causes the issue
data = get_data('juice')
s = setup(data, target='Purchase')
```

### Support Channels

- **Slack Community**: https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w
- **GitHub Discussions**: https://github.com/pycaret/pycaret/discussions
- **GitHub Issues**: https://github.com/pycaret/pycaret/issues (for bugs)
- **Stack Overflow**: Tag with `pycaret`

---

**© 2025 PyCaret. Licensed under MIT License.**
