# PyCaret Installation & Setup Guide

<div align="center">

![PyCaret Logo](../images/logo.png)

**Complete Installation and Configuration Guide**
**Version 3.4.0**

</div>

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Environment Setup](#environment-setup)
4. [GPU Configuration](#gpu-configuration)
5. [Docker Setup](#docker-setup)
6. [IDE Configuration](#ide-configuration)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 7+, macOS 10.12+, Ubuntu 16.04+
- **Python**: 3.9, 3.10, 3.11, or 3.12
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk Space**: 2 GB free space
- **Internet**: Required for installation

### Recommended Requirements

- **RAM**: 16 GB or more
- **CPU**: Multi-core processor (4+ cores)
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Storage**: SSD for faster I/O operations

### Supported Platforms

| Platform | Architecture | Support Level |
|----------|-------------|---------------|
| Windows | x64 | ✅ Fully Supported |
| macOS | x64, ARM64 (M1/M2) | ✅ Fully Supported |
| Linux | x64 | ✅ Fully Supported |
| Linux | ARM64 | ⚠️ Experimental |

---

## Installation Methods

### Method 1: PyPI (Recommended)

#### Basic Installation

```bash
pip install pycaret
```

**Installation time:** ~5-10 minutes

**What's included:**
- Core machine learning functionality
- Basic preprocessing
- Common models (Random Forest, Decision Trees, etc.)
- Standard visualization

#### Full Installation

```bash
pip install pycaret[full]
```

**What's included (additional):**
- XGBoost, CatBoost, LightGBM
- Advanced tuning (Optuna, Hyperopt)
- MLflow integration
- SHAP interpretability
- Data profiling
- Dashboard functionality
- All optional dependencies

#### Selective Installation

Install only what you need:

```bash
# Analysis tools (SHAP, profiling, dashboards)
pip install pycaret[analysis]

# Additional models (XGBoost, CatBoost, etc.)
pip install pycaret[models]

# Hyperparameter tuners (Optuna, Ray Tune)
pip install pycaret[tuners]

# MLOps tools (MLflow, Gradio)
pip install pycaret[mlops]

# Parallel processing (Dask, Fugue)
pip install pycaret[parallel]

# Development tools
pip install pycaret[dev]

# Testing tools
pip install pycaret[test]

# Multiple extras
pip install pycaret[analysis,models,mlops]
```

### Method 2: Conda

```bash
# Create new environment
conda create -n pycaret_env python=3.10
conda activate pycaret_env

# Install PyCaret
pip install pycaret[full]

# Or use conda-forge
conda install -c conda-forge pycaret
```

### Method 3: Development Version

Install the latest development version from GitHub:

```bash
pip install git+https://github.com/pycaret/pycaret.git@master --upgrade
```

⚠️ **Warning:** Development version may be unstable.

### Method 4: Docker

See [Docker Setup](#docker-setup) section.

---

## Environment Setup

### Virtual Environment (Recommended)

#### Using venv

```bash
# Create virtual environment
python -m venv pycaret_env

# Activate (Windows)
pycaret_env\Scripts\activate

# Activate (macOS/Linux)
source pycaret_env/bin/activate

# Install PyCaret
pip install pycaret[full]

# Deactivate when done
deactivate
```

#### Using conda

```bash
# Create environment
conda create -n pycaret_env python=3.10

# Activate
conda activate pycaret_env

# Install
pip install pycaret[full]

# Deactivate
conda deactivate
```

### Jupyter Notebook Setup

```bash
# Install Jupyter
pip install jupyter notebook

# Register environment as kernel
python -m ipykernel install --user --name=pycaret_env --display-name "Python (PyCaret)"

# Start Jupyter
jupyter notebook
```

### JupyterLab Setup

```bash
# Install JupyterLab
pip install jupyterlab

# Install extensions
pip install jupyterlab-execute-time
pip install aquirdturtle_collapsible_headings

# Start JupyterLab
jupyter lab
```

### Google Colab

PyCaret works out-of-the-box on Google Colab:

```python
# Install in Colab
!pip install pycaret[full]

# Import and use
from pycaret.classification import *
```

---

## GPU Configuration

### NVIDIA GPU Setup

#### Prerequisites

1. **NVIDIA GPU** with compute capability 3.5+
2. **NVIDIA Drivers** (Latest recommended)
3. **CUDA Toolkit** 11.0+
4. **cuDNN** 8.0+

#### Installation Steps

**Step 1: Install CUDA Toolkit**

Download from: https://developer.nvidia.com/cuda-downloads

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi
```

**Step 2: Install cuDNN**

Download from: https://developer.nvidia.com/cudnn

**Step 3: Install GPU-enabled libraries**

```bash
# XGBoost with GPU
pip install xgboost

# LightGBM with GPU
pip install lightgbm --install-option=--gpu

# CatBoost (GPU support built-in)
pip install catboost

# RAPIDS cuML (optional, for GPU-accelerated scikit-learn)
pip install cuml-cu11
```

**Step 4: Enable GPU in PyCaret**

```python
from pycaret.classification import setup

s = setup(
    data=data,
    target='target',
    use_gpu=True  # Enable GPU
)
```

#### Supported GPU Models

| Model | GPU Support | Library |
|-------|-------------|---------|
| XGBoost | ✅ Yes | xgboost |
| LightGBM | ✅ Yes | lightgbm |
| CatBoost | ✅ Yes | catboost |
| Scikit-learn models | ✅ Yes (with cuML) | cuml |
| Neural Networks | ✅ Yes | torch/tensorflow |

### Intel CPU Optimization

Use Intel's scikit-learn extension for better CPU performance:

```bash
# Install Intel extension
pip install scikit-learn-intelex

# Enable in PyCaret
from pycaret.classification import setup

s = setup(
    data=data,
    target='target',
    engine='sklearnex'
)
```

---

## Docker Setup

### Pre-built Images

#### Slim Version (Basic)

```bash
# Pull image
docker pull pycaret/slim

# Run container
docker run -p 8888:8888 pycaret/slim
```

#### Full Version (All features)

```bash
# Pull image
docker pull pycaret/full

# Run container
docker run -p 8888:8888 pycaret/full
```

#### GPU Version

```bash
# Pull GPU image
docker pull pycaret/full:gpu

# Run with GPU support
docker run --gpus all -p 8888:8888 pycaret/full:gpu
```

### Custom Dockerfile

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyCaret
RUN pip install --no-cache-dir pycaret[full]

# Install Jupyter
RUN pip install --no-cache-dir jupyter

# Expose Jupyter port
EXPOSE 8888

# Set working directory
WORKDIR /workspace

# Start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

### Build and Run

```bash
# Build image
docker build -t my-pycaret .

# Run container
docker run -p 8888:8888 -v $(pwd):/workspace my-pycaret
```

### Docker Compose

```yaml
version: '3.8'

services:
  pycaret:
    image: pycaret/full
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/workspace
      - ./data:/data
    environment:
      - JUPYTER_ENABLE_LAB=yes
    restart: unless-stopped
```

Run with:
```bash
docker-compose up
```

---

## IDE Configuration

### VS Code

#### Installation

1. Install VS Code: https://code.visualstudio.com/
2. Install Python extension
3. Install Jupyter extension

#### Configuration

**.vscode/settings.json**
```json
{
    "python.defaultInterpreterPath": "./pycaret_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "jupyter.jupyterServerType": "local"
}
```

### PyCharm

#### Installation

1. Install PyCharm: https://www.jetbrains.com/pycharm/
2. Create new project
3. Configure interpreter to use virtual environment

#### Configuration

**Settings → Project → Python Interpreter**
- Select existing virtual environment
- Or create new Conda environment

### Spyder

```bash
# Install Spyder in your environment
pip install spyder

# Launch Spyder
spyder
```

---

## Verification

### Basic Verification

```python
# Check version
import pycaret
print(f"PyCaret version: {pycaret.__version__}")

# Expected output: 3.4.0
```

### Comprehensive Check

```python
# Check all modules
from pycaret.classification import setup as clf_setup
from pycaret.regression import setup as reg_setup
from pycaret.clustering import setup as clu_setup
from pycaret.anomaly import setup as ano_setup
from pycaret.time_series import setup as ts_setup

print("✅ All modules imported successfully!")

# Test basic functionality
from pycaret.datasets import get_data
data = get_data('juice')
print(f"✅ Dataset loaded: {data.shape}")

# Test setup
s = clf_setup(data, target='Purchase', session_id=123, verbose=False)
print("✅ Setup completed successfully!")
```

### GPU Verification

```python
# Check GPU availability
import torch
print(f"PyTorch GPU available: {torch.cuda.is_available()}")

# Check XGBoost GPU
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")

# Check RAPIDS cuML (if installed)
try:
    import cuml
    print(f"✅ cuML available: {cuml.__version__}")
except ImportError:
    print("⚠️ cuML not installed (optional)")
```

### Check Installed Extras

```python
# Check optional dependencies
optional_packages = {
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'catboost': 'CatBoost',
    'optuna': 'Optuna',
    'shap': 'SHAP',
    'mlflow': 'MLflow',
    'ydata_profiling': 'YData Profiling'
}

for package, name in optional_packages.items():
    try:
        __import__(package)
        print(f"✅ {name} installed")
    except ImportError:
        print(f"❌ {name} not installed")
```

---

## Troubleshooting

### Common Installation Issues

#### Issue 1: Dependency Conflicts

**Error:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**Solution:**
```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows

# Install PyCaret first
pip install pycaret[full]

# Then install other packages
```

#### Issue 2: NumPy Version Conflict

**Error:**
```
RuntimeError: module compiled against API version ... but this version of numpy is ...
```

**Solution:**
```bash
# Uninstall and reinstall NumPy
pip uninstall numpy
pip install "numpy>=1.21,<1.27"
```

#### Issue 3: Scikit-learn Version Issue

**Error:**
```
ImportError: cannot import name 'X' from 'sklearn'
```

**Solution:**
```bash
# Install compatible version
pip install "scikit-learn<1.5"
```

#### Issue 4: LightGBM Installation Fails

**Solution (Windows):**
```bash
# Install Visual C++ Build Tools first
# Download from: https://visualstudio.microsoft.com/downloads/

# Then install LightGBM
pip install lightgbm
```

**Solution (macOS):**
```bash
# Install libomp
brew install libomp

# Install LightGBM
pip install lightgbm
```

#### Issue 5: Memory Error

**Error:**
```
MemoryError: Unable to allocate ... for an array with shape ...
```

**Solution:**
```python
# Use low memory mode
from pycaret.classification import setup

s = setup(
    data=data,
    target='target',
    low_memory=True,
    data_split_stratify=False
)
```

### Platform-Specific Issues

#### Windows

**Long Path Issue:**
```bash
# Enable long paths in Windows 10+
# Run as Administrator in PowerShell:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**SSL Certificate Error:**
```bash
# Update certifi
pip install --upgrade certifi
```

#### macOS

**Command Line Tools:**
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

**M1/M2 Compatibility:**
```bash
# Use Rosetta for some packages
arch -x86_64 pip install package_name

# Or use native ARM64 Python
pip install package_name
```

#### Linux

**Missing Libraries:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++
```

### Getting Help

If you encounter issues:

1. **Check Documentation**: https://pycaret.gitbook.io/
2. **Search GitHub Issues**: https://github.com/pycaret/pycaret/issues
3. **Ask on Slack**: https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w
4. **Post on GitHub Discussions**: https://github.com/pycaret/pycaret/discussions

When asking for help, include:
- Python version: `python --version`
- PyCaret version: `python -c "import pycaret; print(pycaret.__version__)"`
- OS and version
- Complete error message
- Minimal reproducible example

---

## Next Steps

After successful installation:

1. 📖 Read the [User Guide](../user-guide/店長向け操作ガイド.md)
2. 💻 Try [Quick Start Examples](https://pycaret.gitbook.io/docs/get-started/quickstart)
3. 📚 Explore [API Documentation](../api-reference/classification-api.md)
4. 🎓 Watch [Video Tutorials](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g)
5. 🚀 Build your first project!

---

**© 2025 PyCaret. Licensed under MIT License.**
