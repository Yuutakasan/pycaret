# PyCaret FAQ / よくある質問

<div align="center">

![PyCaret Logo](../images/logo.png)

**Frequently Asked Questions**
**Version 3.4.0**

</div>

---

## 📚 Table of Contents / 目次

### General / 一般
- [What is PyCaret?](#what-is-pycaret)
- [PyCaretとは何ですか？](#pycaretとは何ですか)

### Installation / インストール
- [How do I install PyCaret?](#how-do-i-install-pycaret)
- [インストール方法は？](#インストール方法は)

### Usage / 使用方法
- [Getting Started](#getting-started)
- [はじめに](#はじめに-1)

### Performance / パフォーマンス
- [Speed and Optimization](#speed-and-optimization)
- [速度と最適化](#速度と最適化)

### Advanced Topics / 高度なトピック
- [Advanced Features](#advanced-features)
- [高度な機能](#高度な機能)

---

## General / 一般

### What is PyCaret?

**Q: What is PyCaret?**

**A:** PyCaret is an open-source, low-code machine learning library in Python that automates the machine learning workflow. It's designed to:
- Reduce coding time by replacing hundreds of lines with a few
- Automate model selection and hyperparameter tuning
- Provide easy-to-use APIs for both beginners and experts
- Support multiple ML tasks: classification, regression, clustering, anomaly detection, and time series

**Q: How does PyCaret differ from scikit-learn?**

**A:** PyCaret is built on top of scikit-learn and other libraries, providing:
- **Higher abstraction**: One function for multiple tasks
- **Automation**: Auto model selection and tuning
- **Comparison**: Built-in model comparison
- **Visualization**: Integrated plotting and evaluation
- **MLOps**: Built-in experiment tracking and deployment

Example comparison:
```python
# Scikit-learn (50+ lines)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# ... more imports and code

# PyCaret (5 lines)
from pycaret.classification import *
s = setup(data, target='target')
best = compare_models()
predictions = predict_model(best)
```

### PyCaretとは何ですか？

**Q: PyCaretとは何ですか？**

**A:** PyCaretは、機械学習のワークフローを自動化するPythonのオープンソース・ローコードライブラリです。以下の特徴があります：
- 数百行のコードを数行に削減
- モデル選択とハイパーパラメータチューニングの自動化
- 初心者から専門家まで使いやすいAPI
- 複数のMLタスクをサポート：分類、回帰、クラスタリング、異常検知、時系列

**Q: PyCaretの商用利用は可能ですか？**

**A:** はい、PyCaretはMITライセンスで提供されており、商用利用を含むあらゆる用途で無料でお使いいただけます。

**Q: どのような業界で使われていますか？**

**A:**
- 🏦 金融：信用リスク評価、不正検知
- 🏥 医療：病気予測、患者分類
- 🛒 小売：需要予測、顧客セグメンテーション
- 📱 テック：ユーザー行動分析、レコメンデーション
- 🏭 製造：予知保全、品質管理

---

## Installation / インストール

### How do I install PyCaret?

**Q: What are the system requirements?**

**A:**
- **Python**: 3.9, 3.10, 3.11, or 3.12
- **OS**: Windows 7+, macOS 10.12+, Ubuntu 16.04+
- **RAM**: 4 GB minimum (8 GB+ recommended)
- **Disk**: 2 GB free space

**Q: How do I install the basic version?**

**A:**
```bash
pip install pycaret
```

**Q: How do I install all features?**

**A:**
```bash
pip install pycaret[full]
```

**Q: Can I use PyCaret with Conda?**

**A:** Yes:
```bash
conda create -n pycaret_env python=3.10
conda activate pycaret_env
pip install pycaret[full]
```

**Q: Does PyCaret work on Apple M1/M2?**

**A:** Yes, PyCaret is compatible with Apple Silicon. Install using:
```bash
pip install pycaret[full]
```

### インストール方法は？

**Q: 基本的なインストール方法は？**

**A:**
```bash
pip install pycaret
```

**Q: すべての機能をインストールするには？**

**A:**
```bash
pip install pycaret[full]
```

**Q: Google Colabで使えますか？**

**A:** はい、Google Colabでそのまま使用できます：
```python
!pip install pycaret[full]
from pycaret.classification import *
```

**Q: インストールにどのくらい時間がかかりますか？**

**A:**
- 基本版：5〜10分
- フル版：10〜15分
（インターネット速度により変動）

---

## Getting Started

**Q: How do I start my first project?**

**A:** Follow these steps:

```python
# 1. Import module
from pycaret.classification import *

# 2. Load data
from pycaret.datasets import get_data
data = get_data('juice')

# 3. Initialize setup
s = setup(data, target='Purchase', session_id=123)

# 4. Compare models
best = compare_models()

# 5. Make predictions
predictions = predict_model(best)

# 6. Save model
save_model(best, 'my_model')
```

**Q: What datasets are available for practice?**

**A:** PyCaret provides 50+ datasets:
```python
from pycaret.datasets import get_data

# Classification
data = get_data('juice')        # Customer purchase
data = get_data('credit')       # Credit risk
data = get_data('diabetes')     # Medical diagnosis

# Regression
data = get_data('insurance')    # Insurance premium
data = get_data('concrete')     # Material strength
data = get_data('diamond')      # Diamond pricing

# Time Series
data = get_data('airline')      # Airline passengers
data = get_data('gold')         # Gold prices

# List all datasets
from pycaret.datasets import get_data
get_data('index')
```

**Q: How do I use my own CSV data?**

**A:**
```python
import pandas as pd

# Load your data
data = pd.read_csv('my_data.csv')

# Setup and use
from pycaret.classification import *
s = setup(data, target='target_column')
```

### はじめに

**Q: 最初のプロジェクトを始めるには？**

**A:** 以下の手順に従ってください：

```python
# 1. モジュールのインポート
from pycaret.classification import *

# 2. データの読み込み
import pandas as pd
data = pd.read_csv('データ.csv')

# 3. セットアップ
s = setup(data, target='目的変数', session_id=123)

# 4. モデルの比較
best = compare_models()

# 5. 予測の実行
predictions = predict_model(best)

# 6. モデルの保存
save_model(best, 'モデル名')
```

**Q: プログラミング経験がなくても使えますか？**

**A:** はい、基本的なPythonの知識があれば使用できます。Jupyter NotebookやGoogle Colabを使うことをお勧めします。

**Q: 日本語データでも使えますか？**

**A:** はい、UTF-8エンコードのCSVファイルであれば日本語データも問題なく使用できます：
```python
# 日本語データの読み込み
data = pd.read_csv('データ.csv', encoding='utf-8')

# 日本語カラム名も使用可能
s = setup(data, target='購入有無')
```

---

## Usage Questions

**Q: How do I handle missing values?**

**A:** PyCaret handles missing values automatically:
```python
s = setup(
    data=data,
    target='target',
    numeric_imputation='mean',        # or 'median', 'mode', 'knn'
    categorical_imputation='mode'     # or 'constant'
)
```

**Q: How do I handle imbalanced data?**

**A:**
```python
s = setup(
    data=data,
    target='target',
    fix_imbalance=True,
    fix_imbalance_method='smote'  # or 'adasyn'
)
```

**Q: How do I improve model accuracy?**

**A:**

1. **Feature engineering:**
```python
s = setup(
    data=data,
    target='target',
    polynomial_features=True,
    feature_interaction=True,
    feature_ratio=True
)
```

2. **Hyperparameter tuning:**
```python
tuned = tune_model(best, n_iter=100, optimize='F1')
```

3. **Ensemble methods:**
```python
# Bagging
bagged = ensemble_model(best)

# Blending
blender = blend_models([model1, model2, model3])

# Stacking
stacker = stack_models([model1, model2, model3])
```

**Q: How do I select specific models?**

**A:**
```python
# Include only specific models
best = compare_models(include=['rf', 'xgboost', 'lightgbm'])

# Exclude certain models
best = compare_models(exclude=['knn', 'svm'])
```

**Q: How do I use custom evaluation metrics?**

**A:**
```python
from sklearn.metrics import make_scorer, f1_score

# Create custom scorer
f1_scorer = make_scorer(f1_score, average='weighted')

# Add metric
add_metric('custom_f1', 'F1 Score', f1_scorer)

# Use in tuning
tuned = tune_model(model, optimize='custom_f1')
```

---

## Speed and Optimization

**Q: How can I speed up training?**

**A:**

1. **Use turbo mode:**
```python
best = compare_models(turbo=True)
```

2. **Reduce folds:**
```python
s = setup(data, target='target', fold=3)  # Default is 10
```

3. **Set time limit:**
```python
best = compare_models(budget_time=0.5)  # 30 seconds
```

4. **Use parallel processing:**
```python
s = setup(data, target='target', n_jobs=-1)  # Use all cores
```

5. **Enable GPU:**
```python
s = setup(data, target='target', use_gpu=True)
```

**Q: Which models are fastest?**

**A:** Speed ranking (fastest to slowest):
1. Logistic Regression (`'lr'`)
2. Ridge Classifier (`'ridge'`)
3. Naive Bayes (`'nb'`)
4. Decision Tree (`'dt'`)
5. LightGBM (`'lightgbm'`)
6. Random Forest (`'rf'`)
7. XGBoost (`'xgboost'`)
8. Neural Networks (`'mlp'`)

### 速度と最適化

**Q: 学習を高速化するには？**

**A:**

1. **ターボモードを使用:**
```python
best = compare_models(turbo=True)
```

2. **フォールド数を削減:**
```python
s = setup(data, target='target', fold=3)
```

3. **GPU を有効化:**
```python
s = setup(data, target='target', use_gpu=True)
```

**Q: メモリ不足エラーが出る場合は？**

**A:**
```python
# ローメモリモード
s = setup(
    data=data,
    target='target',
    low_memory=True,
    data_split_stratify=False
)

# またはデータをサンプリング
sample_data = data.sample(frac=0.5, random_state=123)
s = setup(sample_data, target='target')
```

---

## Advanced Features

**Q: How do I use PyCaret with MLflow?**

**A:**
```python
s = setup(
    data=data,
    target='target',
    log_experiment=True,
    experiment_name='my_experiment'
)

# Train models - automatically logged to MLflow
best = compare_models()

# View in MLflow UI
!mlflow ui
```

**Q: How do I deploy a model?**

**A:**

**Save locally:**
```python
save_model(final_model, 'model_name')
```

**Deploy to cloud:**
```python
# AWS
deploy_model(
    model=final_model,
    model_name='production_model',
    platform='aws',
    authentication={'bucket': 'my-bucket'}
)

# Azure
deploy_model(
    model=final_model,
    model_name='production_model',
    platform='azure',
    authentication={'subscription_id': 'xxx'}
)
```

**Create API:**
```python
from pycaret.classification import create_api

create_api(final_model, 'api_name')
# Creates FastAPI application
```

**Q: How do I interpret model predictions?**

**A:**

**SHAP values:**
```python
interpret_model(model)  # Overall feature importance

interpret_model(model, plot='correlation', observation=0)  # Single prediction
```

**Feature importance:**
```python
plot_model(model, plot='feature')
```

**Partial dependence:**
```python
from pycaret.classification import dashboard
dashboard(model)  # Interactive dashboard
```

**Q: How do I handle time series data?**

**A:**
```python
from pycaret.time_series import *

# Setup for forecasting
s = setup(
    data=data,
    target='sales',
    fh=30,              # Forecast 30 periods
    fold=5,
    session_id=123
)

# Compare models
best = compare_models()

# Forecast
forecast = predict_model(best, fh=30)

# Plot forecast
plot_model(best, plot='forecast')
```

**Q: Can I use custom models?**

**A:** Yes:
```python
from sklearn.base import BaseEstimator, ClassifierMixin

# Create custom model
class MyCustomModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # Training logic
        return self

    def predict(self, X):
        # Prediction logic
        return predictions

# Add to PyCaret
custom = create_model(MyCustomModel())
```

### 高度な機能

**Q: クロスバリデーションの設定方法は？**

**A:**
```python
s = setup(
    data=data,
    target='target',
    fold_strategy='stratifiedkfold',  # or 'kfold', 'groupkfold'
    fold=10,
    fold_shuffle=True
)
```

**Q: カスタム前処理パイプラインを使用できますか？**

**A:**
```python
from sklearn.preprocessing import PowerTransformer

# カスタムパイプライン
custom_pipeline = PowerTransformer()

s = setup(
    data=data,
    target='target',
    custom_pipeline=custom_pipeline,
    custom_pipeline_position=0  # Pipeline position
)
```

---

## Model-Specific Questions

**Q: Which model should I use?**

**A:** Depends on your use case:

| Use Case | Recommended Models |
|----------|-------------------|
| **High accuracy** | XGBoost, LightGBM, CatBoost |
| **Fast training** | Logistic Regression, Naive Bayes |
| **Interpretability** | Decision Tree, Logistic Regression |
| **Large datasets** | LightGBM, Linear models |
| **Small datasets** | Random Forest, SVM |
| **Imbalanced data** | XGBoost with `scale_pos_weight` |

**Q: How do I handle categorical variables with many categories?**

**A:**
```python
s = setup(
    data=data,
    target='target',
    high_cardinality_features=['zip_code', 'user_id'],
    high_cardinality_method='frequency'  # or 'clustering'
)
```

**Q: Can I use PyCaret for multi-label classification?**

**A:** Not directly, but you can:
1. Use binary relevance (train separate model for each label)
2. Use classifier chains
3. Convert to multi-class if possible

---

## Troubleshooting

**Q: I get "Target is constant" error**

**A:**
```python
# Check target values
print(data['target'].value_counts())

# Ensure at least 2 different values exist
data = data[data['target'].notna()]
```

**Q: Setup is very slow**

**A:**
```python
# Disable HTML display
s = setup(data, target='target', html=False, verbose=False)

# Reduce preprocessing
s = setup(
    data=data,
    target='target',
    profile=False,
    polynomial_features=False
)
```

**Q: Predictions don't match training accuracy**

**A:** Possible causes:
1. **Overfitting**: Use cross-validation scores, not training scores
2. **Data leakage**: Check `ignore_features`
3. **Different data distribution**: Check data drift

**Q: Where can I get help?**

**A:**
- 📖 Documentation: https://pycaret.gitbook.io/
- 💬 Slack: https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w
- 🐛 GitHub Issues: https://github.com/pycaret/pycaret/issues
- 💭 Discussions: https://github.com/pycaret/pycaret/discussions

---

## Resources

### Learning Resources
- [Official Documentation](https://pycaret.gitbook.io/)
- [Video Tutorials](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g)
- [Example Notebooks](https://github.com/pycaret/pycaret/tree/master/examples)
- [Blog Posts](https://pycaret.gitbook.io/docs/learn-pycaret/official-blog)

### Community
- [Slack Community](https://join.slack.com/t/pycaret/shared_invite/zt-row9phbm-BoJdEVPYnGf7_NxNBP307w)
- [GitHub Discussions](https://github.com/pycaret/pycaret/discussions)
- [LinkedIn](https://www.linkedin.com/company/pycaret/)

### Contributing
- [Contribution Guide](https://github.com/pycaret/pycaret/blob/master/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/pycaret/pycaret/blob/master/CODE_OF_CONDUCT.md)

---

**© 2025 PyCaret. Licensed under MIT License.**
