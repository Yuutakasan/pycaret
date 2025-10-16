# ABC Analysis Module Documentation
# ABC分析モジュール ドキュメント

## 📋 Overview / 概要

The ABC Analysis Module provides comprehensive product classification and Pareto analysis capabilities for PyCaret. This module implements industry-standard ABC classification with advanced features for retail and inventory analytics.

ABC分析モジュールは、PyCaret向けの包括的な製品分類とパレート分析機能を提供します。このモジュールは、小売および在庫分析のための高度な機能を備えた業界標準のABC分類を実装しています。

**File Location / ファイルの場所:**
- Module: `/mnt/d/github/pycaret/src/analysis/abc_analysis.py`
- 821 lines of production-ready code
- Full bilingual support (English/Japanese)

## ✨ Key Features / 主な機能

### 1️⃣ Category-wise ABC Classification (カテゴリ別ABC分類)

```python
from src.analysis import ABCAnalyzer, MetricType

analyzer = ABCAnalyzer(language='ja')
result = analyzer.classify(
    data=df,
    value_col='revenue',
    item_col='product_id',
    metric_type=MetricType.REVENUE
)
```

**Supports multiple metrics:**
- **Revenue (売上)**: Sales-based classification
- **Profit (利益)**: Profit-based classification
- **Quantity (数量)**: Volume-based classification
- **Margin (利益率)**: Margin-based classification
- **Custom (カスタム)**: Custom metric support

### 2️⃣ Pareto Analysis with 80/20 Rule (パレート分析)

Automatically calculates cumulative percentages and identifies:
- **A-category**: Top 80% of value (typically 20% of items)
- **B-category**: Next 15% of value (typically 30% of items)
- **C-category**: Remaining 5% of value (typically 50% of items)

```python
# View Pareto data
print(result.pareto_data.head())

# Visualize Pareto chart
analyzer.plot_abc_chart(result, save_path='pareto_chart.png')
```

### 3️⃣ Product Ranking by Multiple Metrics (複数指標による製品ランキング)

```python
result = analyzer.multi_metric_classification(
    data=df,
    item_col='product_id',
    metrics={
        'revenue': 'sales_amount',
        'profit': 'profit_amount',
        'quantity': 'units_sold'
    },
    weights={
        'revenue': 0.5,
        'profit': 0.3,
        'quantity': 0.2
    }
)
```

**Features:**
- Composite scoring with custom weights
- Independent ABC classification per metric
- Final classification based on weighted score

### 4️⃣ Cross-Category Comparison (カテゴリ間比較)

```python
comparison = analyzer.cross_category_comparison(
    data=df,
    value_col='revenue',
    item_col='product_id',
    category_col='product_category'
)

# Visualize comparison
analyzer.plot_category_comparison(
    comparison,
    save_path='category_comparison.png'
)
```

**Analyzes:**
- ABC distribution across categories
- Item counts per category and ABC class
- Value percentages per category

### 5️⃣ Time-based ABC Shifts Detection (時系列ABC推移検出)

```python
shifts = analyzer.detect_abc_shifts(
    data=df,
    value_col='revenue',
    item_col='product_id',
    time_col='date',
    periods=['2024-01', '2024-02', '2024-03']
)
```

**Detects:**
- Products moving between ABC categories
- New products entering the analysis
- Stable vs. shifting classifications
- Trend patterns (A→B, B→C, etc.)

### 6️⃣ Store-Specific ABC Patterns (店舗別ABCパターン)

```python
store_results = analyzer.store_specific_abc(
    data=df,
    value_col='revenue',
    item_col='product_id',
    store_col='store_id'
)

# Access results per store
for store_id, result in store_results.items():
    print(f"Store {store_id} Summary:")
    print(result.summary)
```

**Provides:**
- Independent ABC analysis per store
- Store-level performance comparison
- Location-based optimization insights

### 7️⃣ Visual ABC Matrix Generation (ビジュアルABCマトリックス)

```python
# Generate multi-metric result first
multi_result = analyzer.multi_metric_classification(...)

# Create ABC matrix
analyzer.plot_abc_matrix(
    multi_metric_result=multi_result,
    metric1_col='revenue_ABC',
    metric2_col='profit_ABC',
    save_path='abc_matrix.png'
)
```

**Creates:**
- Heatmap-style visualization
- Cross-metric ABC comparison
- Count-based matrix cells

## 🎯 Quick Start / クイックスタート

### Basic Usage

```python
from src.analysis import quick_abc_analysis
import pandas as pd

# Load your data
df = pd.read_csv('sales_data.csv')

# Perform ABC analysis
result = quick_abc_analysis(
    data=df,
    value_col='revenue',
    item_col='product_id',
    language='en',
    plot=True
)

# View summary
print(result.summary)
```

### Advanced Configuration

```python
from src.analysis import ABCAnalyzer, ABCThresholds

# Custom thresholds (default: 80/95/100)
custom_thresholds = ABCThresholds(
    a_threshold=0.70,  # Top 70%
    b_threshold=0.90,  # Next 20%
    c_threshold=1.00   # Remaining 10%
)

analyzer = ABCAnalyzer(
    thresholds=custom_thresholds,
    language='ja'
)

result = analyzer.classify(df, 'revenue', 'product_id')
```

## 📊 Output Structure / 出力構造

### ABCResult Object

```python
@dataclass
class ABCResult:
    data: pd.DataFrame          # Full classified data
    summary: pd.DataFrame       # Category-wise summary
    pareto_data: pd.DataFrame   # Pareto analysis data
    metric_type: MetricType     # Metric used
    thresholds: ABCThresholds   # Thresholds applied
```

### Summary DataFrame Columns

| Column (EN) | Column (JA) | Description |
|-------------|-------------|-------------|
| ABC_Category | ABC分類 | A, B, or C classification |
| Item_Count | アイテム数 | Number of items in category |
| Total_Value | 合計値 | Sum of values in category |
| Average_Value | 平均値 | Mean value per item |
| Item_Percentage | - | Percentage of total items |
| Value_Percentage | 値の割合 | Percentage of total value |

### Classified Data Columns

| Column (EN) | Column (JA) | Description |
|-------------|-------------|-------------|
| Rank | 順位 | Item ranking by value |
| ABC_Category | ABC分類 | Classification result |
| Cumulative_Percentage | 累積割合 | Cumulative value % |
| Value_Percentage | 値の割合 | Individual value % |

## 🔧 API Reference / API リファレンス

### Main Classes

#### `ABCAnalyzer`

**Constructor:**
```python
ABCAnalyzer(
    thresholds: Optional[ABCThresholds] = None,
    language: str = 'en'
)
```

**Key Methods:**

1. **classify()** - Main ABC classification method
2. **multi_metric_classification()** - Multi-metric analysis
3. **detect_abc_shifts()** - Time-series shift detection
4. **store_specific_abc()** - Per-store analysis
5. **cross_category_comparison()** - Category comparison
6. **plot_abc_chart()** - Pareto chart visualization
7. **plot_abc_matrix()** - ABC matrix heatmap
8. **plot_category_comparison()** - Category comparison charts

### Helper Functions

```python
# Quick analysis function
quick_abc_analysis(
    data: pd.DataFrame,
    value_col: str,
    item_col: str,
    language: str = 'en',
    plot: bool = True
) -> ABCResult

# Alias
abc_classify = quick_abc_analysis
```

## 🌟 Use Cases / 使用例

### 1. Inventory Management (在庫管理)

```python
# Classify products by sales volume
result = analyzer.classify(
    inventory_df,
    value_col='sales_volume',
    item_col='sku',
    metric_type=MetricType.QUANTITY
)

# Focus on A-category items for priority stocking
a_items = result.data[result.data['ABC_Category'] == 'A']
```

### 2. Procurement Optimization (調達最適化)

```python
# Multi-metric classification for procurement decisions
procurement = analyzer.multi_metric_classification(
    df,
    item_col='product_id',
    metrics={
        'revenue': 'sales',
        'profit_margin': 'margin',
        'turnover': 'inventory_turns'
    },
    weights={'revenue': 0.4, 'profit_margin': 0.4, 'turnover': 0.2}
)
```

### 3. Customer Segmentation (顧客セグメンテーション)

```python
# ABC analysis on customer value
customer_result = analyzer.classify(
    customer_df,
    value_col='lifetime_value',
    item_col='customer_id',
    metric_type=MetricType.CUSTOM
)
```

### 4. Store Performance Analysis (店舗パフォーマンス分析)

```python
# Compare ABC patterns across stores
store_results = analyzer.store_specific_abc(
    sales_df,
    value_col='revenue',
    item_col='product_id',
    store_col='store_id'
)

# Identify high-performing stores
for store, result in store_results.items():
    a_count = (result.data['ABC_Category'] == 'A').sum()
    print(f"{store}: {a_count} A-category products")
```

## 📈 Visualization Examples / 可視化の例

### 1. Pareto Chart
![Pareto Chart Example](example_pareto.png)

### 2. ABC Matrix
![ABC Matrix Example](example_matrix.png)

### 3. Category Comparison
![Category Comparison Example](example_comparison.png)

## 🔄 Integration with PyCaret / PyCaret統合

```python
# Can be used with PyCaret's data preparation
from pycaret.datasets import get_data

# Load sample data
data = get_data('retail')

# Perform ABC analysis
from src.analysis import ABCAnalyzer

analyzer = ABCAnalyzer(language='en')
result = analyzer.classify(data, 'Sales', 'Product_ID')

# Use results for feature engineering in PyCaret
data['ABC_Category'] = result.data['ABC_Category']
```

## ⚠️ Important Notes / 重要な注意事項

1. **Data Requirements:**
   - Value column must be numeric
   - Item column should have unique identifiers
   - Negative values trigger warnings but are allowed

2. **Performance:**
   - Optimized for datasets up to 1M rows
   - Group operations are vectorized for speed
   - Visualization scales well up to 1000 items

3. **Threshold Customization:**
   - Default 80/15/5 rule is industry standard
   - Adjust based on your business context
   - Validate thresholds before use

4. **Language Support:**
   - Column names adapt to selected language
   - Messages and labels are bilingual
   - Charts use appropriate fonts for Japanese

## 🐛 Troubleshooting / トラブルシューティング

### Common Issues

**Issue 1: ValueError on missing columns**
```python
# Solution: Check column names
print(df.columns.tolist())
```

**Issue 2: Warning about negative values**
```python
# Solution: Filter or transform negative values
df = df[df['revenue'] >= 0]
```

**Issue 3: Empty result**
```python
# Solution: Check data has values
print(df['value_col'].describe())
```

## 📚 References / 参考文献

1. **ABC Analysis Theory:**
   - Pareto Principle (80/20 rule)
   - Inventory categorization methods
   - Multi-criteria decision making

2. **Implementation Standards:**
   - ISO 9001 quality management principles
   - Retail analytics best practices
   - Data-driven inventory optimization

## 📝 License

MIT License - See project root for details

## 👥 Contributing

Contributions welcome! Please follow PyCaret's contribution guidelines.

## 📧 Support

For issues and questions:
- GitHub Issues: https://github.com/pycaret/pycaret/issues
- Documentation: https://pycaret.gitbook.io/

---

**Module Version:** 1.0.0
**Last Updated:** 2025-10-08
**Maintained by:** PyCaret Development Team
