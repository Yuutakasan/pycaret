# PyCaret Dashboard Interpretation Guide

<div align="center">

![PyCaret Logo](../images/logo.png)

**Complete Guide to Understanding PyCaret Dashboards and Visualizations**
**Version 3.4.0**

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Classification Dashboards](#classification-dashboards)
3. [Regression Dashboards](#regression-dashboards)
4. [Clustering Dashboards](#clustering-dashboards)
5. [Anomaly Detection Dashboards](#anomaly-detection-dashboards)
6. [Time Series Dashboards](#time-series-dashboards)
7. [Interactive Dashboards](#interactive-dashboards)
8. [Custom Visualizations](#custom-visualizations)
9. [Exporting and Sharing](#exporting-and-sharing)

---

## Overview

PyCaret provides comprehensive visualization capabilities for model evaluation and interpretation. This guide explains how to read and interpret each dashboard type.

### Generating Dashboards

```python
# Method 1: Interactive evaluation (Jupyter only)
evaluate_model(model)

# Method 2: Specific plots
plot_model(model, plot='auc')

# Method 3: Interactive dashboard (requires explainerdashboard)
dashboard(model)

# Method 4: SHAP interpretations
interpret_model(model)
```

---

## Classification Dashboards

### 1. AUC-ROC Curve

**What it shows:** Model's ability to distinguish between classes

```python
plot_model(model, plot='auc')
```

**How to read:**

![AUC-ROC Example](https://via.placeholder.com/600x400?text=AUC-ROC+Curve)

- **X-axis:** False Positive Rate (FPR)
- **Y-axis:** True Positive Rate (TPR/Recall)
- **Diagonal line:** Random classifier (AUC = 0.5)
- **Curve:** Your model's performance

**Interpretation:**

| AUC Score | Interpretation |
|-----------|---------------|
| 0.90-1.00 | ⭐⭐⭐⭐⭐ Excellent |
| 0.80-0.90 | ⭐⭐⭐⭐ Good |
| 0.70-0.80 | ⭐⭐⭐ Fair |
| 0.60-0.70 | ⭐⭐ Poor |
| 0.50-0.60 | ⭐ Fail |

**Example scenario:**

```
AUC = 0.92 for credit card fraud detection
✅ Excellent! Model correctly identifies 92% of fraud cases
   while minimizing false alarms
```

### 2. Confusion Matrix

**What it shows:** Actual vs predicted classifications

```python
plot_model(model, plot='confusion_matrix')
```

**How to read:**

```
                Predicted
              No    Yes
Actual  No   [TN]  [FP]
        Yes  [FN]  [TP]

TN = True Negatives  (Correctly predicted No)
FP = False Positives (Incorrectly predicted Yes)
FN = False Negatives (Incorrectly predicted No)
TP = True Positives  (Correctly predicted Yes)
```

**Real example:**

```
Customer Churn Prediction (1000 customers)

                Predicted
              Stay  Churn
Actual  Stay  [850]  [50]   ← 850 correctly predicted to stay
        Churn  [30]  [70]   ← 70 correctly predicted to churn

Accuracy = (850 + 70) / 1000 = 92%
```

**What to look for:**

- ✅ **High diagonal values** (TN and TP) = Good
- ❌ **High off-diagonal values** (FP and FN) = Bad

### 3. Precision-Recall Curve

**What it shows:** Trade-off between precision and recall

```python
plot_model(model, plot='pr')
```

**How to read:**

- **Precision:** Of all predicted positives, how many are correct?
  ```
  Precision = TP / (TP + FP)
  ```

- **Recall:** Of all actual positives, how many did we find?
  ```
  Recall = TP / (TP + FN)
  ```

**Use cases:**

| Scenario | Optimize For | Why |
|----------|-------------|-----|
| Spam detection | **Precision** | Avoid blocking legitimate emails |
| Cancer diagnosis | **Recall** | Catch all possible cases |
| Fraud detection | **Balance** | Catch fraud without too many false alarms |

**Example:**

```
Email Spam Filter

High Precision (0.98):
- 98% of emails marked as spam are actually spam
- Few legitimate emails blocked
- Some spam might slip through

High Recall (0.95):
- 95% of all spam emails are caught
- Might block some legitimate emails
- Very few spam emails reach inbox
```

### 4. Classification Report

**What it shows:** Comprehensive metrics for each class

```python
plot_model(model, plot='class_report')
```

**Metrics explained:**

| Metric | Formula | What it means |
|--------|---------|---------------|
| **Precision** | TP/(TP+FP) | How many predicted positives are correct? |
| **Recall** | TP/(TP+FN) | How many actual positives did we find? |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean of precision and recall |
| **Support** | Count | Number of samples in each class |

**Example report:**

```
              Precision  Recall  F1-Score  Support
Class 0 (No)     0.94     0.96     0.95      500
Class 1 (Yes)    0.85     0.80     0.82      200

Accuracy:                          0.91      700
Macro avg:       0.90     0.88     0.89      700
Weighted avg:    0.91     0.91     0.91      700
```

**Interpretation:**

- Class 0 has better performance (higher scores)
- Class 1 is harder to predict (lower scores)
- Overall accuracy is 91%
- Model is slightly better at predicting "No" than "Yes"

### 5. Feature Importance

**What it shows:** Which features contribute most to predictions

```python
plot_model(model, plot='feature')
```

**How to read:**

```
Feature Importance

Monthly_Charges     ████████████████  0.25
Contract_Type       ████████████      0.18
Tenure              ██████████        0.15
Internet_Service    ████████          0.12
...
```

**Interpretation:**

- **Top features:** Most influential in predictions
- **Low importance:** Consider removing to reduce complexity
- **Zero importance:** Definitely can remove

**Example scenario:**

```
Customer Churn Prediction

Top 3 Features:
1. Monthly_Charges (25%)    ← Customers with high charges churn more
2. Contract_Type (18%)      ← Month-to-month contracts churn more
3. Tenure (15%)             ← New customers churn more

Action: Focus retention efforts on high-charge,
        month-to-month, new customers
```

### 6. Learning Curve

**What it shows:** How model performance changes with training data size

```python
plot_model(model, plot='learning')
```

**How to read:**

```
                              ┌─ Training Score
Performance                   │  (should be high)
    ^                    ╱────┘
    │               ╱────┘
    │          ╱────┴─────── Validation Score
    │     ╱────             (should increase and
    │╱────                   converge with training)
    └─────────────────────>
       Training Set Size
```

**Diagnosis:**

| Pattern | Issue | Solution |
|---------|-------|----------|
| **Large gap between curves** | Overfitting | Add more data, reduce complexity |
| **Both curves low** | Underfitting | Add features, use complex model |
| **Curves converged and high** | ✅ Good fit | Model is working well |

**Example:**

```
Scenario 1: Overfitting
Training Score:   95% ────┐
Validation Score: 70% ─┐  │ ← 25% gap
                       └──┘
Solution: Get more training data or simplify model

Scenario 2: Good Fit
Training Score:   88% ──┐
Validation Score: 85% ─┘│ ← 3% gap, both high
                        └─
Solution: Model is ready for production!
```

---

## Regression Dashboards

### 1. Residual Plot

**What it shows:** Prediction errors distribution

```python
plot_model(model, plot='residuals')
```

**How to read:**

```
Residuals
    ^
    │    ·  ·     ·
    │  ·  ·  ·  ·     ← Random scatter = Good
  0 ├──·──·──·──·──
    │  ·     ·  ·
    │    ·  ·
    └─────────────> Predicted Values

Good: Random scatter around zero
Bad:  Pattern (curve, funnel shape)
```

**Patterns to watch:**

| Pattern | Issue | What it means |
|---------|-------|---------------|
| **Random scatter** | ✅ None | Model is good |
| **Curved pattern** | ❌ Non-linearity | Need polynomial features |
| **Funnel shape** | ❌ Heteroscedasticity | Variance increases with prediction |
| **Outliers** | ⚠️ Unusual cases | May need special handling |

**Example:**

```
House Price Prediction

Good Residuals:
   Errors randomly distributed around $0
   ✅ Model predicts well across all price ranges

Bad Residuals (Funnel):
   Small errors for cheap houses
   Large errors for expensive houses
   ❌ Model struggles with luxury properties
   Solution: Transform target variable (log) or use separate model
```

### 2. Prediction Error Plot

**What it shows:** Actual vs predicted values

```python
plot_model(model, plot='error')
```

**How to read:**

```
Actual Values
    ^      ·
    │    ·   ·
    │  ·   ·   ·     ← Points close to diagonal = Good predictions
    │·   ·   ·   ·
    │  ·   ·   ·
    └─────────────> Predicted Values
         Diagonal = Perfect predictions
```

**Interpretation:**

- **Points on diagonal:** Perfect predictions
- **Points above diagonal:** Under-predicted
- **Points below diagonal:** Over-predicted
- **Tight clustering:** Good model
- **Wide scatter:** Poor model

**Example:**

```
Sales Forecasting

Perfect predictions (diagonal):
$100K actual → $100K predicted ✅

Under-prediction (above diagonal):
$150K actual → $100K predicted ❌
Missing $50K in forecast

Over-prediction (below diagonal):
$80K actual → $120K predicted ❌
Over-forecasted by $40K
```

### 3. Feature Importance (Regression)

```python
plot_model(model, plot='feature')
```

**Example:**

```
House Price Prediction

Feature Importance:
Location        ██████████████████ 35%  ← Most important
Square_Feet     ████████████       22%
Bedrooms        ██████             12%
Age             ████               8%
...

Interpretation:
- Location determines 35% of price variation
- Size (square feet) is second most important
- Number of bedrooms less important than size
- Age has small impact
```

---

## Clustering Dashboards

### 1. Elbow Plot

**What it shows:** Optimal number of clusters

```python
plot_model(model, plot='elbow')
```

**How to read:**

```
Distortion Score
    ^
    │╲
    │ ╲
    │  ╲___    ← "Elbow" point = optimal clusters
    │      ────────
    └─────────────> Number of Clusters
      2  3  4  5  6
```

**Finding the elbow:**

```
Example: Customer Segmentation

2 clusters: Distortion = 1000  (high)
3 clusters: Distortion = 400   (big drop)
4 clusters: Distortion = 350   (small drop) ← Elbow!
5 clusters: Distortion = 330   (tiny drop)

Recommendation: Use 4 clusters
```

### 2. Silhouette Plot

**What it shows:** Cluster quality

```python
plot_model(model, plot='silhouette')
```

**How to read:**

```
Cluster 0  ████████████████     ← Wide, positive = Good
Cluster 1  ██████████████       ← Good
Cluster 2  ████                 ← Narrow = Weak cluster

Silhouette Score: 0.7 (Good!)
```

**Silhouette score interpretation:**

| Score | Quality | Interpretation |
|-------|---------|---------------|
| 0.7-1.0 | ⭐⭐⭐⭐⭐ Excellent | Strong, well-separated clusters |
| 0.5-0.7 | ⭐⭐⭐⭐ Good | Reasonable structure |
| 0.25-0.5 | ⭐⭐⭐ Fair | Weak structure |
| < 0.25 | ⭐⭐ Poor | No meaningful clusters |

### 3. Cluster Distribution

```python
plot_model(model, plot='distribution')
```

**Example:**

```
Customer Segments (E-commerce)

Cluster 0 (High-value): 15%
- High spending, frequent purchases
- Target for premium products

Cluster 1 (Regular): 60%
- Moderate spending, occasional purchases
- Target for promotions

Cluster 2 (Inactive): 25%
- Low spending, rare purchases
- Target for re-engagement campaigns
```

---

## Anomaly Detection Dashboards

### 1. Outlier Distribution

```python
plot_model(model, plot='tsne')
```

**How to read:**

- **Normal points:** Clustered together
- **Anomalies:** Far from clusters
- **Borderline:** Near cluster edges

**Example:**

```
Credit Card Fraud Detection

Normal transactions (blue):  99.2%
- Clustered in center
- Similar patterns

Anomalies (red): 0.8%
- Scattered far from cluster
- Unusual spending patterns
- Flagged for review
```

---

## Time Series Dashboards

### 1. Forecast Plot

```python
from pycaret.time_series import *
plot_model(model, plot='forecast')
```

**Components:**

```
Sales Forecast

Historical Data  │ Forecast
                 │
    ──────────── │ ─ ─ ─ ─ ─  ← Point forecast
         ╱╲      │   ╱╲  ╱╲
        ╱  ╲     │  ╱  ╲╱  ╲
    ───╲  ╱─────┼─────────────
        ╲╱       │ ░░░░░░░░░  ← Confidence interval
                 │
                 Now
```

**Interpretation:**

- **Solid line:** Historical actuals
- **Dashed line:** Point forecast (most likely)
- **Shaded area:** Confidence interval (uncertainty range)

**Example:**

```
Monthly Sales Forecast

January Forecast: $150K ± $20K
- Point estimate: $150K
- 95% confidence: $130K - $170K
- Wide interval = high uncertainty

Interpretation:
- Most likely: $150K
- Pessimistic: $130K (plan for this)
- Optimistic: $170K
- Use $130K for conservative planning
```

### 2. Components Plot

```python
plot_model(model, plot='decomp')
```

**Shows four components:**

```
1. Observed (Original data)
   ─────╱╲─╱╲─╱╲─

2. Trend (Long-term direction)
   ────────╱╱╱╱╱

3. Seasonal (Repeating patterns)
   ─╱╲─╱╲─╱╲─╱╲─

4. Residual (Noise)
   ─·─·─·─·─·─·─
```

**Example:**

```
Retail Sales Decomposition

Trend: Increasing 📈
- Business growing 15% per year

Seasonal: Strong pattern 🔄
- Peak in December (holidays)
- Low in January (post-holiday)
- Regular pattern every year

Residual: Small noise ✅
- Model captures most variation
- Random fluctuations minimal
```

### 3. Actual vs Prediction

```python
plot_model(model, plot='insample')
```

**How to read:**

```
Sales (in thousands)

    ─── Actual
    ─ ─ Predicted

       ╱╲    ╱╲
      ╱  ╲  ╱  ╲    ← Predicted follows actual
     ╱    ╲╱    ╲
────╱            ╲──

Good: Lines overlap closely
Bad:  Large gaps between lines
```

**Metrics to check:**

- **MAE** (Mean Absolute Error): Average error in same units
- **RMSE** (Root Mean Square Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Error as percentage

**Example:**

```
Daily Website Traffic Forecast

MAE = 100 visitors
- On average, off by 100 visitors per day

RMSE = 150 visitors
- Some days have larger errors

MAPE = 5%
- Typically within 5% of actual

Interpretation:
Forecast: 2,000 visitors
Actual likely: 1,900 - 2,100 (±5%)
✅ Good for planning server capacity
```

---

## Interactive Dashboards

### ExplainerDashboard

```python
from pycaret.classification import dashboard

dashboard(model)
```

**Features:**

1. **Model Performance Tab**
   - Confusion matrix
   - ROC curves
   - Precision-Recall
   - Metrics summary

2. **Feature Importance Tab**
   - Global importance
   - Permutation importance
   - SHAP values

3. **What-If Analysis**
   - Change feature values
   - See prediction change
   - Understand model behavior

4. **Individual Predictions**
   - SHAP force plots
   - Feature contributions
   - Why this prediction?

**Example use case:**

```
Loan Approval Model Dashboard

Scenario: Denied application
Customer: Age=25, Income=$40K, Credit_Score=650

What-If Analysis:
"What credit score needed for approval?"
- Current: 650 → Denied
- Try: 700 → Denied
- Try: 750 → Approved ✅

Insight: Customer needs to improve credit score to 750
```

---

## Custom Visualizations

### Creating Custom Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions
predictions = predict_model(model)

# Custom visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=predictions, x='actual', y='prediction_label')
plt.title('Custom Prediction Plot')
plt.show()
```

### Combining Multiple Plots

```python
from pycaret.classification import *

# Create subplot grid
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: AUC
plot_model(model, plot='auc', save=True)

# Plot 2: Confusion Matrix
plot_model(model, plot='confusion_matrix', save=True)

# Plot 3: Feature Importance
plot_model(model, plot='feature', save=True)

# Plot 4: Learning Curve
plot_model(model, plot='learning', save=True)
```

---

## Exporting and Sharing

### Save Individual Plots

```python
# Save as PNG
plot_model(model, plot='auc', save=True)
# Saves to: AUC.png

# Save with custom name
plot_model(model, plot='confusion_matrix', save='cm_final.png')
```

### Create HTML Report

```python
# Export dashboard to HTML
from pycaret.classification import dashboard

dashboard(model, mode='external')
# Opens in web browser and saves HTML
```

### Generate PDF Report

```python
# Using matplotlib
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('model_report.pdf') as pdf:
    # Page 1: AUC
    plot_model(model, plot='auc')
    pdf.savefig()
    plt.close()

    # Page 2: Confusion Matrix
    plot_model(model, plot='confusion_matrix')
    pdf.savefig()
    plt.close()

    # Page 3: Feature Importance
    plot_model(model, plot='feature')
    pdf.savefig()
    plt.close()
```

---

## Dashboard Best Practices

### 1. Choose Right Visualizations

| Task | Essential Plots |
|------|----------------|
| **Classification** | AUC, Confusion Matrix, Feature Importance |
| **Regression** | Residuals, Prediction Error, Feature Importance |
| **Clustering** | Elbow, Silhouette, Distribution |
| **Time Series** | Forecast, Decomposition, Actual vs Predicted |

### 2. Interpretation Checklist

Before deploying a model, check:

- ✅ **Confusion Matrix**: Errors acceptable?
- ✅ **Learning Curve**: No overfitting?
- ✅ **Feature Importance**: Makes business sense?
- ✅ **Residuals**: Randomly distributed?
- ✅ **Cross-validation**: Consistent across folds?

### 3. Stakeholder Communication

**For Technical Audience:**
- Show all metrics
- Include statistical tests
- Discuss assumptions and limitations

**For Business Audience:**
- Focus on confusion matrix (real numbers)
- Translate metrics to business impact
- Use what-if scenarios
- Highlight actionable insights

**Example:**

```
Technical:
"Model achieves 0.92 AUC with 85% recall at 95% precision"

Business:
"Out of 100 fraud cases, we'll catch 85 while only flagging
5 legitimate transactions for review"
```

---

## Summary

### Quick Reference

| Need | Function |
|------|----------|
| **All plots** | `evaluate_model(model)` |
| **Specific plot** | `plot_model(model, plot='name')` |
| **Interactive** | `dashboard(model)` |
| **Interpretation** | `interpret_model(model)` |
| **Save plot** | `plot_model(model, plot='name', save=True)` |

### Plot Names Reference

**Classification:**
- `'auc'`, `'pr'`, `'confusion_matrix'`, `'error'`, `'class_report'`
- `'boundary'`, `'rfe'`, `'learning'`, `'manifold'`, `'calibration'`
- `'feature'`, `'lift'`, `'gain'`

**Regression:**
- `'residuals'`, `'error'`, `'cooks'`, `'rfe'`, `'learning'`
- `'vc'`, `'manifold'`, `'feature'`, `'tree'`

**Clustering:**
- `'cluster'`, `'tsne'`, `'elbow'`, `'silhouette'`, `'distance'`, `'distribution'`

**Time Series:**
- `'forecast'`, `'insample'`, `'residuals'`, `'diagnostics'`, `'decomp'`

---

**© 2025 PyCaret. Licensed under MIT License.**
