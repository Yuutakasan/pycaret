# 次のステップ実行ガイド

## 🚀 現在の状態

### ✅ 完了した実装

1. **包括的売上インパクト分析**
   - 109特徴量の詳細分析完了
   - Top 20特徴量の特定（+797%～+18%）
   - CSV出力: `output/comprehensive_sales_impact_analysis.csv`

2. **ダッシュボード統合**
   - `店舗別包括ダッシュボード_v6.1_提案強化.ipynb` に分析セル追加
   - Cell 18: 包括的売上インパクト分析（全特徴量）

3. **Step5統合**
   - `Step5_CategoryWise_Compare_with_Overfitting.ipynb` にTop 20特徴量統合
   - Cell 4: TOP_20_FEATURES, EXCLUDE_NEGATIVE_FEATURES定義
   - 6箇所のcompare_models()にコメント追加

4. **A/Bテスト計画**
   - 4種類のテスト設計完了
   - 実施スケジュール策定
   - 期待ROI算出（総合900%）

---

## 📋 次のステップ（優先順位順）

### ステップ1: ダッシュボード実行【最優先】

**目的**: 包括的売上インパクト分析の結果を確認

**実行手順**:
```bash
# 1. JupyterLab起動
cd /mnt/d/github/pycaret/work
jupyter lab

# 2. ノートブック起動
# 店舗別包括ダッシュボード_v6.1_提案強化.ipynb を開く

# 3. Cell 1-17を順に実行（既存セル）
# 4. Cell 18を実行（新規追加された包括的インパクト分析）
```

**期待される出力**:
```
================================================================================
📊 包括的売上インパクト分析（格納率80%以上の全特徴量）
================================================================================

✅ 格納率80%以上のカラム: 132個
   分析対象カラム: 109個

✅ 分析完了: 109個の特徴量を分析

================================================================================
🏆 売上インパクトランキング Top 20
================================================================================
季節変動指数_変化率_月    +797.56% (+1,308円) [カテゴリカル]
季節変動指数_変化量_月    +797.56% (+1,308円) [カテゴリカル]
売上数量                +561.20% (+1,927円) [連続値（Q75 vs Q25）]
...

✅ 売上増加要因 Top 5:
  季節変動指数_変化率_月: +797.56% (+1,308円)
  ...

⚠️ 売上減少要因 Top 5:
  季節_上昇期: -70.54% (-712円)
  ...
```

**確認ポイント**:
- ✅ Top 20特徴量が正しく表示されるか
- ✅ カテゴリ別サマリーが生成されるか
- ✅ 推奨アクションが表示されるか

---

### ステップ2: Step5実行（GPU高速化）

**目的**: カテゴリ別予測モデル構築とTop 20特徴量の効果検証

**実行手順**:
```bash
# 1. JupyterLab起動（同じセッション）
# Step5_CategoryWise_Compare_with_Overfitting.ipynb を開く

# 2. Cell 1: 環境セットアップ
# 3. Cell 2: GPU高速化設定
# 4. Cell 3: カテゴリ戦略読み込み
# 5. Cell 4: Top 20特徴量統合（新規）← ここで特徴量確認
# 6. Cell 5: データ読み込み
# 7. Cell 6: オーバーフィッティング検出関数
# 8. Cell 7-9: グループA/B/C分析
# 9. Cell 10: サマリー
# 10. Cell 11-13: 店舗別分析
```

**実行時間**:
- **GPU有効時**: 約8分（XGBoost/CatBoost GPU）
- **CPU時**: 約40分

**期待される出力**:
```
================================================================================
📊 包括的売上インパクト分析 - Top 20特徴量
================================================================================

✅ Top 20特徴量をモデルに統合します
   重点特徴量: 20個
   除外特徴量: 5個

🏆 Top 5特徴量:
  1. 季節変動指数_変化率_月
  2. 季節変動指数_変化量_月
  3. 売上数量
  4. 寒くなった_7d
  5. 気温差_拡大

⚠️ 除外する負のインパクト特徴量:
  - 季節_上昇期
  - 暖かくなった_7d
  ...

Group A: 280:チケット・カード
✅ GPU対応モデル: XGBoost, CatBoost
Comparing models...
  XGBoost GPU: 1.03 sec/fold (✅ GPU有効)
  CatBoost GPU: 7.39 sec/fold (✅ GPU有効)

Best model: XGBoost
  Train R²: 0.78
  Test R²: 0.75
  R² Gap: 0.03 (None - 過学習なし)

📊 Overfitting Detection:
  ✅ No overfitting detected
  R² Gap: 3.0% < 5% (閾値)
  MAE increase: 8.2% < 30% (閾値)
```

**確認ポイント**:
- ✅ GPU有効化（XGBoost 1-2秒/fold、CatBoost 7-10秒/fold）
- ✅ Top 20特徴量がvalidate_features()で確認される
- ✅ 過学習検出が正しく機能
- ✅ 各カテゴリで Test R² > 0.70（Group A）、> 0.60（Group B）

---

### ステップ3: 予測精度の比較

**目的**: Top 20特徴量統合前後の精度向上を定量評価

**実行手順**:
```python
# Jupyter Notebook内で実行

# Before（既存の結果を読み込み）
baseline_results = pd.read_csv('output/category_compare_models_results.csv')

# After（Step5実行後）
new_results = compare_models(include=GPU_MODELS + ['et', 'rf'], sort='R2')

# 比較
improvement = (new_results['R2'].mean() - baseline_results['R2'].mean()) / baseline_results['R2'].mean()

print(f"📊 予測精度向上: {improvement:.2%}")
print(f"   Before: R² = {baseline_results['R2'].mean():.4f}")
print(f"   After:  R² = {new_results['R2'].mean():.4f}")
```

**期待される効果**:
- **予測精度向上**: +5-10%
- **過学習削減**: R² Gap -2-3%
- **実行時間短縮**: 40分 → 8分（GPU有効時）

---

### ステップ4: A/Bテスト準備

**目的**: Test 1（季節変動指数監視）の実施準備

**実行手順**:
```bash
# 1. 店舗選定スクリプト作成
cd /mnt/d/github/pycaret/work/scripts
python3 << 'PYTHON'
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('output/06_final_enriched_20250701_20250930.csv', encoding='utf-8-sig')

# 店舗別集計
store_features = df.groupby('店舗').agg({
    '売上金額': 'mean',
    '売上数量': 'mean',
    '季節変動指数_月': 'std'
}).reset_index()

# 2クラスタに分類
kmeans = KMeans(n_clusters=2, random_state=42)
store_features['cluster'] = kmeans.fit_predict(store_features[['売上金額', '売上数量', '季節変動指数_月']])

# 各クラスタから50店舗ずつ選定
control_stores = store_features[store_features['cluster'] == 0].sample(min(50, len(store_features[store_features['cluster'] == 0])))
treatment_stores = store_features[store_features['cluster'] == 1].sample(min(50, len(store_features[store_features['cluster'] == 1])))

# 保存
control_stores.to_csv('output/ab_test1_control_stores.csv', index=False, encoding='utf-8-sig')
treatment_stores.to_csv('output/ab_test1_treatment_stores.csv', index=False, encoding='utf-8-sig')

print(f"✅ Control群: {len(control_stores)}店舗")
print(f"✅ Treatment群: {len(treatment_stores)}店舗")
PYTHON
```

**成果物**:
- `output/ab_test1_control_stores.csv`
- `output/ab_test1_treatment_stores.csv`

---

### ステップ5: モニタリングダッシュボード構築

**目的**: A/Bテストのリアルタイム監視

**実行手順**:
```bash
# scripts/ab_test_monitoring.py を作成
cat > scripts/ab_test_monitoring.py << 'EOF'
#!/usr/bin/env python3
"""A/Bテスト モニタリングダッシュボード"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_monitoring_dashboard():
    # 週次売上データ読み込み
    df = pd.read_csv('output/06_final_enriched_20250701_20250930.csv', encoding='utf-8-sig')
    control_stores = pd.read_csv('output/ab_test1_control_stores.csv', encoding='utf-8-sig')
    treatment_stores = pd.read_csv('output/ab_test1_treatment_stores.csv', encoding='utf-8-sig')

    # 週次集計
    df['week'] = pd.to_datetime(df['日付']).dt.isocalendar().week
    df['test_group'] = df['店舗'].apply(lambda x: 'control' if x in control_stores['店舗'].values else ('treatment' if x in treatment_stores['店舗'].values else 'none'))

    weekly = df[df['test_group'] != 'none'].groupby(['week', 'test_group']).agg({
        '売上金額': 'mean'
    }).reset_index()

    # プロット
    fig = go.Figure()

    for group in ['control', 'treatment']:
        data = weekly[weekly['test_group'] == group]
        fig.add_trace(go.Scatter(
            x=data['week'],
            y=data['売上金額'],
            mode='lines+markers',
            name=group.capitalize()
        ))

    fig.update_layout(
        title='A/Bテスト週次モニタリング',
        xaxis_title='週番号',
        yaxis_title='平均売上金額（円）'
    )

    fig.write_html('output/ab_test_monitoring.html')
    print("✅ モニタリングダッシュボード生成: output/ab_test_monitoring.html")

if __name__ == '__main__':
    create_monitoring_dashboard()
