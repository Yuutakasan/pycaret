#!/usr/bin/env python3
"""動的売上予測システムのGPUエラーを修正（CPU版に適切にフォールバック）"""

import nbformat

# CPU/GPU適応型のモデリングコード
adaptive_modeling = """# %% [markdown]
# # 6. PyCaretによる包括的なモデル比較と最適化（GPU対応版）

# %%
print("="*80)
print("🚀 PyCaretモデリング - GPU/CPU自動選択")
print("="*80)

# PyCaretのインポート
from pycaret.regression import *

# GPU使用可能かチェック
GPU_AVAILABLE = False
try:
    import cuml
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ cuMLが利用可能です")

    # GPUメモリ確認
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"📊 GPU メモリ: {info.used/1e9:.1f}/{info.total/1e9:.1f} GB使用中")
    except:
        pass

    # LightGBM GPU対応チェック
    try:
        import lightgbm as lgb
        # テストトレーニングでGPU動作確認
        import numpy as np
        test_data = lgb.Dataset(np.random.rand(100, 10), label=np.random.rand(100))
        params = {'device': 'gpu', 'verbosity': -1}
        lgb.train(params, test_data, num_boost_round=1)
        print("✅ LightGBM GPU版が利用可能です")
        LIGHTGBM_GPU = True
    except Exception as e:
        print(f"⚠️ LightGBM GPU版は利用できません: {str(e)[:100]}")
        print("   CPU版で処理します")
        LIGHTGBM_GPU = False
        GPU_AVAILABLE = False  # PyCaretのGPUモードもオフ

except ImportError:
    print("⚠️ cuMLが利用できません - CPU版で処理します")

# GPU/CPU両対応のデータフレーム変換
if is_gpu_df:
    # cuDFの場合、一時的にpandasに変換
    df_modeling = df.to_pandas()
    print("📊 GPU DataFrame → CPU DataFrame変換完了")
else:
    df_modeling = df.copy()

# 特徴量と目的変数の準備
print("\\n📊 特徴量エンジニアリング中...")

# 日付特徴量を数値化
df_modeling['年'] = df_modeling['日付'].dt.year
df_modeling['月'] = df_modeling['日付'].dt.month
df_modeling['日'] = df_modeling['日付'].dt.day
df_modeling['曜日'] = df_modeling['日付'].dt.dayofweek
df_modeling['週'] = df_modeling['日付'].dt.isocalendar().week

# 予測に使用しない列を除外
exclude_cols = ['日付', '商品名', '店舗', 'フェイスくくり大分類', 'フェイスくくり中分類',
                'フェイスくくり小分類', '売上金額']

# 目的変数
target = '売上数量'

# 特徴量の選択（数値型のみ）
feature_cols = [col for col in df_modeling.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols and col != target]

print(f"\\n使用する特徴量: {len(feature_cols)}個")
print(f"目的変数: {target}")

# モデリング用データの準備
modeling_data = df_modeling[feature_cols + [target]].copy()

# 欠損値を除去
modeling_data = modeling_data.dropna()
print(f"\\nモデリングデータ: {len(modeling_data):,}行")

# データ分割（時系列を考慮した分割）
train_size = int(len(modeling_data) * 0.8)
train_data = modeling_data.iloc[:train_size].copy()
test_data = modeling_data.iloc[train_size:].copy()

print(f"訓練データ: {len(train_data):,}行")
print(f"テストデータ: {len(test_data):,}行")

# %%
print("="*80)
print("🔧 PyCaretセットアップ")
print("="*80)

if GPU_AVAILABLE:
    print("🚀 GPU加速モードで実行")
else:
    print("💻 CPU版で実行（より安定した処理）")

# PyCaretのセットアップ（GPU/CPU自動選択）
exp = setup(
    data=train_data,
    target=target,
    session_id=123,
    verbose=False,
    html=False,
    n_jobs=-1,  # 全CPUコアを使用
    use_gpu=GPU_AVAILABLE,  # GPU使用（LightGBMがサポートしている場合のみ）
    normalize=True,
    transformation=True,
    polynomial_features=False,  # 計算コスト考慮
    feature_selection=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.9
)

print("\\n✅ PyCaretセットアップ完了")

# %%
print("="*80)
print("🏆 全モデル比較")
print("="*80)

if GPU_AVAILABLE:
    print("\\n🚀 GPU対応モデルを優先的に使用します")
else:
    print("\\n💻 CPU版で全モデルを比較します")
    print("   ※ この処理には10〜30分かかる場合があります")

# 全てのモデルを比較
all_models = compare_models(
    n_select=10,  # 上位10モデルを選択
    sort='MAE',
    verbose=True,
    errors='ignore'
)

print("\\n✅ モデル比較完了")
print(f"\\n選択された上位10モデル:")
for i, model in enumerate(all_models, 1):
    print(f"  {i}. {model.__class__.__name__}")

# %%
print("="*80)
print("🔧 ベストモデルのチューニング")
print("="*80)

# 最良モデルを選択
best_model = all_models[0]
print(f"\\nベストモデル: {best_model.__class__.__name__}")

# ハイパーパラメータチューニング
print("\\nハイパーパラメータチューニング中...")

tuned_model = tune_model(
    best_model,
    optimize='MAE',
    n_iter=50,
    verbose=False
)

print("\\n✅ チューニング完了")

# %%
print("="*80)
print("📊 モデル評価")
print("="*80)

# テストデータでの予測
predictions = predict_model(tuned_model, data=test_data)

# 評価指標の計算
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

actual = predictions[target]
predicted = predictions['prediction_label']

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)
mape = mean_absolute_percentage_error(actual, predicted) * 100

print("\\n【予測精度】")
print(f"  MAE (平均絶対誤差): {mae:.2f}個")
print(f"  RMSE (二乗平均平方根誤差): {rmse:.2f}個")
print(f"  R² (決定係数): {r2:.4f}")
print(f"  MAPE (平均絶対パーセント誤差): {mape:.2f}%")

print("\\n【解釈】")
print(f"  ✓ 予測値は平均して実測値から{mae:.1f}個の誤差がある")
print(f"  ✓ モデルは売上変動の{r2*100:.1f}%を説明できている")
print(f"  ✓ 予測誤差は平均{mape:.1f}%")

# 実測値 vs 予測値のプロット
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 散布図
axes[0].scatter(actual, predicted, alpha=0.5, s=20, color='steelblue')
axes[0].plot([actual.min(), actual.max()], [actual.min(), actual.max()],
             'r--', linewidth=2, label='完璧な予測')
axes[0].set_title(f'実測値 vs 予測値 (R²={r2:.4f})', fontsize=14, fontweight='bold')
axes[0].set_xlabel('実測値（個）', fontsize=11)
axes[0].set_ylabel('予測値（個）', fontsize=11)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 残差プロット
residuals = actual - predicted
axes[1].scatter(predicted, residuals, alpha=0.5, s=20, color='coral')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[1].set_title('残差プロット', fontsize=14, fontweight='bold')
axes[1].set_xlabel('予測値（個）', fontsize=11)
axes[1].set_ylabel('残差（実測値 - 予測値）', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
print("="*80)
print("🎯 特徴量重要度分析")
print("="*80)

try:
    plot_model(tuned_model, plot='feature', display_format='streamlit')
    print("\\n✅ 特徴量重要度を表示しました")
except Exception as e:
    print(f"\\n⚠️ 特徴量重要度プロットのエラー: {e}")

# %%
print("="*80)
print("💾 モデル保存")
print("="*80)

# モデルの保存
model_path = 'output/pycaret_models/best_sales_prediction_model'
Path('output/pycaret_models').mkdir(parents=True, exist_ok=True)

save_model(tuned_model, model_path)

print(f"\\n✅ モデルを保存しました: {model_path}.pkl")
print("\\n💡 次回の使用方法:")
print(f"  loaded_model = load_model('{model_path}')")
print(f"  predictions = predict_model(loaded_model, data=new_data)")

# モデル情報の保存
model_info = {
    'model_name': tuned_model.__class__.__name__,
    'mae': float(mae),
    'rmse': float(rmse),
    'r2': float(r2),
    'mape': float(mape),
    'train_size': len(train_data),
    'test_size': len(test_data),
    'features': feature_cols,
    'target': target,
    'gpu_enabled': GPU_AVAILABLE,
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

import json
with open('output/pycaret_models/model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f"\\n✅ モデル情報を保存しました")

# %%
print("="*80)
print("🚀 予測実行デモ")
print("="*80)

# テストデータの最初の10件で予測デモ
demo_data = test_data.head(10).copy()
demo_predictions = predict_model(tuned_model, data=demo_data)

result_df = pd.DataFrame({
    '実測値': demo_predictions[target].values,
    '予測値': demo_predictions['prediction_label'].values.round(0),
    '誤差': (demo_predictions['prediction_label'].values - demo_predictions[target].values).round(1),
    '誤差率(%)': ((demo_predictions['prediction_label'].values - demo_predictions[target].values) /
                 demo_predictions[target].values * 100).round(1)
})

print("\\n【予測デモ（上位10件）】")
print(result_df.to_string(index=False))

print("\\n" + "="*80)
print("✅ モデリング完了")
print("="*80)
print(f"\\n📊 最終モデル: {tuned_model.__class__.__name__}")
print(f"📈 予測精度 (MAE): {mae:.2f}個")
print(f"📈 決定係数 (R²): {r2:.4f}")
print(f"🖥️ 実行モード: {'GPU加速' if GPU_AVAILABLE else 'CPU版（安定動作）'}")
print(f"💾 保存先: {model_path}.pkl")
"""

print("=" * 70)
print("動的売上予測システムのGPUエラーを修正")
print("=" * 70)

try:
    with open('動的売上予測システム - PyCaret 3.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # GPU加速版のセルを探して置き換え
    replaced = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and '🚀 GPU加速版モデリング - PyCaretとcuML' in cell.source:
            # 既存のセルを置き換え
            cell.source = adaptive_modeling
            print(f"✓ セル {i}: GPU/CPU適応型に置き換えました")
            replaced = True
            break

    if not replaced:
        print("⚠️ GPU加速版セルが見つかりませんでした")

    # 保存
    with open('動的売上予測システム - PyCaret 3.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n✅ 修正完了")
    print("\n改善内容:")
    print("  🔧 LightGBM GPU対応チェックを追加")
    print("  🔧 GPU未対応の場合は自動的にCPU版に切り替え")
    print("  🔧 より安定した処理を実現")
    print("  🔧 エラーメッセージを改善")

    print("\n" + "=" * 70)
    print("GPU/CPU適応型モデリング設定完了")
    print("=" * 70)
    print("\n💡 動作:")
    print("  - LightGBM GPU版が利用可能な場合 → GPU加速")
    print("  - GPU未対応の場合 → CPU版で安定動作")
    print("  - どちらの場合も最適な性能を発揮")

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
