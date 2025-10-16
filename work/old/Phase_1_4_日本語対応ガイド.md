# 📋 Phase 1-4 日本語表示対応ガイド

## 🎯 目的

Phase 1-4の店舗別包括ダッシュボードで、グラフの日本語が正しく表示されるようにします。

---

## ⚡ クイックスタート（3ステップ）

### 1️⃣ 必要なライブラリをインストール

```bash
# 日本語フォント対応
pip install japanize-matplotlib

# インタラクティブウィジェット（店舗選択用）
pip install ipywidgets

# Jupyter Notebookで有効化
jupyter nbextension enable --py widgetsnbextension
```

### 2️⃣ 各Phase ノートブックの最初のコードセルを以下に置き換え

Phase 1, 2, 3, 4 すべてのノートブックで、**最初のimportセル**を以下のコードに置き換えてください：

```python
# 警告抑制
import warnings
warnings.filterwarnings('ignore')

# 基本ライブラリ
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns

# インタラクティブウィジェット
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    WIDGETS_AVAILABLE = True
    print("✅ ipywidgets利用可能")
except ImportError:
    WIDGETS_AVAILABLE = False
    print("⚠️ ipywidgets未インストール")

# 日本語フォント
try:
    import japanize_matplotlib
    print("✅ 日本語表示: japanize_matplotlib")
except ImportError:
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠️ 代替フォント設定")

# 共通設定
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

# seaborn設定
sns.set_style('whitegrid')
sns.set_palette('husl')

# pandas設定
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', 50)

print("\n" + "="*80)
print("🏪 店舗別包括ダッシュボード v5.0".center(80))
print("="*80)
print(f"\n✅ 環境設定完了")
print(f"   実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
print(f"   pandas: {pd.__version__}")
print(f"   matplotlib: {plt.matplotlib.__version__}")
```

### 3️⃣ 店舗選択ウィジェット機能を追加

データ読み込みセルの後に、以下のセルを**新規追加**してください：

```python
# 🎯 店舗選択ウィジェット

# 店舗一覧
stores = sorted(df_enriched['店舗'].unique())
DEFAULT_STORE = stores[0]

print(f"\n🏪 利用可能な店舗 ({len(stores)}店舗):")
for i, store in enumerate(stores, 1):
    print(f"   {i}. {store}")

# 店舗選択ウィジェット
if WIDGETS_AVAILABLE:
    print("\n" + "="*80)
    print("🎯 以下のドロップダウンから分析対象店舗を選択してください")
    print("="*80)

    store_dropdown = widgets.Dropdown(
        options=stores,
        value=DEFAULT_STORE,
        description='分析対象店舗:',
        disabled=False,
        style={'description_width': '120px'},
        layout=widgets.Layout(width='500px')
    )

    info_label = widgets.HTML(
        value=f"<b>💡 ヒント:</b> 店舗を変更すると、以降のすべての分析が選択した店舗で再計算されます。"
    )

    display(widgets.VBox([store_dropdown, info_label]))

    # 選択された店舗
    MY_STORE = store_dropdown.value
else:
    # ウィジェットが使えない場合
    MY_STORE = DEFAULT_STORE
    print(f"\n🎯 分析対象店舗: {MY_STORE} (デフォルト)")

# 店舗データフィルタリング
my_df = df_enriched[df_enriched['店舗'] == MY_STORE].copy()

print(f"\n✅ 選択された店舗: {MY_STORE}")
print(f"   対象データ: {len(my_df):,}行")
```

---

## 🔧 詳細設定（オプション）

### グラフタイトル・ラベルを日本語化

グラフ作成時に、以下のように日本語を使用できます：

```python
# 例: 売上トレンドグラフ
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily['日付'], daily['売上金額'], marker='o', label='今年')
ax.plot(daily['日付'], daily['昨年同日_売上'], marker='s', linestyle='--', label='昨年')
ax.set_title('売上トレンド', fontsize=16, fontweight='bold')
ax.set_xlabel('日付', fontsize=12)
ax.set_ylabel('売上金額 (円)', fontsize=12)
ax.legend(loc='best', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### 凡例・注釈も日本語で

```python
ax.axhline(y=target_sales, color='red', linestyle='--', label='目標売上')
ax.annotate('ピーク', xy=(peak_date, peak_sales), xytext=(peak_date, peak_sales*1.1),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red', fontweight='bold')
```

---

## 📊 実務での使い方

### 毎日の運用フロー

1. **朝（開店前）**
   - Jupyter Notebookを開く
   - Phase 1のエグゼクティブサマリーで昨日の実績確認
   - アラートがあれば即座に対応

2. **午前中**
   - Phase 1の需要予測で明日の発注計画
   - Phase 2の異常検知で在庫状況チェック

3. **午後**
   - Phase 1の気象連動で明日の天気に応じた発注調整
   - Phase 3のトレンド検知で成長商品・衰退商品を確認

4. **夕方**
   - Phase 1の客数・客単価分解で今日の傾向分析
   - Phase 4のプロモーション効果測定でキャンペーン評価

5. **週1回**
   - Phase 1の店舗間比較でベンチマーク確認
   - Phase 3のベストプラクティス抽出でトップ店の施策を学ぶ
   - Phase 4のWhat-ifシミュレーションで次週の戦略立案

### 店舗選択の活用

**マルチ店舗管理の場合:**
1. 店舗選択ドロップダウンで店舗Aを選択
2. Phase 1-4の分析を実行
3. 結果をスクリーンショット or PDFで保存
4. 別の店舗を選択して繰り返し
5. 全店舗の結果を比較・検討

**エリアマネージャーの場合:**
- 各店舗の強み・弱みを客観的に把握
- トップ店舗の成功施策を他店に横展開
- 問題店舗には集中的にサポート

---

## 🐛 トラブルシューティング

### Q1. 日本語が文字化けする

**A1.** 以下の順で試してください：

```bash
# 方法1: japanize-matplotlibをインストール
pip install japanize-matplotlib

# 方法2: システムフォントを確認
fc-list | grep -i gothic  # Linux/Mac
# Windowsの場合: C:\Windows\Fonts を確認

# 方法3: 環境変数を設定
export LANG=ja_JP.UTF-8
```

### Q2. ウィジェットが表示されない

**A2.** Jupyter Notebookの拡張機能を有効化：

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
jupyter notebook  # 再起動
```

### Q3. グラフが小さい・見にくい

**A3.** 図のサイズを調整：

```python
plt.rcParams['figure.figsize'] = (20, 14)  # 幅20インチ、高さ14インチ
plt.rcParams['font.size'] = 12  # フォントサイズ拡大
```

### Q4. データが古い

**A4.** データを最新化：

```bash
cd work
python batch_convert.py --input-dir input --output-dir output --debug
python enrich_features_v2.py --input output/06_* --output output/06_final_enriched_*.csv
```

---

## 📚 参考資料

### Phase別の機能一覧

| Phase | 主な機能 | 使うタイミング |
|-------|---------|--------------|
| Phase 1 | エグゼクティブサマリー、需要予測、気象連動発注 | 毎日（朝・午前） |
| Phase 2 | 異常検知、在庫最適化、前年比較、イベント予測 | 毎日（午前・午後） |
| Phase 3 | AI特徴量選択、トレンド検知、欠品検知、ベストプラクティス、バスケット分析 | 週1回（深掘り分析） |
| Phase 4 | 時間帯分析、プロモーション測定、What-ifシミュレーション、クロスセル自動化 | 週1回（戦略立案） |

### 推奨ライブラリバージョン

```txt
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
scipy>=1.10.0
japanize-matplotlib>=1.1.3
ipywidgets>=8.0.0
pycaret>=3.0.0  # Phase 1の需要予測で使用
mlxtend>=0.22.0  # Phase 3のバスケット分析で使用
statsmodels>=0.14.0  # Phase 4の統計分析で使用
```

---

## 💡 実務Tips

### 見やすいグラフを作るコツ

1. **色使い**: 同系色でグラデーション
2. **凡例**: 必ず日本語で説明
3. **グリッド**: `alpha=0.3`で薄く
4. **タイトル**: 太字(`fontweight='bold'`)で目立たせる
5. **軸ラベル**: 単位（円、個、%）を明記

### データ更新の自動化

cronやタスクスケジューラーで毎日自動実行：

```bash
# 毎日早朝5時にデータ更新
0 5 * * * cd /path/to/pycaret/work && python batch_convert.py && python enrich_features_v2.py
```

### 結果の共有

```python
# グラフをPNG保存
plt.savefig('output/executive_summary.png', dpi=150, bbox_inches='tight')

# PDFレポート生成
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('output/weekly_report.pdf') as pdf:
    # 各グラフを作成してpdf.savefig()
    pass
```

---

## 🎉 まとめ

この設定により、Phase 1-4のすべてのダッシュボードで：

✅ **日本語が正しく表示**される
✅ **店舗を自由に選択**できる
✅ **実務で即使える**分析結果が得られる

**あなたの店舗運営を、データで、科学で、最強にします！** 🚀
