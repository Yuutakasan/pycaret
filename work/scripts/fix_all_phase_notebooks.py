#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1-4ノートブック一括修正スクリプト

日本語フォント設定と店舗選択ウィジェットを全Phaseノートブックに追加します。
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# 修正対象ノートブック
NOTEBOOKS = [
    '店舗別包括ダッシュボード_v5.0_Phase1.ipynb',
    '店舗別包括ダッシュボード_v5.0_Phase2.ipynb',
    '店舗別包括ダッシュボード_v5.0_Phase3.ipynb',
    '店舗別包括ダッシュボード_v5.0_Phase4.ipynb'
]

# 新しいヘッダーセル（日本語フォント対応）
NEW_HEADER_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 警告抑制\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# 基本ライブラリ\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from datetime import datetime, timedelta\n",
        "from pathlib import Path\n",
        "\n",
        "# 可視化\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# インタラクティブウィジェット\n",
        "try:\n",
        "    import ipywidgets as widgets\n",
        "    from IPython.display import display, HTML, clear_output\n",
        "    WIDGETS_AVAILABLE = True\n",
        "    print(\"✅ ipywidgets利用可能\")\n",
        "except ImportError:\n",
        "    WIDGETS_AVAILABLE = False\n",
        "    print(\"⚠️ ipywidgets未インストール - 一部機能制限\")\n",
        "\n",
        "# 日本語フォント設定\n",
        "try:\n",
        "    import japanize_matplotlib\n",
        "    japanize_matplotlib.japanize()\n",
        "    print(\"✅ 日本語表示: japanize_matplotlib\")\n",
        "except ImportError:\n",
        "    # 代替フォント設定\n",
        "    import matplotlib.font_manager as fm\n",
        "    # 日本語フォントを検索\n",
        "    japanese_fonts = [f.name for f in fm.fontManager.ttflist if 'Gothic' in f.name or 'Noto Sans CJK' in f.name or 'IPA' in f.name]\n",
        "    if japanese_fonts:\n",
        "        plt.rcParams['font.family'] = japanese_fonts[0]\n",
        "        print(f\"✅ 日本語表示: {japanese_fonts[0]}\")\n",
        "    else:\n",
        "        plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'sans-serif']\n",
        "        print(\"⚠️ 代替フォント設定（フォント検出失敗）\")\n",
        "    plt.rcParams['axes.unicode_minus'] = False\n",
        "\n",
        "# matplotlib共通設定\n",
        "plt.rcParams['axes.unicode_minus'] = False\n",
        "plt.rcParams['figure.figsize'] = (18, 12)\n",
        "plt.rcParams['figure.dpi'] = 100\n",
        "plt.rcParams['savefig.dpi'] = 150\n",
        "plt.rcParams['font.size'] = 11\n",
        "\n",
        "# seaborn設定\n",
        "sns.set_style('whitegrid')\n",
        "sns.set_palette('husl')\n",
        "\n",
        "# pandas設定\n",
        "pd.set_option('display.unicode.east_asian_width', True)\n",
        "pd.set_option('display.max_columns', 50)\n",
        "pd.set_option('display.max_rows', 200)\n",
        "pd.set_option('display.width', 120)\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"🏪 店舗別包括ダッシュボード v5.0\".center(80))\n",
        "print(\"=\"*80)\n",
        "print(f\"\\n✅ 環境設定完了\")\n",
        "print(f\"   実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\")\n",
        "print(f\"   pandas: {pd.__version__}\")\n",
        "print(f\"   matplotlib: {plt.matplotlib.__version__}\")"
    ]
}

# 店舗選択ウィジェットセル
STORE_SELECTION_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 🎯 店舗選択ウィジェット\n",
        "\n",
        "# 店舗一覧\n",
        "stores = sorted(df_enriched['店舗'].unique())\n",
        "DEFAULT_STORE = stores[0]\n",
        "\n",
        "print(f\"\\n🏪 利用可能な店舗 ({len(stores)}店舗):\")\n",
        "for i, store in enumerate(stores, 1):\n",
        "    print(f\"   {i}. {store}\")\n",
        "\n",
        "# 店舗選択ウィジェット\n",
        "if WIDGETS_AVAILABLE:\n",
        "    print(\"\\n\" + \"=\"*80)\n",
        "    print(\"🎯 以下のドロップダウンから分析対象店舗を選択してください\")\n",
        "    print(\"=\"*80)\n",
        "    \n",
        "    store_dropdown = widgets.Dropdown(\n",
        "        options=stores,\n",
        "        value=DEFAULT_STORE,\n",
        "        description='分析対象店舗:',\n",
        "        disabled=False,\n",
        "        style={'description_width': '120px'},\n",
        "        layout=widgets.Layout(width='500px')\n",
        "    )\n",
        "    \n",
        "    info_label = widgets.HTML(\n",
        "        value=\"<b>💡 ヒント:</b> 店舗を変更すると、以降のすべての分析が選択した店舗で再計算されます。\"\n",
        "    )\n",
        "    \n",
        "    display(widgets.VBox([store_dropdown, info_label]))\n",
        "    \n",
        "    # 選択された店舗\n",
        "    MY_STORE = store_dropdown.value\n",
        "else:\n",
        "    # ウィジェットが使えない場合\n",
        "    MY_STORE = DEFAULT_STORE\n",
        "    print(f\"\\n🎯 分析対象店舗: {MY_STORE} (デフォルト)\")\n",
        "\n",
        "# 店舗データフィルタリング\n",
        "my_df = df_enriched[df_enriched['店舗'] == MY_STORE].copy()\n",
        "\n",
        "print(f\"\\n✅ 選択された店舗: {MY_STORE}\")\n",
        "print(f\"   対象データ: {len(my_df):,}行\")"
    ]
}

# データ検証セル
DATA_VALIDATION_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 🔍 データ検証（存在チェック）\n",
        "\n",
        "def validate_data_column(df, col_name, analysis_name=\"分析\"):\n",
        "    \"\"\"データカラムの存在と有効性をチェック\"\"\"\n",
        "    if col_name not in df.columns:\n",
        "        print(f\"⚠️ {analysis_name}: '{col_name}' カラムが存在しません\")\n",
        "        return False\n",
        "    \n",
        "    non_null_count = df[col_name].notna().sum()\n",
        "    coverage = non_null_count / len(df) * 100\n",
        "    \n",
        "    if coverage < 50:\n",
        "        print(f\"⚠️ {analysis_name}: '{col_name}' のカバレッジが低い ({coverage:.1f}%)\")\n",
        "        return False\n",
        "    \n",
        "    return True\n",
        "\n",
        "print(\"\\n🔍 データ検証中...\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "# 必須カラムチェック\n",
        "required_cols = ['日付', '売上金額', '店舗']\n",
        "for col in required_cols:\n",
        "    if col in df_enriched.columns:\n",
        "        print(f\"✅ 必須カラム '{col}' - 存在\")\n",
        "    else:\n",
        "        print(f\"❌ 必須カラム '{col}' - 不足\")\n",
        "\n",
        "# オプションカラムチェック\n",
        "optional_cols = {\n",
        "    '気象データ': ['最高気温', '降水量'],\n",
        "    '前年データ': ['昨年同日_売上', '昨年同日_客数'],\n",
        "    '時間帯データ': ['時刻', '時間']\n",
        "}\n",
        "\n",
        "for category, cols in optional_cols.items():\n",
        "    has_any = any(col in df_enriched.columns for col in cols)\n",
        "    if has_any:\n",
        "        available_cols = [col for col in cols if col in df_enriched.columns]\n",
        "        print(f\"✅ {category}: {', '.join(available_cols)}\")\n",
        "    else:\n",
        "        print(f\"⚠️ {category}: 利用不可（代替ロジック使用）\")\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"✅ データ検証完了\\n\")"
    ]
}


def backup_notebook(notebook_path):
    """ノートブックをバックアップ"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = notebook_path.with_suffix(f'.backup_{timestamp}.ipynb')
    shutil.copy2(notebook_path, backup_path)
    print(f"   📋 バックアップ: {backup_path.name}")
    return backup_path


def fix_notebook(notebook_path):
    """ノートブックを修正"""
    print(f"\n{'='*80}")
    print(f"🔧 修正中: {notebook_path.name}")
    print(f"{'='*80}")

    # バックアップ
    backup_path = backup_notebook(notebook_path)

    try:
        # ノートブック読み込み
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        cells = nb['cells']
        print(f"   📊 セル数: {len(cells)}")

        # 1番目のセル（タイトル）は維持
        # 2番目のセル（importセル）を置き換え
        if len(cells) > 1 and cells[1].get('cell_type') == 'code':
            print(f"   🔄 ヘッダーセルを置換（セル1）")
            cells[1] = NEW_HEADER_CELL

        # 3番目のセル（データ読み込み）の後に店舗選択セルを挿入
        # データ読み込みセルを探す
        data_load_index = None
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if 'df_enriched' in source and 'read_csv' in source:
                    data_load_index = i
                    break

        if data_load_index is not None:
            # 既存の店舗選択セルがあるかチェック
            has_store_selection = False
            if data_load_index + 1 < len(cells):
                next_cell_source = ''.join(cells[data_load_index + 1].get('source', []))
                if '店舗選択' in next_cell_source or 'store_dropdown' in next_cell_source:
                    has_store_selection = True

            if not has_store_selection:
                print(f"   ➕ 店舗選択セルを挿入（セル{data_load_index + 1}の後）")
                cells.insert(data_load_index + 1, STORE_SELECTION_CELL.copy())
                cells.insert(data_load_index + 2, DATA_VALIDATION_CELL.copy())
            else:
                print(f"   ✅ 店舗選択セルは既に存在（スキップ）")

        # すべてのコードセルで MY_STORE = DEFAULT_STORE を確認
        my_store_count = 0
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source_str = ''.join(source)
                    if 'MY_STORE = DEFAULT_STORE' in source_str:
                        # この行を削除（店舗選択ウィジェットに任せる）
                        new_source = [line for line in source if 'MY_STORE = DEFAULT_STORE' not in line]
                        if len(new_source) < len(source):
                            cell['source'] = new_source
                            my_store_count += 1

        if my_store_count > 0:
            print(f"   🗑️  MY_STORE=DEFAULT_STORE を{my_store_count}箇所削除")

        # 修正したノートブックを保存
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

        print(f"   ✅ 修正完了: {notebook_path.name}")
        print(f"   📊 最終セル数: {len(cells)}")

        return True

    except Exception as e:
        print(f"   ❌ エラー: {str(e)}")
        # エラー時はバックアップから復元
        shutil.copy2(backup_path, notebook_path)
        print(f"   ↩️  バックアップから復元")
        return False


def main():
    """メイン処理"""
    print("\n" + "="*80)
    print("🔧 Phase 1-4 ノートブック一括修正スクリプト")
    print("="*80)
    timestamp = datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')
    print(f"実行日時: {timestamp}")

    work_dir = Path('/mnt/d/github/pycaret/work')

    print(f"\n📂 作業ディレクトリ: {work_dir}")

    success_count = 0
    fail_count = 0

    for notebook_name in NOTEBOOKS:
        notebook_path = work_dir / notebook_name

        if not notebook_path.exists():
            print(f"\n⚠️ スキップ: {notebook_name} が見つかりません")
            fail_count += 1
            continue

        if fix_notebook(notebook_path):
            success_count += 1
        else:
            fail_count += 1

    # サマリー
    print("\n" + "="*80)
    print("📋 修正サマリー")
    print("="*80)
    print(f"✅ 成功: {success_count}個")
    print(f"❌ 失敗: {fail_count}個")

    if success_count > 0:
        print("\n💡 次のステップ:")
        print("   1. Jupyter Notebookで各ノートブックを開く")
        print("   2. 'Restart & Run All' で全セル実行")
        print("   3. 日本語表示と店舗選択が正常動作するか確認")
        print("   4. エラーがあれば .backup_*.ipynb から復元")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
