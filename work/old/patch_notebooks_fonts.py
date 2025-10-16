#!/usr/bin/env python3
"""
ノートブックに日本語フォント設定を注入
- 先頭セルでJP_FP = font_setup.setup_fonts()
- 全グラフAPIにfontproperties=JP_FP / prop=JP_FPを追加
"""

import json
import shutil
import re
from pathlib import Path
from datetime import datetime


# 先頭セル: フォント設定
FONT_SETUP_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 日本語フォント設定\n",
        "import font_setup\n",
        "JP_FP = font_setup.setup_fonts()\n"
    ]
}

# 基本ライブラリ先読みセル
PRELOAD_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 基本ライブラリ先読み\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from datetime import datetime, timedelta\n",
        "from pathlib import Path\n",
        "\n",
        "# 可視化\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Plotly（利用可能な場合）\n",
        "try:\n",
        "    import plotly.graph_objects as go\n",
        "    import plotly.express as px\n",
        "    import plotly.io as pio\n",
        "    PLOTLY_AVAILABLE = True\n",
        "except ImportError:\n",
        "    PLOTLY_AVAILABLE = False\n",
        "\n",
        "# ipywidgets\n",
        "try:\n",
        "    import ipywidgets as widgets\n",
        "    from IPython.display import display, HTML, clear_output\n",
        "    WIDGETS_AVAILABLE = True\n",
        "except ImportError:\n",
        "    WIDGETS_AVAILABLE = False\n",
        "\n",
        "# matplotlib共通設定\n",
        "plt.rcParams['figure.figsize'] = (18, 12)\n",
        "plt.rcParams['figure.dpi'] = 100\n",
        "plt.rcParams['savefig.dpi'] = 150\n",
        "plt.rcParams['font.size'] = 11\n",
        "\n",
        "# seaborn\n",
        "sns.set_style('whitegrid')\n",
        "sns.set_palette('husl')\n",
        "\n",
        "# pandas\n",
        "pd.set_option('display.unicode.east_asian_width', True)\n",
        "pd.set_option('display.max_columns', 50)\n",
        "pd.set_option('display.max_rows', 200)\n",
        "pd.set_option('display.width', 120)\n",
        "\n",
        "print('\\n' + '='*80)\n",
        "print('🏪 店舗別包括ダッシュボード v5.0'.center(80))\n",
        "print('='*80)\n",
        "print(f'\\n✅ 環境設定完了')\n",
        "print(f'   実行日時: {datetime.now().strftime(\"%Y年%m月%d日 %H:%M:%S\")}')\n",
        "print(f'   pandas: {pd.__version__}')\n",
        "print(f'   matplotlib: {plt.matplotlib.__version__}')\n",
        "print(f'   Plotly: {\"利用可能\" if PLOTLY_AVAILABLE else \"未インストール\"}')\n",
        "print(f'   ipywidgets: {\"利用可能\" if WIDGETS_AVAILABLE else \"未インストール\"}')\n"
    ]
}


def inject_fontproperties(source_lines):
    """
    グラフAPIにfontproperties=JP_FP / prop=JP_FPを注入

    対象:
    - plt.suptitle, plt.title, ax.set_title
    - plt.xlabel, plt.ylabel, ax.set_xlabel, ax.set_ylabel
    - plt.text, ax.text, plt.annotate, ax.annotate
    - legend(...) → prop=JP_FP
    """
    modified = False
    new_lines = []

    for line in source_lines:
        original_line = line

        # 1. タイトル系: fontweight='bold'を削除し、fontproperties=JP_FPを追加
        # plt.suptitle(...), plt.title(...), ax.set_title(...)
        if re.search(r'(plt\.suptitle|plt\.title|ax\.set_title|fig\.suptitle)\s*\(', line):
            # fontweight='bold'を削除
            line = re.sub(r",\s*fontweight=['\"]bold['\"]", "", line)

            # fontproperties=JP_FPがまだなければ追加
            if 'fontproperties=' not in line:
                # 閉じ括弧の前に追加
                line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', line)
                modified = True

        # 2. 軸ラベル: plt.xlabel, plt.ylabel, ax.set_xlabel, ax.set_ylabel
        elif re.search(r'(plt\.xlabel|plt\.ylabel|ax\.set_xlabel|ax\.set_ylabel)\s*\(', line):
            if 'fontproperties=' not in line:
                line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', line)
                modified = True

        # 3. テキスト・注釈: plt.text, ax.text, plt.annotate, ax.annotate
        elif re.search(r'(plt\.text|ax\.text|plt\.annotate|ax\.annotate)\s*\(', line):
            if 'fontproperties=' not in line:
                line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', line)
                modified = True

        # 4. 凡例: legend(...) → prop=JP_FP
        elif re.search(r'\.legend\s*\(', line):
            if 'prop=' not in line:
                # 空の引数リスト: legend() → legend(prop=JP_FP)
                if re.search(r'\.legend\s*\(\s*\)', line):
                    line = re.sub(r'\.legend\s*\(\s*\)', r'.legend(prop=JP_FP)', line)
                    modified = True
                # 引数あり: legend(...) → legend(..., prop=JP_FP)
                else:
                    line = re.sub(r'\)(\s*)$', r', prop=JP_FP)\1', line)
                    modified = True

        new_lines.append(line)

    return new_lines, modified


def inject_my_store_fallback(source_lines):
    """
    DEFAULT_STORE 定義後に MY_STORE を未定義時にデフォルトへ設定するフォールバックを注入し、
    ウィジェット非対応ブランチでも印字前に MY_STORE を確実に定義する。
    """
    modified = False
    new_lines = []
    i = 0
    while i < len(source_lines):
        line = source_lines[i]
        new_lines.append(line)

        # 1) DEFAULT_STORE の直後にフォールバックを差し込む
        if 'DEFAULT_STORE' in line and '=' in line:
            fallback = [
                "try:\n",
                "    MY_STORE\n",
                "except NameError:\n",
                "    MY_STORE = DEFAULT_STORE\n",
            ]
            new_lines.extend(fallback)
            modified = True

        # 2) ウィジェット else ブランチの直後で MY_STORE を設定
        if line.strip().startswith('else:'):
            # 次行以降に MY_STORE を含む print があれば、その前に代入を挿入
            j = i + 1
            inserted = False
            while j < len(source_lines) and source_lines[j].strip().startswith(('#', 'print', 'display', 'info_label', 'store_dropdown', 'layout', 'style')) is False:
                # 進める（空行やコメント等を飛ばす準備）
                break
            # 常に挿入（冪等ではないが何度も実行されない想定）
            new_lines.append('    MY_STORE = DEFAULT_STORE\n')
            modified = True

        i += 1

    return new_lines, modified


def patch_notebook(notebook_path):
    """ノートブックにフォント設定を適用"""
    print(f"\n{'='*80}")
    print(f"🔧 適用中: {notebook_path.name}")
    print(f"{'='*80}")

    # バックアップ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = notebook_path.with_suffix(f'.backup_{timestamp}.ipynb')
    shutil.copy2(notebook_path, backup_path)
    print(f"   📋 バックアップ: {backup_path.name}")

    try:
        # ノートブック読み込み
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        cells = nb['cells']
        original_count = len(cells)
        print(f"   📊 元のセル数: {original_count}")

        # 1. 既存のフォント設定セルを削除
        cells_to_remove = []
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                if any(keyword in source for keyword in [
                    'import font_setup',
                    'JP_FP = font_setup',
                    'japanize_matplotlib',
                    '日本語フォント設定'
                ]) and len(source) < 500:
                    cells_to_remove.append(i)

        for i in sorted(cells_to_remove, reverse=True):
            cells.pop(i)
            print(f"   🗑️ 既存フォント設定セル削除: セル{i}")

        # 2. 先頭に新しいセルを挿入
        insert_pos = next((i+1 for i,c in enumerate(cells) if c.get('cell_type')=='markdown'), 0)
        cells.insert(insert_pos, FONT_SETUP_CELL)
        cells.insert(insert_pos+1, PRELOAD_CELL)
        print(f"   ✅ フォント設定セル挿入: 位置{insert_pos}")
        print(f"   ✅ ライブラリ先読みセル挿入: 位置{insert_pos+1}")

        # 3. 全コードセルにfontproperties注入
        injection_count = 0
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source_lines = cell.get('source', [])
                # 先に MY_STORE フォールバックを注入
                source_lines2, mod_store = inject_my_store_fallback(source_lines)
                # フォント指定注入
                new_lines, mod_font = inject_fontproperties(source_lines2)

                if mod_store or mod_font:
                    cell['source'] = new_lines
                    injection_count += 1

        print(f"   ✅ fontproperties注入: {injection_count}セル")

        # 4. ノートブック保存
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

        new_count = len(cells)
        print(f"   📊 新しいセル数: {new_count} (+{new_count - original_count})")
        print(f"   ✅ 適用完了")
        return True

    except Exception as e:
        print(f"   ❌ エラー: {e}")
        shutil.copy2(backup_path, notebook_path)
        print(f"   🔄 バックアップから復元")
        return False


def main():
    """メイン処理"""
    print("\n" + "="*80)
    print("🎨 ノートブック日本語フォント適用スクリプト v5.0")
    print("="*80)
    print("\n📋 適用内容:")
    print("   1. JP_FP = font_setup.setup_fonts() を先頭に挿入")
    print("   2. 全グラフAPIにfontproperties=JP_FP / prop=JP_FPを注入")
    print("   3. fontweight='bold'を削除（日本語フォントと相性悪い）")
    print()

    work_dir = Path.cwd()
    print(f"📂 作業ディレクトリ: {work_dir}")

    # font_setup.py確認
    if not (work_dir / 'font_setup.py').exists():
        print(f"\n❌ font_setup.py が見つかりません")
        return

    # Phase 1-4検索
    notebooks = sorted(work_dir.glob('店舗別包括ダッシュボード_v5.0_Phase[1-4].ipynb'))
    if not notebooks:
        print("\n❌ Phase 1-4のノートブックが見つかりません")
        return

    print(f"\n🔍 対象: {len(notebooks)}個")
    for nb in notebooks:
        print(f"   • {nb.name}")

    # 適用実行
    print("\n" + "="*80)
    print("🚀 適用開始")
    print("="*80)

    success_count = sum(1 for nb in notebooks if patch_notebook(nb))

    # 完了
    print("\n" + "="*80)
    print("✅ 適用完了")
    print("="*80)
    print(f"\n   成功: {success_count}/{len(notebooks)}")
    print(f"\n📝 次のステップ:")
    print(f"   1. Jupyter Labでノートブックを開く")
    print(f"   2. Kernel → Restart Kernel and Run All Cells")
    print(f"   3. 最初の2セルで以下を確認:")
    print(f"      ✅ 日本語フォント: IPAGothic (または他のフォント)")
    print(f"      ✅ 環境設定完了")
    print(f"   4. グラフタイトル（店舗名部分）が正しく表示されるか確認")


if __name__ == "__main__":
    main()
