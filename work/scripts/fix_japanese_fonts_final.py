#!/usr/bin/env python3
"""
日本語フォント設定を確実に適用する最終修正スクリプト
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


# 完全な日本語フォント設定セル
JAPANESE_FONT_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 日本語フォント設定（確実版）\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.font_manager as fm\n",
        "\n",
        "# 方法1: japanize_matplotlibを使用\n",
        "try:\n",
        "    import japanize_matplotlib\n",
        "    japanize_matplotlib.japanize()\n",
        "    print(\"✅ 日本語フォント: japanize_matplotlib使用\")\n",
        "except ImportError:\n",
        "    print(\"⚠️ japanize_matplotlibが見つかりません。代替フォントを設定します。\")\n",
        "\n",
        "# 方法2: システムフォントを検索して設定（確実性向上）\n",
        "japanese_fonts = [f.name for f in fm.fontManager.ttflist \n",
        "                  if any(keyword in f.name for keyword in ['Gothic', 'ゴシック', 'Noto Sans CJK', 'IPA', 'Meiryo', 'Yu Gothic'])]\n",
        "\n",
        "if japanese_fonts:\n",
        "    # 優先順位: IPAゴシック > Noto Sans CJK JP > その他\n",
        "    priority_fonts = ['IPAGothic', 'IPAexGothic', 'Noto Sans CJK JP', 'Yu Gothic', 'Meiryo']\n",
        "    selected_font = None\n",
        "    \n",
        "    for pf in priority_fonts:\n",
        "        if pf in japanese_fonts:\n",
        "            selected_font = pf\n",
        "            break\n",
        "    \n",
        "    if not selected_font:\n",
        "        selected_font = japanese_fonts[0]\n",
        "    \n",
        "    plt.rcParams['font.family'] = selected_font\n",
        "    print(f\"✅ 日本語フォント設定: {selected_font}\")\n",
        "else:\n",
        "    # フォールバック: よく使われるフォント名をリストで指定\n",
        "    plt.rcParams['font.family'] = ['IPAGothic', 'IPAexGothic', 'Noto Sans CJK JP', 'Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']\n",
        "    print(\"⚠️ 日本語フォントが見つかりません。フォールバックフォントを使用します。\")\n",
        "\n",
        "# マイナス記号の文字化け対策\n",
        "plt.rcParams['axes.unicode_minus'] = False\n",
        "\n",
        "# フォント設定確認用のテスト\n",
        "print(f\"現在のフォント設定: {plt.rcParams['font.family']}\")\n",
        "print(\"\\n利用可能な日本語フォント（上位10件）:\")\n",
        "for i, font in enumerate(japanese_fonts[:10], 1):\n",
        "    print(f\"  {i}. {font}\")\n"
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
    """ノートブックのヘッダーセルを完全な日本語フォント設定に置き換え"""
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
        print(f"   📊 総セル数: {len(cells)}")

        # 既存のヘッダーセルを探す（最初のコードセル）
        header_index = None
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                # importやフォント設定を含むセルを検索
                if 'import' in source and ('matplotlib' in source or 'japanize' in source):
                    header_index = i
                    break

        if header_index is None:
            # 最初のmarkdownセルの次に挿入
            for i, cell in enumerate(cells):
                if cell.get('cell_type') == 'markdown':
                    header_index = i + 1
                    break

            if header_index is None:
                header_index = 0

        # 日本語フォント設定セルを最優先で挿入
        if header_index == 0:
            cells.insert(0, JAPANESE_FONT_CELL)
            print(f"   ✅ 日本語フォント設定セルを先頭に挿入")
        else:
            # 既存のヘッダーセルより前に挿入
            cells.insert(header_index, JAPANESE_FONT_CELL)
            print(f"   ✅ 日本語フォント設定セルを位置{header_index}に挿入")

        # ノートブック保存
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

        print(f"   ✅ 修正完了: {notebook_path.name}")
        return True

    except Exception as e:
        print(f"   ❌ エラー: {e}")
        # エラー時はバックアップから復元
        shutil.copy2(backup_path, notebook_path)
        print(f"   🔄 バックアップから復元しました")
        return False


def main():
    """メイン処理"""
    print("\n" + "="*80)
    print("🔧 日本語フォント設定 最終修正スクリプト")
    print("="*80)
    print("\n📋 修正内容:")
    print("   • 確実な日本語フォント設定セルを最優先で挿入")
    print("   • システムフォントを自動検索して最適なフォントを選択")
    print("   • 複数のフォールバックオプションを設定")
    print()

    # 作業ディレクトリ
    work_dir = Path.cwd()
    print(f"📂 作業ディレクトリ: {work_dir}")

    # Phase 1-4のノートブックを検索
    notebooks = sorted(work_dir.glob('店舗別包括ダッシュボード_v5.0_Phase[1-4].ipynb'))

    if not notebooks:
        print("\n❌ Phase 1-4のノートブックが見つかりません")
        return

    print(f"\n🔍 対象ノートブック: {len(notebooks)}個")
    for nb in notebooks:
        print(f"   • {nb.name}")

    # 修正実行
    print("\n" + "="*80)
    print("🚀 修正開始")
    print("="*80)

    success_count = 0
    for notebook_path in notebooks:
        if fix_notebook(notebook_path):
            success_count += 1

    # 完了
    print("\n" + "="*80)
    print("✅ 修正完了")
    print("="*80)
    print(f"\n   成功: {success_count}/{len(notebooks)} ノートブック")
    print(f"\n📝 重要な注意事項:")
    print(f"   1. Jupyter Labでノートブックを開く")
    print(f"   2. 【必須】Kernel → Restart Kernel and Run All Cells を実行")
    print(f"   3. 最初のセルで「✅ 日本語フォント設定」が表示されることを確認")
    print(f"   4. グラフタイトルの店舗名が正しく表示されることを確認")
    print(f"\n💡 それでも文字化けする場合:")
    print(f"   pip install --upgrade japanize-matplotlib")


if __name__ == "__main__":
    main()
