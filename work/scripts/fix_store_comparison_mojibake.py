"""
店舗間比較グラフの文字化け修正

問題: pandas.plot(x='店舗') で作成されるy軸のticklabels（店舗名）が日本語フォントを使用していない
修正: 各グラフ作成後に明示的にticklabelsにfontpropertiesを適用
"""

import json
import re
from pathlib import Path

def fix_store_comparison_mojibake(notebook_path):
    """店舗間比較のグラフにフォント設定を追加"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Cell 19を探す（店舗間比較のセル）
    target_cell_idx = None
    for idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if '店舗間パフォーマンス比較' in source and 'store_summary.plot' in source:
                target_cell_idx = idx
                break

    if target_cell_idx is None:
        print("⚠️ 店舗間比較のセルが見つかりません")
        return 0

    cell = nb['cells'][target_cell_idx]
    source_lines = cell['source']

    # 修正が必要な箇所を検出して修正
    new_source = []
    modified = False

    i = 0
    while i < len(source_lines):
        line = source_lines[i]

        # パターン1: ax1.grid(axis='x', alpha=0.3) の直後にticklabels設定を追加
        if 'ax1.grid(axis=' in line and 'alpha=0.3' in line:
            new_source.append(line)
            # 次の行をチェック（既に修正済みかどうか）
            if i + 1 < len(source_lines) and 'ax1.set_yticklabels' not in source_lines[i + 1]:
                # ticklabels設定を追加
                new_source.append('for label in ax1.get_yticklabels():\n')
                new_source.append('    label.set_fontproperties(JP_FP)\n')
                modified = True
            i += 1
            continue

        # パターン2: ax2.grid(axis='x', alpha=0.3) の直後にticklabels設定を追加
        elif 'ax2.grid(axis=' in line and 'alpha=0.3' in line:
            new_source.append(line)
            if i + 1 < len(source_lines) and 'ax2.set_yticklabels' not in source_lines[i + 1]:
                new_source.append('for label in ax2.get_yticklabels():\n')
                new_source.append('    label.set_fontproperties(JP_FP)\n')
                modified = True
            i += 1
            continue

        # パターン3: ax3.grid(axis='x', alpha=0.3) の直後にticklabels設定を追加
        elif 'ax3.grid(axis=' in line and 'alpha=0.3' in line:
            new_source.append(line)
            if i + 1 < len(source_lines) and 'ax3.set_yticklabels' not in source_lines[i + 1]:
                new_source.append('for label in ax3.get_yticklabels():\n')
                new_source.append('    label.set_fontproperties(JP_FP)\n')
                modified = True
            i += 1
            continue

        else:
            new_source.append(line)
            i += 1

    if modified:
        cell['source'] = new_source

        # 保存
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

        return 1

    return 0


if __name__ == '__main__':
    notebook_path = '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase1.ipynb'

    print("\n" + "="*80)
    print("🔧 店舗間比較グラフの文字化け修正".center(80))
    print("="*80)

    path = Path(notebook_path)
    if not path.exists():
        print(f"❌ ファイルが見つかりません: {notebook_path}")
        exit(1)

    print(f"\n📝 {path.name}")
    count = fix_store_comparison_mojibake(notebook_path)

    if count > 0:
        print(f"✅ 店舗間比較グラフに日本語フォント設定を追加しました")
        print(f"\n修正内容:")
        print(f"  ・ax1（平均日商比較）のy軸ticklabelsにJP_FP適用")
        print(f"  ・ax2（平均客単価比較）のy軸ticklabelsにJP_FP適用")
        print(f"  ・ax3（ギャップ可視化）のy軸ticklabelsにJP_FP適用")
    else:
        print(f"ℹ️ 既に修正済み、または修正不要です")

    print("\n" + "="*80)
