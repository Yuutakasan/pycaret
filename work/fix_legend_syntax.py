#!/usr/bin/env python3
"""
legend()のSyntaxErrorを修正
- legend(, prop=JP_FP) → legend(prop=JP_FP)
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def fix_legend_syntax(notebook_path):
    """legend(, prop=JP_FP)を修正"""
    print(f"\n修正中: {notebook_path.name}")

    # バックアップ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = notebook_path.with_suffix(f'.backup_legend_{timestamp}.ipynb')
    shutil.copy2(notebook_path, backup_path)
    print(f"  バックアップ: {backup_path.name}")

    try:
        # ノートブック読み込み
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        fixed_count = 0
        for cell in nb['cells']:
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                new_source = []

                for line in source:
                    # legend(, prop=JP_FP) を legend(prop=JP_FP) に修正
                    if 'legend(, prop=JP_FP)' in line:
                        line = line.replace('legend(, prop=JP_FP)', 'legend(prop=JP_FP)')
                        fixed_count += 1
                        print(f"  ✅ 修正: legend(, prop=JP_FP) → legend(prop=JP_FP)")
                    new_source.append(line)

                if new_source != source:
                    cell['source'] = new_source

        # 保存
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

        print(f"  ✅ 完了: {fixed_count}箇所を修正")
        return True

    except Exception as e:
        print(f"  ❌ エラー: {e}")
        shutil.copy2(backup_path, notebook_path)
        return False


def main():
    print("="*80)
    print("🔧 legend() SyntaxError修正スクリプト")
    print("="*80)

    work_dir = Path.cwd()
    notebooks = sorted(work_dir.glob('店舗別包括ダッシュボード_v5.0_Phase[1-4].ipynb'))

    if not notebooks:
        print("❌ 対象ファイルが見つかりません")
        return

    print(f"\n対象: {len(notebooks)}個のノートブック")

    success_count = sum(1 for nb in notebooks if fix_legend_syntax(nb))

    print("\n" + "="*80)
    print(f"✅ 完了: {success_count}/{len(notebooks)} 成功")
    print("="*80)


if __name__ == "__main__":
    main()
