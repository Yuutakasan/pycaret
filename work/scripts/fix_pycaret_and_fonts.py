#!/usr/bin/env python3
"""
Phase 1-4 ノートブックのPyCaret設定とフォント問題を修正
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def fix_pycaret_setup_cell(cells):
    """PyCaretセットアップセルを修正"""
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue

        source = ''.join(cell.get('source', []))

        # PyCaretセットアップセルを検出
        if 'from pycaret.regression import' in source and 'setup(' in source:
            print(f"   🔧 PyCaretセットアップセルを修正（セル{i}）")

            # fold_strategy='timeseries'があるか確認
            if "fold_strategy='timeseries'" in source:
                # data_split_shuffle と fold_shuffle を追加
                if 'data_split_shuffle' not in source:
                    # fold=3の次の行に挿入
                    source = source.replace(
                        "fold=3,",
                        "fold=3,\n            data_split_shuffle=False,\n            fold_shuffle=False,"
                    )

                    # ソースを更新
                    cell['source'] = source.split('\n')
                    if not cell['source'][-1].endswith('\n'):
                        cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]

                    print(f"      ✅ data_split_shuffle=False, fold_shuffle=False を追加")
                    return True

    return False


def remove_jp_fp_references(cells):
    """JP_FPへの不要な参照を削除"""
    modified = False

    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue

        source = ''.join(cell.get('source', []))

        # fontproperties=JP_FP を削除
        if 'fontproperties=JP_FP' in source:
            print(f"   🔧 fontproperties=JP_FP参照を削除（セル{i}）")

            # 置換
            source = source.replace(', fontproperties=JP_FP', '')
            source = source.replace('fontproperties=JP_FP, ', '')
            source = source.replace('fontproperties=JP_FP', '')
            source = source.replace(', prop=JP_FP', '')
            source = source.replace('prop=JP_FP, ', '')
            source = source.replace('prop=JP_FP', '')

            # ソースを更新
            cell['source'] = source.split('\n')
            if not cell['source'][-1].endswith('\n'):
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]

            modified = True

    return modified


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

        # 修正実行
        pycaret_fixed = fix_pycaret_setup_cell(cells)
        fonts_fixed = remove_jp_fp_references(cells)

        if pycaret_fixed or fonts_fixed:
            # ノートブック保存
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)

            print(f"   ✅ 修正完了: {notebook_path.name}")
            if pycaret_fixed:
                print(f"      • PyCaret timeseries設定を修正")
            if fonts_fixed:
                print(f"      • 未定義フォント参照を削除")
        else:
            print(f"   ℹ️  修正不要: {notebook_path.name}")

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
    print("🔧 Phase 1-4 ノートブック修正スクリプト")
    print("="*80)
    print("\n📋 修正内容:")
    print("   1. PyCaretのtimeseries設定にdata_split_shuffle/fold_shuffleを追加")
    print("   2. 未定義のJP_FPフォント参照を削除")
    print()

    # 作業ディレクトリ
    work_dir = Path.cwd()
    print(f"📂 作業ディレクトリ: {work_dir}")

    # Phase 1-4のノートブックを検索
    notebooks = sorted(work_dir.glob('店舗別包括ダッシュボード_v5.0_Phase*.ipynb'))

    if not notebooks:
        print("\n❌ Phase 1-4のノートブックが見つかりません")
        return

    print(f"\n🔍 対象ノートブック: {len(notebooks)}個")
    for nb in notebooks:
        print(f"   • {nb.name}")

    # 修正実行（自動実行）
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
    print(f"\n📝 注意事項:")
    print(f"   • バックアップは各ノートブックと同じフォルダに保存されています")
    print(f"   • 問題があれば .backup_* ファイルから復元できます")
    print(f"\n🎯 次のステップ:")
    print(f"   1. Jupyter Labでノートブックを開く")
    print(f"   2. 「Kernel」→「Restart Kernel and Run All Cells」を実行")
    print(f"   3. エラーがないか確認")


if __name__ == "__main__":
    main()
