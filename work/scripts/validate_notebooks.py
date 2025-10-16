"""
ノートブックの構文検証スクリプト
- Pythonコードセルの構文チェック
- インデントエラー検出
- 実行可能性の確認
"""

import json
import ast
import sys
from pathlib import Path

def validate_notebook(notebook_path):
    """ノートブックのすべてのコードセルを検証"""

    print(f"\n{'='*80}")
    print(f"📋 検証中: {notebook_path.name}")
    print(f"{'='*80}\n")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    errors = []
    cell_count = 0
    code_cell_count = 0

    for idx, cell in enumerate(nb['cells']):
        cell_count += 1

        if cell.get('cell_type') != 'code':
            continue

        code_cell_count += 1
        source = cell.get('source', [])

        if not source:
            continue

        # リストをコードに結合
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source

        # 空白のみの場合はスキップ
        if not code.strip():
            continue

        # マジックコマンド（%や!で始まる）を除去
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('%') or stripped.startswith('!'):
                # マジックコマンドをコメントに変換
                cleaned_lines.append('# ' + line)
            else:
                cleaned_lines.append(line)

        cleaned_code = '\n'.join(cleaned_lines)

        # 構文チェック
        try:
            ast.parse(cleaned_code)
            print(f"✅ Cell {idx + 1}: OK")
        except SyntaxError as e:
            error_msg = f"❌ Cell {idx + 1}, Line {e.lineno}: {e.msg}"
            print(error_msg)
            errors.append({
                'cell': idx + 1,
                'line': e.lineno,
                'error': e.msg,
                'text': e.text
            })
        except IndentationError as e:
            error_msg = f"❌ Cell {idx + 1}, Line {e.lineno}: IndentationError - {e.msg}"
            print(error_msg)
            errors.append({
                'cell': idx + 1,
                'line': e.lineno,
                'error': f"IndentationError: {e.msg}",
                'text': e.text
            })
        except Exception as e:
            error_msg = f"⚠️ Cell {idx + 1}: {type(e).__name__} - {str(e)}"
            print(error_msg)

    print(f"\n{'='*80}")
    print(f"📊 検証結果:")
    print(f"  総セル数: {cell_count}")
    print(f"  コードセル数: {code_cell_count}")
    print(f"  エラー数: {len(errors)}")

    if errors:
        print(f"\n⚠️ エラー詳細:")
        for err in errors:
            print(f"  Cell {err['cell']}, Line {err['line']}: {err['error']}")
            if err.get('text'):
                print(f"    → {err['text'].strip()}")
    else:
        print(f"\n✅ すべてのコードセルが正常です！")

    print(f"{'='*80}\n")

    return len(errors) == 0, errors


def main():
    notebooks = [
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase1.ipynb'),
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb'),
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb'),
        Path('/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb')
    ]

    print("\n" + "="*80)
    print("🔍 ノートブック構文検証".center(80))
    print("="*80)

    all_valid = True
    summary = {}

    for nb_path in notebooks:
        if not nb_path.exists():
            print(f"\n⚠️ ファイルが見つかりません: {nb_path}")
            continue

        is_valid, errors = validate_notebook(nb_path)
        summary[nb_path.name] = {
            'valid': is_valid,
            'error_count': len(errors),
            'errors': errors
        }

        if not is_valid:
            all_valid = False

    # 最終サマリー
    print("\n" + "="*80)
    print("📋 最終検証サマリー".center(80))
    print("="*80 + "\n")

    for nb_name, result in summary.items():
        status = "✅ 正常" if result['valid'] else f"❌ {result['error_count']}個のエラー"
        print(f"{nb_name}: {status}")

    print("\n" + "="*80)

    if all_valid:
        print("✅ すべてのノートブックが正常です！".center(80))
        print("="*80 + "\n")
        return 0
    else:
        print("⚠️ 一部のノートブックにエラーがあります".center(80))
        print("="*80 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
