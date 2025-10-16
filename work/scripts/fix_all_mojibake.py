"""
すべての文字化け箇所を完全修正
日本語テキストにfontproperties=JP_FPが無い箇所を全て検出・修正
"""

import json
import re
from pathlib import Path

def fix_mojibake(notebook_path):
    """文字化け修正: 日本語テキストにJP_FPを強制適用"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fixed_count = 0

    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if not isinstance(source, list):
            continue

        new_source = []

        for line in source:
            new_line = line
            modified = False

            # 日本語を含むかチェック
            has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', line))

            if not has_japanese:
                new_source.append(line)
                continue

            # 1. plt.suptitle に fontproperties=JP_FP が無い
            if 'plt.suptitle' in line and 'fontproperties' not in line:
                # 最後の ) の前に fontproperties=JP_FP を挿入
                new_line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', new_line)
                modified = True

            # 2. ax.set_title に fontproperties=JP_FP が無い
            elif re.search(r'ax\d*\.set_title\(', line) and 'fontproperties' not in line:
                new_line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', new_line)
                modified = True

            # 3. ax.set_xlabel に fontproperties=JP_FP が無い
            elif re.search(r'ax\d*\.set_xlabel\(', line) and 'fontproperties' not in line:
                new_line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', new_line)
                modified = True

            # 4. ax.set_ylabel に fontproperties=JP_FP が無い
            elif re.search(r'ax\d*\.set_ylabel\(', line) and 'fontproperties' not in line:
                new_line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', new_line)
                modified = True

            # 5. ax.text に fontproperties=JP_FP が無い（複数行の場合もカバー）
            elif re.search(r'ax\d*\.text\(', line) and 'fontproperties' not in line:
                # 同じ行に閉じ括弧がある場合
                if ')' in line:
                    new_line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', new_line)
                    modified = True

            # 6. fig.text に fontproperties=JP_FP が無い
            elif 'fig.text(' in line and 'fontproperties' not in line:
                if ')' in line:
                    new_line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', new_line)
                    modified = True

            # 7. ax.legend に prop=JP_FP が無い (legendは特別)
            elif re.search(r'ax\d*\.legend\(', line) and 'prop=' not in line and 'fontproperties' not in line:
                if ')' in line:
                    new_line = re.sub(r'\)(\s*)$', r', prop=JP_FP)\1', new_line)
                    modified = True

            # 8. plt.xlabel に fontproperties=JP_FP が無い
            elif 'plt.xlabel(' in line and 'fontproperties' not in line:
                new_line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', new_line)
                modified = True

            # 9. plt.ylabel に fontproperties=JP_FP が無い
            elif 'plt.ylabel(' in line and 'fontproperties' not in line:
                new_line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', new_line)
                modified = True

            # 10. plt.title に fontproperties=JP_FP が無い
            elif 'plt.title(' in line and 'fontproperties' not in line:
                new_line = re.sub(r'\)(\s*)$', r', fontproperties=JP_FP)\1', new_line)
                modified = True

            new_source.append(new_line)
            if modified:
                fixed_count += 1

        if new_source != source:
            cell['source'] = new_source

    if fixed_count > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return fixed_count

    return 0


def detect_mojibake_risk(notebook_path):
    """文字化けリスクを検出（警告のみ）"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    risks = []

    for cell_idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source

        lines = code.split('\n')

        for line_idx, line in enumerate(lines):
            # 日本語を含むかチェック
            has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', line))

            if not has_japanese:
                continue

            # グラフ関連の関数で日本語を使用しているかチェック
            plot_functions = [
                'suptitle', 'set_title', 'set_xlabel', 'set_ylabel',
                'text', 'legend', 'xlabel', 'ylabel', 'title', 'annotate'
            ]

            for func in plot_functions:
                if func in line and 'fontproperties' not in line and 'prop=' not in line:
                    risks.append({
                        'cell': cell_idx + 1,
                        'line': line_idx + 1,
                        'function': func,
                        'text': line.strip()[:80]
                    })

    return risks


if __name__ == '__main__':
    notebooks = [
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase1.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb'
    ]

    print("\n" + "="*80)
    print("🔧 文字化け完全修正".center(80))
    print("="*80)

    total_fixed = 0

    # ステップ1: 修正実行
    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        count = fix_mojibake(nb_path)
        if count > 0:
            print(f"  ✅ {count}箇所にJP_FPを追加")
            total_fixed += count
        else:
            print(f"  ℹ️ 修正箇所なし")

    print("\n" + "="*80)
    print(f"✅ 合計 {total_fixed}箇所を修正".center(80))
    print("="*80)

    # ステップ2: リスク検出
    print("\n" + "="*80)
    print("🔍 文字化けリスク最終チェック".center(80))
    print("="*80)

    total_risks = 0

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        risks = detect_mojibake_risk(nb_path)

        if risks:
            print(f"  ⚠️ {len(risks)}箇所で文字化けリスク検出:")
            for risk in risks[:5]:
                print(f"    Cell {risk['cell']}, Line {risk['line']}: {risk['function']}")
                print(f"      → {risk['text']}")
            total_risks += len(risks)
        else:
            print(f"  ✅ 文字化けリスクなし")

    print("\n" + "="*80)
    if total_risks == 0:
        print("✅ すべての日本語にフォント設定適用済み".center(80))
    else:
        print(f"⚠️ {total_risks}箇所で要確認".center(80))
    print("="*80)
