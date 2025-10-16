"""
残存文字化けリスクを完全修正
- 複数行にわたるax.text()
- HTML内のタイトル
- legend=False のケース
"""

import json
import re
from pathlib import Path

def fix_multiline_text(notebook_path):
    """複数行にわたるテキスト要素にJP_FP追加"""

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
        i = 0

        while i < len(source):
            line = source[i]

            # 日本語を含むかチェック
            has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', line))

            # ax.text() で fontproperties が無い場合
            if has_japanese and re.search(r'ax\d*\.text\(', line) and 'fontproperties' not in line:
                # 次の行を確認して閉じ括弧を探す
                j = i
                full_statement = line
                while j < len(source) - 1 and ')' not in source[j]:
                    j += 1
                    full_statement += source[j]

                # 閉じ括弧の前に fontproperties=JP_FP を追加
                if j > i:
                    # 複数行にわたる場合
                    for k in range(i, j):
                        new_source.append(source[k])

                    # 最後の行に fontproperties を追加
                    last_line = source[j]
                    if ')' in last_line and 'fontproperties' not in last_line:
                        last_line = re.sub(r'\)', r', fontproperties=JP_FP)', last_line, count=1)
                        fixed_count += 1
                    new_source.append(last_line)
                    i = j + 1
                    continue

            # Plotly HTML の <title> タグ（これは問題ない、スキップ）
            if '<title>' in line:
                new_source.append(line)
                i += 1
                continue

            # <div class="section-title"> （これもHTML、スキップ）
            if 'section-title' in line:
                new_source.append(line)
                i += 1
                continue

            # legend=False のケース（フォント不要）
            if 'legend=False' in line or 'legend=F' in line:
                new_source.append(line)
                i += 1
                continue

            new_source.append(line)
            i += 1

        if new_source != source:
            cell['source'] = new_source

    if fixed_count > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        return fixed_count

    return 0


if __name__ == '__main__':
    notebooks = [
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase1.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase2.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase3.ipynb',
        '/mnt/d/github/pycaret/work/店舗別包括ダッシュボード_v5.0_Phase4.ipynb'
    ]

    print("\n" + "="*80)
    print("🔧 残存文字化けリスク修正".center(80))
    print("="*80)

    total_fixed = 0

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")
        count = fix_multiline_text(nb_path)
        if count > 0:
            print(f"  ✅ {count}箇所にJP_FPを追加")
            total_fixed += count
        else:
            print(f"  ℹ️ 修正箇所なし")

    print("\n" + "="*80)
    print(f"✅ 合計 {total_fixed}箇所を修正".center(80))
    print("="*80)
