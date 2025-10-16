"""
Phase1のalert_text, kpi_text, action_textにfontproperties追加
"""

import json

def fix_phase1_text_fonts():
    """Phase1のCell 12のテキスト要素にJP_FP追加"""

    notebook_path = '店舗別包括ダッシュボード_v5.0_Phase1.ipynb'

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fixed_count = 0

    for cell in nb['cells']:
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if not isinstance(source, list):
            continue

        # Cell 12を特定（alert_text, kpi_text, action_textを含む）
        code = ''.join(source)
        if 'alert_text' not in code or 'kpi_text' not in code:
            continue

        new_source = []

        for line in source:
            new_line = line

            # ax4.text(...) - alert_text用
            if 'ax4.text(0.1, 0.5, alert_text' in line and 'fontproperties' not in line:
                # bbox=の後ろに fontproperties=JP_FP を追加
                if 'bbox=' in line:
                    new_line = line.replace('bbox=dict', 'fontproperties=JP_FP, bbox=dict')
                    fixed_count += 1

            # ax5.text(...) - kpi_text用
            elif 'ax5.text(0.1, 0.5, kpi_text' in line and 'fontproperties' not in line:
                if 'bbox=' in line:
                    new_line = line.replace('bbox=dict', 'fontproperties=JP_FP, bbox=dict')
                    fixed_count += 1

            # ax6.text(...) - action_text用
            elif 'ax6.text(0.1, 0.5, action_text' in line and 'fontproperties' not in line:
                if 'bbox=' in line:
                    new_line = line.replace('bbox=dict', 'fontproperties=JP_FP, bbox=dict')
                    fixed_count += 1

            new_source.append(new_line)

        if new_source != source:
            cell['source'] = new_source

    if fixed_count > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"✅ Phase1: {fixed_count}箇所にJP_FPを追加")
        return fixed_count

    print(f"ℹ️ Phase1: 修正箇所なし")
    return 0


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔧 Phase1 テキスト要素フォント修正".center(80))
    print("="*80 + "\n")

    count = fix_phase1_text_fonts()

    print("\n" + "="*80)
    print(f"✅ 完了: {count}箇所修正".center(80))
    print("="*80)
