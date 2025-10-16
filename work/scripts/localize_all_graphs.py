"""
全グラフの完全日本語化と文字化け対策

対応項目:
1. すべての英語テキストを日本語に翻訳
2. すべてのテキスト要素にfontproperties=JP_FPを適用
3. グラフ内に解釈ガイドの注釈を追加（コメントではなく）
"""

import json
import re
from pathlib import Path

# 英語→日本語の翻訳マップ
TRANSLATIONS = {
    'Executive Summary': '経営サマリー',
    'YoY Growth': '前年比成長率',
    'Last 30d': '過去30日間',
    'Avg Spend': '平均客単価',
    'ALERTS': 'アラート',
    'KPIs': '重要指標',
    "TODAY'S ACTIONS": '本日のアクション',
    'Revenue': '売上高',
    'Customers': '顧客数',
    'Items Sold': '販売点数',
    'Growth Rate': '成長率',
    'Alert Level': 'アラートレベル',
    'High': '高',
    'Medium': '中',
    'Low': '低',
    'Critical': '重大',
    'Normal': '正常',
    'Warning': '警告',
    'Trend': 'トレンド',
    'Forecast': '予測',
    'Actual': '実績',
    'Target': '目標',
    'Daily': '日次',
    'Weekly': '週次',
    'Monthly': '月次',
    'Hourly': '時間帯別',
    'Store': '店舗',
    'Product': '商品',
    'Category': 'カテゴリ',
    'Time': '時刻',
    'Date': '日付',
    'Value': '値',
    'Count': '件数',
    'Amount': '金額',
    'Rate': '率',
    'Total': '合計',
    'Average': '平均',
    'Max': '最大',
    'Min': '最小',
}

def localize_graphs(notebook_path):
    """グラフを完全に日本語化"""

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
        modified = False

        for line in source:
            new_line = line

            # 1. plt.suptitle の英語を日本語に変換
            if 'plt.suptitle' in line:
                for eng, jpn in TRANSLATIONS.items():
                    if f"'{eng}" in line or f'"{eng}' in line:
                        new_line = new_line.replace(f"'{eng}", f"'{jpn}")
                        new_line = new_line.replace(f'"{eng}', f'"{jpn}')
                        modified = True

                # fontproperties=JP_FPが無ければ追加
                if 'fontproperties' not in new_line and 'JP_FP' not in new_line:
                    # 閉じ括弧の前にfontpropertiesを追加
                    new_line = new_line.rstrip().rstrip(')')
                    if ',' in new_line:
                        new_line += ', fontproperties=JP_FP)\n'
                    else:
                        # suptitle(title)のような単純な形式
                        new_line += ', fontproperties=JP_FP)\n'
                    modified = True

            # 2. ax.set_title の英語を日本語に変換
            if '.set_title(' in line:
                for eng, jpn in TRANSLATIONS.items():
                    if f"'{eng}" in line or f'"{eng}' in line:
                        new_line = new_line.replace(f"'{eng}", f"'{jpn}")
                        new_line = new_line.replace(f'"{eng}', f'"{jpn}')
                        modified = True

                # fontproperties=JP_FPを追加
                if 'fontproperties' not in new_line:
                    new_line = new_line.rstrip().rstrip(')')
                    new_line += ', fontproperties=JP_FP)\n'
                    modified = True

            # 3. ax.set_xlabel, ax.set_ylabel の日本語化
            if '.set_xlabel(' in line or '.set_ylabel(' in line:
                for eng, jpn in TRANSLATIONS.items():
                    if f"'{eng}" in line or f'"{eng}' in line:
                        new_line = new_line.replace(f"'{eng}", f"'{jpn}")
                        new_line = new_line.replace(f'"{eng}', f'"{jpn}')
                        modified = True

                if 'fontproperties' not in new_line:
                    new_line = new_line.rstrip().rstrip(')')
                    new_line += ', fontproperties=JP_FP)\n'
                    modified = True

            # 4. ax.text の英語を日本語に変換
            if '.text(' in line and 'ALERTS' in line:
                # ALERTSテキストを日本語化
                new_line = new_line.replace('ALERTS:', 'アラート:')
                new_line = new_line.replace('"ALERTS\\n"', '"アラート\\n"')
                modified = True

                # fontpropertiesを追加
                if 'fontproperties' not in new_line:
                    new_line = new_line.rstrip().rstrip(')')
                    new_line += ', fontproperties=JP_FP)\n'
                    modified = True

            # 5. 凡例（legend）の日本語化
            if '.legend(' in line:
                if 'fontproperties' not in new_line and 'prop=' not in new_line:
                    new_line = new_line.rstrip().rstrip(')')
                    new_line += ', prop=JP_FP)\n'
                    modified = True

            # 6. figテキスト（fig.text）の日本語化
            if 'fig.text(' in line:
                for eng, jpn in TRANSLATIONS.items():
                    if f"'{eng}" in line or f'"{eng}' in line:
                        new_line = new_line.replace(f"'{eng}", f"'{jpn}")
                        new_line = new_line.replace(f'"{eng}', f'"{jpn}')
                        modified = True

                if 'fontproperties' not in new_line:
                    new_line = new_line.rstrip().rstrip(')')
                    new_line += ', fontproperties=JP_FP)\n'
                    modified = True

            # 7. Plotlyのタイトル・軸ラベル日本語化
            if 'update_layout' in line or 'update_xaxes' in line or 'update_yaxes' in line:
                for eng, jpn in TRANSLATIONS.items():
                    if f"'{eng}" in line or f'"{eng}' in line:
                        new_line = new_line.replace(f"'{eng}", f"'{jpn}")
                        new_line = new_line.replace(f'"{eng}', f'"{jpn}')
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


def add_graph_annotations(notebook_path):
    """グラフ内に解釈ガイドの注釈を追加（コメントではなく実際のグラフ表示内）"""

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
            new_source.append(line)

            # Executive Summary（経営サマリー）グラフに注釈追加
            if 'plt.suptitle' in line and '経営サマリー' in line:
                # 次の行を確認して、すでに注釈がなければ追加
                if i + 1 < len(source) and 'fig.text' not in source[i + 1]:
                    # グラフの説明注釈を追加
                    indent = len(line) - len(line.lstrip())
                    annotation = ' ' * indent + 'fig.text(0.5, 0.92, "📊 このグラフの見方: 店舗の主要指標を一目で把握できます", ha="center", fontsize=10, fontproperties=JP_FP, style="italic", color="gray")\n'
                    new_source.append(annotation)
                    fixed_count += 1

            # YoY Growth（前年比成長率）グラフに注釈追加
            if '.set_title(' in line and '前年比成長率' in line:
                if i + 1 < len(source) and '.annotate(' not in source[i + 1]:
                    # 成長率の解釈ガイドを追加
                    indent = len(line) - len(line.lstrip())
                    # ax番号を取得
                    ax_match = re.search(r'(ax\d+)', line)
                    if ax_match:
                        ax_name = ax_match.group(1)
                        annotation = f'{" " * indent}{ax_name}.text(0.5, 0.95, "緑: 成長 | 赤: 減少", transform={ax_name}.transAxes, ha="center", va="top", fontsize=9, fontproperties=JP_FP, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))\n'
                        new_source.append(annotation)
                        fixed_count += 1

            # 平均客単価グラフに注釈追加
            if '.set_title(' in line and '平均客単価' in line:
                if i + 1 < len(source) and '.text(' not in source[i + 1]:
                    indent = len(line) - len(line.lstrip())
                    ax_match = re.search(r'(ax\d+)', line)
                    if ax_match:
                        ax_name = ax_match.group(1)
                        annotation = f'{" " * indent}{ax_name}.text(0.5, 0.95, "💡 目標客単価と比較してください", transform={ax_name}.transAxes, ha="center", va="top", fontsize=9, fontproperties=JP_FP, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))\n'
                        new_source.append(annotation)
                        fixed_count += 1

            # アラートパネルに注釈追加
            if '.text(' in line and 'アラート:' in line:
                if i + 1 < len(source) and 'ℹ️' not in source[i + 1]:
                    indent = len(line) - len(line.lstrip())
                    ax_match = re.search(r'(ax\d+)', line)
                    if ax_match:
                        ax_name = ax_match.group(1)
                        annotation = f'{" " * indent}{ax_name}.text(0.5, 0.05, "ℹ️ 優先度順に対応してください", transform={ax_name}.transAxes, ha="center", va="bottom", fontsize=8, fontproperties=JP_FP, style="italic", color="darkred")\n'
                        new_source.append(annotation)
                        fixed_count += 1

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
    print("🌐 全グラフ日本語化・文字化け対策".center(80))
    print("="*80)

    for nb_path in notebooks:
        path = Path(nb_path)
        if not path.exists():
            continue

        print(f"\n📝 {path.name}")

        # ステップ1: 英語→日本語翻訳とフォント適用
        count1 = localize_graphs(nb_path)
        if count1 > 0:
            print(f"  ✅ {count1}箇所を日本語化・フォント適用")

        # ステップ2: グラフ内注釈追加
        count2 = add_graph_annotations(nb_path)
        if count2 > 0:
            print(f"  ✅ {count2}個の解釈ガイド注釈を追加")

        if count1 == 0 and count2 == 0:
            print(f"  ℹ️ 修正箇所なし")

    print("\n" + "="*80)
    print("✅ 完了".center(80))
    print("="*80)
