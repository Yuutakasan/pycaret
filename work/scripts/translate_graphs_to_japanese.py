#!/usr/bin/env python3
"""
グラフの日本語化と注釈追加スクリプト

目的:
1. 全グラフのタイトル・ラベルを日本語化
2. 各グラフに判断基準の注釈を追加
3. 日本人にわかりやすい表現に統一
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


# 英語→日本語の翻訳マッピング
TRANSLATIONS = {
    # グラフタイトル
    'Executive Summary': '経営サマリー',
    'Sales Trend': '売上推移',
    'YoY Growth': '前年比成長率',
    'Avg Customer Spend': '平均客単価',
    'Feature Importance': '特徴量重要度',
    'Sales Prediction': '売上予測',
    'Customer Count Trend': '客数推移',
    'Avg Spend per Customer Trend': '客単価推移',
    'Customer Count YoY Growth': '客数前年比',
    'Avg Spend YoY Growth': '客単価前年比',
    'Avg Daily Sales Comparison': '平均日商比較',
    'Gap from Top Store': 'トップ店舗とのギャップ',

    # 軸ラベル
    'Sales (JPY)': '売上金額（円）',
    'Sales': '売上',
    'Count': '客数',
    'JPY': '円',
    'Importance Score': '重要度スコア',
    'Gap (JPY)': 'ギャップ（円）',

    # 凡例
    'Today': '今年',
    'Last Year': '昨年',
    'This Year': '今年',
    'Target': '目標',

    # その他
    'ALERTS': 'アラート',
    'No Critical Alerts': '重要なアラートなし',
    "KPIs (Latest)": '主要指標（最新）',
    "TODAY'S ACTIONS": '本日のアクション',
    'Check TOP10 inventory': 'TOP10商品の在庫確認',
    'Adjust orders by weather': '天気に応じた発注調整',
    'Analyze YoY negative items': '前年比マイナス商品の分析',
    'Compare with other stores': '他店との比較',
}


def add_graph_annotations(source_code):
    """
    グラフコードに日本語の注釈とガイドを追加
    """
    annotations = []

    # エグゼクティブサマリーの注釈
    if '経営サマリー' in source_code or 'Executive Summary' in source_code:
        annotations.append("""
# 📊 グラフの見方ガイド
#
# 【売上推移グラフ】（左上・大）
#   ・青線（今年）が赤線（昨年）より上 → 好調
#   ・青線が赤点線（目標）を下回る → 要改善
#   ✅ 判断基準: 昨年比+5%以上なら優秀、-5%以下なら対策必須
#
# 【前年比成長率】（右上）
#   ・緑のバー → プラス成長（良好）
#   ・赤のバー → マイナス成長（要注意）
#   ✅ 判断基準: 連続3日以上赤なら要因分析が必要
#
# 【平均客単価】（右中）
#   ・線が上昇トレンド → 客単価向上施策が効果的
#   ・赤点線（平均）を下回る → セット販売・まとめ買い促進が必要
#   ✅ 判断基準: 平均±10%の範囲内なら正常
""")

    # 需要予測の注釈
    if '特徴量重要度' in source_code or 'Feature Importance' in source_code:
        annotations.append("""
# 📊 グラフの見方ガイド
#
# 【特徴量重要度グラフ】
#   ・棒が長い項目 → 売上予測に大きく影響する要素
#   ・上位3つの要素に注目して施策を考える
#
#   例）「最高気温」が上位 → 気温による商品入替が効果的
#       「曜日」が上位 → 曜日別の品揃え変更が重要
#       「昨年同日_売上」が上位 → 前年データを参考にした発注が有効
#
#   ✅ 判断基準: 重要度0.1以上の要素に集中して対策を打つ
""")

    # 客数・客単価分解の注釈
    if '客数推移' in source_code or 'Customer Count Trend' in source_code:
        annotations.append("""
# 📊 グラフの見方ガイド
#
# 【売上の3要素分解】
#   売上 = 客数 × 客単価 で分解して原因を特定
#
# 【客数推移】（左上）
#   ・今年（青）が昨年（紫点線）を上回る → 集客好調
#   ・下回る → チラシ・SNS・キャンペーンで集客強化
#   ✅ 判断基準: 前年比-10%以下なら即座に集客施策が必要
#
# 【客単価推移】（右上）
#   ・今年（オレンジ）が昨年を上回る → セット販売等が効果的
#   ・下回る → まとめ買い促進・関連商品陳列が必要
#   ✅ 判断基準: 前年比-5%以下なら商品構成の見直しが必要
#
# 【前年比グラフ】（下段）
#   ・緑 → プラス、赤 → マイナス
#   ・どちらが主要因かを見極めて対策を打つ
""")

    # 店舗間比較の注釈
    if '平均日商比較' in source_code or 'Daily Sales Comparison' in source_code:
        annotations.append("""
# 📊 グラフの見方ガイド
#
# 【店舗間比較】
#   ・赤色 = あなたの店舗（★マーク）
#   ・水色 = 他店舗
#
# 【平均日商比較】（左）
#   ・上位店舗との差 = 改善余地
#   ✅ 判断基準:
#      - トップ店の80%以上 → 優秀
#      - 60-80% → 改善の余地あり
#      - 60%未満 → 抜本的な見直しが必要
#
# 【平均客単価比較】（中）
#   ・客単価が低い → セット販売・高単価商品の推奨販売
#   ✅ 判断基準: 全店平均の90%以上を目標
#
# 【トップ店とのギャップ】（右）
#   ・このギャップを埋めると得られる増収額
#   ・具体的な改善施策: トップ店の成功事例を真似る
""")

    return '\n'.join(annotations) + '\n\n' if annotations else ''


def translate_to_japanese(source_lines):
    """
    ソースコードの英語表現を日本語に翻訳
    """
    modified = False
    new_lines = []

    for line in source_lines:
        original_line = line

        # 翻訳マッピングに基づいて置換
        for eng, jpn in TRANSLATIONS.items():
            if f"'{eng}'" in line or f'"{eng}"' in line:
                # クォート付きの文字列を置換
                line = line.replace(f"'{eng}'", f"'{jpn}'")
                line = line.replace(f'"{eng}"', f'"{jpn}"')
                modified = True

        new_lines.append(line)

    return new_lines, modified


def process_notebook(notebook_path):
    """
    ノートブック全体を処理して日本語化と注釈追加
    """
    print(f"\n{'='*80}")
    print(f"🔧 処理中: {notebook_path.name}")
    print(f"{'='*80}")

    # バックアップ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = notebook_path.with_suffix(f'.backup_jp_{timestamp}.ipynb')
    shutil.copy2(notebook_path, backup_path)
    print(f"   📋 バックアップ: {backup_path.name}")

    try:
        # ノートブック読み込み
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        cells = nb['cells']
        translation_count = 0
        annotation_count = 0

        for cell in cells:
            if cell.get('cell_type') == 'code':
                source_lines = cell.get('source', [])
                source_code = ''.join(source_lines)

                # グラフコードかチェック
                is_graph_code = any(keyword in source_code for keyword in [
                    'plt.', 'ax.', 'fig.', 'subplot'
                ])

                if is_graph_code:
                    # 注釈を追加
                    annotations = add_graph_annotations(source_code)
                    if annotations:
                        # セルの先頭に注釈を追加
                        source_lines = [annotations] + source_lines
                        annotation_count += 1

                    # 英語を日本語に翻訳
                    source_lines, modified = translate_to_japanese(source_lines)
                    if modified:
                        translation_count += 1

                    cell['source'] = source_lines

        # ノートブック保存
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)

        print(f"   ✅ 翻訳完了: {translation_count}セル")
        print(f"   ✅ 注釈追加: {annotation_count}セル")
        print(f"   ✅ 処理完了")
        return True

    except Exception as e:
        print(f"   ❌ エラー: {e}")
        shutil.copy2(backup_path, notebook_path)
        print(f"   🔄 バックアップから復元")
        return False


def main():
    """メイン処理"""
    print("\n" + "="*80)
    print("🎌 グラフ日本語化 & 注釈追加スクリプト v1.0")
    print("="*80)
    print("\n📋 実施内容:")
    print("   1. 全グラフのタイトル・ラベルを日本語化")
    print("   2. 各グラフに「見方ガイド」と「判断基準」を追加")
    print("   3. 日本人にわかりやすい表現に統一")
    print()

    work_dir = Path.cwd().parent if Path.cwd().name == 'scripts' else Path.cwd()
    print(f"📂 作業ディレクトリ: {work_dir}")

    # Phase 1-4検索
    notebooks = sorted(work_dir.glob('店舗別包括ダッシュボード_v5.0_Phase[1-4].ipynb'))
    if not notebooks:
        print("\n❌ Phase 1-4のノートブックが見つかりません")
        return

    print(f"\n🔍 対象: {len(notebooks)}個")
    for nb in notebooks:
        print(f"   • {nb.name}")

    # 処理実行
    print("\n" + "="*80)
    print("🚀 処理開始")
    print("="*80)

    success_count = sum(1 for nb in notebooks if process_notebook(nb))

    # 完了
    print("\n" + "="*80)
    print("✅ 処理完了")
    print("="*80)
    print(f"\n   成功: {success_count}/{len(notebooks)}")
    print(f"\n📝 変更内容:")
    print(f"   • 全グラフタイトルが日本語に")
    print(f"   • 軸ラベル・凡例が日本語に")
    print(f"   • 各グラフに判断基準の注釈を追加")
    print(f"\n💡 次のステップ:")
    print(f"   1. Jupyter Labでノートブックを開く")
    print(f"   2. Kernel → Restart Kernel and Run All Cells")
    print(f"   3. グラフの注釈と日本語表示を確認")


if __name__ == "__main__":
    main()
