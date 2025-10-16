"""
Plotlyのフォント警告を修正するスクリプト

問題: findfont警告が大量に表示される
原因: font_family_strに存在しないフォント（Arial, Liberation Sans等）を指定
解決: 実際に存在する日本語フォントのみを指定
"""

import json
import re
from pathlib import Path

def fix_plotly_fonts_in_font_setup():
    """font_setup.py のPlotlyフォント設定を修正"""

    font_setup_path = Path('/mnt/d/github/pycaret/work/font_setup.py')

    print("📝 font_setup.py の修正中...")
    print("="*80)

    with open(font_setup_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修正前のfont_family_str設定を探す
    old_pattern = r"font_family_str = f'{font_family}.*?sans-serif'"

    # 新しい設定（実在するフォントのみ）
    new_config = """font_family_str = font_family  # システムに存在する日本語フォントのみを使用"""

    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_config, content)
        print("✅ font_family_str を修正（存在しないフォントを除外）")

        with open(font_setup_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print("✅ font_setup.py の修正完了")
    else:
        print("⚠️ 修正対象のパターンが見つかりませんでした")

    print("="*80)


def verify_font_setup():
    """修正後の設定を確認"""

    print("\n🔍 修正内容の確認:")
    print("="*80)

    font_setup_path = Path('/mnt/d/github/pycaret/work/font_setup.py')

    with open(font_setup_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # setup_plotly_fonts関数の該当行を表示
    in_function = False
    for i, line in enumerate(lines, 1):
        if 'def setup_plotly_fonts' in line:
            in_function = True

        if in_function:
            if 'font_family_str' in line:
                print(f"  行 {i}: {line.rstrip()}")
                break

            if line.strip() and not line.strip().startswith('#') and 'def ' in line and 'setup_plotly_fonts' not in line:
                break

    print("="*80)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔧 Plotlyフォント警告修正スクリプト".center(80))
    print("="*80)

    # font_setup.py を修正
    fix_plotly_fonts_in_font_setup()

    # 修正内容を確認
    verify_font_setup()

    print("\n✅ 修正完了")
    print("\n📋 次のステップ:")
    print("   1. ノートブックのカーネルを再起動")
    print("   2. セルを再実行してフォント警告が消えたことを確認")
    print("="*80)
