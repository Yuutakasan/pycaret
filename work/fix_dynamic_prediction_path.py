#!/usr/bin/env python3
"""動的売上予測システムのファイルパスを修正"""

import nbformat

notebook_file = '動的売上予測システム - PyCaret 3.ipynb'

print("=" * 70)
print("動的売上予測システムのファイルパス修正")
print("=" * 70)

try:
    with open(notebook_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    modified = False

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            # データパスのセルを探す
            if 'possible_paths' in cell.source and 'pos_data_long_format_with_weather.csv' in cell.source:
                # 古いパスを新しいパスに置き換え
                old_source = cell.source

                new_source = """# データパスの設定（複数のパスを試す）
possible_paths = [
    'output/06_final_enriched_20250701_20250930.csv',  # 最新の特徴量付きデータ
    'output/06_cleaned_20250701_20250930.csv',  # クリーニング済みデータ
    'output/06_converted_20250701_20250930.csv'  # 変換済みデータ
]

df = None
is_gpu_df = False

for file_path in possible_paths:
    if Path(file_path).exists():
        df, is_gpu_df = load_data_optimized(file_path)
        print(f"📊 使用ファイル: {file_path}")
        print(f"📊 使用モード: {'GPU (cuDF)' if is_gpu_df else 'CPU (pandas)'}")
        break
    else:
        print(f"⚠️ ファイルが見つかりません: {file_path}")

if df is None:
    print("❌ データファイルが見つかりません")
    print("以下のいずれかの場所にCSVファイルを配置してください:")
    for path in possible_paths:
        print(f"  - {path}")
    raise FileNotFoundError("データファイルが見つかりません")"""

                cell.source = new_source

                print(f"✓ セル {i}: ファイルパスを更新")
                print("  新しいパス:")
                print("    - output/06_final_enriched_20250701_20250930.csv")
                print("    - output/06_cleaned_20250701_20250930.csv")
                print("    - output/06_converted_20250701_20250930.csv")
                modified = True

    if modified:
        # 保存
        with open(notebook_file, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("\n✅ 修正完了")
    else:
        print("○ 該当セルが見つかりませんでした")

    print("\n" + "=" * 70)
    print("ファイルパス更新完了")
    print("=" * 70)

except Exception as e:
    print(f"✗ エラー: {e}")
    import traceback
    traceback.print_exc()
