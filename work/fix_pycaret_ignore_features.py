#!/usr/bin/env python3
"""PyCaretのignore_features追加"""
import json, re
from pathlib import Path
from datetime import datetime

for nb_path in Path.cwd().glob('店舗別包括ダッシュボード_v5.0_Phase1.ipynb'):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            
            # PyCaretのsetup()を検出
            if 'reg = setup(' in source and 'from pycaret.regression import' in source:
                # ignore_features追加
                if 'ignore_features=' not in source:
                    # data=product_daily の次の行に追加
                    source = source.replace(
                        'data=product_daily,',
                        'data=product_daily,\n            ignore_features=[\'商品名\'],'
                    )
                    cell['source'] = source.split('\n')
                    if not cell['source'][-1].endswith('\n'):
                        cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                    modified = True
                    print(f"✅ {nb_path.name}: ignore_features=['商品名'] を追加")
    
    if modified:
        backup = nb_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb')
        nb_path.rename(backup)
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"   バックアップ: {backup.name}")

print("\n✅ 修正完了")
