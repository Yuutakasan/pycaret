#!/usr/bin/env python3
"""
統合した06ファイルを変換して特徴量を付与

1. CSVをワイド→ロング形式に変換
2. 特徴量を付与
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# batch_convert.pyから関数をインポート
sys.path.insert(0, str(Path(__file__).parent))
from batch_convert import analyze_excel_structure, convert_wide_to_long


def setup_logging() -> logging.Logger:
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def main():
    """メイン処理"""
    logger = setup_logging()

    try:
        logger.info("=" * 60)
        logger.info("06統合ファイルの変換・特徴量付与開始")
        logger.info("=" * 60)

        # 入力ファイル
        input_file = Path("work/output/06_【POS情報】店別－商品別実績_TX秋葉原駅_TX六町駅_TXつくば駅_20250701_20250930.csv")
        converted_file = Path("work/output/06_converted_20250701_20250930.csv")
        enriched_file = Path("work/output/06_enriched_20250701_20250930.csv")

        # Step 1: CSVを読み込み
        logger.info(f"読み込み中: {input_file}")
        df_raw = pd.read_csv(input_file, encoding='utf-8-sig')
        logger.info(f"  {len(df_raw):,}行 × {len(df_raw.columns)}列")

        # Step 2: Excelと同じ構造解析
        logger.info("構造解析中...")

        # ヘッダー行と日付行を検出
        header_idx = None
        date_idx = None

        for idx in range(min(20, len(df_raw))):
            row_values = df_raw.iloc[idx].astype(str).tolist()

            # 日付行の検出（YYYY-MM-DD または YYYY/MM/DD）
            if date_idx is None:
                date_count = sum(1 for val in row_values if '/' in val or '-' in val)
                if date_count > 5:
                    date_idx = idx
                    logger.info(f"  日付行を検出: 行{idx + 1}")

            # ヘッダー行の検出（「店舗」「商品」などのキーワード）
            if header_idx is None:
                header_keywords = ['店舗', '商品', 'フェイス', '分類']
                if any(keyword in str(row_values) for keyword in header_keywords):
                    header_idx = idx
                    logger.info(f"  ヘッダー行を検出: 行{idx + 1}")

            if header_idx is not None and date_idx is not None:
                break

        if header_idx is None or date_idx is None:
            logger.error("ヘッダー行または日付行が検出できませんでした")
            sys.exit(1)

        # Step 3: ワイド→ロング形式に変換
        logger.info("ワイド→ロング形式に変換中...")

        # ID列を検出
        header_row = df_raw.iloc[header_idx].astype(str).tolist()
        id_columns = []
        for col, val in enumerate(header_row):
            if any(keyword in val for keyword in ['店舗', '商品', 'フェイス', '分類']):
                id_columns.append(val)

        logger.info(f"  ID列: {id_columns}")

        # 変換実行
        df_long = convert_wide_to_long(df_raw, header_idx, date_idx, id_columns, logger)

        # 保存
        logger.info(f"変換結果を保存中: {converted_file}")
        df_long.to_csv(converted_file, index=False, encoding='utf-8-sig')
        logger.info(f"  {len(df_long):,}行 × {len(df_long.columns)}列")

        logger.info("=" * 60)
        logger.info(f"✅ 変換完了")
        logger.info(f"出力ファイル: {converted_file}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("次のステップ: 特徴量付与")
        logger.info(f"python3 work/enrich_features_v2.py {converted_file} {enriched_file} \\")
        logger.info(f"  --store-locations work/stores.csv \\")
        logger.info(f"  --past-year-data work/output/01_【売上情報】店別実績_20250903143116（20240901-20250831）.csv")

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ エラー: {e}")
        logger.error("=" * 60)

        import traceback
        logger.debug(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()
