#!/usr/bin/env python3
"""
06_convertedファイルから必要な列だけを抽出してクリーンなデータを作成
"""

import sys
from pathlib import Path

import pandas as pd

from work_utils import configure_logging


def main():
    """メイン処理"""
    logger = configure_logging(name="work.clean_06_data")

    try:
        logger.info("=" * 60)
        logger.info("06_convertedファイルのクリーンアップ")
        logger.info("=" * 60)

        # 入力ファイル
        input_file = Path("work/output/06_converted_20250701_20250930.csv")
        output_file = Path("work/output/06_cleaned_20250701_20250930.csv")

        logger.info(f"読み込み中: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8-sig', low_memory=False)
        logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

        # 必要な列を特定
        required_columns = [
            '店舗',
            'フェイスくくり大分類',
            'フェイスくくり中分類',
            'フェイスくくり小分類',
            '商品名',
            '日付',
            '売上数量',
            '売上金額'
        ]

        # 存在する列だけを抽出
        available_columns = [col for col in required_columns if col in df.columns]
        logger.info(f"抽出する列: {available_columns}")

        df_clean = df[available_columns].copy()

        # データ型の変換
        df_clean['日付'] = pd.to_datetime(df_clean['日付'], errors='coerce')

        # 数値列の変換
        for col in ['売上数量', '売上金額']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # NaNを除去
        logger.info("NaN値を除去中...")
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['日付'])
        logger.info(f"  日付がNaNの行を除去: {initial_rows - len(df_clean):,}行")

        # 日付でソート
        df_clean = df_clean.sort_values('日付')

        # 保存
        logger.info(f"保存中: {output_file}")
        df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')

        logger.info("=" * 60)
        logger.info(f"✅ 完了: {len(df_clean):,}行 × {len(df_clean.columns)}列")
        logger.info(f"出力ファイル: {output_file}")
        logger.info("=" * 60)

        # データのサンプルを表示
        logger.info("\nデータサンプル:")
        logger.info(df_clean.head(3).to_string())

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ エラー: {e}")
        logger.error("=" * 60)

        import traceback
        logger.debug(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()
