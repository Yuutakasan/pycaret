#!/usr/bin/env python3
"""
06_【POS情報】店別－商品別実績ファイルを統合

同一形式のExcelファイルを読み込み、重複を除いて統合し、
開始日付から終了日付のファイル名で保存する。
"""

import logging
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


def setup_logging() -> logging.Logger:
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def extract_dates_from_filename(filename: str) -> tuple:
    """
    ファイル名から日付を抽出

    例: 06_【POS情報】店別－商品別実績_TX秋葉原駅_TX六町駅_TXつくば駅_20250701_20250730.xlsx
    → (20250701, 20250730)
    """
    pattern = r'_(\d{8})_(\d{8})\.xlsx$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def main():
    """メイン処理"""
    logger = setup_logging()

    input_dir = Path("work/input")
    output_dir = Path("work/output")

    try:
        logger.info("=" * 60)
        logger.info("06_【POS情報】店別－商品別実績ファイルの統合開始")
        logger.info("=" * 60)

        # 対象ファイルを検索
        pattern = "06_【POS情報】店別－商品別実績_TX秋葉原駅_TX六町駅_TXつくば駅_*.xlsx"
        files = sorted(input_dir.glob(pattern))

        if len(files) == 0:
            logger.error(f"対象ファイルが見つかりません: {pattern}")
            sys.exit(1)

        logger.info(f"対象ファイル: {len(files)}件")
        for f in files:
            logger.info(f"  - {f.name}")

        # 各ファイルの日付範囲を抽出
        all_start_dates = []
        all_end_dates = []

        for file_path in files:
            start_date, end_date = extract_dates_from_filename(file_path.name)
            if start_date and end_date:
                all_start_dates.append(start_date)
                all_end_dates.append(end_date)
                logger.debug(f"{file_path.name}: {start_date} - {end_date}")

        # 全体の日付範囲を決定
        overall_start = min(all_start_dates)
        overall_end = max(all_end_dates)
        logger.info(f"全体の日付範囲: {overall_start} - {overall_end}")

        # データフレームのリスト
        dfs = []

        # 各ファイルを読み込み
        for file_path in files:
            logger.info(f"読み込み中: {file_path.name}")

            try:
                # Excelファイルを読み込み（ヘッダー行を自動検出）
                df = pd.read_excel(file_path)
                logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

                # 列名を表示（デバッグ用）
                logger.debug(f"  列名: {df.columns.tolist()[:5]}...")

                dfs.append(df)

            except Exception as e:
                logger.error(f"  エラー: {e}")
                continue

        if len(dfs) == 0:
            logger.error("読み込めるファイルがありませんでした")
            sys.exit(1)

        # データフレームを結合
        logger.info("データフレームを結合中...")
        df_merged = pd.concat(dfs, ignore_index=True)
        logger.info(f"結合後: {len(df_merged):,}行 × {len(df_merged.columns)}列")

        # 重複を除去
        logger.info("重複を除去中...")
        initial_rows = len(df_merged)
        df_merged = df_merged.drop_duplicates()
        removed_rows = initial_rows - len(df_merged)
        logger.info(f"  重複除去: {removed_rows:,}行削除")
        logger.info(f"  最終行数: {len(df_merged):,}行")

        # 日付列がある場合はソート
        date_columns = [col for col in df_merged.columns if '日付' in col or '年月日' in col]
        if date_columns:
            sort_col = date_columns[0]
            logger.info(f"日付列 '{sort_col}' でソート中...")
            df_merged[sort_col] = pd.to_datetime(df_merged[sort_col], errors='coerce')
            df_merged = df_merged.sort_values(sort_col)

        # 出力ファイル名を生成
        output_filename = f"06_【POS情報】店別－商品別実績_TX秋葉原駅_TX六町駅_TXつくば駅_{overall_start}_{overall_end}.csv"
        output_path = output_dir / output_filename

        # CSV形式で保存
        logger.info(f"保存中: {output_path}")
        df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')

        logger.info("=" * 60)
        logger.info(f"✅ 完了: {len(df_merged):,}行 × {len(df_merged.columns)}列")
        logger.info(f"出力ファイル: {output_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ エラー: {e}")
        logger.error("=" * 60)

        import traceback
        logger.debug(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()
