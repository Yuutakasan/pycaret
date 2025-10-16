#!/usr/bin/env python3
"""
変換済みの06_ファイルを統合

batch_convert.pyで変換されたCSVファイルを統合し、
重複を除いて1つのファイルにまとめる。
"""

import re
import sys
from pathlib import Path

import pandas as pd

from work_utils import configure_logging


def extract_dates_from_filename(filename: str) -> tuple:
    """ファイル名から日付を抽出"""
    pattern = r'_(\d{8})_(\d{8})\.csv$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def main():
    """メイン処理"""
    logger = configure_logging(name="work.merge_converted_06")

    output_dir = Path("work/output")

    try:
        logger.info("=" * 60)
        logger.info("変換済み06_ファイルの統合")
        logger.info("=" * 60)

        # 対象ファイルを検索
        pattern = "06_【POS情報】店別－商品別実績_TX秋葉原駅_TX六町駅_TXつくば駅_*.csv"
        files = sorted(output_dir.glob(pattern))

        if len(files) == 0:
            logger.error(f"対象ファイルが見つかりません: {pattern}")
            sys.exit(1)

        logger.info(f"対象ファイル: {len(files)}件")
        for f in files:
            logger.info(f"  - {f.name}")

        # 日付範囲を取得
        all_start_dates = []
        all_end_dates = []
        for file_path in files:
            start_date, end_date = extract_dates_from_filename(file_path.name)
            if start_date and end_date:
                all_start_dates.append(start_date)
                all_end_dates.append(end_date)

        overall_start = min(all_start_dates)
        overall_end = max(all_end_dates)
        logger.info(f"全体の日付範囲: {overall_start} - {overall_end}")

        # データフレームのリスト
        dfs = []

        # 各ファイルを読み込み
        for file_path in files:
            logger.info(f"読み込み中: {file_path.name}")

            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                logger.info(f"  {len(df):,}行 × {len(df.columns)}列")
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

        # 日付でソート
        if '日付' in df_merged.columns:
            logger.info("日付列でソート中...")
            df_merged['日付'] = pd.to_datetime(df_merged['日付'], errors='coerce')
            df_merged = df_merged.sort_values('日付')

        # 出力ファイル名を生成
        output_filename = f"06_converted_{overall_start}_{overall_end}.csv"
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
