#!/usr/bin/env python3
"""
06_【POS情報】店別－商品別実績ファイルを統合・変換・特徴量付与

1. 同一形式のExcelファイルを統合
2. ワイド→ロング形式に変換
3. 特徴量を付与
"""

import logging
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def setup_logging() -> logging.Logger:
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def extract_dates_from_filename(filename: str) -> tuple:
    """ファイル名から日付を抽出"""
    pattern = r'_(\d{8})_(\d{8})\.xlsx$'
    match = re.search(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def analyze_excel_structure(df: pd.DataFrame, logger: logging.Logger) -> dict:
    """Excel構造を分析"""
    header_idx = None
    date_idx = None

    for idx in range(min(20, len(df))):
        row_values = df.iloc[idx].astype(str).tolist()

        # 日付行の検出
        if date_idx is None:
            date_count = sum(1 for val in row_values if '/' in val or '-' in val)
            if date_count > 5:
                date_idx = idx
                logger.info(f"  日付行を検出: 行{idx + 1}")

        # ヘッダー行の検出
        if header_idx is None:
            header_keywords = ['店舗', '商品', 'フェイス', '分類']
            if any(keyword in str(row_values) for keyword in header_keywords):
                header_idx = idx
                logger.info(f"  ヘッダー行を検出: 行{idx + 1}")

        if header_idx is not None and date_idx is not None:
            break

    return {"header_idx": header_idx, "date_idx": date_idx}


def convert_wide_to_long(df: pd.DataFrame, header_idx: int, date_idx: int, id_columns: list, logger: logging.Logger) -> pd.DataFrame:
    """ワイド→ロング形式に変換"""

    # 日付行とヘッダー行を取得
    date_row = df.iloc[date_idx].astype(str).tolist()
    header_row = df.iloc[header_idx].astype(str).tolist()

    # 日付とメトリックのマッピングを作成
    date_metric_map = {}
    for col_idx, (date_val, metric_val) in enumerate(zip(date_row, header_row)):
        if '/' in date_val or '-' in date_val:
            try:
                date_obj = pd.to_datetime(date_val)
                date_str = date_obj.strftime('%Y-%m-%d')

                if date_str not in date_metric_map:
                    date_metric_map[date_str] = {}

                metric_name = metric_val if metric_val != 'nan' and metric_val.strip() else f'指標{col_idx}'
                date_metric_map[date_str][col_idx] = metric_name
            except:
                continue

    logger.info(f"  {len(date_metric_map)}日分のデータを検出")

    # データ部分を抽出
    data_df = df.iloc[header_idx + 1:].copy()
    data_df.columns = header_row

    # ID列を前方埋め
    for col in id_columns:
        if col in data_df.columns:
            data_df[col] = data_df[col].replace('', pd.NA)
            data_df[col] = data_df[col].ffill()

    # 最後のID列（商品名）がない行を削除
    if id_columns:
        last_id_col = id_columns[-1]
        if last_id_col in data_df.columns:
            data_df = data_df[data_df[last_id_col].notna()]

    # 集計行を除去
    if id_columns and id_columns[0] in data_df.columns:
        data_df = data_df[~data_df[id_columns[0]].astype(str).str.contains('総合計|^合計', na=False, regex=True)]

    # ロング形式に変換
    long_data = []

    for date_str, metric_cols in date_metric_map.items():
        for _, row in data_df.iterrows():
            record = {'日付': date_str}

            # ID列を追加
            for id_col in id_columns:
                if id_col in row:
                    record[id_col] = row[id_col]

            # メトリック値を追加
            for col_idx, metric_name in metric_cols.items():
                if col_idx < len(row):
                    val = row.iloc[col_idx]
                    if pd.notna(val) and val != '':
                        try:
                            record[metric_name] = float(val)
                        except:
                            record[metric_name] = val

            long_data.append(record)

    df_long = pd.DataFrame(long_data)
    logger.info(f"  ロング形式に変換完了: {len(df_long):,}行")

    return df_long


def main():
    """メイン処理"""
    logger = setup_logging()

    input_dir = Path("work/input")
    output_dir = Path("work/output")

    try:
        logger.info("=" * 60)
        logger.info("Step 1: 06_ファイルの統合")
        logger.info("=" * 60)

        # 対象ファイルを検索
        pattern = "06_【POS情報】店別－商品別実績_TX秋葉原駅_TX六町駅_TXつくば駅_*.xlsx"
        files = sorted(input_dir.glob(pattern))

        if len(files) == 0:
            logger.error(f"対象ファイルが見つかりません: {pattern}")
            sys.exit(1)

        logger.info(f"対象ファイル: {len(files)}件")

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

        # 各ファイルを読み込み・変換
        all_long_dfs = []

        for file_path in files:
            logger.info(f"\n処理中: {file_path.name}")

            # Excelを読み込み
            df = pd.read_excel(file_path, header=None)
            logger.info(f"  読み込み: {len(df):,}行 × {len(df.columns)}列")

            # 構造解析
            structure = analyze_excel_structure(df, logger)
            header_idx = structure["header_idx"]
            date_idx = structure["date_idx"]

            if header_idx is None or date_idx is None:
                logger.warning(f"  構造を検出できませんでした。スキップします")
                continue

            # ID列を検出
            header_row = df.iloc[header_idx].astype(str).tolist()
            id_columns = []
            for col, val in enumerate(header_row):
                if any(keyword in val for keyword in ['店舗', '商品', 'フェイス', '分類']):
                    id_columns.append(val)

            logger.info(f"  ID列: {id_columns}")

            # ワイド→ロング変換
            df_long = convert_wide_to_long(df, header_idx, date_idx, id_columns, logger)
            all_long_dfs.append(df_long)

        # 統合
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: データの統合")
        logger.info("=" * 60)

        df_merged = pd.concat(all_long_dfs, ignore_index=True)
        logger.info(f"統合後: {len(df_merged):,}行 × {len(df_merged.columns)}列")

        # 重複除去
        initial_rows = len(df_merged)
        df_merged = df_merged.drop_duplicates()
        logger.info(f"重複除去: {initial_rows - len(df_merged):,}行削除")

        # 日付でソート
        df_merged['日付'] = pd.to_datetime(df_merged['日付'])
        df_merged = df_merged.sort_values('日付')

        # 変換済みファイルを保存
        converted_file = output_dir / f"06_converted_{overall_start}_{overall_end}.csv"
        logger.info(f"\n保存中: {converted_file}")
        df_merged.to_csv(converted_file, index=False, encoding='utf-8-sig')
        logger.info(f"  {len(df_merged):,}行 × {len(df_merged.columns)}列")

        logger.info("\n" + "=" * 60)
        logger.info("Step 3: 特徴量付与")
        logger.info("=" * 60)

        # 特徴量付与スクリプトを実行
        enriched_file = output_dir / f"06_enriched_{overall_start}_{overall_end}.csv"
        past_year_file = output_dir / "01_【売上情報】店別実績_20250903143116（20240901-20250831）.csv"
        stores_file = Path("work/stores.csv")

        cmd = [
            "python3", "work/enrich_features_v2.py",
            str(converted_file),
            str(enriched_file),
            "--store-locations", str(stores_file),
            "--past-year-data", str(past_year_file)
        ]

        logger.info(f"実行コマンド: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        logger.info("\n" + "=" * 60)
        logger.info("✅ 全ての処理が完了しました")
        logger.info(f"最終出力: {enriched_file}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ エラー: {e}")
        logger.error("=" * 60)

        import traceback
        logger.error(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()
