#!/usr/bin/env python3
"""
POSデータ変換プログラム（GPU高速化版）

このプログラムは、ワイド形式のPOSデータをロング形式に変換し、
天気情報とPB/NBフラグを追加します。

主な高速化手法:
1. CUDFを使用したGPUデータ処理
2. 非同期処理によるAPI呼び出しの並列化
3. NumbaによるJITコンパイル最適化
4. バッチ処理とメモリ効率化
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# GPU加速ライブラリ
try:
    import cudf
    import cupy as cp
    from numba import cuda, jit, prange
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("警告: CUDF/CuPyが利用できません。CPUモードで動作します。")

# 非同期HTTP
import aiohttp
import asyncio
from aiohttp import ClientSession, TCPConnector

# 並列処理
from joblib import Parallel, delayed
import dask.dataframe as dd
from dask import delayed as dask_delayed


# ログ設定
def setup_logging(debug: bool = False) -> logging.Logger:
    """ログ設定を初期化する。"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


# 定数
DATE_PAT = re.compile(r"\d{4}/\d{1,2}/\d{1,2}")
DATE_PAT_LOOSE = re.compile(r"\d{4}[/年]\d{1,2}[/月]\d{1,2}")
METRICS = ["納品数量", "売上数量", "売上金額"]
BASE_COLS = [
    "店舗",
    "フェイスくくり大分類",
    "フェイスくくり中分類",
    "フェイスくくり小分類",
    "商品名",
]
WEEKDAYS_JP = ["月", "火", "水", "木", "金", "土", "日"]


class GPUDataProcessor:
    """GPU加速データ処理クラス"""
    
    def __init__(self, use_gpu: bool = True):
        """
        Args:
            use_gpu: GPU使用フラグ
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            # GPUメモリプール設定
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=8 * 1024**3)  # 8GB制限
    
    def read_file_gpu(self, path: Path, logger: logging.Logger) -> pd.DataFrame:
        """GPUメモリを活用したファイル読み込み"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")
        
        try:
            if self.use_gpu:
                # CUDFで直接読み込み
                if path.suffix.lower() in [".xls", ".xlsx"]:
                    logger.info(f"Excelファイルを読み込み中（GPU）: {path.name}")
                    # Excelは一旦pandasで読んでからCUDFに変換
                    df_pd = pd.read_excel(path, header=None)
                    df = cudf.from_pandas(df_pd)
                else:
                    logger.info(f"CSVファイルを読み込み中（GPU）: {path.name}")
                    try:
                        df = cudf.read_csv(path, sep="\t", header=None)
                    except:
                        df = cudf.read_csv(path, header=None)
                
                logger.debug(f"GPU読み込み完了: {len(df)}行 × {len(df.columns)}列")
                # 処理後はPandasに戻す（互換性のため）
                return df.to_pandas()
            else:
                # CPU処理
                if path.suffix.lower() in [".xls", ".xlsx"]:
                    logger.info(f"Excelファイルを読み込み中（CPU）: {path.name}")
                    df = pd.read_excel(path, header=None)
                else:
                    logger.info(f"CSVファイルを読み込み中（CPU）: {path.name}")
                    try:
                        df = pd.read_csv(path, sep="\t", header=None, encoding="utf-8")
                    except:
                        df = pd.read_csv(path, header=None, encoding="utf-8")
                
                logger.debug(f"読み込み完了: {df.shape[0]}行 × {df.shape[1]}列")
                return df
                
        except Exception as e:
            raise ValueError(f"ファイルの読み込みに失敗しました: {e}")
    
    @jit(nopython=True, parallel=True)
    def _find_header_row_numba(self, col0_arr, col1_arr, col4_arr):
        """Numba JITコンパイルによる高速ヘッダー検索"""
        for idx in prange(len(col0_arr)):
            if col0_arr[idx] == "店舗" and "フェイス" in col1_arr[idx] and col4_arr[idx] == "商品名":
                return idx
        return -1
    
    def process_hierarchical_data_gpu(
        self, df: pd.DataFrame, start_row: int, col_map: dict, logger: logging.Logger
    ) -> pd.DataFrame:
        """GPU加速による階層データ処理"""
        
        if self.use_gpu:
            logger.info("GPUを使用してデータを処理中...")
            
            # CUDFデータフレームに変換
            gdf = cudf.from_pandas(df.iloc[start_row:])
            
            # 最初の総合計・合計行をスキップ（GPU上で処理）
            skip_mask = gdf.iloc[:, 0].astype(str).str.strip().isin(["総合計", "合計", ""])
            first_valid_idx = (~skip_mask).argmax()
            
            if first_valid_idx > 0:
                logger.info(f"最初の{first_valid_idx}行（総合計・合計）をスキップ")
                gdf = gdf.iloc[first_valid_idx:]
            
            # 列名を設定
            all_cols = []
            for i in range(len(gdf.columns)):
                if i < len(BASE_COLS):
                    all_cols.append(BASE_COLS[i])
                elif i in col_map:
                    all_cols.append(col_map[i])
                else:
                    all_cols.append(f"_drop_{i}")
            
            gdf.columns = all_cols[:len(gdf.columns)]
            
            # 不要な列を削除
            keep_cols = BASE_COLS + [c for c in gdf.columns if isinstance(c, tuple)]
            gdf = gdf[keep_cols]
            
            # GPU上で前方埋め処理
            for col in BASE_COLS:
                if col in gdf.columns:
                    # 空文字列をNullに変換してfillna
                    gdf[col] = gdf[col].replace("", None)
                    gdf[col] = gdf[col].fillna(method='ffill')
            
            # 商品名がない行を削除
            gdf = gdf[gdf["商品名"].notna()]
            
            # 中間集計行を削除（GPU上でマスク演算）
            mask_summary = (
                (gdf["フェイスくくり大分類"] == "合計") |
                (gdf["フェイスくくり中分類"] == "合計") |
                (gdf["フェイスくくり小分類"] == "合計")
            )
            gdf = gdf[~mask_summary]
            
            # Pandasに戻す
            result = gdf.to_pandas()
            logger.info(f"GPU処理完了: {len(result)}行")
            
        else:
            # CPU処理（元の実装）
            logger.info("CPUを使用してデータを処理中...")
            data = df.iloc[start_row:].copy()
            
            # 最初の総合計・合計行をスキップ
            skip_rows = 0
            for i in range(min(10, len(data))):
                first_col = str(data.iloc[i, 0]).strip()
                if first_col in ["総合計", "合計", ""]:
                    skip_rows += 1
                else:
                    break
            
            if skip_rows > 0:
                logger.info(f"最初の{skip_rows}行（総合計・合計）をスキップ")
                data = data.iloc[skip_rows:]
            
            # 列名を設定
            all_cols = []
            for i in range(len(data.columns)):
                if i < len(BASE_COLS):
                    all_cols.append(BASE_COLS[i])
                elif i in col_map:
                    all_cols.append(col_map[i])
                else:
                    all_cols.append(f"_drop_{i}")
            
            data.columns = all_cols[:len(data.columns)]
            
            # 不要な列を削除
            keep_cols = BASE_COLS + [c for c in data.columns if isinstance(c, tuple)]
            data = data[keep_cols]
            
            # 階層データの処理
            for col in BASE_COLS:
                if col in data.columns:
                    data[col] = data[col].replace("", pd.NA)
                    data[col] = data[col].ffill()
            
            # 商品名がない行を削除
            data = data[data["商品名"].notna()]
            
            # 中間集計行を削除
            mask_summary = (
                (data["フェイスくくり大分類"] == "合計") |
                (data["フェイスくくり中分類"] == "合計") |
                (data["フェイスくくり小分類"] == "合計")
            )
            data = data[~mask_summary]
            
            result = data
            logger.info(f"CPU処理完了: {len(result)}行")
        
        return result
    
    def convert_to_long_format_gpu(self, data: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
        """GPU加速によるワイド形式からロング形式への変換"""
        
        logger.info("データをロング形式に変換中...")
        
        if self.use_gpu:
            # CUDFで処理
            gdf = cudf.from_pandas(data)
            
            # ID列と値列を分離
            id_cols = BASE_COLS
            value_cols = [c for c in gdf.columns if isinstance(c, tuple)]
            
            # GPU上でmelt操作
            melted = gdf.melt(
                id_vars=id_cols, 
                value_vars=value_cols, 
                var_name="date_metric", 
                value_name="値"
            )
            
            # タプルを分解（GPU上で処理）
            # date_metricはタプルなので、文字列に変換してから処理
            date_metric_str = melted["date_metric"].astype(str)
            melted["日付"] = date_metric_str.str.extract(r"'([^']+)'", expand=False)
            melted["指標"] = date_metric_str.str.extract(r"', '([^']+)'", expand=False)
            melted = melted.drop("date_metric", axis=1)
            
            # 欠損値を削除
            melted = melted.dropna(subset=["値"])
            
            # GPU上でpivot操作
            result = melted.pivot_table(
                index=id_cols + ["日付"], 
                columns="指標", 
                values="値", 
                aggfunc="first"
            ).reset_index()
            
            # Pandasに戻す
            result = result.to_pandas()
            
        else:
            # CPU処理（元の実装）
            id_cols = BASE_COLS
            value_cols = [c for c in data.columns if isinstance(c, tuple)]
            
            melted = data.melt(
                id_vars=id_cols, 
                value_vars=value_cols, 
                var_name="date_metric", 
                value_name="値"
            )
            
            melted[["日付", "指標"]] = pd.DataFrame(
                melted["date_metric"].tolist(), index=melted.index
            )
            melted = melted.drop("date_metric", axis=1)
            
            melted = melted.dropna(subset=["値"])
            
            result = melted.pivot_table(
                index=id_cols + ["日付"], 
                columns="指標", 
                values="値", 
                aggfunc="first"
            ).reset_index()
        
        # 曜日列を追加（ベクトル化処理）
        result["曜日"] = pd.to_datetime(result["日付"]).dt.day_name().map({
            "Monday": "月", "Tuesday": "火", "Wednesday": "水",
            "Thursday": "木", "Friday": "金", "Saturday": "土", "Sunday": "日"
        })
        
        # PB/NBフラグを追加（ベクトル化処理）
        result["PB/NBフラグ"] = np.where(
            result["商品名"].str.contains("◎", na=False), "PB", "NB"
        )
        
        # 列の順序を整理
        result = result[BASE_COLS + ["日付", "曜日", "PB/NBフラグ"] + METRICS]
        
        # 数値型に変換
        for metric in METRICS:
            result[metric] = pd.to_numeric(result[metric], errors="coerce")
        
        logger.info(f"変換完了: {len(result)}行")
        
        # PB商品の統計を表示
        pb_count = (result["PB/NBフラグ"] == "PB").sum()
        nb_count = (result["PB/NBフラグ"] == "NB").sum()
        logger.info(f"  PB商品: {pb_count:,}行")
        logger.info(f"  NB商品: {nb_count:,}行")
        
        return result


class AsyncWeatherAPI:
    """非同期天気情報API"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.semaphore = asyncio.Semaphore(10)  # 同時接続数制限
        
    async def fetch_weather_async(
        self, session: ClientSession, location_data: tuple, date_str: str
    ) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        非同期で天気情報を取得

        Args:
            location_data: (緯度, 経度) のタプル、または住所文字列
            date_str: 日付文字列 "YYYY/MM/DD"

        Returns:
            (天気, 最高気温, 最低気温) のタプル
        """

        async with self.semaphore:
            try:
                # 日付をフォーマット
                date_obj = datetime.strptime(date_str, "%Y/%m/%d")
                date_formatted = date_obj.strftime("%Y-%m-%d")

                # 緯度経度を取得
                if isinstance(location_data, tuple) and len(location_data) == 2:
                    # 緯度経度が直接提供されている場合
                    lat, lon = location_data
                    self.logger.debug(f"緯度経度を使用: ({lat}, {lon})")

                else:
                    # 住所から geocoding（後方互換性）
                    location = str(location_data)

                    # 住所から簡易的な地域名を抽出
                    simplified_location = location
                    if "東京都" in location:
                        simplified_location = "東京"
                    elif "茨城県" in location:
                        simplified_location = "つくば"
                    elif "千葉県" in location:
                        simplified_location = "千葉"
                    elif "神奈川県" in location:
                        simplified_location = "横浜"

                    # Geocoding（非同期）
                    geocoding_url = "https://nominatim.openstreetmap.org/search"
                    geocoding_params = {
                        "q": simplified_location,
                        "format": "json",
                        "limit": 1,
                        "countrycodes": "jp",
                    }

                    headers = {
                        "User-Agent": "POSDataConverter/2.0",
                        "Accept": "application/json",
                    }

                    async with session.get(
                        geocoding_url, params=geocoding_params, headers=headers, timeout=10
                    ) as geo_response:
                        if geo_response.status != 200:
                            return None, None, None

                        geo_data = await geo_response.json()
                        if not geo_data:
                            return None, None, None

                        lat = float(geo_data[0]["lat"])
                        lon = float(geo_data[0]["lon"])

                # Weather API（Open-Meteo Archive API - JMAモデル使用）
                weather_url = "https://archive-api.open-meteo.com/v1/archive"
                weather_params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": date_formatted,
                    "end_date": date_formatted,
                    "daily": "weathercode,temperature_2m_max,temperature_2m_min",
                    "timezone": "Asia/Tokyo",
                }

                async with session.get(
                    weather_url, params=weather_params, timeout=10
                ) as weather_response:
                    if weather_response.status != 200:
                        return None, None, None

                    weather_data = await weather_response.json()

                    if "daily" in weather_data:
                        daily = weather_data["daily"]

                        weather_code = daily["weathercode"][0] if daily["weathercode"] else None
                        weather = self._convert_weather_code(weather_code)

                        max_temp = daily["temperature_2m_max"][0] if daily["temperature_2m_max"] else None
                        min_temp = daily["temperature_2m_min"][0] if daily["temperature_2m_min"] else None

                        self.logger.debug(
                            f"天気取得成功: {date_str} ({lat:.4f}, {lon:.4f}) - {weather}, "
                            f"最高{max_temp}℃, 最低{min_temp}℃"
                        )

                        return weather, max_temp, min_temp

                return None, None, None

            except asyncio.TimeoutError:
                self.logger.error(f"タイムアウト: {location_data}, {date_str}")
                return None, None, None
            except Exception as e:
                self.logger.error(f"エラー: {location_data}, {date_str} - {e}")
                return None, None, None
    
    def _convert_weather_code(self, code: Optional[int]) -> str:
        """天気コードを日本語に変換"""
        if code is None:
            return "不明"
        
        weather_map = {
            0: "晴れ", 1: "晴れ", 2: "薄曇り", 3: "曇り",
            45: "霧", 48: "霧",
            51: "小雨", 53: "雨", 55: "雨",
            61: "小雨", 63: "雨", 65: "大雨",
            71: "小雪", 73: "雪", 75: "大雪",
            80: "にわか雨", 81: "にわか雨", 82: "激しい雨",
            95: "雷雨",
        }
        
        return weather_map.get(code, "不明")
    
    async def fetch_weather_batch(
        self, locations_dates: List[Tuple[any, str]]
    ) -> Dict[str, Tuple[Optional[str], Optional[float], Optional[float]]]:
        """
        バッチで天気情報を取得

        Args:
            locations_dates: [(location_data, date_str), ...] のリスト
                            location_data は (緯度, 経度) または 住所文字列

        Returns:
            キャッシュキー -> (天気, 最高気温, 最低気温) の辞書
        """

        # TCPコネクションプール設定
        connector = TCPConnector(limit=100, limit_per_host=30)

        async with ClientSession(connector=connector) as session:
            tasks = []
            for location_data, date_str in locations_dates:
                task = self.fetch_weather_async(session, location_data, date_str)
                tasks.append(task)

            # 非同期で並列実行
            results = await asyncio.gather(*tasks)

        # 結果を辞書形式で返す
        weather_dict = {}
        for (location_data, date_str), result in zip(locations_dates, results):
            # キャッシュキーを生成
            if isinstance(location_data, tuple):
                cache_key = f"{location_data[0]:.4f}_{location_data[1]:.4f}_{date_str}"
            else:
                cache_key = f"{location_data}_{date_str}"

            weather_dict[cache_key] = result

        return weather_dict


def add_weather_info_parallel(
    result: pd.DataFrame,
    store_locations: Dict[str, any],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    並列処理による天気情報の追加

    Args:
        result: データフレーム
        store_locations: 店舗名 -> (緯度, 経度) または 住所 のマッピング
        logger: ロガー

    Returns:
        天気情報が追加されたデータフレーム
    """

    if not store_locations:
        logger.info("店舗情報がないため、天気情報の追加をスキップします")
        result["天気"] = "不明"
        result["最高気温"] = pd.NA
        result["最低気温"] = pd.NA
        return result

    logger.info("天気情報を並列取得中...")

    # ユニークな店舗と日付の組み合わせを取得
    unique_combinations = result[["店舗", "日付"]].drop_duplicates()
    total = len(unique_combinations)
    logger.info(f"処理対象: {total}件の店舗×日付の組み合わせ")

    # 店舗名を緯度経度または住所にマッピング
    locations_dates = []
    store_to_location_map = {}  # キャッシュ用

    for store, date in unique_combinations.values:
        # 店舗名のクリーニング（コード除去）
        clean_store = store.split(":")[-1].strip() if ":" in store else store

        # キャッシュから取得
        if clean_store in store_to_location_map:
            location_data = store_to_location_map[clean_store]
        else:
            # 店舗情報を検索
            location_data = None

            # 完全一致
            if clean_store in store_locations:
                location_data = store_locations[clean_store]
            elif clean_store + "店" in store_locations:
                location_data = store_locations[clean_store + "店"]
            else:
                # 部分一致で検索
                for map_store, loc_data in store_locations.items():
                    # 店舗名の正規化（スペース、全角半角の違いを吸収）
                    map_store_normalized = map_store.replace(" ", "").replace("　", "").replace("店", "")
                    clean_store_normalized = clean_store.replace(" ", "").replace("　", "").replace("店", "")

                    if clean_store_normalized in map_store_normalized or map_store_normalized in clean_store_normalized:
                        location_data = loc_data
                        logger.debug(f"部分一致: {clean_store} -> {map_store}")
                        break

            # キャッシュに保存
            store_to_location_map[clean_store] = location_data

        if location_data:
            locations_dates.append((location_data, date))
        else:
            logger.warning(f"店舗情報が見つかりません: {clean_store}")

    logger.info(f"天気取得対象: {len(locations_dates)}件（店舗マッチング: {len(store_to_location_map)}店舗）")

    # 非同期処理で天気情報を取得
    weather_api = AsyncWeatherAPI(logger)

    # イベントループで実行
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    weather_cache = loop.run_until_complete(
        weather_api.fetch_weather_batch(locations_dates)
    )
    loop.close()

    logger.info(f"天気情報取得完了: {len(weather_cache)}件")

    # データフレームに天気情報を追加
    def get_weather_for_row(row):
        store = row["店舗"]
        date = row["日付"]

        # 店舗名のクリーニング
        clean_store = store.split(":")[-1].strip() if ":" in store else store

        # キャッシュから位置情報を取得
        location_data = store_to_location_map.get(clean_store)

        if location_data:
            # キャッシュキーを生成
            if isinstance(location_data, tuple):
                cache_key = f"{location_data[0]:.4f}_{location_data[1]:.4f}_{date}"
            else:
                cache_key = f"{location_data}_{date}"

            return weather_cache.get(cache_key, ("不明", None, None))
        else:
            return ("不明", None, None)

    # ベクトル化処理
    weather_info = result.apply(get_weather_for_row, axis=1, result_type="expand")
    result["天気"] = weather_info[0]
    result["最高気温"] = weather_info[1]
    result["最低気温"] = weather_info[2]

    logger.info("✓ 天気情報の追加が完了しました")
    return result


# 既存の関数を再利用（変更なし）
def find_header_row(df: pd.DataFrame, logger: logging.Logger) -> int:
    """データフレームからヘッダー行を検出する。"""
    logger.info("ヘッダー行を検索中...")
    
    for idx in range(min(30, len(df))):
        row = df.iloc[idx]
        
        if len(row) >= 5:
            col0 = str(row.iloc[0]).strip()
            col1 = str(row.iloc[1]).strip()
            col4 = str(row.iloc[4]).strip()
            
            if col0 == "店舗" and "フェイス" in col1 and col4 == "商品名":
                logger.info(f"✓ ヘッダー行を発見: 行{idx}")
                logger.debug(f"  内容: {row[:10].tolist()}")
                return idx
    
    raise ValueError("ヘッダー行が見つかりません。ファイル形式を確認してください。")


def find_date_row(
    df: pd.DataFrame, header_idx: int, logger: logging.Logger
) -> tuple[int, list[tuple[int, str]]]:
    """日付が記載された行を検出し、日付と列位置のマッピングを返す。"""
    logger.info("日付行を検索中...")
    
    for offset in [-2, -1]:
        row_idx = header_idx + offset
        if 0 <= row_idx < len(df):
            row = df.iloc[row_idx]
            dates = []
            
            col_idx = 5
            while col_idx < len(row):
                val = row.iloc[col_idx]
                if pd.notna(val):
                    val_str = str(val).strip()
                    match = re.search(r"(\d{4}/\d{1,2}/\d{1,2})", val_str)
                    if match:
                        dates.append((col_idx, match.group(1)))
                        logger.debug(f"  列{col_idx}: {match.group(1)}")
                col_idx += 3
            
            if dates:
                logger.info(f"日付行を発見: 行{row_idx} ({len(dates)}個の日付)")
                return row_idx, dates
    
    raise ValueError("日付が見つかりません。ファイル形式を確認してください。")


def build_column_mapping(
    dates: list[tuple[int, str]], logger: logging.Logger
) -> dict[int, tuple[str, str]]:
    """日付列から数値列へのマッピングを作成する。"""
    col_map = {}
    
    for col_idx, date_str in dates:
        for offset, metric in enumerate(METRICS):
            col_map[col_idx + offset] = (date_str, metric)
    
    logger.info(f"列マッピング完了: {len(col_map)}列")
    return col_map


def load_store_locations(
    file_path: Optional[str], logger: logging.Logger
) -> Dict[str, tuple]:
    """
    店舗名と位置情報のマッピングファイルを読み込む。

    Returns:
        店舗名 -> (緯度, 経度) のマッピング辞書
        緯度経度がない場合は 店舗名 -> 住所 のマッピング
    """
    if not file_path:
        return {}

    try:
        logger.info(f"店舗情報を読み込み中: {file_path}")
        df = pd.read_csv(file_path, header=0, encoding="utf-8")

        # 列名を確認
        columns = [col.strip() for col in df.columns]
        logger.debug(f"列名: {columns}")

        # 緯度経度がある場合
        if len(columns) >= 4 and '緯度' in columns and '経度' in columns:
            store_col = columns[0]
            lat_col = '緯度'
            lon_col = '経度'

            # 店舗名 -> (緯度, 経度) のマッピング
            store_map = {}
            for _, row in df.iterrows():
                store_name = str(row[store_col]).strip()
                lat = row[lat_col]
                lon = row[lon_col]

                if pd.notna(lat) and pd.notna(lon):
                    store_map[store_name] = (float(lat), float(lon))

            logger.info(f"✓ {len(store_map)}店舗の緯度経度情報を読み込みました")
            return store_map

        # 住所のみの場合（後方互換性）
        elif len(columns) >= 2:
            store_col = columns[0]
            address_col = columns[1]
            store_map = dict(zip(df[store_col], df[address_col]))
            logger.info(f"✓ {len(store_map)}店舗の住所情報を読み込みました")
            return store_map

        else:
            logger.warning("店舗情報ファイルの形式が正しくありません（2列以上必要）")
            return {}

    except Exception as e:
        logger.error(f"店舗情報の読み込みに失敗しました: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}


def main():
    """メイン処理関数。"""
    parser = argparse.ArgumentParser(
        description="POSデータをワイド形式からロング形式に変換する（GPU高速化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python wide_to_long_gpu.py input.xlsx output.csv
  python wide_to_long_gpu.py --no-gpu input.xlsx output.csv  # CPU処理
  python wide_to_long_gpu.py --store-locations stores.csv input.xlsx output.csv

注意事項:
  - GPU版はCUDA対応GPUとCUDF/CuPyのインストールが必要です
  - 大規模データの処理が大幅に高速化されます
  - 天気情報は非同期処理により並列取得されます
        """,
    )
    parser.add_argument("input", help="入力ファイル（Excel または CSV）")
    parser.add_argument("output", help="出力CSVファイル")
    parser.add_argument(
        "--debug", action="store_true", help="デバッグモード（詳細ログを出力）"
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="GPU使用を無効化（CPU処理）"
    )
    parser.add_argument(
        "--store-locations",
        help="店舗情報ファイル（CSV形式、1列目:店舗名、2列目:住所）",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="天気情報の取得をスキップ（APIエラー時の回避用）",
    )
    parser.add_argument(
        "--test-weather", action="store_true", help="テスト用のダミー天気情報を使用"
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.debug)
    
    try:
        logger.info("=" * 60)
        logger.info("POSデータ変換を開始（GPU高速化版）")
        logger.info("=" * 60)
        
        # GPU使用状況を表示
        use_gpu = not args.no_gpu
        if use_gpu and GPU_AVAILABLE:
            logger.info("✓ GPU加速: 有効")
            # GPU情報を表示
            if cuda.is_available():
                logger.info(f"  GPU: {cuda.get_current_device().name}")
        else:
            logger.info("✗ GPU加速: 無効（CPU処理）")
        
        # プロセッサインスタンス作成
        processor = GPUDataProcessor(use_gpu=use_gpu)
        
        # ファイル読み込み（GPU最適化）
        df = processor.read_file_gpu(args.input, logger)
        
        # ヘッダー行を探す
        header_idx = find_header_row(df, logger)
        
        # 日付行を探す
        date_idx, dates = find_date_row(df, header_idx, logger)
        
        # 列マッピングを作成
        col_map = build_column_mapping(dates, logger)
        
        # データを処理（GPU最適化）
        data = processor.process_hierarchical_data_gpu(df, header_idx + 1, col_map, logger)
        
        # ロング形式に変換（GPU最適化）
        result = processor.convert_to_long_format_gpu(data, logger)
        
        # 店舗情報を読み込み
        store_locations = load_store_locations(args.store_locations, logger)
        
        # 天気情報を追加（並列処理）
        if args.skip_weather:
            logger.info("--skip-weather オプションにより天気情報の取得をスキップします")
            result["天気"] = "不明"
            result["最高気温"] = pd.NA
            result["最低気温"] = pd.NA
        elif args.test_weather:
            logger.info("--test-weather オプションによりダミー天気情報を使用します")
            import random
            
            weather_options = ["晴れ", "曇り", "雨", "小雨"]
            result["天気"] = result.apply(
                lambda x: random.choice(weather_options), axis=1
            )
            result["最高気温"] = result.apply(
                lambda x: round(random.uniform(20, 35), 1), axis=1
            )
            result["最低気温"] = result.apply(
                lambda x: round(random.uniform(15, 25), 1), axis=1
            )
        else:
            result = add_weather_info_parallel(result, store_locations, logger)
        
        # 最終的な列の順序
        final_cols = (
            BASE_COLS
            + ["日付", "曜日", "PB/NBフラグ", "天気", "最高気温", "最低気温"]
            + METRICS
        )
        result = result[final_cols]
        
        # 保存（高速化）
        logger.info(f"保存中: {args.output}")
        result.to_csv(args.output, index=False, encoding="utf-8")
        
        # 結果のサマリーを表示
        logger.info("=" * 60)
        logger.info(f"✅ 成功: {len(result):,}行を変換しました")
        logger.info(f"   期間: {result['日付'].min()} ～ {result['日付'].max()}")
        logger.info(f"   店舗数: {result['店舗'].nunique()}")
        logger.info(f"   商品数: {result['商品名'].nunique()}")
        
        # PB/NB統計
        pb_products = result[result["PB/NBフラグ"] == "PB"]["商品名"].nunique()
        nb_products = result[result["PB/NBフラグ"] == "NB"]["商品名"].nunique()
        logger.info(f"   PB商品数: {pb_products}")
        logger.info(f"   NB商品数: {nb_products}")
        
        # 天気情報の統計
        if args.store_locations and not args.skip_weather:
            weather_counts = result["天気"].value_counts()
            logger.info("   天気の内訳:")
            for weather, count in weather_counts.items():
                logger.info(f"     {weather}: {count:,}行")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ エラー: {e}")
        logger.error("=" * 60)
        
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        
        sys.exit(1)


if __name__ == "__main__":
    main()