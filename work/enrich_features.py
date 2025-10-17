#!/usr/bin/env python3
"""
特徴量付与スクリプト

CSVデータに以下の特徴量を追加:
- カレンダー特徴（曜日、休日、連休など）
- 気象特徴（気温、降水量、天気）
- 季節変動指数・客数指数（前年データから計算）
- 時系列特徴（ラグ、移動平均、変化量、変化率）
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 天気API
import aiohttp
from aiohttp import ClientSession, TCPConnector
import urllib.request


def setup_logging(debug: bool = False) -> logging.Logger:
    """ログ設定"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_japan_holidays(logger: logging.Logger) -> set:
    """
    内閣府の公式CSVから日本の祝日を読み込む

    Returns:
        祝日の日付セット (datetime.date)
    """
    url = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"
    cache_path = Path(__file__).parent / ".holidays_cache.csv"

    # キャッシュの有効期限（30日）
    cache_valid = False
    if cache_path.exists():
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if cache_age < 30 * 24 * 3600:  # 30日以内
            cache_valid = True
            logger.debug("祝日キャッシュを使用します")

    # キャッシュがない、または古い場合はダウンロード
    if not cache_valid:
        try:
            logger.info("内閣府の祝日CSVをダウンロード中...")
            urllib.request.urlretrieve(url, cache_path)
            logger.info("✓ 祝日データをダウンロードしました")
        except Exception as e:
            logger.warning(f"祝日データのダウンロードに失敗: {e}")
            if not cache_path.exists():
                logger.warning("祝日データが利用できません")
                return set()

    # CSVを読み込み
    try:
        # Shift-JISでエンコードされているため、エンコーディングを指定
        df = pd.read_csv(cache_path, encoding='shift_jis')

        # 最初の列が日付（例: 国民の祝日・休日月日）
        date_col = df.columns[0]

        # 日付をdatetime.dateに変換
        holidays = set()
        for date_str in df[date_col]:
            try:
                # "YYYY/M/D" または "YYYY-M-D" 形式に対応
                if '/' in str(date_str):
                    date_obj = datetime.strptime(str(date_str), "%Y/%m/%d").date()
                else:
                    date_obj = datetime.strptime(str(date_str), "%Y-%m-%d").date()
                holidays.add(date_obj)
            except:
                continue

        logger.info(f"✓ {len(holidays)}件の祝日を読み込みました")
        return holidays

    except Exception as e:
        logger.error(f"祝日CSVの読み込みエラー: {e}")
        return set()


# ============================================================
# A. カレンダー特徴
# ============================================================

def create_calendar_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    カレンダー特徴を作成

    追加される列:
    - 月、日、週番号、曜日、年内日数
    - 休日フラグ（0=平日、1=休日）
    - 休日タイプ（0=平日、1=土曜、2=日曜、3=祝日）
    - 連休関連フラグ
    - 大型連休フラグ
    - 月内位置フラグ
    - 学校休みフラグ
    """
    logger.info("カレンダー特徴を作成中...")

    df = df.copy()
    df['日付'] = pd.to_datetime(df['日付'])

    # 基本時間特徴
    df['年'] = df['日付'].dt.year
    df['月'] = df['日付'].dt.month
    df['日'] = df['日付'].dt.day
    df['曜日'] = df['日付'].dt.dayofweek  # 0=月曜
    df['週番号'] = df['日付'].dt.isocalendar().week
    df['年内日数'] = df['日付'].dt.dayofyear

    # 祝日判定（内閣府公式CSV使用）
    holidays = load_japan_holidays(logger)
    if holidays:
        df['祝日フラグ'] = df['日付'].apply(lambda x: int(x.date() in holidays))
    else:
        logger.warning("祝日データが利用できないため、祝日フラグは0に設定されます")
        df['祝日フラグ'] = 0

    df['土曜フラグ'] = (df['曜日'] == 5).astype(int)
    df['日曜フラグ'] = (df['曜日'] == 6).astype(int)
    df['週末フラグ'] = ((df['曜日'] >= 5)).astype(int)
    df['平日フラグ'] = ((df['祝日フラグ'] == 0) & (df['週末フラグ'] == 0)).astype(int)

    # 休日フラグ（0=平日、1=休日）※ユーザー要件
    df['休日フラグ'] = ((df['祝日フラグ'] == 1) | (df['週末フラグ'] == 1)).astype(int)

    # 休日タイプ
    def get_holiday_type(row):
        if row['祝日フラグ'] == 1:
            return 3
        elif row['土曜フラグ'] == 1:
            return 1
        elif row['日曜フラグ'] == 1:
            return 2
        else:
            return 0

    df['休日タイプ'] = df.apply(get_holiday_type, axis=1)

    # 休日前後
    df['休日前日'] = df['休日フラグ'].shift(-1).fillna(0).astype(int)
    df['休日翌日'] = df['休日フラグ'].shift(1).fillna(0).astype(int)

    # 連休判定
    def count_consecutive_holidays(date, holiday_series):
        """連続休日数をカウント"""
        idx = holiday_series[holiday_series.index == date].index[0]
        count = 0

        # 前方カウント
        i = idx
        while i < len(holiday_series) and holiday_series.iloc[i] == 1:
            count += 1
            i += 1

        # 後方カウント
        i = idx - 1
        while i >= 0 and holiday_series.iloc[i] == 1:
            count += 1
            i -= 1

        return count

    df['連休日数'] = df.apply(
        lambda row: count_consecutive_holidays(row.name, df['休日フラグ']), axis=1
    )
    df['連休フラグ'] = (df['連休日数'] >= 3).astype(int)

    # 大型連休
    df['GW'] = (((df['月'] == 4) & (df['日'] >= 29)) |
                ((df['月'] == 5) & (df['日'] <= 7))).astype(int)
    df['盆休み'] = ((df['月'] == 8) & (df['日'] >= 11) & (df['日'] <= 16)).astype(int)
    df['年末年始'] = (((df['月'] == 12) & (df['日'] >= 28)) |
                      ((df['月'] == 1) & (df['日'] <= 3))).astype(int)

    # 月内位置
    df['給料日'] = (df['日'] == 25).astype(int)
    df['給料日直後'] = ((df['日'] >= 25) & (df['日'] <= 27)).astype(int)
    df['月初'] = (df['日'] <= 5).astype(int)
    df['月末'] = (df['日'] >= 25).astype(int)
    df['月中旬'] = ((df['日'] >= 10) & (df['日'] <= 20)).astype(int)

    # 学校休み
    df['夏休み'] = (((df['月'] == 7) & (df['日'] >= 21)) | (df['月'] == 8)).astype(int)
    df['冬休み'] = (((df['月'] == 12) & (df['日'] >= 25)) |
                    ((df['月'] == 1) & (df['日'] <= 7))).astype(int)
    df['春休み'] = (((df['月'] == 3) & (df['日'] >= 25)) |
                    ((df['月'] == 4) & (df['日'] <= 7))).astype(int)
    df['学校休み'] = (df['夏休み'] | df['冬休み'] | df['春休み']).astype(int)

    # 四半期・季節
    df['四半期'] = (df['月'] - 1) // 3 + 1

    def get_season(month):
        if month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        elif month in [9, 10, 11]:
            return 3
        else:
            return 4

    df['季節'] = df['月'].apply(get_season)

    logger.info(f"✓ カレンダー特徴を作成しました（{len(df)}行）")

    return df


# ============================================================
# B. 気象特徴（API取得）
# ============================================================

class WeatherAPI:
    """Open-Meteo Archive API（天気取得）"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.semaphore = asyncio.Semaphore(10)

    async def fetch_weather_async(
        self, session: ClientSession, lat: float, lon: float, date_str: str
    ):
        """非同期で天気情報を取得"""

        async with self.semaphore:
            try:
                # 日付フォーマット変換
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                date_formatted = date_obj.strftime("%Y-%m-%d")

                # Open-Meteo Archive API
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": date_formatted,
                    "end_date": date_formatted,
                    "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "timezone": "Asia/Tokyo",
                }

                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        return None, None, None, None

                    data = await response.json()

                    if "daily" in data:
                        daily = data["daily"]

                        weather_code = daily["weathercode"][0] if daily["weathercode"] else None
                        weather = self._convert_weather_code(weather_code)
                        max_temp = daily["temperature_2m_max"][0]
                        min_temp = daily["temperature_2m_min"][0]
                        precipitation = daily["precipitation_sum"][0]

                        return weather, max_temp, min_temp, precipitation

                return None, None, None, None

            except Exception as e:
                self.logger.error(f"天気取得エラー ({date_str}): {e}")
                return None, None, None, None

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
        self, lat: float, lon: float, dates: List[str]
    ) -> Dict[str, tuple]:
        """バッチで天気情報を取得"""

        connector = TCPConnector(limit=100, limit_per_host=30)

        async with ClientSession(connector=connector) as session:
            tasks = []
            for date_str in dates:
                task = self.fetch_weather_async(session, lat, lon, date_str)
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        weather_dict = {}
        for date_str, result in zip(dates, results):
            weather_dict[date_str] = result

        return weather_dict


def add_weather_features(
    df: pd.DataFrame,
    store_locations: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    気象特徴を追加

    追加される列:
    - 天気、最高気温、最低気温、降水量
    - 平均気温、気温差
    - 気温閾値フラグ
    - 降水フラグ
    """
    logger.info("気象特徴を取得中...")

    df = df.copy()
    df['日付'] = pd.to_datetime(df['日付'])

    # 店舗の緯度経度を取得
    if '店舗' not in df.columns:
        logger.warning("店舗列がありません。気象情報をスキップします")
        return df

    # ユニークな店舗を取得
    unique_stores = df['店舗'].unique()

    # 各店舗の気象情報を取得
    for store in unique_stores:
        # 店舗の緯度経度を検索
        store_clean = store.split(":")[-1].strip() if ":" in store else store

        lat, lon = None, None
        for _, row in store_locations.iterrows():
            if store_clean in row.iloc[0]:
                lat = row.iloc[2]  # 緯度
                lon = row.iloc[3]  # 経度
                break

        if lat is None or lon is None:
            logger.warning(f"店舗 {store} の緯度経度が見つかりません")
            continue

        # 該当店舗のデータ
        store_mask = df['店舗'] == store
        dates = df[store_mask]['日付'].dt.strftime('%Y-%m-%d').unique().tolist()

        logger.info(f"  店舗 {store}: {len(dates)}日分の天気を取得中...")

        # 天気API呼び出し
        weather_api = WeatherAPI(logger)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        weather_cache = loop.run_until_complete(
            weather_api.fetch_weather_batch(lat, lon, dates)
        )
        loop.close()

        # データフレームに追加
        for date_str, (weather, max_temp, min_temp, precip) in weather_cache.items():
            mask = (df['店舗'] == store) & (df['日付'] == pd.to_datetime(date_str))
            df.loc[mask, '天気'] = weather
            df.loc[mask, '最高気温'] = max_temp
            df.loc[mask, '最低気温'] = min_temp
            df.loc[mask, '降水量'] = precip

    # 派生特徴
    df['平均気温'] = (df['最高気温'] + df['最低気温']) / 2
    df['気温差'] = df['最高気温'] - df['最低気温']

    # 閾値フラグ
    df['猛暑日'] = (df['最高気温'] >= 35).astype(int)
    df['真夏日'] = (df['最高気温'] >= 30).astype(int)
    df['夏日'] = (df['最高気温'] >= 25).astype(int)
    df['快適温度'] = ((df['平均気温'] >= 18) & (df['平均気温'] <= 25)).astype(int)

    # 降水フラグ
    df['降雨フラグ'] = (df['降水量'] > 1).astype(int)
    df['強雨'] = (df['降水量'] > 10).astype(int)

    logger.info("✓ 気象特徴を追加しました")

    return df


# ============================================================
# C. 季節変動指数・客数指数
# ============================================================

def calculate_seasonal_customer_index(
    df_current: pd.DataFrame,
    df_past_year: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    季節変動指数と客数指数を計算

    季節変動指数 = 前年同月の平均売上 / 前年の年間平均売上
    客数指数 = 前年同月の平均客数 / 前年の年間平均客数
    """
    logger.info("季節変動指数・客数指数を計算中...")

    df = df_current.copy()
    df['日付'] = pd.to_datetime(df['日付'])
    df['月'] = df['日付'].dt.month

    # 前年データの準備
    df_past = df_past_year.copy()
    df_past['日付'] = pd.to_datetime(df_past['日付'])
    df_past['月'] = df_past['日付'].dt.month

    # 店舗ごとに計算
    if '店舗' in df.columns:
        for store in df['店舗'].unique():
            past_store = df_past[df_past['店舗'] == store]

            if len(past_store) == 0:
                logger.warning(f"店舗 {store} の前年データがありません")
                continue

            # 年間平均
            yearly_avg_sales = past_store['売上'].mean()
            yearly_avg_customers = past_store['客数'].mean()

            # 月次平均
            monthly_avg_sales = past_store.groupby('月')['売上'].mean()
            monthly_avg_customers = past_store.groupby('月')['客数'].mean()

            # 季節変動指数
            seasonal_index = (monthly_avg_sales / yearly_avg_sales).to_dict()

            # 客数指数
            customer_index = (monthly_avg_customers / yearly_avg_customers).to_dict()

            # 現在のデータに付与
            store_mask = df['店舗'] == store
            df.loc[store_mask, '季節変動指数'] = df.loc[store_mask, '月'].map(seasonal_index)
            df.loc[store_mask, '客数指数'] = df.loc[store_mask, '月'].map(customer_index)

    else:
        # 店舗列がない場合は全体で計算
        yearly_avg_sales = df_past['売上'].mean()
        yearly_avg_customers = df_past['客数'].mean()

        monthly_avg_sales = df_past.groupby('月')['売上'].mean()
        monthly_avg_customers = df_past.groupby('月')['客数'].mean()

        seasonal_index = (monthly_avg_sales / yearly_avg_sales).to_dict()
        customer_index = (monthly_avg_customers / yearly_avg_customers).to_dict()

        df['季節変動指数'] = df['月'].map(seasonal_index)
        df['客数指数'] = df['月'].map(customer_index)

    logger.info("✓ 季節変動指数・客数指数を計算しました")

    return df


# ============================================================
# D. 時系列特徴
# ============================================================

def add_time_series_features(
    df: pd.DataFrame,
    target_columns: List[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    時系列特徴を追加

    追加される特徴:
    - ラグ（1,7,14日前）
    - 移動平均（3,7,14日）
    - 変化量
    - 変化率
    """
    logger.info("時系列特徴を作成中...")

    df = df.copy()

    for col in target_columns:
        if col not in df.columns:
            continue

        logger.info(f"  {col} の時系列特徴を作成中...")

        # ラグ
        for lag in [1, 7, 14]:
            df[f'{col}_t-{lag}'] = df[col].shift(lag)

        # 移動平均
        for window in [3, 7, 14]:
            df[f'{col}_MA{window}'] = df[col].rolling(window).mean()

        # 変化量
        for lag in [1, 7, 14]:
            lag_col = f'{col}_t-{lag}'
            if lag_col in df.columns:
                df[f'{col}_変化量_{lag}d'] = df[col] - df[lag_col]

        # 変化率
        for lag in [1, 7, 14]:
            lag_col = f'{col}_t-{lag}'
            if lag_col in df.columns:
                df[f'{col}_変化率_{lag}d'] = (df[col] - df[lag_col]) / df[lag_col].replace(0, np.nan)

    logger.info("✓ 時系列特徴を作成しました")

    return df


# ============================================================
# メイン処理
# ============================================================

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="CSVデータに特徴量を追加",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python enrich_features.py input.csv output.csv --store-locations stores.csv --past-year-data past_year.csv
  python enrich_features.py input.csv output.csv --skip-weather
        """
    )

    parser.add_argument("input", help="入力CSVファイル")
    parser.add_argument("output", help="出力CSVファイル")
    parser.add_argument("--store-locations", help="店舗情報CSV（緯度経度）")
    parser.add_argument("--past-year-data", help="前年データCSV（季節変動指数計算用）")
    parser.add_argument("--skip-weather", action="store_true", help="天気情報の取得をスキップ")
    parser.add_argument("--skip-timeseries", action="store_true", help="時系列特徴の作成をスキップ")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()
    logger = setup_logging(args.debug)

    try:
        logger.info("=" * 60)
        logger.info("特徴量付与処理を開始")
        logger.info("=" * 60)

        # データ読み込み
        logger.info(f"入力ファイル: {args.input}")
        df = pd.read_csv(args.input, encoding='utf-8-sig')
        logger.info(f"読み込み完了: {len(df):,}行 × {len(df.columns)}列")

        # カレンダー特徴
        df = create_calendar_features(df, logger)

        # 気象特徴
        if not args.skip_weather and args.store_locations:
            store_locations = pd.read_csv(args.store_locations, encoding='utf-8-sig')
            df = add_weather_features(df, store_locations, logger)
        elif not args.skip_weather:
            logger.warning("--store-locationsが指定されていないため、天気情報をスキップします")

        # 季節変動指数・客数指数
        if args.past_year_data:
            df_past_year = pd.read_csv(args.past_year_data, encoding='utf-8-sig')
            df = calculate_seasonal_customer_index(df, df_past_year, logger)
        else:
            logger.warning("--past-year-dataが指定されていないため、季節変動指数・客数指数をスキップします")

        # 時系列特徴
        if not args.skip_timeseries:
            target_columns = []
            if '売上' in df.columns:
                target_columns.append('売上')
            if '客数' in df.columns:
                target_columns.append('客数')
            if '最高気温' in df.columns:
                target_columns.extend(['最高気温', '最低気温', '平均気温'])

            if target_columns:
                df = add_time_series_features(df, target_columns, logger)

        # 保存
        logger.info(f"保存中: {args.output}")
        df.to_csv(args.output, index=False, encoding='utf-8-sig')

        logger.info("=" * 60)
        logger.info(f"✅ 完了: {len(df):,}行 × {len(df.columns)}列")
        logger.info(f"出力ファイル: {args.output}")
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
