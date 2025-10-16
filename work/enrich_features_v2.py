#!/usr/bin/env python3
"""
特徴量付与スクリプト v2（完全版 250-300特徴量対応）

CSVデータに以下の特徴量を追加:
- カレンダー特徴（詳細な休日・連休情報）
- 気象特徴（気温、降水量、天気 + 変化量・変化率）
- 季節変動指数・客数指数（前年データから計算）
- 時系列特徴（ラグ、移動平均、変化量、変化率、トレンド）
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import urllib.request

# 天気API
import aiohttp
from aiohttp import ClientSession, TCPConnector


def setup_logging(debug: bool = False) -> logging.Logger:
    """ログ設定"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_japan_holidays(logger: logging.Logger) -> Set[datetime]:
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
# A. カレンダー特徴（詳細版）
# ============================================================

def is_holiday_or_weekend(date: datetime, holidays: Set[datetime]) -> bool:
    """休日または週末かどうか"""
    return date.date() in holidays or date.weekday() >= 5


def count_consecutive_holidays(date: datetime, holidays: Set[datetime]) -> tuple:
    """
    連続休日数をカウント

    Returns:
        (連休日数, 連休開始日, 連休終了日, 連休中の何日目か)
    """
    if not is_holiday_or_weekend(date, holidays):
        return 0, None, None, 0

    # 連休の開始日を探す
    start_date = date
    current = date - timedelta(days=1)
    while is_holiday_or_weekend(current, holidays):
        start_date = current
        current -= timedelta(days=1)

    # 連休の終了日を探す
    end_date = date
    current = date + timedelta(days=1)
    while is_holiday_or_weekend(current, holidays):
        end_date = current
        current += timedelta(days=1)

    # 連休日数
    total_days = (end_date - start_date).days + 1

    # 連休中の何日目か
    day_in_holiday = (date - start_date).days + 1

    return total_days, start_date, end_date, day_in_holiday


def create_calendar_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    カレンダー特徴を作成（詳細版）

    追加される列（約50個）:
    - 基本時間特徴
    - 休日タイプ（詳細）
    - 休日の前後関係
    - 連休情報（詳細）
    - 大型連休
    - 月内位置
    - 学校カレンダー
    - 季節・四半期
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
    df['週末フラグ'] = (df['曜日'] >= 5).astype(int)
    df['平日フラグ'] = ((df['祝日フラグ'] == 0) & (df['週末フラグ'] == 0)).astype(int)

    # 休日フラグ（0=平日、1=休日）※ユーザー要件
    df['休日フラグ'] = ((df['祝日フラグ'] == 1) | (df['週末フラグ'] == 1)).astype(int)

    # 休日タイプ
    def get_holiday_type(row):
        if row['祝日フラグ'] == 1:
            return 3  # 祝日
        elif row['土曜フラグ'] == 1:
            return 1  # 土曜
        elif row['日曜フラグ'] == 1:
            return 2  # 日曜
        else:
            return 0  # 平日

    df['休日タイプ'] = df.apply(get_holiday_type, axis=1)

    # 休日前後
    df['休日前日'] = df['休日フラグ'].shift(-1).fillna(0).astype(int)
    df['休日翌日'] = df['休日フラグ'].shift(1).fillna(0).astype(int)
    df['休日前々日'] = df['休日フラグ'].shift(-2).fillna(0).astype(int)

    # 連休判定（詳細）
    holiday_info = df['日付'].apply(lambda x: count_consecutive_holidays(x, holidays))
    df['連休日数'] = holiday_info.apply(lambda x: x[0])
    df['連休中日番号'] = holiday_info.apply(lambda x: x[3])

    df['連休フラグ'] = (df['連休日数'] >= 3).astype(int)
    df['連休初日'] = ((df['連休フラグ'] == 1) & (df['連休中日番号'] == 1)).astype(int)
    df['連休2日目'] = ((df['連休フラグ'] == 1) & (df['連休中日番号'] == 2)).astype(int)
    df['連休最終日前日'] = ((df['連休フラグ'] == 1) &
                            (df['連休中日番号'] == df['連休日数'] - 1)).astype(int)
    df['連休最終日'] = ((df['連休フラグ'] == 1) &
                        (df['連休中日番号'] == df['連休日数'])).astype(int)

    # 大型連休
    df['GW'] = (((df['月'] == 4) & (df['日'] >= 29)) |
                ((df['月'] == 5) & (df['日'] <= 7))).astype(int)
    df['GW前半'] = (((df['月'] == 4) & (df['日'] >= 29)) |
                    ((df['月'] == 5) & (df['日'] <= 2))).astype(int)
    df['GW後半'] = ((df['月'] == 5) & (df['日'] >= 3) & (df['日'] <= 7)).astype(int)
    df['盆休み'] = ((df['月'] == 8) & (df['日'] >= 11) & (df['日'] <= 16)).astype(int)
    df['年末年始'] = (((df['月'] == 12) & (df['日'] >= 28)) |
                      ((df['月'] == 1) & (df['日'] <= 3))).astype(int)
    df['シルバーウィーク'] = ((df['月'] == 9) & (df['連休日数'] >= 3)).astype(int)

    # 月内位置
    df['給料日'] = (df['日'] == 25).astype(int)
    df['給料日直後'] = ((df['日'] >= 25) & (df['日'] <= 27)).astype(int)
    df['月初'] = (df['日'] <= 5).astype(int)
    df['月初3日'] = (df['日'] <= 3).astype(int)

    # 月末（各月の最終日を計算）
    df['月末3日'] = (df['日付'].dt.is_month_end |
                     (df['日付'] + timedelta(days=1)).dt.is_month_end |
                     (df['日付'] + timedelta(days=2)).dt.is_month_end).astype(int)
    df['月末'] = (df['日'] >= 25).astype(int)
    df['月中旬'] = ((df['日'] >= 10) & (df['日'] <= 20)).astype(int)

    # 学校カレンダー
    df['夏休み'] = (((df['月'] == 7) & (df['日'] >= 21)) | (df['月'] == 8)).astype(int)
    df['冬休み'] = (((df['月'] == 12) & (df['日'] >= 25)) |
                    ((df['月'] == 1) & (df['日'] <= 7))).astype(int)
    df['春休み'] = (((df['月'] == 3) & (df['日'] >= 25)) |
                    ((df['月'] == 4) & (df['日'] <= 7))).astype(int)
    df['学校休み'] = (df['夏休み'] | df['冬休み'] | df['春休み']).astype(int)
    df['新学期'] = (((df['月'] == 4) | (df['月'] == 9)) & (df['日'] <= 7)).astype(int)

    # 四半期・季節
    df['四半期'] = (df['月'] - 1) // 3 + 1

    def get_season(month):
        if month in [3, 4, 5]:
            return 1  # 春
        elif month in [6, 7, 8]:
            return 2  # 夏
        elif month in [9, 10, 11]:
            return 3  # 秋
        else:
            return 4  # 冬

    df['季節'] = df['月'].apply(get_season)

    logger.info(f"✓ カレンダー特徴を作成しました（{len(df)}行）")

    return df


# ============================================================
# B. 気象特徴（API取得 + 詳細派生）
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
    気象特徴を追加（詳細版）

    追加される列（約80個）:
    - 基本気象データ（取得）
    - 気温派生特徴
    - 気温閾値フラグ
    - 降水フラグ
    - 気温ラグ・移動平均
    - 気温変化量・変化率
    - 気温変化フラグ
    - 降水ラグ・累積
    - 降水変化量・フラグ
    - 気温トレンド
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
        # NaN値をスキップ
        if pd.isna(store):
            continue

        # 店舗の緯度経度を検索
        store_str = str(store)
        store_clean = store_str.split(":")[-1].strip() if ":" in store_str else store_str

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

    # === 気温派生特徴 ===
    df['平均気温'] = (df['最高気温'] + df['最低気温']) / 2
    df['気温差'] = df['最高気温'] - df['最低気温']

    # === 気温閾値フラグ ===
    df['猛暑日'] = (df['最高気温'] >= 35).astype(int)
    df['真夏日'] = (df['最高気温'] >= 30).astype(int)
    df['夏日'] = (df['最高気温'] >= 25).astype(int)
    df['冬日'] = (df['最低気温'] < 0).astype(int)
    df['真冬日'] = (df['最高気温'] < 0).astype(int)
    df['快適温度'] = ((df['平均気温'] >= 18) & (df['平均気温'] <= 25)).astype(int)
    df['暑い'] = (df['平均気温'] > 28).astype(int)
    df['やや暑い'] = ((df['平均気温'] > 25) & (df['平均気温'] <= 28)).astype(int)
    df['寒い'] = (df['平均気温'] < 10).astype(int)
    df['やや寒い'] = ((df['平均気温'] >= 10) & (df['平均気温'] < 15)).astype(int)

    # === 降水フラグ ===
    df['降雨フラグ'] = (df['降水量'] > 1).astype(int)
    df['弱雨'] = ((df['降水量'] > 1) & (df['降水量'] <= 5)).astype(int)
    df['普通雨'] = ((df['降水量'] > 5) & (df['降水量'] <= 10)).astype(int)
    df['強雨'] = ((df['降水量'] > 10) & (df['降水量'] <= 30)).astype(int)
    df['豪雨'] = (df['降水量'] > 30).astype(int)

    # === 気温ラグ ===
    for col in ['最高気温', '最低気温', '平均気温']:
        df[f'{col}_t-1'] = df[col].shift(1)
        df[f'{col}_t-7'] = df[col].shift(7)
        df[f'{col}_t-14'] = df[col].shift(14)

    # === 気温移動平均 ===
    for col in ['最高気温', '平均気温']:
        df[f'{col}_MA3'] = df[col].rolling(3).mean()
        df[f'{col}_MA7'] = df[col].rolling(7).mean()
        df[f'{col}_MA14'] = df[col].rolling(14).mean()

    # === 気温変化量 ===
    df['最高気温_変化量_1d'] = df['最高気温'] - df['最高気温_t-1']
    df['最高気温_変化量_7d'] = df['最高気温'] - df['最高気温_t-7']
    df['最高気温_変化量_14d'] = df['最高気温'] - df['最高気温_t-14']
    df['平均気温_変化量_1d'] = df['平均気温'] - df['平均気温_t-1']
    df['平均気温_変化量_7d'] = df['平均気温'] - df['平均気温_t-7']
    df['平均気温_変化量_vs_MA7'] = df['平均気温'] - df['平均気温_MA7']

    # === 気温変化率 ===
    # 注意: 気温はケルビン換算（+273.15）して計算
    df['最高気温_変化率_1d'] = (df['最高気温'] - df['最高気温_t-1']) / (df['最高気温_t-1'] + 273.15)
    df['最高気温_変化率_7d'] = (df['最高気温'] - df['最高気温_t-7']) / (df['最高気温_t-7'] + 273.15)

    # === 気温変化フラグ ===
    df['気温上昇_急_1d'] = (df['最高気温_変化量_1d'] > 5).astype(int)
    df['気温下降_急_1d'] = (df['最高気温_変化量_1d'] < -5).astype(int)
    df['暖かくなった_7d'] = (df['平均気温_変化量_7d'] > 3).astype(int)
    df['寒くなった_7d'] = (df['平均気温_変化量_7d'] < -3).astype(int)
    df['気温安定'] = (df['最高気温_変化量_1d'].abs() < 2).astype(int)

    # === 気温差（日較差）特徴 ===
    df['気温差_t-1'] = df['気温差'].shift(1)
    df['気温差_MA7'] = df['気温差'].rolling(7).mean()
    df['気温差_変化量_1d'] = df['気温差'] - df['気温差_t-1']
    df['気温差_大'] = (df['気温差'] > 10).astype(int)
    df['気温差_拡大'] = (df['気温差_変化量_1d'] > 3).astype(int)

    # === 降水ラグ・累積 ===
    df['降水量_t-1'] = df['降水量'].shift(1)
    df['降水量_t-7'] = df['降水量'].shift(7)
    df['降水量_累積3日'] = df['降水量'].rolling(3).sum()
    df['降水量_累積7日'] = df['降水量'].rolling(7).sum()
    df['降水量_累積14日'] = df['降水量'].rolling(14).sum()

    # === 降水変化量 ===
    df['降水量_変化量_1d'] = df['降水量'] - df['降水量_t-1']
    df['降水量_変化量_7d'] = df['降水量'] - df['降水量_t-7']

    # === 降水フラグ ===
    df['降水_開始'] = ((df['降水量'] > 1) & (df['降水量_t-1'] <= 1)).astype(int)
    df['降水_終了'] = ((df['降水量'] <= 1) & (df['降水量_t-1'] > 1)).astype(int)

    # 連続降雨日数・晴天日数
    def count_consecutive_days(series, threshold, above=True):
        """連続日数をカウント"""
        result = []
        count = 0
        for val in series:
            if pd.isna(val):
                result.append(0)
                count = 0
            elif (above and val > threshold) or (not above and val <= threshold):
                count += 1
                result.append(count)
            else:
                count = 0
                result.append(0)
        return result

    df['連続降雨日数'] = count_consecutive_days(df['降水量'], 1, above=True)
    df['連続晴天日数'] = count_consecutive_days(df['降水量'], 1, above=False)

    # === 気温トレンド ===
    df['気温トレンド_7d'] = (df['平均気温'] - df['平均気温_t-7']) / 7
    df['気温トレンド_14d'] = (df['平均気温'] - df['平均気温_t-14']) / 14

    logger.info("✓ 気象特徴を追加しました")

    return df


# ============================================================
# C. 時系列特徴（汎用関数）
# ============================================================

def add_time_series_features(
    df: pd.DataFrame,
    column: str,
    periods: List[int] = [1, 7, 14],
    ma_windows: List[int] = [7, 14, 28],
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    時系列特徴の一括計算（変化量・変化率含む）

    Parameters:
    df: DataFrame
    column: str - 対象カラム名
    periods: list - ラグ期間
    ma_windows: list - 移動平均の窓サイズ
    """
    if column not in df.columns:
        if logger:
            logger.warning(f"列 {column} が見つかりません。スキップします")
        return df

    result = df.copy()

    # === ラグ ===
    for p in periods:
        result[f'{column}_t-{p}'] = result[column].shift(p)

    # === 移動平均 ===
    for w in ma_windows:
        result[f'{column}_MA{w}'] = result[column].rolling(w).mean()

    # === 移動標準偏差 ===
    for w in [7, 14]:
        result[f'{column}_STD{w}'] = result[column].rolling(w).std()
        if f'{column}_MA{w}' in result.columns:
            result[f'{column}_CV{w}'] = result[f'{column}_STD{w}'] / result[f'{column}_MA{w}'].replace(0, np.nan)

    # === 変化量（絶対差） ===
    for p in periods:
        lag_col = f'{column}_t-{p}'
        if lag_col in result.columns:
            result[f'{column}_変化量_{p}d'] = result[column] - result[lag_col]

    # MA比較
    for w in [7, 14]:
        ma_col = f'{column}_MA{w}'
        if ma_col in result.columns:
            result[f'{column}_変化量_vs_MA{w}'] = result[column] - result[ma_col]

    # 週次変化（MA7の1週間差）
    if f'{column}_MA7' in result.columns:
        result[f'{column}_変化量_週次'] = result[f'{column}_MA7'] - result[f'{column}_MA7'].shift(7)

    # === 変化率（相対差） ===
    for p in periods:
        lag_col = f'{column}_t-{p}'
        if lag_col in result.columns:
            # ゼロ除算回避
            result[f'{column}_変化率_{p}d'] = (result[column] - result[lag_col]) / result[lag_col].replace(0, np.nan)
            result[f'{column}_変化率_{p}d_pct'] = result[f'{column}_変化率_{p}d'] * 100

    # MA比較
    for w in [7, 14]:
        ma_col = f'{column}_MA{w}'
        if ma_col in result.columns:
            result[f'{column}_変化率_vs_MA{w}'] = (result[column] - result[ma_col]) / result[ma_col].replace(0, np.nan)

    # 週次成長率
    if f'{column}_MA7' in result.columns:
        ma7_lag7 = result[f'{column}_MA7'].shift(7)
        result[f'{column}_成長率_週次'] = (result[f'{column}_MA7'] - ma7_lag7) / ma7_lag7.replace(0, np.nan)

    # === 方向性フラグ ===
    for p in [1, 7]:
        change_col = f'{column}_変化量_{p}d'
        if change_col in result.columns:
            result[f'{column}_増加_{p}d'] = (result[change_col] > 0).astype(int)
            result[f'{column}_減少_{p}d'] = (result[change_col] < 0).astype(int)

    # 急変動フラグ
    if f'{column}_変化率_7d' in result.columns:
        result[f'{column}_急増_7d'] = (result[f'{column}_変化率_7d'] > 0.2).astype(int)
        result[f'{column}_急減_7d'] = (result[f'{column}_変化率_7d'] < -0.2).astype(int)

    # MA超過フラグ
    for w in [7, 14]:
        ma_col = f'{column}_MA{w}'
        if ma_col in result.columns:
            result[f'{column}_MA{w}超'] = (result[column] > result[ma_col]).astype(int)

    # === 極値・レンジ ===
    for w in [7, 14]:
        result[f'{column}_MAX{w}'] = result[column].rolling(w).max()
        result[f'{column}_MIN{w}'] = result[column].rolling(w).min()
        result[f'{column}_レンジ{w}'] = result[f'{column}_MAX{w}'] - result[f'{column}_MIN{w}']

    # === トレンド（線形回帰の傾き） ===
    def calc_trend(x):
        if len(x) < 2 or x.isna().any():
            return np.nan
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(range(len(x)), x)
        return slope

    for w in [7, 14]:
        result[f'{column}_トレンド_{w}d'] = result[column].rolling(w).apply(calc_trend, raw=False)

    return result


# ============================================================
# D. 季節変動指数・客数指数
# ============================================================

def calculate_seasonal_customer_index(
    df_current: pd.DataFrame,
    df_past_year: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    季節変動指数と客数指数を計算
    + 昨年同日の売上、客数、客単価を追加

    季節変動指数 = 前年同月の平均売上 / 前年の年間平均売上
    客数指数 = 前年同月の平均客数 / 前年の年間平均客数
    """
    logger.info("季節変動指数・客数指数・昨年同日データを計算中...")

    df = df_current.copy()
    df['日付'] = pd.to_datetime(df['日付'])
    df['月'] = df['日付'].dt.month
    df['日'] = df['日付'].dt.day
    df['週'] = df['日付'].dt.isocalendar().week

    # 前年データの準備
    df_past = df_past_year.copy()
    df_past['日付'] = pd.to_datetime(df_past['日付'])
    df_past['月'] = df_past['日付'].dt.month
    df_past['日'] = df_past['日付'].dt.day
    df_past['週'] = df_past['日付'].dt.isocalendar().week

    # 昨年同日のマッピング用辞書を作成（店舗 × 月日 → 売上/客数/客単価）
    if '店舗' in df_past.columns:
        df_past['月日'] = df_past['月'].astype(str) + '-' + df_past['日'].astype(str)
        past_year_dict = {}

        for _, row in df_past.iterrows():
            store = row['店舗']
            month_day = row['月日']
            key = (store, month_day)

            past_year_dict[key] = {
                '売上': row.get('売上', np.nan),
                '客数': row.get('客数', np.nan),
                '客単価': row.get('客単価', np.nan)
            }

    # 店舗ごとに計算
    if '店舗' in df.columns and '店舗' in df_past.columns:
        # 昨年同日データの列を初期化
        df['昨年同日_売上'] = np.nan
        df['昨年同日_客数'] = np.nan
        df['昨年同日_客単価'] = np.nan

        for store in df['店舗'].unique():
            past_store = df_past[df_past['店舗'] == store]

            if len(past_store) == 0:
                logger.warning(f"店舗 {store} の前年データがありません")
                continue

            # 年間平均
            yearly_avg_sales = past_store['売上'].mean()
            yearly_avg_customers = past_store['客数'].mean() if '客数' in past_store.columns else 0

            # 月次平均
            monthly_avg_sales = past_store.groupby('月')['売上'].mean()
            monthly_avg_customers = past_store.groupby('月')['客数'].mean() if '客数' in past_store.columns else {}

            # 週次平均
            weekly_avg_sales = past_store.groupby('週')['売上'].mean()

            # 季節変動指数
            seasonal_index_monthly = (monthly_avg_sales / yearly_avg_sales).to_dict()
            seasonal_index_weekly = (weekly_avg_sales / yearly_avg_sales).to_dict()

            # 客数指数
            if yearly_avg_customers > 0:
                customer_index = (monthly_avg_customers / yearly_avg_customers).to_dict()
            else:
                customer_index = {}

            # 現在のデータに付与
            store_mask = df['店舗'] == store
            df.loc[store_mask, '季節変動指数_月'] = df.loc[store_mask, '月'].map(seasonal_index_monthly)
            df.loc[store_mask, '季節変動指数_週'] = df.loc[store_mask, '週'].map(seasonal_index_weekly)
            if customer_index:
                df.loc[store_mask, '客数指数'] = df.loc[store_mask, '月'].map(customer_index)

            # 昨年同日データをマッピング
            for idx in df[store_mask].index:
                month = df.loc[idx, '月']
                day = df.loc[idx, '日']
                month_day = f"{month}-{day}"
                key = (store, month_day)

                if key in past_year_dict:
                    df.loc[idx, '昨年同日_売上'] = past_year_dict[key]['売上']
                    df.loc[idx, '昨年同日_客数'] = past_year_dict[key]['客数']
                    df.loc[idx, '昨年同日_客単価'] = past_year_dict[key]['客単価']

    else:
        # 店舗列がない場合は全体で計算
        yearly_avg_sales = df_past['売上'].mean()
        yearly_avg_customers = df_past['客数'].mean() if '客数' in df_past.columns else 0

        monthly_avg_sales = df_past.groupby('月')['売上'].mean()
        monthly_avg_customers = df_past.groupby('月')['客数'].mean() if '客数' in df_past.columns else {}
        weekly_avg_sales = df_past.groupby('週')['売上'].mean()

        seasonal_index_monthly = (monthly_avg_sales / yearly_avg_sales).to_dict()
        seasonal_index_weekly = (weekly_avg_sales / yearly_avg_sales).to_dict()

        if yearly_avg_customers > 0:
            customer_index = (monthly_avg_customers / yearly_avg_customers).to_dict()
        else:
            customer_index = {}

        df['季節変動指数_月'] = df['月'].map(seasonal_index_monthly)
        df['季節変動指数_週'] = df['週'].map(seasonal_index_weekly)
        if customer_index:
            df['客数指数'] = df['月'].map(customer_index)

    # 昨年同日との比較特徴量
    if '昨年同日_売上' in df.columns and '売上' in df.columns:
        df['昨年同日比_売上_変化量'] = df['売上'] - df['昨年同日_売上']
        df['昨年同日比_売上_変化率'] = (df['売上'] - df['昨年同日_売上']) / df['昨年同日_売上'].replace(0, np.nan)
        df['昨年同日比_売上_増加'] = (df['昨年同日比_売上_変化量'] > 0).astype(int)
        df['昨年同日比_売上_減少'] = (df['昨年同日比_売上_変化量'] < 0).astype(int)

    if '昨年同日_客数' in df.columns and '客数' in df.columns:
        df['昨年同日比_客数_変化量'] = df['客数'] - df['昨年同日_客数']
        df['昨年同日比_客数_変化率'] = (df['客数'] - df['昨年同日_客数']) / df['昨年同日_客数'].replace(0, np.nan)
        df['昨年同日比_客数_増加'] = (df['昨年同日比_客数_変化量'] > 0).astype(int)
        df['昨年同日比_客数_減少'] = (df['昨年同日比_客数_変化量'] < 0).astype(int)

    if '昨年同日_客単価' in df.columns and '客単価' in df.columns:
        df['昨年同日比_客単価_変化量'] = df['客単価'] - df['昨年同日_客単価']
        df['昨年同日比_客単価_変化率'] = (df['客単価'] - df['昨年同日_客単価']) / df['昨年同日_客単価'].replace(0, np.nan)
        df['昨年同日比_客単価_増加'] = (df['昨年同日比_客単価_変化量'] > 0).astype(int)
        df['昨年同日比_客単価_減少'] = (df['昨年同日比_客単価_変化量'] < 0).astype(int)

    # 季節変動指数の変化量・変化率
    df['季節変動指数_月_t-1'] = df['季節変動指数_月'].shift(1)
    df['季節変動指数_変化量_月'] = df['季節変動指数_月'] - df['季節変動指数_月_t-1']
    df['季節変動指数_変化率_月'] = (df['季節変動指数_変化量_月'] / df['季節変動指数_月_t-1'].replace(0, np.nan))

    df['季節変動指数_週_t-1'] = df['季節変動指数_週'].shift(1)
    df['季節変動指数_変化量_週'] = df['季節変動指数_週'] - df['季節変動指数_週_t-1']
    df['季節変動指数_変化率_週'] = (df['季節変動指数_変化量_週'] / df['季節変動指数_週_t-1'].replace(0, np.nan))

    # 季節フラグ
    df['季節_ピーク期'] = (df['季節変動指数_月'] > 1.2).astype(int)
    df['季節_オフ期'] = (df['季節変動指数_月'] < 0.8).astype(int)
    df['季節_上昇期'] = (df['季節変動指数_変化量_週'] > 0.05).astype(int)
    df['季節_下降期'] = (df['季節変動指数_変化量_週'] < -0.05).astype(int)

    logger.info("✓ 季節変動指数・客数指数・昨年同日データを計算しました")

    return df


# ============================================================
# メイン処理
# ============================================================

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="CSVデータに特徴量を追加（完全版 250-300特徴量）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python enrich_features_v2.py input.csv output.csv --store-locations stores.csv --past-year-data past_year.csv
  python enrich_features_v2.py input.csv output.csv --skip-weather
        """
    )

    parser.add_argument("input", help="入力CSVファイル")
    parser.add_argument("output", help="出力CSVファイル")
    parser.add_argument("--store-locations", help="店舗情報CSV（緯度経度）")
    parser.add_argument("--past-year-data", help="前年データCSV（季節変動指数計算用）")
    parser.add_argument("--skip-weather", action="store_true", help="天気情報の取得をスキップ")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()
    logger = setup_logging(args.debug)

    try:
        logger.info("=" * 60)
        logger.info("特徴量付与処理を開始（完全版）")
        logger.info("=" * 60)

        # データ読み込み
        logger.info(f"入力ファイル: {args.input}")
        df = pd.read_csv(args.input, encoding='utf-8-sig')
        logger.info(f"読み込み完了: {len(df):,}行 × {len(df.columns)}列")

        # A. カレンダー特徴
        df = create_calendar_features(df, logger)

        # B. 気象特徴
        if not args.skip_weather and args.store_locations:
            store_locations = pd.read_csv(args.store_locations, encoding='utf-8-sig')
            df = add_weather_features(df, store_locations, logger)
        elif not args.skip_weather:
            logger.warning("--store-locationsが指定されていないため、天気情報をスキップします")

        # C. 季節変動指数・客数指数
        if args.past_year_data:
            df_past_year = pd.read_csv(args.past_year_data, encoding='utf-8-sig')
            df = calculate_seasonal_customer_index(df, df_past_year, logger)
        else:
            logger.warning("--past-year-dataが指定されていないため、季節変動指数・客数指数をスキップします")

        # D. 時系列特徴（売上・客数・客単価）
        logger.info("時系列特徴を作成中...")

        if '売上' in df.columns:
            logger.info("  売上の時系列特徴を作成中...")
            df = add_time_series_features(df, '売上', periods=[1, 2, 3, 7, 14], ma_windows=[3, 7, 14, 28], logger=logger)

        if '客数' in df.columns:
            logger.info("  客数の時系列特徴を作成中...")
            df = add_time_series_features(df, '客数', periods=[1, 7, 14], ma_windows=[7, 14], logger=logger)

            # 客単価（派生値）
            if '売上' in df.columns:
                df['客単価'] = df['売上'] / df['客数'].replace(0, np.nan)
                logger.info("  客単価の時系列特徴を作成中...")
                df = add_time_series_features(df, '客単価', periods=[1, 7, 14], ma_windows=[7, 14], logger=logger)

        logger.info("✓ 時系列特徴を作成しました")

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
