"""
時系列分析モジュール (Time-Series Analysis Module)
====================================================

このモジュールは、PyCaret の時系列機能を使用した包括的な時系列分析を提供します。

主な機能:
- 月次トレンド分析（YoY比較付き）
- 週次パフォーマンス追跡（WoW変化付き）
- 日次売上パターン（異常検知付き）
- 季節分解
- 成長率計算
- 移動平均（7日間、30日間）
- 店舗レベルのフィルタリング機能
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

try:
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.utils.plotting import plot_series
except ImportError:
    warnings.warn("sktime not available. Some functionality may be limited.")


@dataclass
class TimeSeriesConfig:
    """時系列分析の設定クラス"""
    date_column: str = 'date'
    value_column: str = 'sales'
    store_column: Optional[str] = 'store_id'
    freq: str = 'D'  # Daily frequency
    seasonal_period: int = 7  # Weekly seasonality
    anomaly_threshold: float = 3.0  # Standard deviations for anomaly detection
    ma_windows: List[int] = None  # Moving average windows

    def __post_init__(self):
        if self.ma_windows is None:
            self.ma_windows = [7, 30]


@dataclass
class TrendAnalysisResult:
    """トレンド分析結果"""
    period: str
    current_value: float
    previous_value: float
    change_value: float
    change_percent: float
    trend_direction: str  # 'up', 'down', 'stable'

    def to_dict(self) -> Dict:
        return {
            '期間': self.period,
            '現在値': self.current_value,
            '前期値': self.previous_value,
            '変化量': self.change_value,
            '変化率(%)': self.change_percent,
            'トレンド': self.trend_direction
        }


@dataclass
class SeasonalDecompositionResult:
    """季節分解結果"""
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    observed: pd.Series

    def to_dataframe(self) -> pd.DataFrame:
        """結果をDataFrameに変換"""
        return pd.DataFrame({
            'トレンド': self.trend,
            '季節性': self.seasonal,
            '残差': self.residual,
            '観測値': self.observed
        })


@dataclass
class AnomalyDetectionResult:
    """異常検知結果"""
    anomalies: pd.DataFrame
    anomaly_count: int
    anomaly_dates: List[datetime]
    statistics: Dict

    def to_dict(self) -> Dict:
        return {
            '異常値数': self.anomaly_count,
            '異常日付': [d.strftime('%Y-%m-%d') for d in self.anomaly_dates],
            '統計情報': self.statistics
        }


class TimeSeriesAnalyzer:
    """
    時系列分析クラス

    PyCaret の時系列機能を活用した包括的な時系列分析を提供します。

    Parameters
    ----------
    data : pd.DataFrame
        分析対象のデータフレーム
    config : TimeSeriesConfig, optional
        時系列分析の設定

    Examples
    --------
    >>> analyzer = TimeSeriesAnalyzer(sales_data)
    >>> monthly_trend = analyzer.analyze_monthly_trend()
    >>> anomalies = analyzer.detect_anomalies()
    """

    def __init__(self, data: pd.DataFrame, config: Optional[TimeSeriesConfig] = None):
        self.data = data.copy()
        self.config = config or TimeSeriesConfig()
        self._validate_data()
        self._prepare_data()

    def _validate_data(self) -> None:
        """データの検証"""
        required_columns = [self.config.date_column, self.config.value_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {missing_columns}")

        # 日付カラムの型チェック
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.config.date_column]):
            try:
                self.data[self.config.date_column] = pd.to_datetime(
                    self.data[self.config.date_column]
                )
            except Exception as e:
                raise ValueError(f"日付カラムの変換に失敗しました: {e}")

    def _prepare_data(self) -> None:
        """データの前処理"""
        # 日付でソート
        self.data = self.data.sort_values(self.config.date_column)

        # インデックスを日付に設定
        self.data = self.data.set_index(self.config.date_column)

        # 欠損値の処理
        self.data[self.config.value_column] = self.data[self.config.value_column].fillna(
            method='ffill'
        ).fillna(0)

    def filter_by_store(self, store_id: Union[str, int]) -> 'TimeSeriesAnalyzer':
        """
        特定の店舗でフィルタリング

        Parameters
        ----------
        store_id : str or int
            フィルタリングする店舗ID

        Returns
        -------
        TimeSeriesAnalyzer
            フィルタリングされた新しいアナライザーインスタンス
        """
        if self.config.store_column is None:
            raise ValueError("店舗カラムが設定されていません")

        filtered_data = self.data[
            self.data[self.config.store_column] == store_id
        ].copy()

        return TimeSeriesAnalyzer(filtered_data.reset_index(), self.config)

    def analyze_monthly_trend(self, yoy_comparison: bool = True) -> pd.DataFrame:
        """
        月次トレンド分析（YoY比較付き）

        Parameters
        ----------
        yoy_comparison : bool, default=True
            前年同月比較を含めるかどうか

        Returns
        -------
        pd.DataFrame
            月次トレンド分析結果
        """
        # 月次データに集約
        monthly_data = self.data.resample('M')[self.config.value_column].agg([
            ('合計', 'sum'),
            ('平均', 'mean'),
            ('最大', 'max'),
            ('最小', 'min'),
            ('標準偏差', 'std')
        ])

        # MoM（Month over Month）変化
        monthly_data['MoM変化量'] = monthly_data['合計'].diff()
        monthly_data['MoM変化率(%)'] = monthly_data['合計'].pct_change() * 100

        if yoy_comparison:
            # YoY（Year over Year）変化
            monthly_data['YoY変化量'] = monthly_data['合計'].diff(12)
            monthly_data['YoY変化率(%)'] = monthly_data['合計'].pct_change(12) * 100

        # トレンド方向の判定
        monthly_data['トレンド'] = monthly_data['MoM変化率(%)'].apply(
            self._determine_trend_direction
        )

        return monthly_data.round(2)

    def analyze_weekly_performance(self, wow_comparison: bool = True) -> pd.DataFrame:
        """
        週次パフォーマンス追跡（WoW変化付き）

        Parameters
        ----------
        wow_comparison : bool, default=True
            前週比較を含めるかどうか

        Returns
        -------
        pd.DataFrame
            週次パフォーマンス分析結果
        """
        # 週次データに集約
        weekly_data = self.data.resample('W')[self.config.value_column].agg([
            ('合計', 'sum'),
            ('平均', 'mean'),
            ('取引日数', 'count'),
            ('最大日次売上', 'max'),
            ('最小日次売上', 'min')
        ])

        if wow_comparison:
            # WoW（Week over Week）変化
            weekly_data['WoW変化量'] = weekly_data['合計'].diff()
            weekly_data['WoW変化率(%)'] = weekly_data['合計'].pct_change() * 100

        # 週次成長率
        weekly_data['成長率(%)'] = weekly_data['合計'].pct_change() * 100

        # パフォーマンスカテゴリ
        weekly_data['パフォーマンス'] = pd.cut(
            weekly_data['成長率(%)'],
            bins=[-np.inf, -5, 5, np.inf],
            labels=['低下', '安定', '成長']
        )

        return weekly_data.round(2)

    def analyze_daily_patterns(self, detect_anomalies: bool = True) -> pd.DataFrame:
        """
        日次売上パターン分析（異常検知付き）

        Parameters
        ----------
        detect_anomalies : bool, default=True
            異常値を検知するかどうか

        Returns
        -------
        pd.DataFrame
            日次パターン分析結果
        """
        daily_data = self.data[[self.config.value_column]].copy()
        daily_data.columns = ['売上']

        # 曜日の追加
        daily_data['曜日'] = daily_data.index.day_name()
        daily_data['曜日番号'] = daily_data.index.dayofweek

        # 月
        daily_data['月'] = daily_data.index.month

        # 日次変化
        daily_data['日次変化量'] = daily_data['売上'].diff()
        daily_data['日次変化率(%)'] = daily_data['売上'].pct_change() * 100

        if detect_anomalies:
            # 異常値の検知（Zスコア法）
            z_scores = np.abs(stats.zscore(daily_data['売上'].dropna()))
            daily_data['異常値'] = False
            daily_data.loc[daily_data['売上'].notna(), '異常値'] = (
                z_scores > self.config.anomaly_threshold
            )
            daily_data['Zスコア'] = np.nan
            daily_data.loc[daily_data['売上'].notna(), 'Zスコア'] = z_scores

        return daily_data.round(2)

    def seasonal_decomposition(
        self,
        model: str = 'additive',
        period: Optional[int] = None
    ) -> SeasonalDecompositionResult:
        """
        季節分解分析

        Parameters
        ----------
        model : str, default='additive'
            分解モデル（'additive' or 'multiplicative'）
        period : int, optional
            季節周期（デフォルトは設定値）

        Returns
        -------
        SeasonalDecompositionResult
            季節分解結果
        """
        period = period or self.config.seasonal_period

        # 欠損値を除外
        series = self.data[self.config.value_column].dropna()

        if len(series) < 2 * period:
            raise ValueError(
                f"データが不足しています。最低 {2 * period} 個のデータポイントが必要です。"
            )

        # 季節分解の実行
        decomposition = seasonal_decompose(
            series,
            model=model,
            period=period,
            extrapolate_trend='freq'
        )

        return SeasonalDecompositionResult(
            trend=decomposition.trend,
            seasonal=decomposition.seasonal,
            residual=decomposition.resid,
            observed=decomposition.observed
        )

    def calculate_growth_rates(
        self,
        periods: List[int] = None
    ) -> pd.DataFrame:
        """
        成長率計算

        Parameters
        ----------
        periods : list of int, optional
            計算する期間のリスト（日数）

        Returns
        -------
        pd.DataFrame
            各期間の成長率
        """
        if periods is None:
            periods = [1, 7, 30, 90, 365]

        growth_data = pd.DataFrame(index=self.data.index)
        growth_data['値'] = self.data[self.config.value_column]

        for period in periods:
            col_name = f'{period}日成長率(%)'
            growth_data[col_name] = (
                growth_data['値'].pct_change(period) * 100
            )

        # CAGR（年複合成長率）の計算
        if len(self.data) > 365:
            first_value = growth_data['値'].iloc[0]
            last_value = growth_data['値'].iloc[-1]
            years = len(self.data) / 365.25
            cagr = (np.power(last_value / first_value, 1 / years) - 1) * 100
            growth_data['CAGR(%)'] = cagr

        return growth_data.round(2)

    def calculate_moving_averages(
        self,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        移動平均の計算（7日間、30日間など）

        Parameters
        ----------
        windows : list of int, optional
            移動平均の窓サイズリスト

        Returns
        -------
        pd.DataFrame
            移動平均データ
        """
        windows = windows or self.config.ma_windows

        ma_data = pd.DataFrame(index=self.data.index)
        ma_data['実績値'] = self.data[self.config.value_column]

        for window in windows:
            # 単純移動平均（SMA）
            ma_data[f'SMA_{window}日'] = (
                ma_data['実績値'].rolling(window=window).mean()
            )

            # 指数移動平均（EMA）
            ma_data[f'EMA_{window}日'] = (
                ma_data['実績値'].ewm(span=window, adjust=False).mean()
            )

        # ゴールデンクロス/デッドクロスの検出（7日と30日の場合）
        if 7 in windows and 30 in windows:
            ma_data['クロス'] = ''

            # ゴールデンクロス（短期が長期を上抜け）
            golden_cross = (
                (ma_data['SMA_7日'] > ma_data['SMA_30日']) &
                (ma_data['SMA_7日'].shift(1) <= ma_data['SMA_30日'].shift(1))
            )
            ma_data.loc[golden_cross, 'クロス'] = 'ゴールデンクロス'

            # デッドクロス（短期が長期を下抜け）
            dead_cross = (
                (ma_data['SMA_7日'] < ma_data['SMA_30日']) &
                (ma_data['SMA_7日'].shift(1) >= ma_data['SMA_30日'].shift(1))
            )
            ma_data.loc[dead_cross, 'クロス'] = 'デッドクロス'

        return ma_data.round(2)

    def detect_anomalies(
        self,
        method: str = 'zscore',
        threshold: Optional[float] = None
    ) -> AnomalyDetectionResult:
        """
        異常検知

        Parameters
        ----------
        method : str, default='zscore'
            異常検知手法（'zscore', 'iqr', 'isolation_forest'）
        threshold : float, optional
            異常値判定の閾値

        Returns
        -------
        AnomalyDetectionResult
            異常検知結果
        """
        threshold = threshold or self.config.anomaly_threshold
        series = self.data[self.config.value_column].dropna()

        if method == 'zscore':
            # Zスコア法
            z_scores = np.abs(stats.zscore(series))
            anomaly_mask = z_scores > threshold
            scores = z_scores

        elif method == 'iqr':
            # IQR法（四分位範囲）
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            anomaly_mask = (series < lower_bound) | (series > upper_bound)
            scores = np.abs((series - series.median()) / IQR)

        elif method == 'isolation_forest':
            # Isolation Forest（PyOD使用）
            try:
                from pyod.models.iforest import IForest

                clf = IForest(contamination=0.1, random_state=42)
                clf.fit(series.values.reshape(-1, 1))
                anomaly_mask = clf.labels_ == 1
                scores = clf.decision_scores_
            except ImportError:
                raise ImportError("PyOD が必要です。pip install pyod でインストールしてください。")
        else:
            raise ValueError(f"サポートされていない手法です: {method}")

        # 異常値のデータフレーム作成
        anomalies_df = pd.DataFrame({
            '日付': series.index,
            '値': series.values,
            'スコア': scores,
            '異常値': anomaly_mask
        })

        anomaly_dates = series[anomaly_mask].index.to_list()

        # 統計情報
        statistics = {
            '総データ数': len(series),
            '異常値数': int(anomaly_mask.sum()),
            '異常値率(%)': round(anomaly_mask.sum() / len(series) * 100, 2),
            '平均値': round(series.mean(), 2),
            '標準偏差': round(series.std(), 2),
            '最小値': round(series.min(), 2),
            '最大値': round(series.max(), 2),
            '検知手法': method,
            '閾値': threshold
        }

        return AnomalyDetectionResult(
            anomalies=anomalies_df[anomalies_df['異常値']],
            anomaly_count=int(anomaly_mask.sum()),
            anomaly_dates=anomaly_dates,
            statistics=statistics
        )

    def get_summary_statistics(self) -> Dict:
        """
        時系列データの要約統計量を取得

        Returns
        -------
        dict
            要約統計情報
        """
        series = self.data[self.config.value_column]

        # 定常性テスト（ADF検定）
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] < 0.05

        return {
            '基本統計': {
                'データ数': len(series),
                '期間開始': series.index.min().strftime('%Y-%m-%d'),
                '期間終了': series.index.max().strftime('%Y-%m-%d'),
                '平均': round(series.mean(), 2),
                '中央値': round(series.median(), 2),
                '標準偏差': round(series.std(), 2),
                '最小値': round(series.min(), 2),
                '最大値': round(series.max(), 2),
                '合計': round(series.sum(), 2)
            },
            '変化統計': {
                '最大上昇': round(series.diff().max(), 2),
                '最大下降': round(series.diff().min(), 2),
                '平均変化': round(series.diff().mean(), 2),
                '変動係数(%)': round((series.std() / series.mean()) * 100, 2)
            },
            '定常性': {
                '定常性': 'はい' if is_stationary else 'いいえ',
                'ADF統計量': round(adf_result[0], 4),
                'p値': round(adf_result[1], 4),
                '臨界値': {
                    k: round(v, 4) for k, v in adf_result[4].items()
                }
            }
        }

    @staticmethod
    def _determine_trend_direction(change_percent: float) -> str:
        """トレンド方向の判定"""
        if pd.isna(change_percent):
            return '不明'
        elif change_percent > 5:
            return '上昇'
        elif change_percent < -5:
            return '下降'
        else:
            return '安定'

    def export_analysis(
        self,
        output_path: str,
        include_all: bool = True
    ) -> None:
        """
        分析結果をExcelにエクスポート

        Parameters
        ----------
        output_path : str
            出力ファイルパス
        include_all : bool, default=True
            すべての分析を含めるかどうか
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 要約統計
            summary = pd.DataFrame([self.get_summary_statistics()])
            summary.to_excel(writer, sheet_name='要約統計', index=False)

            if include_all:
                # 月次トレンド
                monthly = self.analyze_monthly_trend()
                monthly.to_excel(writer, sheet_name='月次トレンド')

                # 週次パフォーマンス
                weekly = self.analyze_weekly_performance()
                weekly.to_excel(writer, sheet_name='週次パフォーマンス')

                # 日次パターン
                daily = self.analyze_daily_patterns()
                daily.to_excel(writer, sheet_name='日次パターン')

                # 移動平均
                ma = self.calculate_moving_averages()
                ma.to_excel(writer, sheet_name='移動平均')

                # 成長率
                growth = self.calculate_growth_rates()
                growth.to_excel(writer, sheet_name='成長率')

                # 異常検知
                anomalies = self.detect_anomalies()
                anomalies.anomalies.to_excel(writer, sheet_name='異常値', index=False)

                # 季節分解
                try:
                    decomp = self.seasonal_decomposition()
                    decomp_df = decomp.to_dataframe()
                    decomp_df.to_excel(writer, sheet_name='季節分解')
                except Exception as e:
                    print(f"季節分解のエクスポートに失敗しました: {e}")


def create_analyzer(
    data: pd.DataFrame,
    date_column: str = 'date',
    value_column: str = 'sales',
    store_column: Optional[str] = None,
    **kwargs
) -> TimeSeriesAnalyzer:
    """
    時系列アナライザーを作成するヘルパー関数

    Parameters
    ----------
    data : pd.DataFrame
        分析対象データ
    date_column : str, default='date'
        日付カラム名
    value_column : str, default='sales'
        値カラム名
    store_column : str, optional
        店舗カラム名
    **kwargs : dict
        TimeSeriesConfigに渡す追加パラメータ

    Returns
    -------
    TimeSeriesAnalyzer
        時系列アナライザーインスタンス
    """
    config = TimeSeriesConfig(
        date_column=date_column,
        value_column=value_column,
        store_column=store_column,
        **kwargs
    )

    return TimeSeriesAnalyzer(data, config)


if __name__ == '__main__':
    # 使用例
    print("時系列分析モジュール")
    print("=" * 50)
    print("\n使用例:")
    print("""
    import pandas as pd
    from analysis.time_series import create_analyzer

    # データ読み込み
    data = pd.read_csv('sales_data.csv')

    # アナライザー作成
    analyzer = create_analyzer(
        data,
        date_column='date',
        value_column='sales',
        store_column='store_id'
    )

    # 月次トレンド分析
    monthly_trend = analyzer.analyze_monthly_trend()
    print(monthly_trend)

    # 異常検知
    anomalies = analyzer.detect_anomalies()
    print(f"検出された異常値: {anomalies.anomaly_count}件")

    # 全分析結果をエクスポート
    analyzer.export_analysis('analysis_results.xlsx')
    """)
