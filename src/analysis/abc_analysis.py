"""
ABC Analysis Module for Product Classification and Pareto Analysis
ABC分析モジュール - 製品分類とパレート分析

This module provides comprehensive ABC analysis functionality including:
- Category-wise ABC classification (売上、利益、数量)
- Pareto analysis with 80/20 rule
- Product ranking by multiple metrics
- Cross-category comparison
- Time-based ABC shifts detection
- Store-specific ABC patterns
- Visual ABC matrix generation

このモジュールは以下の機能を提供します：
- カテゴリ別ABC分類（売上、利益、数量）
- パレート分析（80/20ルール）
- 複数指標による製品ランキング
- カテゴリ間比較
- 時系列ABC推移検出
- 店舗別ABCパターン
- ビジュアルABCマトリックス生成

Author: PyCaret Development Team
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class ABCCategory(Enum):
    """ABC分類カテゴリ / ABC Classification Categories"""
    A = "A"  # Top 80% of value (20% of items) / 価値の80%（アイテムの20%）
    B = "B"  # Next 15% of value (30% of items) / 次の15%の価値（アイテムの30%）
    C = "C"  # Remaining 5% of value (50% of items) / 残り5%の価値（アイテムの50%）


class MetricType(Enum):
    """分析指標タイプ / Analysis Metric Types"""
    REVENUE = "revenue"  # 売上 / Sales Revenue
    PROFIT = "profit"  # 利益 / Profit
    QUANTITY = "quantity"  # 数量 / Quantity
    MARGIN = "margin"  # 利益率 / Profit Margin
    CUSTOM = "custom"  # カスタム指標 / Custom Metric


@dataclass
class ABCThresholds:
    """
    ABC分類の閾値設定 / ABC Classification Thresholds

    Attributes:
        a_threshold: A分類の累積割合閾値（デフォルト: 0.8 = 80%）
        b_threshold: B分類の累積割合閾値（デフォルト: 0.95 = 95%）
        c_threshold: C分類の累積割合閾値（デフォルト: 1.0 = 100%）
    """
    a_threshold: float = 0.80
    b_threshold: float = 0.95
    c_threshold: float = 1.00

    def validate(self) -> None:
        """閾値の妥当性を検証 / Validate threshold values"""
        if not (0 < self.a_threshold < self.b_threshold <= self.c_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: A={self.a_threshold}, B={self.b_threshold}, C={self.c_threshold}. "
                f"Must satisfy: 0 < A < B <= C <= 1.0"
            )


@dataclass
class ABCResult:
    """
    ABC分析結果 / ABC Analysis Result

    Attributes:
        data: ABC分類を含むデータフレーム / DataFrame with ABC classifications
        summary: カテゴリ別サマリー / Category-wise summary
        pareto_data: パレート分析データ / Pareto analysis data
        metric_type: 使用した指標タイプ / Metric type used
        thresholds: 使用した閾値 / Thresholds used
    """
    data: pd.DataFrame
    summary: pd.DataFrame
    pareto_data: pd.DataFrame
    metric_type: MetricType
    thresholds: ABCThresholds


class ABCAnalyzer:
    """
    ABC分析とパレート分析を実行するメインクラス
    Main class for performing ABC and Pareto analysis

    Example:
        >>> analyzer = ABCAnalyzer(language='ja')
        >>> result = analyzer.classify(
        ...     df,
        ...     value_col='revenue',
        ...     item_col='product_id',
        ...     metric_type=MetricType.REVENUE
        ... )
        >>> analyzer.plot_abc_chart(result)
    """

    def __init__(
        self,
        thresholds: Optional[ABCThresholds] = None,
        language: str = 'en'
    ):
        """
        初期化 / Initialize ABC Analyzer

        Args:
            thresholds: ABC分類の閾値（デフォルト: 80/15/5ルール）
            language: 出力言語 ('en' or 'ja')
        """
        self.thresholds = thresholds or ABCThresholds()
        self.thresholds.validate()
        self.language = language

        # 言語別メッセージ / Language-specific messages
        self.messages = {
            'en': {
                'abc_category': 'ABC_Category',
                'cumulative_pct': 'Cumulative_Percentage',
                'item_count': 'Item_Count',
                'total_value': 'Total_Value',
                'avg_value': 'Average_Value',
                'value_pct': 'Value_Percentage',
                'rank': 'Rank',
                'category': 'Category',
                'value': 'Value'
            },
            'ja': {
                'abc_category': 'ABC分類',
                'cumulative_pct': '累積割合',
                'item_count': 'アイテム数',
                'total_value': '合計値',
                'avg_value': '平均値',
                'value_pct': '値の割合',
                'rank': '順位',
                'category': 'カテゴリ',
                'value': '値'
            }
        }

    def _get_message(self, key: str) -> str:
        """メッセージを言語に応じて取得 / Get message based on language"""
        return self.messages.get(self.language, self.messages['en']).get(key, key)

    def classify(
        self,
        data: pd.DataFrame,
        value_col: str,
        item_col: str,
        metric_type: MetricType = MetricType.REVENUE,
        group_cols: Optional[List[str]] = None
    ) -> ABCResult:
        """
        ABC分類を実行 / Perform ABC classification

        Args:
            data: 分析対象データ
            value_col: 値を表すカラム名（売上、利益など）
            item_col: アイテムを識別するカラム名（商品ID など）
            metric_type: 指標タイプ
            group_cols: グループ化するカラム（店舗、カテゴリなど）

        Returns:
            ABCResult: ABC分析結果
        """
        df = data.copy()

        # 入力検証 / Input validation
        self._validate_input(df, value_col, item_col, group_cols)

        # グループ化して集計 / Aggregate by groups
        if group_cols:
            agg_df = df.groupby([item_col] + group_cols).agg({
                value_col: 'sum'
            }).reset_index()
        else:
            agg_df = df.groupby(item_col).agg({
                value_col: 'sum'
            }).reset_index()

        # 値でソート / Sort by value descending
        agg_df = agg_df.sort_values(value_col, ascending=False).reset_index(drop=True)

        # 累積割合を計算 / Calculate cumulative percentage
        total_value = agg_df[value_col].sum()
        agg_df['cumulative_value'] = agg_df[value_col].cumsum()
        agg_df[self._get_message('cumulative_pct')] = (
            agg_df['cumulative_value'] / total_value * 100
        )

        # ABC分類を割り当て / Assign ABC categories
        agg_df[self._get_message('abc_category')] = agg_df.apply(
            lambda row: self._assign_abc_category(
                row[self._get_message('cumulative_pct')] / 100
            ),
            axis=1
        )

        # ランキングを追加 / Add ranking
        agg_df[self._get_message('rank')] = range(1, len(agg_df) + 1)

        # 値の割合を計算 / Calculate value percentage
        agg_df[self._get_message('value_pct')] = (
            agg_df[value_col] / total_value * 100
        )

        # サマリーを生成 / Generate summary
        summary = self._generate_summary(agg_df, value_col)

        # パレートデータを生成 / Generate Pareto data
        pareto_data = self._generate_pareto_data(agg_df, value_col)

        return ABCResult(
            data=agg_df,
            summary=summary,
            pareto_data=pareto_data,
            metric_type=metric_type,
            thresholds=self.thresholds
        )

    def _validate_input(
        self,
        df: pd.DataFrame,
        value_col: str,
        item_col: str,
        group_cols: Optional[List[str]]
    ) -> None:
        """入力データの検証 / Validate input data"""
        # カラムの存在確認 / Check column existence
        required_cols = [value_col, item_col]
        if group_cols:
            required_cols.extend(group_cols)

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # 値カラムが数値型か確認 / Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            raise TypeError(f"Column '{value_col}' must be numeric")

        # 負の値がないか確認 / Check for negative values
        if (df[value_col] < 0).any():
            warnings.warn(f"Column '{value_col}' contains negative values")

    def _assign_abc_category(self, cumulative_pct: float) -> str:
        """累積割合に基づいてABCカテゴリを割り当て / Assign ABC category based on cumulative percentage"""
        if cumulative_pct <= self.thresholds.a_threshold:
            return ABCCategory.A.value
        elif cumulative_pct <= self.thresholds.b_threshold:
            return ABCCategory.B.value
        else:
            return ABCCategory.C.value

    def _generate_summary(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """カテゴリ別サマリーを生成 / Generate category-wise summary"""
        abc_col = self._get_message('abc_category')

        summary = df.groupby(abc_col).agg({
            value_col: ['count', 'sum', 'mean'],
        }).reset_index()

        # カラム名を平坦化 / Flatten column names
        summary.columns = [
            abc_col,
            self._get_message('item_count'),
            self._get_message('total_value'),
            self._get_message('avg_value')
        ]

        # パーセンテージを計算 / Calculate percentages
        total_items = summary[self._get_message('item_count')].sum()
        total_value = summary[self._get_message('total_value')].sum()

        summary['Item_Percentage'] = (
            summary[self._get_message('item_count')] / total_items * 100
        )
        summary[self._get_message('value_pct')] = (
            summary[self._get_message('total_value')] / total_value * 100
        )

        # A, B, C の順序で並べ替え / Sort by A, B, C order
        category_order = ['A', 'B', 'C']
        summary[abc_col] = pd.Categorical(
            summary[abc_col],
            categories=category_order,
            ordered=True
        )
        summary = summary.sort_values(abc_col)

        return summary

    def _generate_pareto_data(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """パレート分析用データを生成 / Generate Pareto analysis data"""
        pareto_df = df[[
            self._get_message('rank'),
            value_col,
            self._get_message('cumulative_pct'),
            self._get_message('abc_category')
        ]].copy()

        return pareto_df

    def multi_metric_classification(
        self,
        data: pd.DataFrame,
        item_col: str,
        metrics: Dict[str, str],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        複数指標によるABC分類 / Multi-metric ABC classification

        Args:
            data: 分析対象データ
            item_col: アイテムを識別するカラム名
            metrics: 指標名と対応するカラム名の辞書
                    例: {'revenue': 'sales', 'profit': 'profit_amount'}
            weights: 各指標の重み（デフォルト: 均等）

        Returns:
            複数指標のABC分類を含むデータフレーム
        """
        if weights is None:
            weights = {k: 1.0 / len(metrics) for k in metrics.keys()}

        # 重みの合計が1になるよう正規化 / Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        results = {}
        composite_scores = pd.Series(0, index=data.index)

        # 各指標でABC分類を実行 / Perform ABC classification for each metric
        for metric_name, col_name in metrics.items():
            result = self.classify(
                data,
                value_col=col_name,
                item_col=item_col,
                metric_type=MetricType.CUSTOM
            )
            results[metric_name] = result

            # 正規化されたスコアを計算 / Calculate normalized scores
            max_val = result.data[col_name].max()
            normalized = result.data[col_name] / max_val

            # 重み付きスコアを加算 / Add weighted scores
            composite_scores += normalized * weights[metric_name]

        # 複合スコアでABC分類 / ABC classification based on composite scores
        composite_df = data[[item_col]].copy()
        composite_df['composite_score'] = composite_scores

        final_result = self.classify(
            composite_df,
            value_col='composite_score',
            item_col=item_col,
            metric_type=MetricType.CUSTOM
        )

        # 各指標のABC分類を結合 / Combine ABC classifications from each metric
        for metric_name, result in results.items():
            final_result.data[f'{metric_name}_ABC'] = result.data[
                self._get_message('abc_category')
            ]

        return final_result.data

    def detect_abc_shifts(
        self,
        data: pd.DataFrame,
        value_col: str,
        item_col: str,
        time_col: str,
        periods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        時系列でのABC分類の変化を検出 / Detect ABC classification shifts over time

        Args:
            data: 時系列データ
            value_col: 値を表すカラム名
            item_col: アイテムを識別するカラム名
            time_col: 時間を表すカラム名
            periods: 分析する期間のリスト（Noneの場合は自動検出）

        Returns:
            ABC分類の変化を含むデータフレーム
        """
        df = data.copy()

        # 時間カラムをdatetime型に変換 / Convert time column to datetime
        df[time_col] = pd.to_datetime(df[time_col])

        if periods is None:
            # 自動的に期間を検出 / Automatically detect periods
            periods = df[time_col].dt.to_period('M').astype(str).unique().tolist()
            periods.sort()

        shifts = []
        previous_abc = None

        for period in periods:
            # 期間ごとにデータをフィルタ / Filter data by period
            period_mask = df[time_col].dt.to_period('M').astype(str) == period
            period_data = df[period_mask]

            if len(period_data) == 0:
                continue

            # ABC分類を実行 / Perform ABC classification
            result = self.classify(
                period_data,
                value_col=value_col,
                item_col=item_col
            )

            current_abc = result.data[[
                item_col,
                self._get_message('abc_category')
            ]].copy()
            current_abc['period'] = period

            # 前期との比較 / Compare with previous period
            if previous_abc is not None:
                merged = current_abc.merge(
                    previous_abc[[item_col, self._get_message('abc_category')]],
                    on=item_col,
                    how='left',
                    suffixes=('_current', '_previous')
                )

                # 変化を検出 / Detect changes
                merged['shift'] = merged.apply(
                    lambda row: self._detect_shift(
                        row[f"{self._get_message('abc_category')}_previous"],
                        row[f"{self._get_message('abc_category')}_current"]
                    ),
                    axis=1
                )

                shifts.append(merged)

            previous_abc = current_abc

        if not shifts:
            return pd.DataFrame()

        return pd.concat(shifts, ignore_index=True)

    def _detect_shift(
        self,
        previous_cat: Optional[str],
        current_cat: str
    ) -> str:
        """ABC分類の変化を検出 / Detect ABC shift"""
        if pd.isna(previous_cat):
            return 'NEW'
        elif previous_cat == current_cat:
            return 'STABLE'
        else:
            return f'{previous_cat}→{current_cat}'

    def store_specific_abc(
        self,
        data: pd.DataFrame,
        value_col: str,
        item_col: str,
        store_col: str
    ) -> Dict[str, ABCResult]:
        """
        店舗別ABC分析 / Store-specific ABC analysis

        Args:
            data: データフレーム
            value_col: 値を表すカラム名
            item_col: アイテムを識別するカラム名
            store_col: 店舗を識別するカラム名

        Returns:
            店舗ごとのABC分析結果の辞書
        """
        results = {}

        for store in data[store_col].unique():
            store_data = data[data[store_col] == store]

            result = self.classify(
                store_data,
                value_col=value_col,
                item_col=item_col
            )

            results[store] = result

        return results

    def cross_category_comparison(
        self,
        data: pd.DataFrame,
        value_col: str,
        item_col: str,
        category_col: str
    ) -> pd.DataFrame:
        """
        カテゴリ間ABC比較 / Cross-category ABC comparison

        Args:
            data: データフレーム
            value_col: 値を表すカラム名
            item_col: アイテムを識別するカラム名
            category_col: カテゴリを識別するカラム名

        Returns:
            カテゴリ別ABC分布の比較データフレーム
        """
        comparison_data = []

        for category in data[category_col].unique():
            cat_data = data[data[category_col] == category]

            result = self.classify(
                cat_data,
                value_col=value_col,
                item_col=item_col
            )

            summary = result.summary.copy()
            summary[self._get_message('category')] = category
            comparison_data.append(summary)

        comparison_df = pd.concat(comparison_data, ignore_index=True)

        return comparison_df

    def plot_abc_chart(
        self,
        result: ABCResult,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        ABCパレート図を作成 / Create ABC Pareto chart

        Args:
            result: ABC分析結果
            figsize: 図のサイズ
            save_path: 保存先パス（Noneの場合は保存しない）

        Returns:
            matplotlib Figure オブジェクト
        """
        fig, ax1 = plt.subplots(figsize=figsize)

        # 棒グラフ（値） / Bar chart (values)
        x = result.pareto_data[self._get_message('rank')]
        y1 = result.data[result.data.columns[1]]  # value column

        color_map = {'A': '#2ecc71', 'B': '#f39c12', 'C': '#e74c3c'}
        colors = [
            color_map[cat]
            for cat in result.pareto_data[self._get_message('abc_category')]
        ]

        ax1.bar(x, y1, color=colors, alpha=0.7, label=self._get_message('value'))
        ax1.set_xlabel(self._get_message('rank'), fontsize=12)
        ax1.set_ylabel(self._get_message('value'), fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # 折れ線グラフ（累積割合） / Line chart (cumulative percentage)
        ax2 = ax1.twinx()
        y2 = result.pareto_data[self._get_message('cumulative_pct')]
        ax2.plot(x, y2, color='#3498db', marker='o', linewidth=2, markersize=4,
                label=self._get_message('cumulative_pct'))
        ax2.set_ylabel(
            self._get_message('cumulative_pct') + ' (%)',
            fontsize=12,
            color='#3498db'
        )
        ax2.tick_params(axis='y', labelcolor='#3498db')
        ax2.set_ylim(0, 105)

        # 80%ラインを追加 / Add 80% line
        ax2.axhline(
            y=self.thresholds.a_threshold * 100,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label='80% Line'
        )
        ax2.axhline(
            y=self.thresholds.b_threshold * 100,
            color='orange',
            linestyle='--',
            linewidth=1.5,
            label='95% Line'
        )

        # タイトルとレジェンド / Title and legend
        title = 'ABC Analysis - Pareto Chart' if self.language == 'en' else 'ABC分析 - パレート図'
        plt.title(title, fontsize=14, fontweight='bold')

        # カスタムレジェンド / Custom legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_abc_matrix(
        self,
        multi_metric_result: pd.DataFrame,
        metric1_col: str,
        metric2_col: str,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        ABC分類マトリックスを作成 / Create ABC classification matrix

        Args:
            multi_metric_result: 複数指標ABC分析結果
            metric1_col: 第1指標のABCカラム名
            metric2_col: 第2指標のABCカラム名
            figsize: 図のサイズ
            save_path: 保存先パス

        Returns:
            matplotlib Figure オブジェクト
        """
        # クロス集計 / Cross tabulation
        matrix = pd.crosstab(
            multi_metric_result[metric1_col],
            multi_metric_result[metric2_col],
            margins=True
        )

        fig, ax = plt.subplots(figsize=figsize)

        # ヒートマップを作成 / Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            cbar_kws={'label': 'Count'},
            ax=ax
        )

        title = 'ABC Classification Matrix' if self.language == 'en' else 'ABC分類マトリックス'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(metric2_col, fontsize=12)
        ax.set_ylabel(metric1_col, fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_category_comparison(
        self,
        comparison_df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        カテゴリ間ABC比較チャートを作成 / Create cross-category ABC comparison chart

        Args:
            comparison_df: cross_category_comparisonの結果
            figsize: 図のサイズ
            save_path: 保存先パス

        Returns:
            matplotlib Figure オブジェクト
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # アイテム数の比較 / Item count comparison
        pivot_items = comparison_df.pivot(
            index=self._get_message('category'),
            columns=self._get_message('abc_category'),
            values=self._get_message('item_count')
        )

        pivot_items.plot(
            kind='bar',
            stacked=True,
            ax=ax1,
            color=['#2ecc71', '#f39c12', '#e74c3c']
        )
        ax1.set_title(
            'Item Count by Category' if self.language == 'en' else 'カテゴリ別アイテム数',
            fontsize=12,
            fontweight='bold'
        )
        ax1.set_ylabel(self._get_message('item_count'))
        ax1.legend(title='ABC')
        ax1.tick_params(axis='x', rotation=45)

        # 値の比較 / Value comparison
        pivot_values = comparison_df.pivot(
            index=self._get_message('category'),
            columns=self._get_message('abc_category'),
            values=self._get_message('value_pct')
        )

        pivot_values.plot(
            kind='bar',
            stacked=True,
            ax=ax2,
            color=['#2ecc71', '#f39c12', '#e74c3c']
        )
        ax2.set_title(
            'Value Percentage by Category' if self.language == 'en' else 'カテゴリ別値の割合',
            fontsize=12,
            fontweight='bold'
        )
        ax2.set_ylabel(self._get_message('value_pct') + ' (%)')
        ax2.legend(title='ABC')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def quick_abc_analysis(
    data: pd.DataFrame,
    value_col: str,
    item_col: str,
    language: str = 'en',
    plot: bool = True
) -> ABCResult:
    """
    クイックABC分析（ヘルパー関数） / Quick ABC analysis (helper function)

    Args:
        data: データフレーム
        value_col: 値を表すカラム名
        item_col: アイテムを識別するカラム名
        language: 出力言語 ('en' or 'ja')
        plot: チャートを表示するかどうか

    Returns:
        ABC分析結果

    Example:
        >>> result = quick_abc_analysis(df, 'revenue', 'product_id', language='ja')
        >>> print(result.summary)
    """
    analyzer = ABCAnalyzer(language=language)
    result = analyzer.classify(data, value_col, item_col)

    if plot:
        analyzer.plot_abc_chart(result)
        plt.show()

    return result


# エイリアス / Aliases for convenience
abc_classify = quick_abc_analysis


if __name__ == "__main__":
    # 使用例 / Usage example
    print("ABC Analysis Module - サンプル実行 / Sample Execution")
    print("=" * 60)

    # サンプルデータを生成 / Generate sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'product_id': [f'P{i:03d}' for i in range(1, 101)],
        'revenue': np.random.pareto(2, 100) * 1000 + 100,
        'profit': np.random.pareto(2, 100) * 200 + 20,
        'quantity': np.random.poisson(50, 100),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food'], 100),
        'store': np.random.choice(['Store_A', 'Store_B', 'Store_C'], 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D')
    })

    # ABC分析を実行 / Perform ABC analysis
    result = quick_abc_analysis(
        sample_data,
        value_col='revenue',
        item_col='product_id',
        language='en',
        plot=False
    )

    print("\n📊 ABC Analysis Summary:")
    print(result.summary)

    print("\n✅ ABC Analysis Module is ready to use!")
    print("モジュールの使用準備が完了しました！")
