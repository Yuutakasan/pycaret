"""
Dashboard Visualization Framework
Interactive plotly dashboards with comprehensive analytics visualization
日本語ラベル対応
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Noto Sans CJK JP', 'IPAexGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# カラーパレット定義
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}


class DashboardVisualizer:
    """
    ダッシュボード可視化フレームワーク
    Interactive dashboards with drill-down capabilities
    """

    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize dashboard visualizer

        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn')
        """
        self.theme = theme
        self.font_family = "Noto Sans CJK JP, sans-serif"

    def create_kpi_cards(
        self,
        kpis: Dict[str, Dict[str, Any]],
        sparkline_data: Optional[Dict[str, List[float]]] = None,
        height: int = 150
    ) -> go.Figure:
        """
        Create KPI cards with sparklines and trend indicators

        Args:
            kpis: Dictionary with KPI data
                  {name: {'value': float, 'target': float, 'unit': str, 'change': float}}
            sparkline_data: Optional sparkline data {name: [values]}
            height: Card height in pixels

        Returns:
            Plotly figure with KPI cards
        """
        n_kpis = len(kpis)
        cols = min(4, n_kpis)
        rows = (n_kpis + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"<b>{name}</b>" for name in kpis.keys()],
            specs=[[{'type': 'indicator'}] * cols for _ in range(rows)],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        for idx, (name, data) in enumerate(kpis.items()):
            row = idx // cols + 1
            col = idx % cols + 1

            value = data['value']
            target = data.get('target', None)
            unit = data.get('unit', '')
            change = data.get('change', 0)

            # Determine trend color
            delta_color = 'green' if change >= 0 else 'red'

            # Create indicator
            indicator = go.Indicator(
                mode="number+delta",
                value=value,
                delta={
                    'reference': value - change if change else None,
                    'relative': False,
                    'valueformat': '.2f',
                    'increasing': {'color': 'green'},
                    'decreasing': {'color': 'red'}
                },
                number={
                    'suffix': f' {unit}',
                    'font': {'size': 32, 'family': self.font_family}
                },
                domain={'x': [0, 1], 'y': [0.3, 1]}
            )

            fig.add_trace(indicator, row=row, col=col)

            # Add sparkline if data provided
            if sparkline_data and name in sparkline_data:
                spark_data = sparkline_data[name]
                x_vals = list(range(len(spark_data)))

                sparkline = go.Scatter(
                    x=x_vals,
                    y=spark_data,
                    mode='lines',
                    line=dict(color=COLOR_PALETTE['primary'], width=2),
                    fill='tozeroy',
                    fillcolor=f"rgba(31, 119, 180, 0.2)",
                    showlegend=False,
                    hovertemplate='値: %{y:.2f}<extra></extra>'
                )

                fig.add_trace(sparkline, row=row, col=col)

                # Update axes for sparkline
                fig.update_xaxes(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    row=row,
                    col=col,
                    domain=[0, 1],
                    range=[0, len(spark_data)-1]
                )
                fig.update_yaxes(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    row=row,
                    col=col,
                    domain=[0, 0.25]
                )

        fig.update_layout(
            height=height * rows,
            template=self.theme,
            font=dict(family=self.font_family),
            showlegend=False,
            margin=dict(t=80, b=20, l=20, r=20)
        )

        return fig

    def create_heatmap(
        self,
        data: pd.DataFrame,
        x_label: str = "列",
        y_label: str = "行",
        title: str = "ヒートマップ",
        colorscale: str = 'RdYlGn',
        annotations: bool = True
    ) -> go.Figure:
        """
        Create interactive heatmap for multi-dimensional analysis

        Args:
            data: DataFrame with numerical data
            x_label: X-axis label
            y_label: Y-axis label
            title: Chart title
            colorscale: Color scale ('RdYlGn', 'Viridis', 'Blues', etc.)
            annotations: Show value annotations

        Returns:
            Plotly heatmap figure
        """
        # Create annotations if requested
        annot_text = None
        if annotations:
            annot_text = [[f"{val:.2f}" for val in row] for row in data.values]

        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns.tolist(),
            y=data.index.tolist(),
            colorscale=colorscale,
            text=annot_text,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=f'{y_label}: %{{y}}<br>{x_label}: %{{x}}<br>値: %{{z:.2f}}<extra></extra>',
            colorbar=dict(title="値")
        ))

        fig.update_layout(
            title={
                'text': f"<b>{title}</b>",
                'font': {'size': 20, 'family': self.font_family}
            },
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme,
            font=dict(family=self.font_family),
            height=max(400, len(data) * 30),
            margin=dict(t=100, b=80, l=100, r=80)
        )

        return fig

    def create_comparison_chart(
        self,
        data: pd.DataFrame,
        categories: List[str],
        values_col: str,
        group_col: str,
        title: str = "比較チャート",
        chart_type: str = 'bar',
        benchmark: Optional[pd.Series] = None
    ) -> go.Figure:
        """
        Create comparison charts (store vs benchmark)

        Args:
            data: DataFrame with comparison data
            categories: Category column name or list
            values_col: Values column name
            group_col: Grouping column name
            title: Chart title
            chart_type: 'bar', 'line', or 'radar'
            benchmark: Optional benchmark series

        Returns:
            Plotly comparison figure
        """
        if chart_type == 'radar':
            return self._create_radar_comparison(data, categories, values_col, group_col, title, benchmark)

        fig = go.Figure()

        # Add traces for each group
        groups = data[group_col].unique()
        colors = px.colors.qualitative.Set2[:len(groups)]

        for idx, group in enumerate(groups):
            group_data = data[data[group_col] == group]

            if chart_type == 'bar':
                trace = go.Bar(
                    name=str(group),
                    x=group_data[categories] if isinstance(categories, str) else categories,
                    y=group_data[values_col],
                    marker_color=colors[idx],
                    text=group_data[values_col].round(2),
                    textposition='outside',
                    hovertemplate='カテゴリ: %{x}<br>値: %{y:.2f}<extra></extra>'
                )
            else:  # line
                trace = go.Scatter(
                    name=str(group),
                    x=group_data[categories] if isinstance(categories, str) else categories,
                    y=group_data[values_col],
                    mode='lines+markers',
                    line=dict(color=colors[idx], width=3),
                    marker=dict(size=8),
                    hovertemplate='カテゴリ: %{x}<br>値: %{y:.2f}<extra></extra>'
                )

            fig.add_trace(trace)

        # Add benchmark line if provided
        if benchmark is not None:
            fig.add_trace(go.Scatter(
                name='ベンチマーク',
                x=benchmark.index.tolist(),
                y=benchmark.values,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='ベンチマーク: %{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title={
                'text': f"<b>{title}</b>",
                'font': {'size': 20, 'family': self.font_family}
            },
            xaxis_title="カテゴリ",
            yaxis_title="値",
            template=self.theme,
            font=dict(family=self.font_family),
            barmode='group' if chart_type == 'bar' else None,
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def _create_radar_comparison(
        self,
        data: pd.DataFrame,
        categories: List[str],
        values_col: str,
        group_col: str,
        title: str,
        benchmark: Optional[pd.Series] = None
    ) -> go.Figure:
        """Create radar/spider chart for comparison"""
        fig = go.Figure()

        groups = data[group_col].unique()
        colors = px.colors.qualitative.Set2[:len(groups)]

        for idx, group in enumerate(groups):
            group_data = data[data[group_col] == group]

            fig.add_trace(go.Scatterpolar(
                r=group_data[values_col].tolist() + [group_data[values_col].iloc[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=str(group),
                line_color=colors[idx],
                hovertemplate='%{theta}<br>値: %{r:.2f}<extra></extra>'
            ))

        if benchmark is not None:
            fig.add_trace(go.Scatterpolar(
                r=benchmark.tolist() + [benchmark.iloc[0]],
                theta=categories + [categories[0]],
                name='ベンチマーク',
                line=dict(color='red', dash='dash', width=2),
                hovertemplate='ベンチマーク: %{r:.2f}<extra></extra>'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, data[values_col].max() * 1.1]
                )
            ),
            title={
                'text': f"<b>{title}</b>",
                'font': {'size': 20, 'family': self.font_family}
            },
            template=self.theme,
            font=dict(family=self.font_family),
            height=500,
            showlegend=True
        )

        return fig

    def create_timeseries_plot(
        self,
        data: pd.DataFrame,
        date_col: str,
        value_cols: Union[str, List[str]],
        title: str = "時系列プロット",
        annotations: Optional[List[Dict]] = None,
        forecast: Optional[pd.DataFrame] = None,
        confidence_intervals: bool = True
    ) -> go.Figure:
        """
        Create time-series plots with annotations

        Args:
            data: DataFrame with time series data
            date_col: Date column name
            value_cols: Value column name(s)
            title: Chart title
            annotations: List of annotations {date, text, color}
            forecast: Optional forecast DataFrame
            confidence_intervals: Show confidence intervals for forecast

        Returns:
            Plotly time series figure
        """
        fig = go.Figure()

        # Convert to list if single column
        if isinstance(value_cols, str):
            value_cols = [value_cols]

        colors = px.colors.qualitative.Set2[:len(value_cols)]

        # Plot actual data
        for idx, col in enumerate(value_cols):
            fig.add_trace(go.Scatter(
                x=data[date_col],
                y=data[col],
                name=col,
                mode='lines+markers',
                line=dict(color=colors[idx], width=2),
                marker=dict(size=6),
                hovertemplate='日付: %{x}<br>値: %{y:.2f}<extra></extra>'
            ))

        # Add forecast if provided
        if forecast is not None:
            for idx, col in enumerate(value_cols):
                if col in forecast.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast[col],
                        name=f'{col} (予測)',
                        mode='lines',
                        line=dict(color=colors[idx], width=2, dash='dash'),
                        hovertemplate='日付: %{x}<br>予測値: %{y:.2f}<extra></extra>'
                    ))

                    # Add confidence intervals
                    if confidence_intervals and f'{col}_lower' in forecast.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast.index.tolist() + forecast.index.tolist()[::-1],
                            y=forecast[f'{col}_upper'].tolist() + forecast[f'{col}_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor=f'rgba({colors[idx]}, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{col} (信頼区間)',
                            showlegend=True,
                            hoverinfo='skip'
                        ))

        # Add annotations
        if annotations:
            for annot in annotations:
                fig.add_vline(
                    x=annot['date'],
                    line_dash="dash",
                    line_color=annot.get('color', 'gray'),
                    annotation_text=annot['text'],
                    annotation_position="top"
                )

        fig.update_layout(
            title={
                'text': f"<b>{title}</b>",
                'font': {'size': 20, 'family': self.font_family}
            },
            xaxis_title="日付",
            yaxis_title="値",
            template=self.theme,
            font=dict(family=self.font_family),
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def create_sankey_diagram(
        self,
        source: List[str],
        target: List[str],
        values: List[float],
        labels: Optional[List[str]] = None,
        title: str = "フロー分析"
    ) -> go.Figure:
        """
        Create Sankey diagrams for flow analysis

        Args:
            source: Source node names
            target: Target node names
            values: Flow values
            labels: Optional custom labels
            title: Chart title

        Returns:
            Plotly Sankey diagram
        """
        # Get unique nodes
        all_nodes = list(set(source + target))
        if labels is None:
            labels = all_nodes

        # Create mapping from names to indices
        node_dict = {name: idx for idx, name in enumerate(all_nodes)}

        # Convert to indices
        source_idx = [node_dict[s] for s in source]
        target_idx = [node_dict[t] for t in target]

        # Create color palette for nodes
        node_colors = px.colors.qualitative.Set2[:len(all_nodes)]

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=source_idx,
                target=target_idx,
                value=values,
                hovertemplate='%{source.label} → %{target.label}<br>値: %{value:.2f}<extra></extra>'
            )
        )])

        fig.update_layout(
            title={
                'text': f"<b>{title}</b>",
                'font': {'size': 20, 'family': self.font_family}
            },
            font=dict(size=12, family=self.font_family),
            height=600,
            template=self.theme
        )

        return fig

    def create_drill_down_dashboard(
        self,
        summary_data: pd.DataFrame,
        detail_data: Dict[str, pd.DataFrame],
        summary_metric: str,
        category_col: str,
        title: str = "ドリルダウン分析"
    ) -> go.Figure:
        """
        Create interactive dashboard with drill-down capabilities

        Args:
            summary_data: Summary level DataFrame
            detail_data: Dictionary of detailed DataFrames by category
            summary_metric: Metric column name in summary
            category_col: Category column name
            title: Dashboard title

        Returns:
            Plotly figure with drill-down capability
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "概要", "カテゴリ別分布",
                "詳細トレンド", "比較分析"
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Summary bar chart
        fig.add_trace(
            go.Bar(
                x=summary_data[category_col],
                y=summary_data[summary_metric],
                name="概要",
                marker_color=COLOR_PALETTE['primary'],
                text=summary_data[summary_metric].round(2),
                textposition='outside',
                hovertemplate='%{x}<br>値: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=summary_data[category_col],
                values=summary_data[summary_metric],
                name="分布",
                hovertemplate='%{label}<br>値: %{value:.2f}<br>割合: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )

        # Detail trend (using first category as example)
        if detail_data:
            first_category = list(detail_data.keys())[0]
            detail_df = detail_data[first_category]

            if 'date' in detail_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=detail_df['date'],
                        y=detail_df[summary_metric] if summary_metric in detail_df.columns else detail_df.iloc[:, 1],
                        name=f"{first_category} トレンド",
                        mode='lines+markers',
                        line=dict(color=COLOR_PALETTE['secondary'], width=2),
                        hovertemplate='日付: %{x}<br>値: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )

        # Comparison analysis (top vs bottom performers)
        sorted_data = summary_data.sort_values(summary_metric, ascending=False)
        top_5 = sorted_data.head(5)

        fig.add_trace(
            go.Bar(
                x=top_5[category_col],
                y=top_5[summary_metric],
                name="トップ5",
                marker_color=COLOR_PALETTE['success'],
                text=top_5[summary_metric].round(2),
                textposition='outside',
                hovertemplate='%{x}<br>値: %{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title={
                'text': f"<b>{title}</b>",
                'font': {'size': 24, 'family': self.font_family}
            },
            template=self.theme,
            font=dict(family=self.font_family),
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )

        return fig

    def create_responsive_layout(
        self,
        figures: List[go.Figure],
        layout: str = 'grid',
        mobile_breakpoint: int = 768
    ) -> str:
        """
        Create responsive HTML layout for multiple figures

        Args:
            figures: List of Plotly figures
            layout: 'grid', 'stack', or 'tabs'
            mobile_breakpoint: Breakpoint for mobile layout (px)

        Returns:
            HTML string with responsive layout
        """
        html_parts = [
            """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {
                        font-family: 'Noto Sans CJK JP', sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }
                    .dashboard-container {
                        max-width: 1400px;
                        margin: 0 auto;
                    }
                    .chart-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                        gap: 20px;
                        margin-bottom: 20px;
                    }
                    .chart-container {
                        background: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        padding: 15px;
                    }
                    @media (max-width: """ + str(mobile_breakpoint) + """px) {
                        .chart-grid {
                            grid-template-columns: 1fr;
                        }
                        body {
                            padding: 10px;
                        }
                    }
                    .tabs {
                        display: flex;
                        border-bottom: 2px solid #ddd;
                        margin-bottom: 20px;
                    }
                    .tab {
                        padding: 10px 20px;
                        cursor: pointer;
                        border: none;
                        background: none;
                        font-size: 16px;
                    }
                    .tab.active {
                        border-bottom: 3px solid #1f77b4;
                        font-weight: bold;
                    }
                    .tab-content {
                        display: none;
                    }
                    .tab-content.active {
                        display: block;
                    }
                </style>
            </head>
            <body>
                <div class="dashboard-container">
            """
        ]

        if layout == 'grid':
            html_parts.append('<div class="chart-grid">')
            for idx, fig in enumerate(figures):
                html_parts.append(f'<div class="chart-container" id="chart-{idx}"></div>')
            html_parts.append('</div>')

        elif layout == 'tabs':
            html_parts.append('<div class="tabs">')
            for idx in range(len(figures)):
                active = 'active' if idx == 0 else ''
                html_parts.append(
                    f'<button class="tab {active}" onclick="showTab({idx})">チャート {idx+1}</button>'
                )
            html_parts.append('</div>')

            for idx, fig in enumerate(figures):
                active = 'active' if idx == 0 else ''
                html_parts.append(
                    f'<div class="tab-content {active}" id="tab-{idx}">'
                    f'<div class="chart-container" id="chart-{idx}"></div>'
                    f'</div>'
                )

        else:  # stack
            for idx, fig in enumerate(figures):
                html_parts.append(f'<div class="chart-container" id="chart-{idx}"></div>')

        html_parts.append('</div>')

        # Add JavaScript for rendering charts
        html_parts.append('<script>')
        for idx, fig in enumerate(figures):
            fig_json = fig.to_json()
            html_parts.append(f'''
                var figure_{idx} = {fig_json};
                Plotly.newPlot('chart-{idx}', figure_{idx}.data, figure_{idx}.layout, {{responsive: true}});
            ''')

        if layout == 'tabs':
            html_parts.append('''
                function showTab(idx) {
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    document.querySelectorAll('.tab')[idx].classList.add('active');
                    document.getElementById('tab-' + idx).classList.add('active');
                }
            ''')

        html_parts.append('</script></body></html>')

        return ''.join(html_parts)


class MatplotlibVisualizer:
    """
    Matplotlib/Seaborn based visualizations for static reports
    """

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """Initialize matplotlib visualizer"""
        plt.style.use('default')
        sns.set_palette("Set2")

    def create_professional_report(
        self,
        data: Dict[str, pd.DataFrame],
        output_path: str,
        title: str = "分析レポート"
    ) -> None:
        """
        Create professional multi-page PDF report

        Args:
            data: Dictionary of DataFrames for different sections
            output_path: Output PDF path
            title: Report title
        """
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(output_path) as pdf:
            # Cover page
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.5, title, ha='center', va='center', size=32, weight='bold')
            fig.text(0.5, 0.4, f'生成日: {datetime.now().strftime("%Y年%m月%d日")}',
                    ha='center', va='center', size=16)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Data pages
            for section_name, df in data.items():
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
                fig.suptitle(section_name, size=20, weight='bold')

                # Summary statistics
                if len(df.select_dtypes(include=[np.number]).columns) > 0:
                    df.select_dtypes(include=[np.number]).describe().plot(
                        kind='bar', ax=axes[0, 0]
                    )
                    axes[0, 0].set_title('統計サマリー')
                    axes[0, 0].tick_params(axis='x', rotation=45)

                # Distribution
                if len(df.select_dtypes(include=[np.number]).columns) > 0:
                    df.select_dtypes(include=[np.number]).iloc[:, 0].hist(
                        ax=axes[0, 1], bins=30, edgecolor='black'
                    )
                    axes[0, 1].set_title('分布')

                # Correlation heatmap
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    sns.heatmap(
                        df[numeric_cols].corr(),
                        ax=axes[1, 0],
                        annot=True,
                        fmt='.2f',
                        cmap='coolwarm',
                        center=0
                    )
                    axes[1, 0].set_title('相関マトリックス')

                # Time series or categorical
                if len(df) > 0:
                    if 'date' in df.columns:
                        df.set_index('date')[numeric_cols[0]].plot(ax=axes[1, 1])
                        axes[1, 1].set_title('時系列トレンド')
                    else:
                        df[numeric_cols[0]].plot(kind='line', ax=axes[1, 1])
                        axes[1, 1].set_title('トレンド')

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()


# Utility functions
def save_dashboard(fig: go.Figure, path: str, format: str = 'html') -> None:
    """
    Save dashboard to file

    Args:
        fig: Plotly figure
        path: Output path
        format: 'html', 'png', 'jpg', 'svg', or 'pdf'
    """
    if format == 'html':
        fig.write_html(path)
    else:
        fig.write_image(path, format=format)


def combine_figures(
    figures: List[go.Figure],
    layout: str = 'vertical',
    titles: Optional[List[str]] = None
) -> go.Figure:
    """
    Combine multiple figures into one

    Args:
        figures: List of figures to combine
        layout: 'vertical', 'horizontal', or 'grid'
        titles: Optional subplot titles

    Returns:
        Combined figure
    """
    n_figs = len(figures)

    if layout == 'vertical':
        rows, cols = n_figs, 1
    elif layout == 'horizontal':
        rows, cols = 1, n_figs
    else:  # grid
        cols = int(np.ceil(np.sqrt(n_figs)))
        rows = int(np.ceil(n_figs / cols))

    specs = [[{'type': 'xy'}] * cols for _ in range(rows)]

    combined_fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles or [f"図 {i+1}" for i in range(n_figs)],
        specs=specs,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    for idx, fig in enumerate(figures):
        row = idx // cols + 1
        col = idx % cols + 1

        for trace in fig.data:
            combined_fig.add_trace(trace, row=row, col=col)

    combined_fig.update_layout(
        height=400 * rows,
        showlegend=True,
        font=dict(family="Noto Sans CJK JP, sans-serif")
    )

    return combined_fig
