# 警告抑制
import warnings
warnings.filterwarnings('ignore')

# 基本ライブラリ
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 統計・ML
from scipy import stats
from sklearn.linear_model import LinearRegression

# インタラクティブウィジェット
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    WIDGETS_AVAILABLE = True
    print("✅ ipywidgets利用可能")
except ImportError:
    WIDGETS_AVAILABLE = False
    print("⚠️ ipywidgets未インストール - 一部機能制限")

# 日本語フォント
try:
    import japanize_matplotlib
    print("✅ 日本語表示: japanize_matplotlib")
except ImportError:
    plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠️ 代替フォント設定")

# 共通設定
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

# seaborn設定
sns.set_style('whitegrid')
sns.set_palette('husl')
sns.set_context('talk')

# pandas設定
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 120)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# plotly設定
import plotly.io as pio
pio.templates.default = "plotly_white"

print("\n" + "="*80)
print("🏪 店舗別包括ダッシュボード v5.0 - 実務対応版".center(80))
print("="*80)
print(f"\n✅ 環境設定完了")
print(f"   実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
print(f"   Python: {pd.__version__}")
print(f"   pandas: {pd.__version__}")
print(f"   matplotlib: {plt.matplotlib.__version__}")
print(f"   ウィジェット: {'利用可能' if WIDGETS_AVAILABLE else '未インストール'}")
