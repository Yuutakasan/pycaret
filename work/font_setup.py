# 日本語フォント設定（全ノートブック共通）
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# フォントプロパティオブジェクトを作成
jp_font = None

try:
    import japanize_matplotlib
    japanize_matplotlib.japanize()  # ← 重要：初期化を明示的に呼ぶ

    # 日本語フォントを検索してFontPropertiesを作成
    japanese_fonts = [f.name for f in fm.fontManager.ttflist
                      if 'Gothic' in f.name or 'Noto Sans CJK' in f.name or 'IPA' in f.name]
    if japanese_fonts:
        jp_font = fm.FontProperties(family=japanese_fonts[0])
    else:
        jp_font = fm.FontProperties(family='IPAGothic')

    print("✅ 日本語表示: japanize_matplotlib")
except ImportError:
    # 代替フォント設定
    japanese_fonts = [f.name for f in fm.fontManager.ttflist
                      if 'Gothic' in f.name or 'Noto Sans CJK' in f.name or 'IPA' in f.name]
    if japanese_fonts:
        plt.rcParams['font.family'] = japanese_fonts[0]
        jp_font = fm.FontProperties(family=japanese_fonts[0])
        print(f"✅ 日本語表示: {japanese_fonts[0]}")
    else:
        plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'MS Gothic', 'Yu Gothic', 'sans-serif']
        jp_font = fm.FontProperties(family='IPAGothic')
        print("⚠️ 代替フォント設定（フォント検出失敗）")
    plt.rcParams['axes.unicode_minus'] = False

# matplotlib共通設定
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

# Plotly日本語設定（可能な場合）
try:
    import plotly.io as pio
    pio.templates["japanese"] = pio.templates["plotly"]
    pio.templates["japanese"].layout.font.family = "Meiryo, Yu Gothic, MS Gothic, sans-serif"
    pio.templates.default = "japanese"
    print("✅ Plotly日本語表示設定完了")
except ImportError:
    pass
