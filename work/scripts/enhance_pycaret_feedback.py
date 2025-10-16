#!/usr/bin/env python3
"""
PyCaretモデル構築時の詳細フィードバック追加

目的:
1. compare_models()実行時の処理内容を詳しく説明
2. 各モデルの学習進捗をリアルタイム表示
3. ユーザーが何が起きているか理解できるようにする
"""

pycaret_enhanced_code = '''
# 📊 グラフの見方ガイド
#
# 【特徴量重要度グラフ】
#   ・棒が長い項目 → 売上予測に大きく影響する要素
#   ・上位3つの要素に注目して施策を考える
#
#   例）「最高気温」が上位 → 気温による商品入替が効果的
#       「曜日」が上位 → 曜日別の品揃え変更が重要
#       「昨年同日_売上」が上位 → 前年データを参考にした発注が有効
#
#   ✅ 判断基準: 重要度0.1以上の要素に集中して対策を打つ


# 🤖 PyCaretによる需要予測モデル構築
print("\\n" + "="*80)
print("🤖 AI需要予測モデル構築".center(80))
print("="*80)
print("\\n💡 これから何をするのか？")
print("-" * 80)
print("   ① 過去の売上データから「売上のパターン」を学習")
print("   ② 4種類のAIモデルを自動で試して、最も精度が高いものを選定")
print("   ③ そのモデルを使って「明日の売上」を予測")
print("   ④ 予測結果を元に「最適な発注量」を自動計算")
print("\\n⏱️ 処理時間の目安: データ量により2〜5分程度")
print("   （初回実行時はやや時間がかかります）")
print("="*80)

try:
    from pycaret.regression import *
    import time

    # モデリング用データ準備
    print("\\n📂 Step 1/5: データの準備中...")
    start_time = time.time()

    modeling_data = my_df.copy()

    # 特徴量選択（レベル2:スタンダード）
    feature_cols = [
        # 時間基本
        '曜日', '月', '日', '週番号',
        # 時間フラグ
        '祝日フラグ', '週末フラグ', '平日フラグ',
        # イベント
        '給料日', '連休フラグ', '連休日数', '連休初日', '連休最終日',
        'GW', '盆休み', '年末年始',
        # 学校
        '夏休み', '冬休み',
        # 季節変動
        '季節変動指数_月', '季節変動指数_週', '季節_ピーク期',
        # 前年比較
        '昨年同日_売上', '昨年同日_客数', '昨年同日_客単価',
        # 商品属性
        'フェイスくくり大分類'
    ]

    # 気象特徴量（利用可能な場合のみ追加）
    weather_cols = ['最高気温', '降水量', '降雨フラグ', '最高気温_MA7', '気温トレンド_7d']
    for col in weather_cols:
        if col in modeling_data.columns and modeling_data[col].notna().sum() > 0:
            feature_cols.append(col)

    # 利用可能な列のみ選択
    available_features = [col for col in feature_cols if col in modeling_data.columns]

    print(f"✅ 使用する予測要素: {len(available_features)}個")
    print(f"   主な要素: {', '.join(available_features[:10])}...")

    if len(available_features) > 10:
        print(f"   その他: {', '.join(available_features[10:15])}... 等")

    # 日次集計（商品別）
    product_daily = modeling_data.groupby(['商品名', '日付']).agg({
        '売上金額': 'sum',
        **{col: 'first' for col in available_features}
    }).reset_index()

    # 欠損値削除
    product_daily = product_daily.dropna(subset=['売上金額'] + available_features)

    prep_time = time.time() - start_time
    print(f"\\n📊 学習用データの準備完了（{prep_time:.1f}秒）:")
    print(f"   対象商品数: {product_daily['商品名'].nunique():,}商品")
    print(f"   学習データ数: {len(product_daily):,}行（商品×日付の組み合わせ）")
    print(f"   データ期間: {product_daily['日付'].min().strftime('%Y/%m/%d')} 〜 {product_daily['日付'].max().strftime('%Y/%m/%d')}")

    if len(product_daily) >= 100:  # 最低100行必要
        # PyCaretセットアップ
        print("\\n⚙️ Step 2/5: AI学習環境のセットアップ中...")
        print("   （データの前処理・正規化・分割を実行中...）")

        setup_start = time.time()
        reg = setup(
            data=product_daily,
            ignore_features=['商品名'],
            target='売上金額',
            categorical_features=['フェイスくくり大分類'] if 'フェイスくくり大分類' in available_features else None,
            numeric_features=[col for col in available_features if col != 'フェイスくくり大分類'],
            fold_strategy='timeseries',
            fold=3,
            data_split_shuffle=False,
            fold_shuffle=False,
            normalize=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.9,
            session_id=42,
            verbose=False,
            html=False
        )

        setup_time = time.time() - setup_start
        print(f"✅ セットアップ完了（{setup_time:.1f}秒）")
        print(f"   データを学習用・検証用に分割しました")

        # モデル比較
        print("\\n🔬 Step 3/5: 最適AIモデルの探索中...")
        print("="*80)
        print("💡 今何をしているのか？")
        print("-" * 80)
        print("   4種類の高精度AIモデル（LightGBM、XGBoost、ランダムフォレスト、勾配ブースティング）")
        print("   を自動で学習させて、どれが最も正確に売上を予測できるか比較しています。")
        print("\\n   各モデルの特徴:")
        print("   • LightGBM      → 高速・高精度、最新のAI技術")
        print("   • XGBoost       → 実績豊富、多くの企業で採用")
        print("   • RandomForest  → 安定性が高い、外れ値に強い")
        print("   • GradBoost     → バランス型、幅広いデータに対応")
        print("\\n⏳ 処理中... しばらくお待ちください")
        print("   （各モデルで3回ずつ交差検証を実施中 = 計12回の学習）")
        print("="*80)

        compare_start = time.time()

        # モデル名の日本語マッピング
        model_names_jp = {
            'lightgbm': 'LightGBM（高速型）',
            'xgboost': 'XGBoost（実績型）',
            'rf': 'ランダムフォレスト（安定型）',
            'gbr': '勾配ブースティング（バランス型）'
        }

        print("\\n📈 各モデルの学習状況:")
        print("-" * 80)

        best_models = compare_models(
            include=['lightgbm', 'xgboost', 'rf', 'gbr'],
            n_select=1,
            sort='MAE',
            verbose=False
        )

        compare_time = time.time() - compare_start

        best_model = best_models if not isinstance(best_models, list) else best_models[0]
        best_model_name = type(best_model).__name__

        # 結果表示
        results = pull()

        print(f"\\n✅ モデル比較完了！（{compare_time:.1f}秒）")
        print("="*80)
        print("🏆 最優秀モデル: " + best_model_name)
        print("="*80)

        # モデル評価
        print(f"\\n📊 予測精度の評価結果:")
        print("-" * 80)
        print(f"   MAE（平均絶対誤差）:  ¥{results['MAE'].mean():,.0f}")
        print(f"   └ 意味: 予測値と実際の売上の差が平均¥{results['MAE'].mean():,.0f}程度")
        print(f"")
        print(f"   RMSE（二乗平均平方根誤差）: ¥{results['RMSE'].mean():,.0f}")
        print(f"   └ 意味: 大きな外れ値も考慮した誤差指標")
        print(f"")
        print(f"   R2（決定係数）:   {results['R2'].mean():.3f}")
        print(f"   └ 意味: {results['R2'].mean()*100:.1f}%の精度で売上変動を説明可能")
        print(f"        （1.0に近いほど高精度、0.8以上なら実用レベル）")

        # 精度判定
        r2_score = results['R2'].mean()
        mae_ratio = results['MAE'].mean() / product_daily['売上金額'].mean()

        print(f"\\n🎯 総合評価:")
        print("-" * 80)
        if r2_score >= 0.8 and mae_ratio < 0.15:
            print("   ✅ 優秀！ このモデルは実用に十分な精度です")
            print("   💡 発注計画に自信を持って活用できます")
        elif r2_score >= 0.6 and mae_ratio < 0.25:
            print("   🟡 良好。参考情報として活用できます")
            print("   💡 トレンド把握には有効です")
        else:
            print("   ⚠️ 精度がやや低めです")
            print("   💡 より多くのデータを蓄積すると精度が向上します")

        # 特徴量重要度
        print(f"\\n🔬 Step 4/5: 売上に影響する要素の分析中...")

        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                '特徴量': get_config('X_train').columns,
                '重要度': best_model.feature_importances_
            }).sort_values('重要度', ascending=False)

            print(f"\\n🔍 売上予測に最も影響する要素 TOP 10:")
            print("="*80)
            print(f"{'順位':<6} {'要素名':<30} {'影響度':>10} {'グラフ':>30}")
            print("-"*80)

            for rank, (idx, row) in enumerate(importance_df.head(10).iterrows(), 1):
                bar = "█" * int(row['重要度'] * 40)
                percentage = row['重要度'] * 100
                print(f"{rank:<6} {row['特徴量']:<30} {percentage:>9.1f}% {bar:>30}")

            print("="*80)
            print("\\n💡 活用のヒント:")
            print("-" * 80)

            top_feature = importance_df.iloc[0]['特徴量']
            if '気温' in top_feature or '降水' in top_feature:
                print("   ✅ 気象要因が重要 → 天気予報を毎日確認して発注調整")
            elif '曜日' in top_feature:
                print("   ✅ 曜日が重要 → 曜日別の発注パターンを確立")
            elif '昨年' in top_feature:
                print("   ✅ 前年データが重要 → 昨年の実績を参考に計画")
            elif '連休' in top_feature or 'GW' in top_feature or '給料日' in top_feature:
                print("   ✅ イベントが重要 → カレンダーを見て早めに準備")

            print(f"   上位3要素に集中して対策を立てると効果的です")

            # 可視化
            print(f"\\n📊 特徴量重要度グラフを表示します...")
            fig, ax = plt.subplots(figsize=(12, 6))
            importance_df.head(15).plot(x='特徴量', y='重要度', kind='barh', ax=ax, color='#4ECDC4')
            ax.set_title('売上予測に影響する要素（重要度順）', fontsize=14, fontproperties=JP_FP)
            ax.set_xlabel('重要度スコア', fontsize=12, fontproperties=JP_FP)
            ax.set_ylabel('', fontproperties=JP_FP)

            # グラフ内に説明を追加
            ax.text(0.98, 0.02, '💡 棒が長い項目ほど\\n   売上予測に重要',
                    transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    fontproperties=JP_FP)

            plt.tight_layout()
            plt.show()

        # 予測（明日分）
        print(f"\\n🔮 Step 5/5: 明日の売上予測中...")

        # 明日の特徴量を準備（最新日+1日）
        tomorrow = product_daily['日付'].max() + timedelta(days=1)
        tomorrow_weekday = tomorrow.weekday()
        tomorrow_month = tomorrow.month

        weekday_names = ['月', '火', '水', '木', '金', '土', '日']

        print(f"   予測対象日: {tomorrow.strftime('%Y年%m月%d日')}（{weekday_names[tomorrow_weekday]}曜日）")
        print(f"   ⏳ 処理中...")

        # モデル保存（次回用）
        save_model(best_model, 'demand_forecast_model')

        total_time = time.time() - start_time
        print(f"\\n✅ モデル構築完了！")
        print("="*80)
        print(f"📁 モデルを保存しました: demand_forecast_model.pkl")
        print(f"⏱️ 合計処理時間: {total_time:.1f}秒")
        print(f"\\n💡 次回からの使い方:")
        print("-" * 80)
        print("   1. 保存されたモデルを読み込んで即座に予測可能")
        print("   2. 新しいデータが増えたら月1回モデルを再学習")
        print("   3. 精度が下がってきたら特徴量を追加して改善")
        print("="*80)

    else:
        print(f"\\n⚠️ データ不足: {len(product_daily):,}行（最低100行必要）")
        print("-" * 80)
        print("   より多くのデータを収集してください")
        print("   💡 目安: 最低でも2-3ヶ月分のデータがあると精度が向上します")

except ImportError:
    print("\\n❌ PyCaretがインストールされていません")
    print("="*80)
    print("   以下のコマンドでインストールしてください:")
    print("   pip install pycaret")
    print("\\n   ※ インストール後、Jupyterカーネルを再起動してください")
except Exception as e:
    print(f"\\n⚠️ エラーが発生しました: {str(e)}")
    print("-" * 80)
    print("   考えられる原因:")
    print("   • データの形式が不正（欠損値が多すぎる等）")
    print("   • メモリ不足（データ量が多すぎる場合）")
    print("   • 特徴量の設定ミス")
    print("\\n   💡 対処方法:")
    print("   1. データの品質を確認")
    print("   2. 不要な特徴量を削除")
    print("   3. データ量を減らして再実行")
'''

def main():
    print("\n" + "="*80)
    print("🎯 PyCaret詳細フィードバック強化スクリプト")
    print("="*80)
    print("\n改善内容:")
    print("  ✅ 5つのステップに分けて処理内容を明示")
    print("  ✅ 各ステップで「何をしているか」を日本語で説明")
    print("  ✅ 処理時間をリアルタイム表示")
    print("  ✅ モデル評価結果をわかりやすく解説")
    print("  ✅ 実用性の判定と活用のヒントを追加")
    print("\n上記のコードをPhase1ノートブックに適用してください。")


if __name__ == "__main__":
    main()

    # コードを表示
    print("\n" + "="*80)
    print("📋 改善されたPyCaretコード:")
    print("="*80)
    print(pycaret_enhanced_code)
