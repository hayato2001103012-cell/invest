import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import yfinance as yf
from datetime import timedelta

# -------------------------------------------
# 1. 設定 & タイトル
# -------------------------------------------
st.title("📈 Google株価予測AIアプリ")
st.write("最新の市場データを取得し、AIが「明日の株価」を予測します。")

# -------------------------------------------
# 2. モデルの読み込み
# -------------------------------------------
@st.cache_resource
def load_model():
    # joblibで読み込む
    model = joblib.load('stock_model.pkl')
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("エラー: stock_model.pkl が見つかりません。同じフォルダに置いてください。")
    st.stop()

# -------------------------------------------
# 3. データの自動取得 (yfinance)
# -------------------------------------------
st.write("Fetching latest data from Yahoo Finance...")

TICKER = 'GOOGL'
# 過去データを長めに取得
data = yf.download(TICKER, period='10y', interval='1d')

# yfinanceのデータ整形
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
    
data.columns = [c.lower() for c in data.columns] # close, openなどに小文字化
data.index.name = 'date'

df = data.copy()

# -------------------------------------------
# 4. 特徴量エンジニアリング
# -------------------------------------------
st.subheader("📊 直近のテクニカル分析")

# 期間を絞る（表示用）
df_recent = df[df.index >= '2020-01-01'].copy()

# テクニカル指標の計算
df_recent['MA25'] = df_recent['close'].rolling(window=25).mean()
df_recent['MA75'] = df_recent['close'].rolling(window=75).mean()
sigma = df_recent['close'].rolling(window=25).std()
df_recent['Upper'] = df_recent['MA25'] + 2 * sigma
df_recent['Lower'] = df_recent['MA25'] - 2 * sigma
df_recent['return'] = df_recent['close'].pct_change()

# RSI
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
df_recent['RSI'] = calculate_rsi(df_recent['close'])

# MACD
ema12 = df_recent['close'].ewm(span=12, adjust=False).mean()
ema26 = df_recent['close'].ewm(span=26, adjust=False).mean()
df_recent['MACD'] = ema12 - ema26
df_recent['Signal'] = df_recent['MACD'].ewm(span=9, adjust=False).mean()

# Volatility & Lags
df_recent['Volatility'] = df_recent['return'].rolling(window=20).std()
df_recent['return_lag1'] = df_recent['return'].shift(1)
df_recent['return_lag2'] = df_recent['return'].shift(2)
df_recent['return_lag3'] = df_recent['return'].shift(3)

# 欠損値削除
df_display = df_recent.copy()
df_ml = df_recent.dropna().copy()

# グラフ描画
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_display.index, df_display['close'], label='Close Price', color='gray')
ax.plot(df_display.index, df_display['MA25'], label='MA25', color='orange')
ax.plot(df_display.index, df_display['MA75'], label='MA75', color='blue')
ax.legend()
st.pyplot(fig)

# -------------------------------------------
# 5. 未来予測
# -------------------------------------------
st.subheader("🔮 未来の株価予測")

if st.button('予測を実行する'):
    # 最新のデータ（今日）を使って明日を予測
    latest_data = df_ml.iloc[[-1]]
    
    features = [
        'close', 'MA25', 'MA75', 'Upper', 'Lower',        
        'RSI', 'MACD', 'Signal', 'Volatility',            
        'return', 'return_lag1', 'return_lag2', 'return_lag3'
    ]
    X_latest = latest_data[features]
    
    # 予測実行
    pred_return = model.predict(X_latest)[0]
    
    # --- 日付の計算 ---
    # 基準日（データの最新日付）
    base_date = latest_data.index[0]
    base_date_str = base_date.strftime('%Y年%m月%d日')
    
    # 予想対象日（基準日の翌日）
    # ※ 土日は考慮していませんが、簡易的に+1日としています
    target_date = base_date + timedelta(days=1)
    if target_date.weekday() == 5: # 土曜なら+2日して月曜に
        target_date += timedelta(days=2)
    elif target_date.weekday() == 6: # 日曜なら+1日して月曜に
        target_date += timedelta(days=1)
    target_date_str = target_date.strftime('%Y年%m月%d日')

    # --- 結果の表示 ---
    st.markdown(f"### 📅 {target_date_str} の予想")
    st.caption(f"（※ {base_date_str} の終値データを基準に算出）")
    
    # 変化率をパーセントに変換
    change_pct = pred_return * 100
    
    # カラムを使って見やすくレイアウト
    col1, col2 = st.columns(2)
    
    with col1:
        if pred_return > 0:
            st.markdown("## 📈 **上昇 (UP)**", unsafe_allow_html=True)
            st.metric(label="予想変動率", value=f"+{change_pct:.2f}%", delta="Bullish")
        else:
            st.markdown("## 📉 **下落 (DOWN)**", unsafe_allow_html=True)
            st.metric(label="予想変動率", value=f"{change_pct:.2f}%", delta="-Bearish")

    with col2:
        # 参考情報：現在の株価
        current_price = latest_data['close'].values[0]
        st.metric(label="現在の株価 (基準日)", value=f"${current_price:.2f}")

    # 注意書き
    st.info("※ この予測はAIの学習に基づくものであり、投資勧誘ではありません。市場が休日の場合は翌営業日の予想となります。")