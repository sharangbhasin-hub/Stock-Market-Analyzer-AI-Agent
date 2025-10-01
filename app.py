import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import openai
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import gspread
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
import time
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import pandas_ta as ta


warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GMAIL_EMAIL = os.getenv("GMAIL_EMAI")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# KITE_API_KEY = os.getenv("KITE_API_KEY") # Add these to your .env file
# KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN") # Add these to your .env file

# Configure the Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Static Data & Mappings ---
STOCK_CATEGORIES = {
    "NIFTY 50 Index": {
        "ticker": "^NSEI",
        "individual_stocks": {
            "Reliance Industries": "RELIANCE.NS",
            "Tata Consultancy Services": "TCS.NS",
        }
    },
    "BANK NIFTY Index": {
        "ticker": "^NSEBANK",
        "individual_stocks": {
            "HDFC Bank": "HDFCBANK.NS",
            "ICICI Bank": "ICICIBANK.NS",
        }
    },
}

# ==============================================================================
# === GLOBAL HELPER FUNCTIONS ==================================================
# ==============================================================================

# NOTE: For intraday functions to work, you need to implement a real-time data provider
# The functions below are placeholders using yfinance, which has delayed data
def fetch_intraday_data(ticker, interval="5m"):
    """Fetches intraday data using yfinance. NOTE: Data is delayed."""
    try:
        data = yf.download(ticker, period="1d", interval=interval)
        if data.empty:
            return None
        # yfinance column names are capitalized, converting to lowercase for consistency
        data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        return data
    except Exception as e:
        st.error(f"Could not fetch intraday data for {ticker}: {e}")
        return None


@st.cache_data(ttl=3600)
def run_pre_market_screener():
    """
    Downloads all NSE stock symbols, fetches their previous day's data,
    and filters them based on price and volume criteria.
    """
    st.write("Running Pre-Market Screener...")
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df_all_stocks = pd.read_csv(url)
        nse_symbols = [f"{symbol}.NS" for symbol in df_all_stocks['SYMBOL']]
        
        # Limit stocks for speed during development
        symbols_to_scan = nse_symbols[:100]
        st.write(f"Found {len(nse_symbols)} total stocks. Scanning the first {len(symbols_to_scan)} for this run...")

        tickers_str = " ".join(symbols_to_scan)
        data = yf.download(tickers_str, period="5d", group_by='ticker', auto_adjust=True)

        screened_list = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(symbols_to_scan):
            try:
                stock_data = data[ticker]
                if not stock_data.empty:
                    last_day = stock_data.iloc[-1]
                    price = last_day['Close']
                    volume = last_day['Volume']
                    
                    if price > 100 and volume > 100000:
                        screened_list[ticker] = {'price': price, 'volume': volume}
            except Exception:
                continue
            progress_bar.progress((i + 1) / len(symbols_to_scan))

        st.write(f"Screening complete. Found {len(screened_list)} stocks meeting your criteria.")
        return screened_list
    except Exception as e:
        st.error(f"An error occurred during the pre-market scan: {e}")
        return {}


@st.cache_data
def fetch_stock_data(ticker, period="1y"):
    """Fetch daily stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            st.error(f"No historical data found for ticker: {ticker}.")
            return None
        return hist
    except Exception as e:
        st.error(f"Error fetching data for '{ticker}': {e}")
        return None

def create_intraday_chart(data, ticker_name):
    """Creates an intraday candlestick chart with VWAP."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=(f'Price Chart for {ticker_name}', 'Volume'),
                        row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Candles'), row=1, col=1)
    if 'vwap' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['vwap'], mode='lines', name='VWAP', line=dict(color='yellow', width=2)), row=1, col=1)
    colors = ['green' if row['close'] >= row['open'] else 'red' for index, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color=colors), row=2, col=1)
    fig.update_layout(title_text=f'Intraday Analysis', xaxis_rangeslider_visible=False, template='plotly_dark')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def create_indicator_charts(data, rsi, macd, ma_data):
    """Create indicator charts for daily analysis."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(data.index[-30:], data['Close'][-30:], label='Close Price', linewidth=2, color='black')
    if len(data) >= 50:
        ma_50_series = data['Close'].rolling(window=50).mean()
        axes[0].plot(data.index[-30:], ma_50_series[-30:], color='blue', linestyle='--', label='MA 50', alpha=0.8)
    axes[0].set_title('Price & Moving Averages')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # ... (Rest of plotting logic for RSI and MACD)
    plt.tight_layout()
    return fig

# ==============================================================================
# === PRIMARY ANALYSIS CLASS ===================================================
# ==============================================================================
class StockAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        self.setup_sentiment_analyzer()

    def setup_sentiment_analyzer(self):
        """Initialize sentiment analysis pipeline"""
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True)
        except Exception as e:
            st.warning(f"Using default sentiment analyzer due to: {e}")
            self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    # --- INDICATOR METHODS ---
    def compute_rsi(self, data, window=14):
        """Calculate RSI (Relative Strength Index)"""
        return data.ta.rsi(length=window).iloc[-1]

    def compute_vwap(self, data):
        """Calculates the Volume Weighted Average Price (VWAP)."""
        if all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            data.ta.vwap(append=True)
        return data

    # --- INTRADAY ANALYSIS METHODS ---
    def run_multi_timeframe_analysis(self, ticker):
        analysis_results = {}
        daily_data = fetch_stock_data(ticker, period="6mo")
        if daily_data is not None and not daily_data.empty:
            daily_data['SMA50'] = daily_data['Close'].rolling(window=50).mean()
            last_price = daily_data['Close'].iloc[-1]
            last_sma50 = daily_data['SMA50'].iloc[-1]
            analysis_results['trend'] = "Uptrend" if last_price > last_sma50 else "Downtrend"
        else:
            analysis_results['trend'] = "Unknown"

        fifteen_min_data = fetch_intraday_data(ticker, interval="15m")
        if fifteen_min_data is not None and not fifteen_min_data.empty:
            analysis_results['support'] = fifteen_min_data['low'].min()
            analysis_results['resistance'] = fifteen_min_data['high'].max()
        else:
            analysis_results['support'] = 0
            analysis_results['resistance'] = 0

        five_min_data = fetch_intraday_data(ticker, interval="5m")
        if five_min_data is not None and not five_min_data.empty:
            analysis_results['5m_data'] = five_min_data
            analysis_results['latest_price'] = five_min_data['close'].iloc[-1]
        else:
            analysis_results['5m_data'] = pd.DataFrame()
            analysis_results['latest_price'] = daily_data['Close'].iloc[-1] if daily_data is not None and not daily_data.empty else 0
        
        return analysis_results

    def detect_breakout_retest(self, five_min_data, resistance):
        if five_min_data.empty or resistance == 0:
            return "Not Analyzed"
        recent_data = five_min_data.tail(20)
        breakout_candle_index = -1
        for i in range(1, len(recent_data)):
            if recent_data['high'].iloc[i] > resistance and recent_data['high'].iloc[i-1] <= resistance:
                breakout_candle_index = i
        if breakout_candle_index == -1:
            return "No Breakout Detected"
        
        retest_candle_index = -1
        for i in range(breakout_candle_index + 1, len(recent_data)):
            if recent_data['low'].iloc[i] <= resistance:
                retest_candle_index = i
                break
        if retest_candle_index == -1:
            return f"Breakout Occurred. Awaiting Retest."
        
        if retest_candle_index < len(recent_data) - 1:
            confirmation_candle = recent_data.iloc[retest_candle_index + 1]
            if confirmation_candle['close'] > resistance:
                return f"✅ Retest Confirmed. Potential Entry."
        return f"Retest in Progress. Awaiting Confirmation."

    def check_candlestick_pattern(self, five_min_data):
        if len(five_min_data) < 3:
            return "Not enough data"
        c1, c2, c3 = five_min_data.iloc[-3], five_min_data.iloc[-2], five_min_data.iloc[-1]
        if (c1['close'] < c1['open'] and abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) and c3['close'] > c3['open'] and c3['close'] > c1['open']):
            return "Morning Star (Bullish)"
        if (c1['close'] > c1['open'] and abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) and c3['close'] < c3['open'] and c3['close'] < c1['open']):
            return "Evening Star (Bearish)"
        return "No significant pattern"

    def run_confirmation_checklist(self, analysis_results):
        checklist = {
            "1. At Key S/R Level": "⚠️ PENDING", "2. Price Rejection": "⚠️ PENDING",
            "3. Chart Pattern Confirmed": "⚠️ PENDING", "4. Candlestick Signal": "⚠️ PENDING",
            "5. Indicator Alignment": "⚠️ PENDING", "FINAL_SIGNAL": "HOLD"
        }
        five_min_df = analysis_results['5m_data']
        if five_min_df.empty: return checklist

        resistance = analysis_results['resistance']
        support = analysis_results['support']
        latest_price = analysis_results['latest_price']

        at_support = abs(latest_price - support) / support < 0.005 if support > 0 else False
        if at_support:
            checklist["1. At Key S/R Level"] = "✅ At Support"
            last_candle = five_min_df.iloc[-1]
            if (last_candle['low'] < support) and (last_candle['close'] > support):
                checklist["2. Price Rejection"] = "✅ Bullish Rejection"
        else:
            checklist["1. At Key S/R Level"] = "❌ Not at a key level"
            checklist["2. Price Rejection"] = "❌ No Rejection"

        pattern_status = self.detect_breakout_retest(five_min_df, resistance)
        checklist["3. Chart Pattern Confirmed"] = "✅ Breakout/Retest" if "✅ Retest Confirmed" in pattern_status else "❌ No Confirmed Pattern"
        
        candle_pattern = self.check_candlestick_pattern(five_min_df)
        checklist["4. Candlestick Signal"] = f"✅ {candle_pattern}" if "No significant" not in candle_pattern else "❌ No Signal"

        rsi = self.compute_rsi(five_min_df.rename(columns={'close': 'Close'}))
        five_min_df = self.compute_vwap(five_min_df)
        vwap = five_min_df['vwap'].iloc[-1] if 'vwap' in five_min_df.columns else latest_price
        
        if (checklist["1. At Key S/R Level"] == "✅ At Support" and rsi < 70 and latest_price > vwap):
            checklist["5. Indicator Alignment"] = "✅ Bullish Alignment"
        else:
            checklist["5. Indicator Alignment"] = "❌ No Alignment"
            
        if sum(1 for v in checklist.values() if "✅ Bullish" in str(v) or "✅ Breakout/Retest" in str(v)) >= 3:
            checklist["FINAL_SIGNAL"] = "BUY"
        
        return checklist

# ==============================================================================
# === MAIN APP LOGIC ===========================================================
# ==============================================================================
def main():
    st.set_page_config(page_title="📈 AI Trading Agent", page_icon="🤖", layout="wide")
    st.title("🤖 AI Intraday & Swing Trading Agent")

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    if 'screened_stocks' not in st.session_state:
        st.session_state.screened_stocks = None

    # --- SIDEBAR ---
    st.sidebar.header("Analysis Mode")
    analysis_mode = st.sidebar.radio("Choose Analysis Type:", ["Swing Analysis", "Intraday Analysis"])
    
    st.sidebar.header("Pre-Market Analysis")
    if st.sidebar.button("Run Daily Stock Screener"):
        st.session_state.screened_stocks = run_pre_market_screener()

    st.sidebar.header("Stock Selection")
    analysis_target = st.sidebar.radio("What do you want to analyze?", ("Index", "Individual Stock from Screener"))

    ticker, ticker_name = None, None
    if analysis_target == "Index":
        selected_category = st.sidebar.selectbox("Choose Index:", list(STOCK_CATEGORIES.keys()))
        ticker = STOCK_CATEGORIES[selected_category]["ticker"]
        ticker_name = selected_category
    else:
        if st.session_state.screened_stocks:
            ticker = st.sidebar.selectbox("Choose Screened Stock:", list(st.session_state.screened_stocks.keys()))
            ticker_name = ticker.replace(".NS", "") if ticker else "None"
        else:
            st.sidebar.warning("Run the screener to see stocks.")

    if ticker:
        st.sidebar.success(f"Selected: {ticker_name}")

    # --- MAIN CONTENT ---
    analyze_button = st.button("🔍 Analyze Now", type="primary")

    if analyze_button and ticker:
        if analysis_mode == "Swing Analysis":
            # Simplified Swing Analysis for brevity
            with st.spinner(f"Running Swing Analysis for {ticker_name}..."):
                st.info("Swing analysis logic would run here.")
        
        elif analysis_mode == "Intraday Analysis":
            with st.spinner(f"Running Multi-Timeframe Analysis for {ticker_name}..."):
                try:
                    analysis = st.session_state.analyzer.run_multi_timeframe_analysis(ticker)
                    if not analysis or analysis['5m_data'].empty:
                        st.warning("Could not retrieve sufficient data for intraday analysis.")
                        return

                    st.success("Multi-Timeframe Analysis Complete!")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Selected Stock", ticker_name)
                    col2.metric("Overall Trend (Daily)", analysis['trend'])
                    col3.metric("Support (15-min)", f"₹{analysis['support']:.2f}")
                    col4.metric("Resistance (15-min)", f"₹{analysis['resistance']:.2f}")

                    five_min_df = analysis['5m_data']
                    
                    pattern_status = st.session_state.analyzer.detect_breakout_retest(five_min_df, analysis['resistance'])
                    st.subheader("📈 Pattern Recognition Status")
                    st.info(pattern_status)

                    st.subheader("✅ Confirmation Checklist")
                    checklist_results = st.session_state.analyzer.run_confirmation_checklist(analysis)
                    final_signal = checklist_results.pop("FINAL_SIGNAL")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for key, value in list(checklist_results.items())[:3]:
                            st.write(f"**{key}:** {value}")
                    with col2:
                        for key, value in list(checklist_results.items())[3:]:
                            st.write(f"**{key}:** {value}")
                    
                    st.subheader("🤖 Final AI Signal")
                    if final_signal == "BUY":
                        st.success(f"**{final_signal}** - High probability bullish setup detected.")
                    else:
                        st.warning(f"**{final_signal}** - Conditions not met for a high-probability trade.")

                    five_min_df_with_vwap = st.session_state.analyzer.compute_vwap(five_min_df.copy())
                    st.subheader("5-Minute Execution Chart")
                    intraday_fig = create_intraday_chart(five_min_df_with_vwap, ticker_name)
                    st.plotly_chart(intraday_fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Intraday Analysis failed: {e}")
                    st.exception(e)

if __name__ == "__main__":
    main()
