import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from transformers import pipeline
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
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Stock Categories (keeping your existing structure)
STOCK_CATEGORIES = {
    "NIFTY 50 Index": {
        "ticker": "^NSEI",
        "individual_stocks": {
            "Reliance Industries": "RELIANCE.NS",
            "Tata Consultancy Services": "TCS.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "Infosys": "INFY.NS",
            "ICICI Bank": "ICICIBANK.NS",
            # ... (keep all your stocks)
        }
    },
    # ... (keep all your categories)
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def run_pre_market_screener():
    """Downloads NSE stocks and filters by price and volume criteria"""
    st.write("Running Pre-Market Screener...")
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df_all_stocks = pd.read_csv(url)
        nse_symbols = [f"{symbol}.NS" for symbol in df_all_stocks['SYMBOL']]
        
        # Scan first 500 stocks (increase for production)
        symbols_to_scan = nse_symbols[:500]
        st.write(f"Scanning {len(symbols_to_scan)} stocks...")

        tickers_str = " ".join(symbols_to_scan)
        data = yf.download(tickers_str, period="5d", group_by='ticker', auto_adjust=True, progress=False)

        screened_list = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(symbols_to_scan):
            try:
                stock_data = data[ticker]
                if not stock_data.empty:
                    last_day = stock_data.iloc[-1]
                    price = last_day['Close']
                    volume = last_day['Volume']
                    
                    # Screening criteria: price > 100, volume > 100000
                    if price > 100 and volume > 100000:
                        screened_list[ticker] = {'price': price, 'volume': volume}
            except:
                continue
            progress_bar.progress((i + 1) / len(symbols_to_scan))

        st.write(f"Found {len(screened_list)} stocks meeting criteria.")
        return screened_list
    except Exception as e:
        st.error(f"Pre-market scan error: {e}")
        return {}

@st.cache_data
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            st.error(f"No historical data for ticker: {ticker}")
            return None
        return hist
    except Exception as e:
        st.error(f"Error fetching data for '{ticker}': {e}")
        return None

def fetch_intraday_data(ticker, interval="5m"):
    """Fetch intraday data for 5min/15min analysis"""
    try:
        stock = yf.Ticker(ticker)
        # Get 5 days of intraday data
        hist = stock.history(period="5d", interval=interval)
        if hist.empty:
            st.warning(f"No intraday data for {ticker}")
            return None
        
        # Rename columns to lowercase for consistency
        hist.columns = [col.lower() for col in hist.columns]
        return hist
    except Exception as e:
        st.warning(f"Intraday data fetch error: {e}")
        return None

def create_intraday_chart(data, ticker_name):
    """Creates intraday chart with VWAP and volume"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(f'{ticker_name} - 5 Min Chart', 'Volume'), 
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['open'], high=data['high'],
        low=data['low'], close=data['close'], name='Price'
    ), row=1, col=1)
    
    # VWAP line
    if 'vwap' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['vwap'], mode='lines',
            name='VWAP', line=dict(color='orange', width=2)
        ), row=1, col=1)
    
    # Volume bars
    colors = ['green' if row['close'] >= row['open'] else 'red' 
              for _, row in data.iterrows()]
    fig.add_trace(go.Bar(
        x=data.index, y=data['volume'], name='Volume',
        marker_color=colors
    ), row=2, col=1)
    
    fig.update_layout(
        height=700, showlegend=True, xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    return fig

def embed_tradingview_widget(ticker):
    """Generates TradingView widget"""
    tv_ticker = f"NSE:{ticker.replace('.NS', '')}" if ".NS" in ticker else ticker
    html_code = f"""
    <div class="tradingview-widget-container" style="height:500px;width:100%;">
      <div id="tradingview_chart" style="height:100%;width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%", "height": 500, "symbol": "{tv_ticker}", "interval": "D",
        "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
        "enable_publishing": false, "allow_symbol_change": true, 
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>"""
    return html_code

def setup_google_sheets():
    """Initialize Google Sheets connection"""
    try:
        scope = ["https://spreadsheets.google.com/feeds", 
                 "https://www.googleapis.com/auth/drive"]
        creds_dict = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        try:
            sheet = client.open("TradingAnalyzerLog").sheet1
        except gspread.SpreadsheetNotFound:
            sheet = client.create("TradingAnalyzerLog").sheet1
            headers = ["Timestamp", "Ticker", "Signal", "Confidence", "RSI", 
                      "Sentiment", "AI Summary"]
            sheet.append_row(headers)
        return sheet
    except Exception as e:
        st.error(f"Google Sheets setup failed: {e}")
        return None

def log_to_sheets(sheet, data):
    """Log data to Google Sheet"""
    if sheet:
        try:
            sheet.append_row(data)
            return True
        except Exception as e:
            st.warning(f"Failed to log to sheets: {e}")
    return False

# ============================================================================
# STOCK ANALYZER CLASS
# ============================================================================

class StockAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        self.setup_sentiment_analyzer()

    def setup_sentiment_analyzer(self):
        """Initialize sentiment analysis pipeline"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
        except Exception as e:
            st.warning(f"Using default sentiment analyzer: {e}")
            self.sentiment_analyzer = pipeline("sentiment-analysis")

    def compute_rsi(self, data, window=14):
        """Calculate RSI"""
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

    def compute_macd(self, data):
        """Calculate MACD"""
        try:
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            return {
                'line': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0,
                'signal': signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0.0,
                'histogram': histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
            }
        except:
            return {'line': 0.0, 'signal': 0.0, 'histogram': 0.0}

    def compute_moving_averages(self, data):
        """Calculate Moving Averages"""
        try:
            current_price = data['Close'].iloc[-1]
            mas = {}
            for period in [20, 25, 50, 200]:
                if len(data) >= period:
                    mas[f'MA_{period}'] = data['Close'].rolling(window=period).mean().iloc[-1]
                else:
                    mas[f'MA_{period}'] = current_price
            mas['current_price'] = current_price
            return mas
        except:
            cp = data['Close'].iloc[-1]
            return {'MA_20': cp, 'MA_25': cp, 'MA_50': cp, 'MA_200': cp, 'current_price': cp}

    def compute_vwap(self, data):
        """Calculate VWAP for intraday data"""
        try:
            data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
            return data
        except:
            data['vwap'] = data['close']
            return data

    def compute_atr(self, data, period=14):
        """Calculate Average True Range"""
        try:
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            data[f'ATRr_{period}'] = true_range.rolling(period).mean()
            return data
        except:
            data[f'ATRr_{period}'] = 0
            return data

    def compute_obv(self, data):
        """Calculate On-Balance Volume"""
        try:
            obv = [0]
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + data['Volume'].iloc[i])
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            data['OBV'] = obv
            return data
        except:
            data['OBV'] = 0
            return data

    def compute_fibonacci_levels(self, data):
        """Calculate Fibonacci retracement levels"""
        try:
            high = data['High'].max()
            low = data['Low'].min()
            diff = high - low
            return {
                '0%': high,
                '23.6%': high - 0.236 * diff,
                '38.2%': high - 0.382 * diff,
                '50%': high - 0.5 * diff,
                '61.8%': high - 0.618 * diff,
                '100%': low
            }
        except:
            return {}

    def check_candlestick_pattern(self, five_min_data):
        """Identifies 3-candle reversal patterns"""
        if len(five_min_data) < 3:
            return "Not enough data"
        
        last3 = five_min_data.tail(3)
        c1, c2, c3 = last3.iloc[0], last3.iloc[1], last3.iloc[2]
        
        # Morning Star (Bullish)
        is_morning_star = (c1['close'] < c1['open'] and
                           abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) and
                           c3['close'] > c3['open'] and
                           c3['close'] > c1['open'])
        if is_morning_star:
            return "Morning Star (Bullish)"
        
        # Evening Star (Bearish)
        is_evening_star = (c1['close'] > c1['open'] and
                           abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) and
                           c3['close'] < c3['open'] and
                           c3['close'] < c1['open'])
        if is_evening_star:
            return "Evening Star (Bearish)"
        
        return "No significant pattern"

    def detect_breakout_retest(self, five_min_data, resistance):
        """Analyzes breakout, retest, and confirmation"""
        if five_min_data.empty or resistance == 0:
            return "Not Analyzed"
        
        recent_data = five_min_data.tail(20)
        breakout_idx = -1
        
        # Detect breakout
        for i in range(1, len(recent_data)):
            if recent_data['high'].iloc[i] > resistance and recent_data['high'].iloc[i-1] <= resistance:
                breakout_idx = i
                break
        
        if breakout_idx == -1:
            return "No Breakout Detected"
        
        # Detect retest
        retest_idx = -1
        for i in range(breakout_idx + 1, len(recent_data)):
            if recent_data['low'].iloc[i] <= resistance:
                retest_idx = i
                break
        
        if retest_idx == -1:
            return f"Breakout at {recent_data.index[breakout_idx].strftime('%H:%M')}. Awaiting Retest."
        
        # Check confirmation
        if retest_idx < len(recent_data) - 1:
            conf_candle = recent_data.iloc[retest_idx + 1]
            if conf_candle['close'] > resistance:
                return f"✅ Retest Confirmed at {recent_data.index[retest_idx + 1].strftime('%H:%M')}. Entry Signal."
        
        return f"Retest at {recent_data.index[retest_idx].strftime('%H:%M')}. Awaiting Confirmation."

    def run_confirmation_checklist(self, analysis_results):
        """5-point confirmation checklist for trade signal"""
        checklist = {
            "1. At Key S/R Level": "⚠️ PENDING",
            "2. Price Rejection": "⚠️ PENDING",
            "3. Chart Pattern Confirmed": "⚠️ PENDING",
            "4. Candlestick Signal": "⚠️ PENDING",
            "5. Indicator Alignment": "⚠️ PENDING",
            "FINAL_SIGNAL": "HOLD"
        }
        
        five_min_df = analysis_results.get('5m_data', pd.DataFrame())
        if five_min_df.empty:
            return checklist
        
        resistance = analysis_results.get('resistance', 0)
        support = analysis_results.get('support', 0)
        latest_price = analysis_results.get('latest_price', 0)
        
        # Check 1 & 2: At key level with rejection
        at_resistance = abs(latest_price - resistance) / resistance < 0.005
        at_support = abs(latest_price - support) / support < 0.005
        
        if at_support:
            checklist["1. At Key S/R Level"] = "✅ At Support"
            last_candle = five_min_df.iloc[-1]
            if last_candle['low'] < support and last_candle['close'] > support:
                checklist["2. Price Rejection"] = "✅ Bullish Rejection"
        elif at_resistance:
            checklist["1. At Key S/R Level"] = "✅ At Resistance"
            last_candle = five_min_df.iloc[-1]
            if last_candle['high'] > resistance and last_candle['close'] < resistance:
                checklist["2. Price Rejection"] = "✅ Bearish Rejection"
        else:
            checklist["1. At Key S/R Level"] = "❌ Not at key level"
            checklist["2. Price Rejection"] = "❌ No Rejection"
        
        # Check 3: Pattern
        pattern_status = self.detect_breakout_retest(five_min_df, resistance)
        if "✅ Retest Confirmed" in pattern_status:
            checklist["3. Chart Pattern Confirmed"] = "✅ Breakout/Retest"
        else:
            checklist["3. Chart Pattern Confirmed"] = "❌ No Confirmed Pattern"
        
        # Check 4: Candlestick
        candle_pattern = self.check_candlestick_pattern(five_min_df)
        checklist["4. Candlestick Signal"] = f"✅ {candle_pattern}" if "No significant" not in candle_pattern else "❌ No Signal"
        
        # Check 5: Indicators
        temp_df = five_min_df.rename(columns={'close': 'Close'})
        rsi = self.compute_rsi(temp_df)
        five_min_df = self.compute_vwap(five_min_df)
        vwap = five_min_df['vwap'].iloc[-1]
        
        if checklist["1. At Key S/R Level"] == "✅ At Support" and rsi < 70 and latest_price > vwap:
            checklist["5. Indicator Alignment"] = "✅ Bullish Alignment"
        elif checklist["1. At Key S/R Level"] == "✅ At Resistance" and rsi > 30 and latest_price < vwap:
            checklist["5. Indicator Alignment"] = "✅ Bearish Alignment"
        else:
            checklist["5. Indicator Alignment"] = "❌ No Alignment"
        
        # Final signal
        bullish_checks = sum(1 for v in checklist.values() if "✅" in str(v) and ("Bullish" in str(v) or "Breakout" in str(v)))
        if bullish_checks >= 3:
            checklist["FINAL_SIGNAL"] = "BUY"
        
        return checklist

    def run_multi_timeframe_analysis(self, ticker):
        """Multi-timeframe analysis: Daily trend, 15-min S/R, 5-min execution"""
        analysis_results = {}
        
        # Daily trend
        daily_data = fetch_stock_data(ticker, period="6mo")
        if daily_data is not None and not daily_data.empty:
            daily_data['SMA50'] = daily_data['Close'].rolling(window=50).mean()
            last_price = daily_data['Close'].iloc[-1]
            last_sma50 = daily_data['SMA50'].iloc[-1]
            analysis_results['trend'] = "Uptrend" if last_price > last_sma50 else "Downtrend"
        else:
            analysis_results['trend'] = "Unknown"
        
        # 15-min S/R
        fifteen_min_data = fetch_intraday_data(ticker, interval="15m")
        if fifteen_min_data is not None and not fifteen_min_data.empty:
            analysis_results['support'] = fifteen_min_data['low'].min()
            analysis_results['resistance'] = fifteen_min_data['high'].max()
        else:
            analysis_results['support'] = 0
            analysis_results['resistance'] = 0
        
        # 5-min execution
        five_min_data = fetch_intraday_data(ticker, interval="5m")
        if five_min_data is not None and not five_min_data.empty:
            analysis_results['5m_data'] = five_min_data
            analysis_results['latest_price'] = five_min_data['close'].iloc[-1]
        else:
            analysis_results['5m_data'] = pd.DataFrame()
            analysis_results['latest_price'] = last_price if daily_data is not None else 0
        
        return analysis_results

    def scrape_news_headlines(self, ticker_name, days=1):
        """Scrape news from NewsAPI"""
        try:
            api_key = "e205d77d7bc14acc8744d3ea10568f50"
            search_query = ticker_name.replace("^", "").replace(".NS", "").replace("NSE", "")
            url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=publishedAt&apiKey={api_key}&pageSize=5"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            news_data = response.json()
            headlines = []
            if news_data.get("status") == "ok" and news_data.get("articles"):
                for article in news_data["articles"]:
                    title = article.get("title")
                    if title and len(title) > 15:
                        headlines.append(title)
                    if len(headlines) >= 5:
                        break
            return headlines if headlines else [f"News unavailable for {search_query}"]
        except Exception as e:
            return [f"News unavailable for {ticker_name}"]

    def analyze_sentiment(self, headlines):
        """Analyze sentiment of headlines"""
        if not headlines or all("unavailable" in h for h in headlines):
            return "Neutral", 0
        try:
            sentiments = []
            for headline in headlines:
                if self.sentiment_analyzer:
                    result = self.sentiment_analyzer(headline)
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            sentiment_scores = {item['label']: item['score'] for item in result[0]}
                            if 'positive' in sentiment_scores:
                                sentiments.append(sentiment_scores['positive'] - sentiment_scores.get('negative', 0))
                        else:
                            label = result[0]['label'].upper()
                            score = result[0]['score']
                            if label in ['POSITIVE', 'POS']:
                                sentiments.append(score)
                            elif label in ['NEGATIVE', 'NEG']:
                                sentiments.append(-score)
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            if avg_sentiment > 0.1:
                return "Positive", avg_sentiment
            elif avg_sentiment < -0.1:
                return "Negative", avg_sentiment
            else:
                return "Neutral", avg_sentiment
        except:
            return "Neutral", 0

    def generate_signal(self, rsi, macd, ma_data, sentiment_score):
        """Generate trading signal"""
        bullish_signals = 0
        bearish_signals = 0
        
        if rsi < 40:
            bullish_signals += 1
        elif rsi > 60:
            bearish_signals += 1
        
        if macd['line'] > macd['signal'] and macd['histogram'] > 0:
            bullish_signals += 1
        elif macd['line'] < macd['signal'] and macd['histogram'] < 0:
            bearish_signals += 1
        
        current_price = ma_data.get('current_price', ma_data['MA_20'])
        if (current_price > ma_data['MA_25'] > ma_data['MA_50'] and ma_data['MA_25'] > ma_data['MA_200']):
            bullish_signals += 1
        elif (current_price < ma_data['MA_25'] < ma_data['MA_50'] and ma_data['MA_25'] < ma_data['MA_200']):
            bearish_signals += 1
        
        if sentiment_score > 0.2:
            bullish_signals += 0.5
        elif sentiment_score < -0.2:
            bearish_signals += 0.5
        
        if bullish_signals >= 2:
            return "Buy", "High" if bullish_signals >= 2.5 else "Medium"
        elif bearish_signals >= 2:
            return "Sell", "High" if bearish_signals >= 2.5 else "Medium"
        else:
            return "Hold", "Medium" if abs(bullish_signals - bearish_signals) > 0.5 else "Low"

    def get_ai_summary(self, ticker_name, rsi, macd, ma_data, signal, confidence, sentiment_label, headlines):
        """Generate AI summary using Gemini"""
        try:
            prompt = f"""Professional stock analysis for {ticker_name}:

Technical Indicators:
- RSI: {rsi:.2f} (oversold <40, overbought >60)
- MACD Line: {macd['line']:.2f}, Signal: {macd['signal']:.2f}, Histogram: {macd['histogram']:.2f}
- MAs: 20D={ma_data['MA_20']:.2f}, 25D={ma_data['MA_25']:.2f}, 50D={ma_data['MA_50']:.2f}, 200D={ma_data['MA_200']:.2f}

Signal: {signal} | Confidence: {confidence} | Sentiment: {sentiment_label}

Recent Headlines: {chr(10).join(headlines[:3])}

Provide 2-3 sentence trading strategy with entry/exit guidance and risk factors."""

            if GOOGLE_API_KEY:
                model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
                response = model.generate_content(prompt)
                if response.text:
                    return response.text.strip()
            raise Exception("API unavailable")
        except Exception as e:
            rsi_status = "oversold" if rsi < 40 else "overbought" if rsi > 60 else "neutral"
            macd_trend = "bullish" if macd['line'] > macd['signal'] else "bearish"
            return (f"Technical analysis shows '{signal}' with {confidence} confidence. "
                    f"RSI at {rsi:.1f} indicates {rsi_status} conditions, MACD shows {macd_trend} momentum. "
                    f"Use proper risk management.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="📈 AI Trading Agent", page_icon="📈", layout="wide")
    
    st.title("📈 AI Stock Market Trading Agent")
    st.markdown("*Intelligent Swing & Intraday Analysis with AI-Powered Insights*")
    
    # Sidebar - Pre-Market Screener
    st.sidebar.header("Pre-Market Analysis")
    if 'screened_stocks' not in st.session_state:
        st.session_state.screened_stocks = None
    
    if st.sidebar.button("🔍 Run Daily Stock Screener"):
        st.session_state.screened_stocks = run_pre_market_screener()
    
    # Sidebar - Analysis Mode Selection
    st.sidebar.header("Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Choose Trading Style:",
        ["Swing Trading", "Intraday Trading"]
    )
    
    # Stock Selection
    st.sidebar.header("Stock Selection")
    ticker = None
    ticker_name = None
    
    if st.session_state.screened_stocks:
        analysis_target = st.sidebar.radio(
            "Analyze:",
            ["Index", "Screened Stock"]
        )
    else:
        analysis_target = "Index"
    
    if analysis_target == "Index":
        selected_category = st.sidebar.selectbox(
            "Choose Index:",
            list(STOCK_CATEGORIES.keys()),
            index=0
        )
        stock_type = st.sidebar.radio("Type:", ["Index", "Individual Stock"])
        
        if stock_type == "Index":
            ticker = STOCK_CATEGORIES[selected_category]["ticker"]
            ticker_name = selected_category
        else:
            individual_stocks = STOCK_CATEGORIES[selected_category]["individual_stocks"]
            selected_stock = st.sidebar.selectbox("Choose Stock:", list(individual_stocks.keys()))
            ticker = individual_stocks[selected_stock]
            ticker_name = selected_stock
    else:
        screened_tickers = list(st.session_state.screened_stocks.keys())
        selected_ticker = st.sidebar.selectbox("Screened Stock:", screened_tickers)
        ticker = selected_ticker
        ticker_name = selected_ticker.replace(".NS", "")
    
    # Display selection
    if ticker:
        st.sidebar.success(f"Selected: {ticker_name}")
        st.sidebar.info(f"Ticker: {ticker}")
    
    # Settings
    st.sidebar.header("Settings")
    period = st.sidebar.selectbox(
        "Data Period:",
        ["5d", "1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "max"],
        index=5
    )
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    
    # Setup Google Sheets
    if 'sheets_client' not in st.session_state:
        st.session_state.sheets_client = setup_google_sheets()
    
    # Main analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Now", type="primary", use_container_width=True)
    
    if analyze_button and ticker:
        # ===================================================================
        # SWING TRADING ANALYSIS
        # ===================================================================
        if analysis_mode == "Swing Trading":
            with st.spinner(f"Running Swing Analysis for {ticker_name}..."):
                try:
                    data = fetch_stock_data(ticker, period)
                    if data is None:
                        st.stop()
                    
                    # Calculate indicators
                    rsi = st.session_state.analyzer.compute_rsi(data)
                    macd = st.session_state.analyzer.compute_macd(data)
                    ma_data = st.session_state.analyzer.compute_moving_averages(data)
                    data = st.session_state.analyzer.compute_atr(data)
                    data = st.session_state.analyzer.compute_obv(data)
                    fib_levels = st.session_state.analyzer.compute_fibonacci_levels(data)
                    
                    latest_atr = data[f'ATRr_14'].iloc[-1]
                    latest_obv = data['OBV'].iloc[-1]
                    
                    # News and sentiment
                    headlines = st.session_state.analyzer.scrape_news_headlines(ticker_name, days=1)
                    sentiment_label, sentiment_score = st.session_state.analyzer.analyze_sentiment(headlines)
                    
                    # Generate signal
                    signal, confidence = st.session_state.analyzer.generate_signal(
                        rsi, macd, ma_data, sentiment_score
                    )
                    
                    # AI summary
                    ai_summary = st.session_state.analyzer.get_ai_summary(
                        ticker_name, rsi, macd, ma_data, signal, confidence,
                        sentiment_label, headlines
                    )
                    
                    # Display results
                    st.success("✅ Swing Analysis Complete!")
                    
                    # Metrics
                    current_price = data['Close'].iloc[-1]
                    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{current_price:.2f}", 
                                 delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
                    with col2:
                        signal_color = {"Buy": "🟢", "Sell": "🔴", "Hold": "🟡"}[signal]
                        st.metric("Trading Signal", f"{signal_color} {signal}", delta=f"Confidence: {confidence}")
                    with col3:
                        rsi_status = "Oversold" if rsi < 40 else "Overbought" if rsi > 60 else "Normal"
                        st.metric("RSI (40:60)", f"{rsi:.2f}", delta=rsi_status)
                    with col4:
                        sentiment_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}[sentiment_label]
                        st.metric("News Sentiment", f"{sentiment_color} {sentiment_label}", 
                                 delta=f"Score: {sentiment_score:.2f}")
                    
                    # AI Summary
                    st.subheader("🤖 AI Strategy Summary")
                    st.info(ai_summary)
                    
                    # Technical Indicators
                    st.subheader("📊 Technical Indicators")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Moving Averages:**")
                        st.write(f"• 20-day MA: ₹{ma_data['MA_20']:.2f}")
                        st.write(f"• 25-day MA: ₹{ma_data['MA_25']:.2f}")
                        st.write(f"• 50-day MA: ₹{ma_data['MA_50']:.2f}")
                        st.write(f"• 200-day MA: ₹{ma_data['MA_200']:.2f}")
                        
                        if ma_data['MA_20'] > ma_data['MA_50'] > ma_data['MA_200']:
                            st.success("📈 Strong Bullish MA Alignment")
                        elif ma_data['MA_20'] < ma_data['MA_50'] < ma_data['MA_200']:
                            st.error("📉 Strong Bearish MA Alignment")
                        else:
                            st.warning("🔄 Mixed MA Signals")
                    
                    with col2:
                        st.write("**MACD:**")
                        st.write(f"• MACD Line: {macd['line']:.4f}")
                        st.write(f"• Signal Line: {macd['signal']:.4f}")
                        st.write(f"• Histogram: {macd['histogram']:.4f}")
                        
                        if macd['line'] > macd['signal'] and macd['histogram'] > 0:
                            st.success("📈 Bullish MACD Crossover")
                        elif macd['line'] < macd['signal'] and macd['histogram'] < 0:
                            st.error("📉 Bearish MACD Crossover")
                        else:
                            st.warning("🔄 MACD Consolidation")
                    
                    # Fibonacci Levels
                    if fib_levels:
                        st.subheader("📏 Fibonacci Retracement Levels")
                        fib_cols = st.columns(6)
                        for i, (level, price) in enumerate(fib_levels.items()):
                            fib_cols[i].metric(level, f"₹{price:.2f}")
                    
                    # Charts
                    st.subheader("📈 Technical Analysis Charts")
                    fig = create_indicator_charts(data, rsi, macd, ma_data)
                    st.pyplot(fig)
                    
                    # TradingView Chart
                    st.subheader("🚀 Live Professional Chart")
                    st.info("Use for advanced indicator analysis")
                    components.html(embed_tradingview_widget(ticker), height=520)
                    
                    # Volume Analysis
                    st.subheader("📊 Volume Analysis")
                    avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                    current_volume = data['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    vol_col1, vol_col2 = st.columns(2)
                    with vol_col1:
                        st.metric("Current Volume", f"{current_volume:,.0f}", 
                                 delta=f"{volume_ratio:.2f}x Average")
                    with vol_col2:
                        if volume_ratio > 1.5:
                            st.success("🔊 High Volume Activity")
                        elif volume_ratio < 0.5:
                            st.warning("🔇 Low Volume Activity")
                        else:
                            st.info("🔉 Normal Volume Activity")
                    
                    # News Headlines
                    st.subheader("📰 Recent News Headlines")
                    for i, headline in enumerate(headlines[:5], 1):
                        st.write(f"{i}. {headline}")
                    
                    # Support & Resistance
                    st.subheader("📏 Support & Resistance Levels")
                    high_20 = data['High'].rolling(window=20).max().iloc[-1]
                    low_20 = data['Low'].rolling(window=20).min().iloc[-1]
                    
                    sr_col1, sr_col2, sr_col3 = st.columns(3)
                    with sr_col1:
                        st.metric("20-Day High", f"₹{high_20:.2f}")
                    with sr_col2:
                        st.metric("20-Day Low", f"₹{low_20:.2f}")
                    with sr_col3:
                        range_pct = ((high_20 - low_20) / low_20) * 100
                        st.metric("Range", f"{range_pct:.1f}%")
                    
                    # Log to Google Sheets
                    log_data = [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        ticker_name, f"{rsi:.2f}", f"{macd['line']:.4f}",
                        f"{macd['signal']:.4f}", f"{ma_data['MA_20']:.2f}",
                        f"{ma_data['MA_25']:.2f}", f"{ma_data['MA_50']:.2f}",
                        f"{ma_data['MA_200']:.2f}", signal, confidence,
                        ai_summary[:500], sentiment_label
                    ]
                    
                    if log_to_sheets(st.session_state.sheets_client, log_data):
                        st.success("✅ Results logged to Google Sheets")
                
                except Exception as e:
                    st.error(f"❌ Swing Analysis failed: {e}")
                    st.exception(e)
        
        # ===================================================================
        # INTRADAY TRADING ANALYSIS
        # ===================================================================
        elif analysis_mode == "Intraday Trading":
            with st.spinner(f"Running Intraday Analysis for {ticker_name}..."):
                try:
                    analysis = st.session_state.analyzer.run_multi_timeframe_analysis(ticker)
                    
                    if analysis:
                        st.success("✅ Multi-Timeframe Analysis Complete!")
                        
                        # High-level findings
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Stock", ticker_name)
                        col2.metric("Trend (Daily)", analysis['trend'])
                        col3.metric("Support (15-min)", f"₹{analysis['support']:.2f}")
                        col4.metric("Resistance (15-min)", f"₹{analysis['resistance']:.2f}")
                        
                        # Pattern Recognition
                        five_min_df = analysis['5m_data']
                        if not five_min_df.empty:
                            resistance_level = analysis['resistance']
                            pattern_status = st.session_state.analyzer.detect_breakout_retest(
                                five_min_df, resistance_level
                            )
                            
                            st.subheader("📈 Pattern Recognition Status")
                            st.info(pattern_status)
                            
                            # Confirmation Checklist
                            st.subheader("✅ 5-Point Confirmation Checklist")
                            checklist_results = st.session_state.analyzer.run_confirmation_checklist(analysis)
                            final_signal = checklist_results.pop("FINAL_SIGNAL")
                            
                            check_col1, check_col2 = st.columns(2)
                            with check_col1:
                                for key, value in list(checklist_results.items())[:3]:
                                    st.write(f"**{key}:** {value}")
                            with check_col2:
                                for key, value in list(checklist_results.items())[3:]:
                                    st.write(f"**{key}:** {value}")
                            
                            # Final Signal
                            st.subheader("🤖 Final AI Signal")
                            if final_signal == "BUY":
                                st.success(f"**{final_signal}** - High probability bullish setup detected.")
                            elif final_signal == "SELL":
                                st.error(f"**{final_signal}** - High probability bearish setup detected.")
                            else:
                                st.warning(f"**{final_signal}** - Conditions not met for high-probability trade.")
                            
                            # Intraday Indicators
                            five_min_df = st.session_state.analyzer.compute_vwap(five_min_df)
                            
                            ind_col1, ind_col2, ind_col3 = st.columns(3)
                            with ind_col1:
                                current_price = five_min_df['close'].iloc[-1]
                                st.metric("Current Price", f"₹{current_price:.2f}")
                            with ind_col2:
                                vwap = five_min_df['vwap'].iloc[-1]
                                st.metric("VWAP", f"₹{vwap:.2f}")
                            with ind_col3:
                                position = "Above VWAP" if current_price > vwap else "Below VWAP"
                                st.metric("Position", position)
                            
                            # 5-Minute Chart
                            st.subheader("📊 5-Minute Execution Chart")
                            intraday_fig = create_intraday_chart(five_min_df, ticker_name)
                            st.plotly_chart(intraday_fig, use_container_width=True)
                            
                            # Trading Plan
                            st.subheader("📋 Intraday Trading Plan")
                            st.write(f"**Entry Zone:** Near support at ₹{analysis['support']:.2f}")
                            st.write(f"**Target:** Resistance at ₹{analysis['resistance']:.2f}")
                            st.write(f"**Stop Loss:** Below ₹{analysis['support'] * 0.98:.2f} (2% below support)")
                            st.write(f"**Risk-Reward Ratio:** {((analysis['resistance'] - current_price) / (current_price - analysis['support'] * 0.98)):.2f}:1")
                        else:
                            st.warning("Could not retrieve 5-minute data for detailed analysis.")
                
                except Exception as e:
                    st.error(f"❌ Intraday Analysis failed: {e}")
                    st.exception(e)
    
    elif analyze_button and not ticker:
        st.sidebar.error("Please select a stock or index to analyze.")
    
    # Information panel
    with st.expander("ℹ️ About This Trading Agent"):
        st.write("""
        **Features:**
        - ✅ Swing Trading: Multi-day trend analysis with RSI, MACD, MAs
        - ✅ Intraday Trading: Multi-timeframe analysis with 5-point confirmation
        - ✅ Pre-market stock screener (price > ₹100, volume > 100K)
        - ✅ Pattern recognition: Breakout, Retest, Confirmation
        - ✅ Candlestick patterns: Morning Star, Evening Star
        - ✅ VWAP, Support/Resistance, Fibonacci levels
        - ✅ AI-powered trade summaries via Google Gemini
        - ✅ News sentiment analysis with FinBERT
        
        **Swing Indicators:**
        - RSI (40:60 levels), MACD, Moving Averages (20/25/50/200)
        - ATR, OBV, Fibonacci retracements
        
        **Intraday Strategy:**
        - Daily chart: Trend identification
        - 15-min chart: Support & Resistance
        - 5-min chart: Precise entry/exit timing
        - 5-point confirmation checklist before entry
        
        **Risk Management:**
        - Daily profit target: ₹500-₹2000
        - Capital per trade: ₹12,500 max
        - Avoid first 10 minutes of trading
        - Use stop-loss on all positions
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*AI Trading Agent powered by yfinance, FinBERT, Google Gemini, and Google Sheets*")

def create_indicator_charts(data, rsi, macd, ma_data):
    """Create swing trading indicator charts"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Price and MAs
    axes[0].plot(data.index[-30:], data['Close'][-30:], label='Close', linewidth=2, color='black')
    for period, color in zip([20, 25, 50, 200], ['orange', 'purple', 'blue', 'red']):
        if len(data) >= period:
            ma_series = data['Close'].rolling(window=period).mean()
            axes[0].plot(data.index[-30:], ma_series[-30:], color=color, 
                        linestyle='--', label=f'MA {period}', alpha=0.8)
    axes[0].set_title('Price & Moving Averages')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RSI
    rsi_series = []
    for i in range(len(data)):
        if i >= 13:
            temp_data = data.iloc[:i+1]
            delta = temp_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            temp_rsi = 100 - (100 / (1 + rs))
            rsi_series.append(temp_rsi.iloc[-1] if not pd.isna(temp_rsi.iloc[-1]) else 50)
        else:
            rsi_series.append(50)
    
    axes[1].plot(data.index[-30:], rsi_series[-30:], label='RSI', color='purple', linewidth=2)
    axes[1].axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Overbought (60)')
    axes[1].axhline(y=40, color='green', linestyle='--', alpha=0.7, label='Oversold (40)')
    axes[1].axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    axes[1].set_title(f'RSI (Current: {rsi:.2f})')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MACD
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    
    axes[2].plot(data.index[-30:], macd_line[-30:], label='MACD', color='blue')
    axes[2].plot(data.index[-30:], signal_line[-30:], label='Signal', color='red')
    axes[2].bar(data.index[-30:], histogram[-30:], label='Histogram', alpha=0.3, color='gray')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_title('MACD')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    main()
