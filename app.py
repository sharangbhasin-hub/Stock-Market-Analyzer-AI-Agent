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
from datetime import datetime, timedelta, time as dt_time
import json
import time
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import pytz

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==============================================================================
# === STOCK CATEGORIES (COMPLETE) ==============================================
# ==============================================================================

STOCK_CATEGORIES = {
    "NIFTY 50 Index": {
        "ticker": "^NSEI",
        "individual_stocks": {
            "Reliance Industries": "RELIANCE.NS",
            "Tata Consultancy Services": "TCS.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "Infosys": "INFY.NS",
            "ICICI Bank": "ICICIBANK.NS",
            "Hindustan Unilever": "HINDUNILVR.NS",
            "ITC": "ITC.NS",
            "State Bank of India": "SBIN.NS",
            "Bharti Airtel": "BHARTIARTL.NS",
            "Kotak Mahindra Bank": "KOTAKBANK.NS",
            "Larsen & Toubro": "LT.NS",
            "Axis Bank": "AXISBANK.NS",
            "Bajaj Finance": "BAJFINANCE.NS",
            "Asian Paints": "ASIANPAINT.NS",
            "Maruti Suzuki": "MARUTI.NS",
            "HCL Technologies": "HCLTECH.NS",
            "Wipro": "WIPRO.NS",
            "Titan Company": "TITAN.NS",
            "Mahindra & Mahindra": "M&M.NS",
            "Sun Pharmaceutical": "SUNPHARMA.NS",
            "Nestle India": "NESTLEIND.NS",
            "UltraTech Cement": "ULTRACEMCO.NS",
            "Adani Ports": "ADANIPORTS.NS",
            "Bajaj Finserv": "BAJAJFINSV.NS",
            "Tech Mahindra": "TECHM.NS",
            "Power Grid Corporation": "POWERGRID.NS",
            "Tata Steel": "TATASTEEL.NS",
            "IndusInd Bank": "INDUSINDBK.NS",
            "JSW Steel": "JSWSTEEL.NS",
            "Cipla": "CIPLA.NS",
            "Coal India": "COALINDIA.NS",
            "NTPC": "NTPC.NS",
            "Grasim Industries": "GRASIM.NS",
            "ONGC": "ONGC.NS",
            "Dr. Reddy's Laboratories": "DRREDDY.NS",
            "Eicher Motors": "EICHERMOT.NS",
            "Britannia Industries": "BRITANNIA.NS",
            "Divi's Laboratories": "DIVISLAB.NS",
            "Bajaj Auto": "BAJAJ-AUTO.NS",
            "Hero MotoCorp": "HEROMOTOCO.NS",
            "Shree Cement": "SHREECEM.NS",
            "Hindalco Industries": "HINDALCO.NS",
            "Tata Consumer Products": "TATACONSUM.NS",
            "Adani Enterprises": "ADANIENT.NS",
            "UPL": "UPL.NS",
            "Bharat Petroleum": "BPCL.NS",
            "IOC": "IOC.NS",
            "SBI Life Insurance": "SBILIFE.NS",
            "Tata Motors": "TATAMOTORS.NS",
            "Apollo Hospitals": "APOLLOHOSP.NS"
        }
    },
    "NIFTY Bank": {
        "ticker": "^NSEBANK",
        "individual_stocks": {
            "HDFC Bank": "HDFCBANK.NS",
            "ICICI Bank": "ICICIBANK.NS",
            "State Bank of India": "SBIN.NS",
            "Kotak Mahindra Bank": "KOTAKBANK.NS",
            "Axis Bank": "AXISBANK.NS",
            "IndusInd Bank": "INDUSINDBK.NS",
            "IDFC First Bank": "IDFCFIRSTB.NS",
            "Bandhan Bank": "BANDHANBNK.NS",
            "Federal Bank": "FEDERALBNK.NS",
            "RBL Bank": "RBLBANK.NS",
            "Punjab National Bank": "PNB.NS",
            "Bank of Baroda": "BANKBARODA.NS"
        }
    },
    "NIFTY IT": {
        "ticker": "^CNXIT",
        "individual_stocks": {
            "Tata Consultancy Services": "TCS.NS",
            "Infosys": "INFY.NS",
            "HCL Technologies": "HCLTECH.NS",
            "Wipro": "WIPRO.NS",
            "Tech Mahindra": "TECHM.NS",
            "LTI Mindtree": "LTIM.NS",
            "Mphasis": "MPHASIS.NS",
            "Coforge": "COFORGE.NS",
            "Persistent Systems": "PERSISTENT.NS",
            "L&T Technology Services": "LTTS.NS"
        }
    },
    "NIFTY Pharma": {
        "ticker": "^CNXPHARMA",
        "individual_stocks": {
            "Sun Pharmaceutical": "SUNPHARMA.NS",
            "Cipla": "CIPLA.NS",
            "Dr. Reddy's Laboratories": "DRREDDY.NS",
            "Divi's Laboratories": "DIVISLAB.NS",
            "Lupin": "LUPIN.NS",
            "Aurobindo Pharma": "AUROPHARMA.NS",
            "Torrent Pharmaceuticals": "TORNTPHARM.NS",
            "Alkem Laboratories": "ALKEM.NS",
            "Biocon": "BIOCON.NS",
            "Cadila Healthcare": "ZYDUSLIFE.NS"
        }
    },
    "NIFTY Auto": {
        "ticker": "^CNXAUTO",
        "individual_stocks": {
            "Maruti Suzuki": "MARUTI.NS",
            "Mahindra & Mahindra": "M&M.NS",
            "Tata Motors": "TATAMOTORS.NS",
            "Bajaj Auto": "BAJAJ-AUTO.NS",
            "Hero MotoCorp": "HEROMOTOCO.NS",
            "Eicher Motors": "EICHERMOT.NS",
            "Ashok Leyland": "ASHOKLEY.NS",
            "TVS Motor": "TVSMOTOR.NS",
            "Bosch": "BOSCHLTD.NS",
            "MRF": "MRF.NS"
        }
    },
    "NIFTY FMCG": {
        "ticker": "^CNXFMCG",
        "individual_stocks": {
            "Hindustan Unilever": "HINDUNILVR.NS",
            "ITC": "ITC.NS",
            "Nestle India": "NESTLEIND.NS",
            "Britannia Industries": "BRITANNIA.NS",
            "Tata Consumer Products": "TATACONSUM.NS",
            "Dabur India": "DABUR.NS",
            "Marico": "MARICO.NS",
            "Godrej Consumer Products": "GODREJCP.NS",
            "Colgate-Palmolive": "COLPAL.NS",
            "United Spirits": "MCDOWELL-N.NS"
        }
    },
    "NIFTY Metal": {
        "ticker": "^CNXMETAL",
        "individual_stocks": {
            "Tata Steel": "TATASTEEL.NS",
            "JSW Steel": "JSWSTEEL.NS",
            "Hindalco Industries": "HINDALCO.NS",
            "Coal India": "COALINDIA.NS",
            "Vedanta": "VEDL.NS",
            "Jindal Steel": "JINDALSTEL.NS",
            "SAIL": "SAIL.NS",
            "NMDC": "NMDC.NS",
            "Hindustan Zinc": "HINDZINC.NS",
            "Nalco": "NATIONALUM.NS"
        }
    },
    "NIFTY Energy": {
        "ticker": "^CNXENERGY",
        "individual_stocks": {
            "Reliance Industries": "RELIANCE.NS",
            "NTPC": "NTPC.NS",
            "Power Grid Corporation": "POWERGRID.NS",
            "ONGC": "ONGC.NS",
            "Bharat Petroleum": "BPCL.NS",
            "IOC": "IOC.NS",
            "Adani Power": "ADANIPOWER.NS",
            "Tata Power": "TATAPOWER.NS",
            "Oil India": "OIL.NS",
            "GAIL": "GAIL.NS"
        }
    }
}

# ==============================================================================
# === HELPER FUNCTIONS =========================================================
# ==============================================================================

@st.cache_data(ttl=3600)
def run_pre_market_screener():
    """
    Downloads all NSE stock symbols, fetches their previous day's data,
    and filters them based on intraday criteria: Price > ₹100, Volume > 100,000
    """
    st.write("🔍 Running Pre-Market Screener...")
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df_all_stocks = pd.read_csv(url)
        nse_symbols = [f"{symbol}.NS" for symbol in df_all_stocks['SYMBOL']]
        
        # For production, remove the limit. For testing, use first 100
        symbols_to_scan = nse_symbols[:100]
        st.write(f"📊 Scanning {len(symbols_to_scan)} stocks...")

        tickers_str = " ".join(symbols_to_scan)
        data = yf.download(tickers_str, period="5d", group_by='ticker', auto_adjust=True, progress=False)

        screened_list = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(symbols_to_scan):
            try:
                stock_data = data[ticker] if len(symbols_to_scan) > 1 else data
                if not stock_data.empty:
                    last_day = stock_data.iloc[-1]
                    price = last_day['Close']
                    volume = last_day['Volume']
                    
                    # Intraday screening criteria
                    if price > 100 and volume > 100000:
                        screened_list[ticker] = {
                            'price': price, 
                            'volume': volume,
                            'change_pct': ((last_day['Close'] - stock_data.iloc[-2]['Close']) / stock_data.iloc[-2]['Close'] * 100) if len(stock_data) > 1 else 0
                        }
            except Exception:
                continue
            progress_bar.progress((i + 1) / len(symbols_to_scan))

        st.success(f"✅ Found {len(screened_list)} stocks meeting criteria")
        return screened_list

    except Exception as e:
        st.error(f"❌ Pre-market scan error: {e}")
        return {}

@st.cache_data
def search_for_ticker(query: str, asset_type: str = "EQUITY") -> dict:
    """Searches Yahoo Finance for a given query, filtered by asset type."""
    asset_type_map = {
        "Equities (Stocks)": "EQUITY",
        "Cryptocurrencies": "CRYPTOCURRENCY",
        "ETFs": "ETF",
        "Mutual Funds": "MUTUALFUND",
        "Indices": "INDEX",
        "Commodities": "COMMODITY",
        "Currencies / Forex": "CURRENCY"
    }
    api_quote_type = asset_type_map.get(asset_type, "EQUITY")
    base_url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {
        'q': query,
        'quotesCount': 10,
        'newsCount': 0,
        'listsCount': 0,
        'enableFuzzyQuery': 'false',
        'quotesQueryId': 'tss_match_phrase_query',
        'multiQuoteQueryId': 'multi_quote_single_token_query',
        'newsQueryId': 'news_cie_vespa',
        'enableCb': 'true',
        'enableNavLinks': 'true',
        'enableEnhancedTrivialQuery': 'true',
        'quoteType': api_quote_type
    }
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        results = data.get('quotes', [])
        if not results:
            return {}
        ticker_options = {}
        for result in results:
            name = result.get('longname') or result.get('shortname')
            if name and 'symbol' in result:
                exchange = result.get('exchDisp', 'N/A')
                display_name = f"{name} ({result['symbol']}) - {exchange}"
                ticker_options[display_name] = result['symbol']
        return ticker_options
    except Exception as e:
        st.warning(f"Could not connect to Yahoo Finance search: {e}")
        return {}

@st.cache_data
def fetch_stock_data(ticker, period="1y", interval="1d"):
    """Fetch stock data with configurable interval for multi-timeframe analysis"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info.get('longName') and not info.get('shortName'):
            st.error(f"Ticker '{ticker}' not found or is invalid.")
            return None
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            st.error(f"No historical data found for ticker: {ticker}.")
            return None
        return hist
    except Exception as e:
        st.error(f"Error fetching data for '{ticker}': {e}")
        return None

def fetch_intraday_data(ticker, interval="5m", period="5d"):
    """
    Fetch intraday data for 5-minute or 15-minute analysis
    Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            st.warning(f"No intraday data available for {ticker}")
            return None
        # Rename columns to lowercase for consistency
        hist.columns = [col.lower() for col in hist.columns]
        return hist
    except Exception as e:
        st.error(f"Error fetching intraday data: {e}")
        return None

def is_market_open():
    """Check if Indian stock market is currently open (9:15 AM to 3:30 PM IST)"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    market_start = dt_time(9, 15)
    market_end = dt_time(15, 30)
    buffer_start = dt_time(9, 25)  # Avoid first 10 minutes
    
    is_weekday = now.weekday() < 5
    is_trading_hours = market_start <= now.time() <= market_end
    past_buffer = now.time() >= buffer_start
    
    return is_weekday and is_trading_hours and past_buffer

def send_email_alert(subject, body, to_email=None):
    """
    Send email alert for trading signals
    """
    if not GMAIL_EMAIL or not GMAIL_APP_PASSWORD:
        st.warning("Email credentials not configured")
        return False
    
    try:
        recipient = to_email if to_email else GMAIL_EMAIL
        
        msg = MIMEMultipart()
        msg['From'] = GMAIL_EMAIL
        msg['To'] = recipient
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        
        return True
    except Exception as e:
        st.error(f"Email error: {e}")
        return False

def setup_google_sheets():
    """Initializes connection to Google Sheets."""
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        try:
            sheet = client.open("TradingAnalyzerLog").sheet1
        except gspread.SpreadsheetNotFound:
            sheet = client.create("TradingAnalyzerLog").sheet1
            headers = ["Timestamp", "Ticker", "Signal", "Confidence", "RSI", "Sentiment", "AI Summary"]
            sheet.append_row(headers)
        return sheet
    except Exception as e:
        st.error(f"Google Sheets setup failed: {e}")
        return None

def log_to_sheets(sheet, data):
    """Logs a row of data to the specified Google Sheet."""
    if sheet:
        try:
            sheet.append_row(data)
            return True
        except Exception as e:
            st.warning(f"Failed to log to sheets: {e}")
    return False

def create_plotly_charts(data, ticker_name):
    """Creates a focused trading chart with Price/EMAs, RSI, and color-coded Volume."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & EMAs', 'RSI', 'Volume'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price and EMAs
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'] if 'Close' in data.columns else data['close'],
            mode='lines',
            name='Price',
            line=dict(color='white', width=2)
        ),
        row=1, col=1
    )
    
    close_col = 'Close' if 'Close' in data.columns else 'close'
    
    for span, color in zip([20, 50, 200], ['#1f77b4', '#ff7f0e', '#d62728']):
        if len(data) >= span:
            ema = data[close_col].ewm(span=span, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ema,
                    mode='lines',
                    name=f'EMA {span}',
                    line=dict(width=1.5, color=color)
                ),
                row=1, col=1
            )
    
    # RSI
    delta = data[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=rsi_series,
            mode='lines',
            name='RSI',
            line=dict(color='#9467bd')
        ),
        row=2, col=1
    )
    
    # RSI levels
    for y_val, dash_style, text in [(70, 'dash', 'Overbought'), (60, 'dot', ''), (40, 'dot', ''), (30, 'dash', 'Oversold')]:
        fig.add_hline(
            y=y_val,
            line_dash=dash_style,
            line_color="grey",
            row=2, col=1,
            annotation_text=text,
            annotation_position="right"
        )
    
    # Volume with color coding
    open_col = 'Open' if 'Open' in data.columns else 'open'
    volume_col = 'Volume' if 'Volume' in data.columns else 'volume'
    
    colors = ['#2ca02c' if row[close_col] >= row[open_col] else '#d62728' for index, row in data.iterrows()]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data[volume_col],
            name='Volume',
            marker_color=colors
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text=f'Technical Analysis for {ticker_name}',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

def embed_tradingview_widget(ticker):
    """Generates and embeds a TradingView Advanced Real-Time Chart widget."""
    if ".NS" in ticker:
        tv_ticker = f"NSE:{ticker.replace('.NS', '')}"
    elif ".BO" in ticker:
        tv_ticker = f"BSE:{ticker.replace('.BO', '')}"
    else:
        tv_ticker = ticker.replace('-', '')
    
    html_code = f"""
    <div class="tradingview-widget-container" style="height:500px;width:100%;">
      <div id="tradingview_chart" style="height:100%;width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%",
        "height": 500,
        "symbol": "{tv_ticker}",
        "interval": "D",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>"""
    return html_code

# ==============================================================================
# === AI/LLM INTEGRATION FUNCTIONS =============================================
# ==============================================================================

def get_ai_analysis_gemini(prompt):
    """
    Get AI analysis using Google Gemini API
    """
    if not GOOGLE_API_KEY:
        return "Gemini API key not configured"
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API error: {str(e)}"

def get_ai_analysis_openrouter(prompt, model="anthropic/claude-3.5-sonnet"):
    """
    Get AI analysis using OpenRouter API (supports multiple models)
    """
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not configured"
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"OpenRouter API error: {str(e)}"

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
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
        except Exception as e:
            st.warning(f"Using default sentiment analyzer due to: {e}")
            self.sentiment_analyzer = pipeline("sentiment-analysis")

    def fetch_stock_data(self, ticker, period="60d"):
        """Fetch stock data using yfinance with longer period for MA calculations"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                st.error(f"No data found for ticker: {ticker}")
                return None
            return hist
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    def compute_rsi(self, data, window=14):
        """Calculate RSI (Relative Strength Index)"""
        try:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            delta = data[close_col].diff()
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
            close_col = 'Close' if 'Close' in data.columns else 'close'
            exp1 = data[close_col].ewm(span=12, adjust=False).mean()
            exp2 = data[close_col].ewm(span=26, adjust=False).mean()
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
        """Calculate Moving Averages including 25-day MA for intraday"""
        try:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            ma_20 = data[close_col].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else np.nan
            ma_25 = data[close_col].rolling(window=25).mean().iloc[-1] if len(data) >= 25 else np.nan
            ma_50 = data[close_col].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else np.nan
            ma_200 = data[close_col].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else np.nan

            current_price = data[close_col].iloc[-1]
            return {
                'MA_20': ma_20 if not pd.isna(ma_20) else current_price,
                'MA_25': ma_25 if not pd.isna(ma_25) else current_price,
                'MA_50': ma_50 if not pd.isna(ma_50) else current_price,
                'MA_200': ma_200 if not pd.isna(ma_200) else current_price
            }
        except:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            current_price = data[close_col].iloc[-1]
            return {
                'MA_20': current_price,
                'MA_25': current_price,
                'MA_50': current_price,
                'MA_200': current_price
            }

    def compute_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands for intraday volatility analysis"""
        try:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            sma = data[close_col].rolling(window=window).mean()
            std = data[close_col].rolling(window=window).std()
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            
            return {
                'upper': upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else data[close_col].iloc[-1],
                'middle': sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else data[close_col].iloc[-1],
                'lower': lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else data[close_col].iloc[-1]
            }
        except:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            current_price = data[close_col].iloc[-1]
            return {'upper': current_price, 'middle': current_price, 'lower': current_price}

    def compute_stochastic_momentum(self, data, k_period=14, d_period=3):
        """Calculate Stochastic Momentum Index (SMI) for intraday trading"""
        try:
            high_col = 'High' if 'High' in data.columns else 'high'
            low_col = 'Low' if 'Low' in data.columns else 'low'
            close_col = 'Close' if 'Close' in data.columns else 'close'
            
            highest_high = data[high_col].rolling(window=k_period).max()
            lowest_low = data[low_col].rolling(window=k_period).min()
            
            k_line = 100 * ((data[close_col] - lowest_low) / (highest_high - lowest_low))
            d_line = k_line.rolling(window=d_period).mean()
            
            crossover = 'none'
            if len(k_line) > 1 and len(d_line) > 1:
                if k_line.iloc[-1] > d_line.iloc[-1] and k_line.iloc[-2] <= d_line.iloc[-2]:
                    crossover = 'bullish'
                elif k_line.iloc[-1] < d_line.iloc[-1] and k_line.iloc[-2] >= d_line.iloc[-2]:
                    crossover = 'bearish'
            
            return {
                'k': k_line.iloc[-1] if not pd.isna(k_line.iloc[-1]) else 50.0,
                'd': d_line.iloc[-1] if not pd.isna(d_line.iloc[-1]) else 50.0,
                'crossover': crossover
            }
        except:
            return {'k': 50.0, 'd': 50.0, 'crossover': 'none'}

    def compute_vwap(self, data):
        """Calculate VWAP (Volume-Weighted Average Price) for intraday stocks only"""
        try:
            high_col = 'High' if 'High' in data.columns else 'high'
            low_col = 'Low' if 'Low' in data.columns else 'low'
            close_col = 'Close' if 'Close' in data.columns else 'close'
            volume_col = 'Volume' if 'Volume' in data.columns else 'volume'
            
            typical_price = (data[high_col] + data[low_col] + data[close_col]) / 3
            vwap = (typical_price * data[volume_col]).cumsum() / data[volume_col].cumsum()
            data['vwap'] = vwap
            return data
        except:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            data['vwap'] = data[close_col]
            return data

    def compute_vwma(self, data, period=20):
        """Calculate VWMA (Volume-Weighted Moving Average) for crossover signals"""
        try:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            volume_col = 'Volume' if 'Volume' in data.columns else 'volume'
            
            vwma = (data[close_col] * data[volume_col]).rolling(window=period).sum() / data[volume_col].rolling(window=period).sum()
            return vwma.iloc[-1] if not pd.isna(vwma.iloc[-1]) else data[close_col].iloc[-1]
        except:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            return data[close_col].iloc[-1]

    def compute_supertrend(self, data, period=10, multiplier=3):
        """Calculate Supertrend indicator for target identification"""
        try:
            high_col = 'High' if 'High' in data.columns else 'high'
            low_col = 'Low' if 'Low' in data.columns else 'low'
            close_col = 'Close' if 'Close' in data.columns else 'close'
            
            # Calculate ATR
            high_low = data[high_col] - data[low_col]
            high_close = abs(data[high_col] - data[close_col].shift())
            low_close = abs(data[low_col] - data[close_col].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Calculate basic upper and lower bands
            hl_avg = (data[high_col] + data[low_col]) / 2
            upper_band = hl_avg + (multiplier * atr)
            lower_band = hl_avg - (multiplier * atr)
            
            # Determine trend
            supertrend = pd.Series(index=data.index, dtype=float)
            trend = pd.Series(index=data.index, dtype=int)
            
            for i in range(period, len(data)):
                if data[close_col].iloc[i] > upper_band.iloc[i-1]:
                    trend.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]
                elif data[close_col].iloc[i] < lower_band.iloc[i-1]:
                    trend.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]
                else:
                    trend.iloc[i] = trend.iloc[i-1] if i > period else 0
                    if trend.iloc[i] == 1:
                        supertrend.iloc[i] = lower_band.iloc[i]
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]
            
            return {
                'value': supertrend.iloc[-1] if not pd.isna(supertrend.iloc[-1]) else data[close_col].iloc[-1],
                'trend': 'uptrend' if trend.iloc[-1] == 1 else 'downtrend'
            }
        except:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            return {'value': data[close_col].iloc[-1], 'trend': 'neutral'}

    def detect_support_resistance(self, data, window=20):
        """Identify support and resistance levels from 15-minute data"""
        try:
            high_col = 'High' if 'High' in data.columns else 'high'
            low_col = 'Low' if 'Low' in data.columns else 'low'
            close_col = 'Close' if 'Close' in data.columns else 'close'
            
            highs = data[high_col].rolling(window=window, center=True).max()
            lows = data[low_col].rolling(window=window, center=True).min()
            
            resistance = highs.iloc[-window:].max()
            support = lows.iloc[-window:].min()
            
            return {
                'resistance': resistance,
                'support': support,
                'current_price': data[close_col].iloc[-1]
            }
        except:
            close_col = 'Close' if 'Close' in data.columns else 'close'
            current_price = data[close_col].iloc[-1]
            return {
                'resistance': current_price * 1.02,
                'support': current_price * 0.98,
                'current_price': current_price
            }

    def check_candlestick_pattern(self, five_min_data):
        """Identifies key 3-candle reversal patterns like Morning Star and Evening Star"""
        if len(five_min_data) < 3:
            return "Not enough data"
        
        close_col = 'Close' if 'Close' in five_min_data.columns else 'close'
        open_col = 'Open' if 'Open' in five_min_data.columns else 'open'
        
        last3 = five_min_data.tail(3)
        c1, c2, c3 = last3.iloc[0], last3.iloc[1], last3.iloc[2]
        
        # Morning Star (Bullish Reversal)
        is_morning_star = (
            c1[close_col] < c1[open_col] and
            abs(c2[close_col] - c2[open_col]) < abs(c1[close_col] - c1[open_col]) and
            c3[close_col] > c3[open_col] and
            c3[close_col] > c1[open_col]
        )
        if is_morning_star:
            return "Morning Star (Bullish)"
        
        # Evening Star (Bearish Reversal)
        is_evening_star = (
            c1[close_col] > c1[open_col] and
            abs(c2[close_col] - c2[open_col]) < abs(c1[close_col] - c1[open_col]) and
            c3[close_col] < c3[open_col] and
            c3[close_col] < c1[open_col]
        )
        if is_evening_star:
            return "Evening Star (Bearish)"
        
        return "No significant pattern"

    def detect_inside_bar_pattern(self, data):
        """Detect Inside Bar setup (15-min, Stocks Only)"""
        if len(data) < 2:
            return {"detected": False, "message": "Insufficient data"}
        
        try:
            high_col = 'High' if 'High' in data.columns else 'high'
            low_col = 'Low' if 'Low' in data.columns else 'low'
            close_col = 'Close' if 'Close' in data.columns else 'close'
            
            mother_bar = data.iloc[-2]
            inside_bar = data.iloc[-1]
            
            is_inside_bar = (
                inside_bar[high_col] <= mother_bar[high_col] and
                inside_bar[low_col] >= mother_bar[low_col]
            )
            
            if is_inside_bar:
                return {
                    "detected": True,
                    "mother_high": mother_bar[high_col],
                    "mother_low": mother_bar[low_col],
                    "current_price": data[close_col].iloc[-1],
                    "buy_trigger": mother_bar[high_col],
                    "sell_trigger": mother_bar[low_col],
                    "message": f"Inside Bar detected. Buy above {mother_bar[high_col]:.2f} or Sell below {mother_bar[low_col]:.2f}"
                }
            else:
                return {"detected": False, "message": "No Inside Bar pattern"}
        except:
            return {"detected": False, "message": "Error detecting Inside Bar"}

    def detect_breakout_retest(self, five_min_data, resistance):
        """Analyzes 5-minute data to detect a breakout, retest, and confirmation."""
        if five_min_data.empty or resistance == 0:
            return "Not Analyzed"
        
        high_col = 'High' if 'High' in five_min_data.columns else 'high'
        low_col = 'Low' if 'Low' in five_min_data.columns else 'low'
        close_col = 'Close' if 'Close' in five_min_data.columns else 'close'
        
        recent_data = five_min_data.tail(20)
        
        breakout_candle_index = -1
        retest_candle_index = -1
        
        # 1. Detect the Breakout
        for i in range(1, len(recent_data)):
            prev_high = recent_data[high_col].iloc[i-1]
            current_high = recent_data[high_col].iloc[i]
            
            if current_high > resistance and prev_high <= resistance:
                breakout_candle_index = i
                break
        
        if breakout_candle_index == -1:
            return "No Breakout Detected"
        
        # 2. Detect the Retest
        for i in range(breakout_candle_index + 1, len(recent_data)):
            current_low = recent_data[low_col].iloc[i]
            
            if current_low <= resistance:
                retest_candle_index = i
                break
        
        if retest_candle_index == -1:
            return f"Breakout Occurred at {recent_data.index[breakout_candle_index].strftime('%H:%M')}. Awaiting Retest."
        
        # 3. Check for Confirmation
        if retest_candle_index < len(recent_data) - 1:
            confirmation_candle = recent_data.iloc[retest_candle_index + 1]
            
            if confirmation_candle[close_col] > resistance:
                return f"✅ Retest Confirmed at {recent_data.index[retest_candle_index + 1].strftime('%H:%M')}. Potential Entry."
        
        return f"Retest in Progress at {recent_data.index[retest_candle_index].strftime('%H:%M')}. Awaiting Confirmation."

    def run_confirmation_checklist(self, analysis_results):
        """Runs the full 5-point confirmation checklist to generate a final trade signal."""
        checklist = {
            "1. At Key S/R Level": "⚠️ PENDING",
            "2. Price Rejection": "⚠️ PENDING",
            "3. Chart Pattern Confirmed": "⚠️ PENDING",
            "4. Candlestick Signal": "⚠️ PENDING",
            "5. Indicator Alignment": "⚠️ PENDING",
            "FINAL_SIGNAL": "HOLD"
        }
        
        five_min_df = analysis_results.get('5m_data')
        if five_min_df is None or five_min_df.empty:
            return checklist
        
        close_col = 'Close' if 'Close' in five_min_df.columns else 'close'
        high_col = 'High' if 'High' in five_min_df.columns else 'high'
        low_col = 'Low' if 'Low' in five_min_df.columns else 'low'
        
        resistance = analysis_results.get('resistance', 0)
        support = analysis_results.get('support', 0)
        latest_price = analysis_results.get('latest_price', 0)
        
        # Check 1 & 2: At a key level with price rejection
        at_resistance = abs(latest_price - resistance) / resistance < 0.005 if resistance > 0 else False
        at_support = abs(latest_price - support) / support < 0.005 if support > 0 else False
        
        if at_support:
            checklist["1. At Key S/R Level"] = "✅ At Support"
            last_candle = five_min_df.iloc[-1]
            if (last_candle[low_col] < support) and (last_candle[close_col] > support):
                checklist["2. Price Rejection"] = "✅ Bullish Rejection"
        elif at_resistance:
            checklist["1. At Key S/R Level"] = "✅ At Resistance"
            last_candle = five_min_df.iloc[-1]
            if (last_candle[high_col] > resistance) and (last_candle[close_col] < resistance):
                checklist["2. Price Rejection"] = "✅ Bearish Rejection"
        else:
            checklist["1. At Key S/R Level"] = "❌ Not at a key level"
            checklist["2. Price Rejection"] = "❌ No Rejection"
        
        # Check 3: Chart Pattern
        pattern_status = self.detect_breakout_retest(five_min_df, resistance)
        if "✅ Retest Confirmed" in pattern_status:
            checklist["3. Chart Pattern Confirmed"] = "✅ Breakout/Retest"
        else:
            checklist["3. Chart Pattern Confirmed"] = f"❌ {pattern_status}"
        
        # Check 4: Candlestick Pattern
        candle_pattern = self.check_candlestick_pattern(five_min_df)
        if "No significant" not in candle_pattern:
            checklist["4. Candlestick Signal"] = f"✅ {candle_pattern}"
        else:
            checklist["4. Candlestick Signal"] = "❌ No Signal"
        
        # Check 5: Indicator Alignment
        rsi = analysis_results.get('rsi', 50)
        five_min_df = self.compute_vwap(five_min_df)
        vwap = five_min_df['vwap'].iloc[-1]
        
        if (checklist["1. At Key S/R Level"] == "✅ At Support" and 
            rsi < 70 and latest_price > vwap):
            checklist["5. Indicator Alignment"] = "✅ Bullish Alignment"
        elif (checklist["1. At Key S/R Level"] == "✅ At Resistance" and
              rsi > 30 and latest_price < vwap):
            checklist["5. Indicator Alignment"] = "✅ Bearish Alignment"
        else:
            checklist["5. Indicator Alignment"] = "❌ No Alignment"
        
        # Final Signal Generation
        bullish_checks = sum(1 for v in checklist.values() if "✅" in str(v) and ("Bullish" in str(v) or "Breakout" in str(v)))
        bearish_checks = sum(1 for v in checklist.values() if "✅" in str(v) and "Bearish" in str(v))
        
        if bullish_checks >= 3:
            checklist["FINAL_SIGNAL"] = "🟢 BUY"
        elif bearish_checks >= 3:
            checklist["FINAL_SIGNAL"] = "🔴 SELL"
        else:
            checklist["FINAL_SIGNAL"] = "⚪ HOLD"
        
        return checklist

    def analyze_for_intraday(self, ticker):
        """Complete intraday analysis with multi-timeframe approach"""
        results = {
            'ticker': ticker,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'market_open': is_market_open()
        }
        
        try:
            # 1. Fetch 1-day data for trend overview
            daily_data = fetch_stock_data(ticker, period="60d", interval="1d")
            if daily_data is None:
                return None
            
            # 2. Fetch 15-min data for S/R levels
            fifteen_min_data = fetch_intraday_data(ticker, interval="15m", period="5d")
            if fifteen_min_data is None:
                st.warning("15-min data unavailable, using daily data")
                fifteen_min_data = daily_data.copy()
                fifteen_min_data.columns = [col.lower() for col in fifteen_min_data.columns]
            
            # 3. Fetch 5-min data for execution signals
            five_min_data = fetch_intraday_data(ticker, interval="5m", period="5d")
            if five_min_data is None:
                st.warning("5-min data unavailable, using 15-min data")
                five_min_data = fifteen_min_data.copy()
            
            # Technical indicators
            results['latest_price'] = daily_data['Close'].iloc[-1]
            results['rsi'] = self.compute_rsi(daily_data)
            results['macd'] = self.compute_macd(daily_data)
            results['moving_averages'] = self.compute_moving_averages(daily_data)
            
            # Intraday-specific indicators
            results['bollinger_bands'] = self.compute_bollinger_bands(five_min_data)
            results['stochastic'] = self.compute_stochastic_momentum(five_min_data)
            five_min_data = self.compute_vwap(five_min_data)
            results['vwap'] = five_min_data['vwap'].iloc[-1]
            results['vwma'] = self.compute_vwma(five_min_data)
            results['supertrend'] = self.compute_supertrend(five_min_data)
            
            # Support & Resistance
            sr_levels = self.detect_support_resistance(fifteen_min_data)
            results['resistance'] = sr_levels['resistance']
            results['support'] = sr_levels['support']
            
            # Pattern detection
            results['candlestick_pattern'] = self.check_candlestick_pattern(five_min_data)
            results['inside_bar'] = self.detect_inside_bar_pattern(fifteen_min_data)
            results['breakout_status'] = self.detect_breakout_retest(five_min_data, sr_levels['resistance'])
            
            # Store data
            results['5m_data'] = five_min_data
            results['15m_data'] = fifteen_min_data
            results['daily_data'] = daily_data
            
            # Run confirmation checklist
            results['confirmation_checklist'] = self.run_confirmation_checklist(results)
            results['signal'] = results['confirmation_checklist']['FINAL_SIGNAL']
            
            return results
            
        except Exception as e:
            st.error(f"Error in intraday analysis: {e}")
            return None

    def analyze_for_swing(self, ticker):
        """Swing trading analysis with daily charts"""
        results = {
            'ticker': ticker,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'mode': 'swing'
        }
        
        try:
            # Fetch 1-year daily data
            daily_data = self.fetch_stock_data(ticker, period="1y")
            if daily_data is None:
                return None
            
            # Technical indicators
            results['latest_price'] = daily_data['Close'].iloc[-1]
            results['rsi'] = self.compute_rsi(daily_data)
            results['macd'] = self.compute_macd(daily_data)
            results['moving_averages'] = self.compute_moving_averages(daily_data)
            
            # 52-week high/low
            results['52w_high'] = daily_data['Close'].max()
            results['52w_low'] = daily_data['Close'].min()
            results['distance_from_52w_high'] = ((results['latest_price'] - results['52w_high']) / results['52w_high']) * 100
            
            # EMA analysis
            ema_100 = daily_data['Close'].ewm(span=100, adjust=False).mean().iloc[-1] if len(daily_data) >= 100 else None
            ema_200 = daily_data['Close'].ewm(span=200, adjust=False).mean().iloc[-1] if len(daily_data) >= 200 else None
            
            results['ema_100'] = ema_100
            results['ema_200'] = ema_200
            
            # Generate swing signal
            signal = "HOLD"
            if results['latest_price'] > results['moving_averages']['MA_50']:
                if results['rsi'] < 70 and results['macd']['histogram'] > 0:
                    signal = "BUY"
            elif results['latest_price'] < results['moving_averages']['MA_50']:
                if results['rsi'] > 30 and results['macd']['histogram'] < 0:
                    signal = "SELL"
            
            results['signal'] = signal
            results['daily_data'] = daily_data
            
            return results
            
        except Exception as e:
            st.error(f"Error in swing analysis: {e}")
            return None

    def scrape_news_headlines(self, ticker_name, days=1):
        """Scrape news headlines from NewsAPI.org"""
        try:
            api_key = NEWSAPI_KEY if NEWSAPI_KEY else "e205d77d7bc14acc8744d3ea10568f50"
            search_query = ticker_name.replace("^", "").replace(".NS", "").replace("NSE", "")
            url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=publishedAt&apiKey={api_key}&pageSize=5"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
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
            return headlines if headlines else ["No recent news found"]
        except Exception as e:
            return [f"Error fetching news: {str(e)}"]

    def analyze_sentiment(self, headlines):
        """Analyze sentiment of news headlines"""
        if not headlines or not self.sentiment_analyzer:
            return {"sentiment": "Neutral", "score": 0.0}
        
        try:
            sentiments = []
            for headline in headlines:
                if len(headline) > 15:
                    result = self.sentiment_analyzer(headline[:512])
                    if isinstance(result[0], list):
                        sentiment_scores = {item['label']: item['score'] for item in result[0]}
                        if 'positive' in sentiment_scores:
                            sentiments.append(sentiment_scores['positive'] - sentiment_scores.get('negative', 0))
                    else:
                        score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
                        sentiments.append(score)
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                if avg_sentiment > 0.1:
                    return {"sentiment": "Positive", "score": avg_sentiment}
                elif avg_sentiment < -0.1:
                    return {"sentiment": "Negative", "score": avg_sentiment}
            
            return {"sentiment": "Neutral", "score": 0.0}
        except Exception as e:
            return {"sentiment": "Error", "score": 0.0}

def analyze_portfolio(tickers_list):
    """Analyze multiple stocks for portfolio view"""
    analyzer = StockAnalyzer()
    portfolio_results = []
    
    progress_bar = st.progress(0)
    for i, ticker in enumerate(tickers_list):
        try:
            data = analyzer.fetch_stock_data(ticker, period="60d")
            if data is not None:
                latest_price = data['Close'].iloc[-1]
                rsi = analyzer.compute_rsi(data)
                macd = analyzer.compute_macd(data)
                
                signal = "HOLD"
                if rsi < 40 and macd['histogram'] > 0:
                    signal = "BUY"
                elif rsi > 60 and macd['histogram'] < 0:
                    signal = "SELL"
                
                portfolio_results.append({
                    'Ticker': ticker,
                    'Price': f"₹{latest_price:.2f}",
                    'RSI': f"{rsi:.2f}",
                    'Signal': signal
                })
        except:
            continue
        
        progress_bar.progress((i + 1) / len(tickers_list))
    
    return pd.DataFrame(portfolio_results)

# ==============================================================================
# === STREAMLIT UI =============================================================
# ==============================================================================

def main():
    st.set_page_config(page_title="AI Trading Agent", page_icon="📈", layout="wide")
    
    st.title("🤖 AI Trading Agent - Intraday & Swing Trading")
    st.markdown("**Multi-Timeframe Analysis | Pattern Recognition | AI-Powered Insights**")
    
    # Initialize session state
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Trading Configuration")
    
    # Trading Mode Selection
    trading_mode = st.sidebar.radio(
        "Select Trading Mode",
        ["Intraday Trading", "Swing Trading"],
        help="Intraday: 5/15-min execution | Swing: Daily charts"
    )
    
    # Market Status
    market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
    st.sidebar.metric("Market Status", market_status)
    
    # Asset Type Selection
    asset_type = st.sidebar.selectbox(
        "Asset Type",
        ["Equities (Stocks)", "Cryptocurrencies", "ETFs", "Indices", "Commodities", "Currencies / Forex"],
        index=0
    )
    
    # Category Selection
    if asset_type == "Equities (Stocks)":
        use_category = st.sidebar.checkbox("Use Stock Categories", value=False)
        if use_category:
            category = st.sidebar.selectbox("Select Category", list(STOCK_CATEGORIES.keys()))
            stock_name = st.sidebar.selectbox("Select Stock", list(STOCK_CATEGORIES[category]["individual_stocks"].keys()))
            selected_ticker = STOCK_CATEGORIES[category]["individual_stocks"][stock_name]
        else:
            selected_ticker = None
    else:
        selected_ticker = None
    
    # Mode-specific info
    if trading_mode == "Intraday Trading":
        st.sidebar.info("📊 **Intraday Active**\n\n- 1-Day: Overview\n- 15-Min: S/R\n- 5-Min: Execution\n\n⚠️ Avoid first 10 min!")
        
        if st.sidebar.button("🔍 Run Pre-Market Screener"):
            with st.spinner("Scanning market..."):
                screened_stocks = run_pre_market_screener()
                if screened_stocks:
                    st.session_state['screened_stocks'] = screened_stocks
        
        if 'screened_stocks' in st.session_state:
            st.sidebar.write("**Top Screened:**")
            for ticker, data in list(st.session_state['screened_stocks'].items())[:5]:
                st.sidebar.write(f"{ticker}: ₹{data['price']:.2f}")
    else:
        st.sidebar.info("📈 **Swing Mode**\n\n- Daily charts\n- Multi-day holds\n- 52-week analysis")
    
    # AI Model Selection
    st.sidebar.subheader("🤖 AI Analysis")
    ai_model = st.sidebar.selectbox(
        "AI Model",
        ["None", "Google Gemini", "OpenRouter (Claude)"],
        help="Enable AI-powered analysis"
    )
    
    # Email Alerts
    st.sidebar.subheader("📧 Alerts")
    enable_email = st.sidebar.checkbox("Enable Email Alerts", value=False)
    if enable_email:
        alert_email = st.sidebar.text_input("Email Address", value=GMAIL_EMAIL or "")
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🤖 AI Insights", "📁 Portfolio", "⚙️ Settings"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if selected_ticker:
                ticker_input = selected_ticker
                st.info(f"Analyzing: {ticker_input}")
            else:
                search_query = st.text_input("Search for Stock/Asset", "")
                
                if search_query:
                    with st.spinner("Searching..."):
                        search_results = search_for_ticker(search_query, asset_type)
                    
                    if search_results:
                        selected = st.selectbox("Select from results:", list(search_results.keys()))
                        ticker_input = search_results[selected]
                    else:
                        st.warning("No results found")
                        ticker_input = st.text_input("Enter ticker manually", "RELIANCE.NS")
                else:
                    ticker_input = st.text_input("Or enter ticker directly", "RELIANCE.NS")
            
            if st.button("📊 Analyze Stock", type="primary"):
                with st.spinner("Analyzing..."):
                    analyzer = StockAnalyzer()
                    
                    if trading_mode == "Intraday Trading":
                        results = analyzer.analyze_for_intraday(ticker_input)
                    else:
                        results = analyzer.analyze_for_swing(ticker_input)
                    
                    if results:
                        st.session_state['analysis_results'] = results
                        st.session_state['analysis_history'].append(results)
                        st.success("✅ Analysis Complete!")
                        
                        # Send email alert if enabled
                        if enable_email and 'signal' in results:
                            if results['signal'] in ['🟢 BUY', '🔴 SELL']:
                                email_body = f"""
                                <h2>Trading Signal Alert</h2>
                                <p><strong>Ticker:</strong> {ticker_input}</p>
                                <p><strong>Signal:</strong> {results['signal']}</p>
                                <p><strong>Price:</strong> ₹{results['latest_price']:.2f}</p>
                                <p><strong>RSI:</strong> {results['rsi']:.2f}</p>
                                <p><strong>Time:</strong> {results['timestamp']}</p>
                                """
                                send_email_alert(f"Trading Signal: {results['signal']}", email_body, alert_email)
        
        with col2:
            if 'analysis_results' in st.session_state:
                results = st.session_state['analysis_results']
                st.metric("Current Price", f"₹{results['latest_price']:.2f}")
                st.metric("RSI", f"{results['rsi']:.2f}")
                
                if 'signal' in results:
                    st.metric("Signal", results['signal'])
        
        # Display Analysis Results
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            if trading_mode == "Intraday Trading":
                st.subheader("📋 Confirmation Checklist")
                checklist = results['confirmation_checklist']
                
                cols = st.columns(5)
                for i, (key, value) in enumerate(checklist.items()):
                    if key != "FINAL_SIGNAL":
                        with cols[i % 5]:
                            st.write(f"**{key}**")
                            st.write(value)
                
                st.markdown(f"### Final Signal: {checklist['FINAL_SIGNAL']}")
                
                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("15-Min Chart")
                    fig_15m = create_plotly_charts(results['15m_data'], ticker_input)
                    st.plotly_chart(fig_15m, use_container_width=True)
                
                with col2:
                    st.subheader("5-Min Chart")
                    fig_5m = create_plotly_charts(results['5m_data'], ticker_input)
                    st.plotly_chart(fig_5m, use_container_width=True)
                
                # Key Levels
                st.subheader("📊 Key Levels")
                col1, col2, col3 = st.columns(3)
                col1.metric("Resistance", f"₹{results['resistance']:.2f}")
                col2.metric("Current", f"₹{results['latest_price']:.2f}")
                col3.metric("Support", f"₹{results['support']:.2f}")
                
                # Indicators
                st.subheader("📈 Indicators")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("VWAP", f"₹{results['vwap']:.2f}")
                col2.metric("VWMA", f"₹{results['vwma']:.2f}")
                col3.metric("Supertrend", results['supertrend']['trend'])
                col4.metric("SMI", f"{results['stochastic']['k']:.2f}")
                
                # Patterns
                st.subheader("🔍 Pattern Analysis")
                st.write(f"**Candlestick:** {results['candlestick_pattern']}")
                st.write(f"**Breakout:** {results['breakout_status']}")
                if results['inside_bar']['detected']:
                    st.success(results['inside_bar']['message'])
            
            else:  # Swing Trading
                st.subheader("Daily Chart Analysis")
                fig_daily = create_plotly_charts(results['daily_data'], ticker_input)
                st.plotly_chart(fig_daily, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                mas = results['moving_averages']
                col1.metric("MA 20", f"₹{mas['MA_20']:.2f}")
                col2.metric("MA 50", f"₹{mas['MA_50']:.2f}")
                col3.metric("MA 200", f"₹{mas['MA_200']:.2f}")
                col4.metric("Signal", results['signal'])
                
                st.subheader("52-Week Analysis")
                col1, col2, col3 = st.columns(3)
                col1.metric("52W High", f"₹{results['52w_high']:.2f}")
                col2.metric("52W Low", f"₹{results['52w_low']:.2f}")
                col3.metric("Distance from High", f"{results['distance_from_52w_high']:.2f}%")
            
            # TradingView Widget
            st.subheader("📈 TradingView Live Chart")
            components.html(embed_tradingview_widget(ticker_input), height=500)
    
    with tab2:
        st.subheader("🤖 AI-Powered Insights")
        
        if 'analysis_results' not in st.session_state:
            st.info("Please run an analysis first")
        elif ai_model == "None":
            st.warning("Please select an AI model from the sidebar")
        else:
            results = st.session_state['analysis_results']
            
            if st.button("Generate AI Analysis"):
                with st.spinner("AI is analyzing..."):
                    # Prepare prompt
                    prompt = f"""
                    Analyze this stock trading data and provide insights:
                    
                    Ticker: {results['ticker']}
                    Price: ₹{results['latest_price']:.2f}
                    RSI: {results['rsi']:.2f}
                    MACD: {results['macd']}
                    Signal: {results.get('signal', 'N/A')}
                    
                    Provide:
                    1. Technical analysis interpretation
                    2. Risk assessment
                    3. Entry/exit recommendations
                    4. Market sentiment analysis
                    """
                    
                    if ai_model == "Google Gemini":
                        ai_response = get_ai_analysis_gemini(prompt)
                    else:
                        ai_response = get_ai_analysis_openrouter(prompt)
                    
                    st.markdown(ai_response)
                    st.session_state['ai_analysis'] = ai_response
            
            if 'ai_analysis' in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state['ai_analysis'])
    
    with tab3:
        st.subheader("📁 Portfolio Analysis")
        
        portfolio_input = st.text_area(
            "Enter tickers (one per line)",
            "RELIANCE.NS\nTCS.NS\nHDFCBANK.NS\nINFY.NS"
        )
        
        if st.button("Analyze Portfolio"):
            tickers = [t.strip() for t in portfolio_input.split('\n') if t.strip()]
            
            with st.spinner(f"Analyzing {len(tickers)} stocks..."):
                portfolio_df = analyze_portfolio(tickers)
                
                if not portfolio_df.empty:
                    st.dataframe(portfolio_df, use_container_width=True)
                    
                    # Download option
                    csv = portfolio_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Portfolio Analysis",
                        csv,
                        "portfolio_analysis.csv",
                        "text/csv"
                    )
    
    with tab4:
        st.subheader("⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**API Configuration**")
            st.text_input("OpenRouter API Key", value=OPENROUTER_API_KEY or "", type="password")
            st.text_input("Google API Key", value=GOOGLE_API_KEY or "", type="password")
            st.text_input("NewsAPI Key", value=NEWSAPI_KEY or "", type="password")
        
        with col2:
            st.write("**Email Configuration**")
            st.text_input("Gmail Email", value=GMAIL_EMAIL or "")
            st.text_input("Gmail App Password", value=GMAIL_APP_PASSWORD or "", type="password")
        
        st.markdown("---")
        st.write("**Analysis History**")
        if st.session_state['analysis_history']:
            st.write(f"Total analyses: {len(st.session_state['analysis_history'])}")
            if st.button("Clear History"):
                st.session_state['analysis_history'] = []
                st.success("History cleared!")
        else:
            st.info("No analysis history yet")

if __name__ == "__main__":
    main()
