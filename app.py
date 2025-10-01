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

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GMAIL_EMAIL = os.getenv("GMAIL_EMAI")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Static Data & Mappings ---
# Comprehensive Indian Stock Index and Individual Stock Mapping
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
            "State Bank of India": "SBIN.NS",
            "ITC": "ITC.NS",
            "Bharti Airtel": "BHARTIARTL.NS",
            "Kotak Mahindra Bank": "KOTAKBANK.NS",
            "Larsen & Toubro": "LT.NS",
            "Axis Bank": "AXISBANK.NS",
            "Maruti Suzuki": "MARUTI.NS",
            "Asian Paints": "ASIANPAINT.NS",
            "Nestle India": "NESTLEIND.NS",
            "HCL Technologies": "HCLTECH.NS",
            "Bajaj Finance": "BAJFINANCE.NS",
            "Wipro": "WIPRO.NS",
            "Ultratech Cement": "ULTRACEMCO.NS",
            "Titan Company": "TITAN.NS",
            "Tata Motors": "TATAMOTORS.NS",
            "Sun Pharmaceutical": "SUNPHARMA.NS",
            "NTPC": "NTPC.NS",
            "Power Grid Corporation": "POWERGRID.NS",
            "Bajaj Finserv": "BAJAJFINSV.NS",
            "Dr. Reddy's Laboratories": "DRREDDY.NS",
            "Tech Mahindra": "TECHM.NS",
            "Oil & Natural Gas Corp": "ONGC.NS",
            "Tata Steel": "TATASTEEL.NS",
            "IndusInd Bank": "INDUSINDBK.NS",
            "Mahindra & Mahindra": "M&M.NS",
            "Adani Enterprises": "ADANIENT.NS",
            "Coal India": "COALINDIA.NS",
            "JSW Steel": "JSWSTEEL.NS",
            "Cipla": "CIPLA.NS",
            "Grasim Industries": "GRASIM.NS",
            "Hero MotoCorp": "HEROMOTOCO.NS",
            "UPL": "UPL.NS",
            "Britannia Industries": "BRITANNIA.NS",
            "Eicher Motors": "EICHERMOT.NS",
            "HDFC Life Insurance": "HDFCLIFE.NS",
            "SBI Life Insurance": "SBILIFE.NS",
            "Divis Laboratories": "DIVISLAB.NS",
            "Hindalco Industries": "HINDALCO.NS",
            "Bajaj Auto": "BAJAJ-AUTO.NS",
            "Shree Cement": "SHREECEM.NS",
            "Apollo Hospitals": "APOLLOHOSP.NS",
            "HDFC Asset Management": "HDFCAMC.NS",
            "Adani Ports": "ADANIPORTS.NS",
            "BPCL": "BPCL.NS"
        }
    },
    "BANK NIFTY Index": {
        "ticker": "^NSEBANK",
        "individual_stocks": {
            "HDFC Bank": "HDFCBANK.NS",
            "ICICI Bank": "ICICIBANK.NS",
            "State Bank of India": "SBIN.NS",
            "Kotak Mahindra Bank": "KOTAKBANK.NS",
            "Axis Bank": "AXISBANK.NS",
            "IndusInd Bank": "INDUSINDBK.NS",
            "Punjab National Bank": "PNB.NS",
            "Bank of Baroda": "BANKBARODA.NS",
            "Federal Bank": "FEDERALBNK.NS",
            "IDFC First Bank": "IDFCFIRSTB.NS",
            "AU Small Finance Bank": "AUBANK.NS",
            "Bandhan Bank": "BANDHANBNK.NS"
        }
    },
    "NIFTY AUTO Index": {
        "ticker": "^CNXAUTO",
        "individual_stocks": {
            "Maruti Suzuki": "MARUTI.NS",
            "Tata Motors": "TATAMOTORS.NS",
            "Mahindra & Mahindra": "M&M.NS",
            "Hero MotoCorp": "HEROMOTOCO.NS",
            "Eicher Motors": "EICHERMOT.NS",
            "Bajaj Auto": "BAJAJ-AUTO.NS",
            "TVS Motor Company": "TVSMOTOR.NS",
            "Ashok Leyland": "ASHOKLEY.NS",
            "Force Motors": "FORCEMOT.NS",
            "MRF": "MRF.NS",
            "Balkrishna Industries": "BALKRISIND.NS",
            "Ceat": "CEATLTD.NS",
            "Apollo Tyres": "APOLLOTYRE.NS",
            "JK Tyre": "JKTYRE.NS",
            "Bharat Forge": "BHARATFORG.NS"
        }
    },
    "NIFTY PHARMA Index": {
        "ticker": "^CNXPHARMA",
        "individual_stocks": {
            "Sun Pharmaceutical": "SUNPHARMA.NS",
            "Dr. Reddy's Laboratories": "DRREDDY.NS",
            "Cipla": "CIPLA.NS",
            "Divis Laboratories": "DIVISLAB.NS",
            "Lupin": "LUPIN.NS",
            "Aurobindo Pharma": "AUROPHARMA.NS",
            "Biocon": "BIOCON.NS",
            "Cadila Healthcare": "ZYDUSLIFE.NS",
            "Alkem Laboratories": "ALKEM.NS",
            "Glenmark Pharmaceuticals": "GLENMARK.NS",
            "Torrent Pharmaceuticals": "TORNTPHARM.NS",
            "Ipca Laboratories": "IPCALAB.NS",
            "Abbott India": "ABBOTINDIA.NS",
            "Pfizer": "PFIZER.NS",
            "GSK Pharma": "GSK.NS"
        }
    },
    "NIFTY METAL Index": {
        "ticker": "^CNXMETAL",
        "individual_stocks": {
            "Tata Steel": "TATASTEEL.NS",
            "JSW Steel": "JSWSTEEL.NS",
            "Hindalco Industries": "HINDALCO.NS",
            "Vedanta": "VEDL.NS",
            "Coal India": "COALINDIA.NS",
            "Steel Authority of India": "SAIL.NS",
            "NMDC": "NMDC.NS",
            "Jindal Steel & Power": "JINDALSTEL.NS",
            "National Aluminium Company": "NATIONALUM.NS",
            "APL Apollo Tubes": "APLAPOLLO.NS",
            "Hindustan Zinc": "HINDZINC.NS",
            "Ratnamani Metals": "RATNAMANI.NS",
            "Welspun Corp": "WELCORP.NS",
            "MOIL": "MOIL.NS"
        }
    },
    "NIFTY IT Index": {
        "ticker": "^CNXIT",
        "individual_stocks": {
            "Tata Consultancy Services": "TCS.NS",
            "Infosys": "INFY.NS",
            "HCL Technologies": "HCLTECH.NS",
            "Wipro": "WIPRO.NS",
            "Tech Mahindra": "TECHM.NS",
            "LTI Mindtree": "LTIM.NS",
            "Mphasis": "MPHASIS.NS",
            "Persistent Systems": "PERSISTENT.NS",
            "Coforge": "COFORGE.NS",
            "L&T Technology Services": "LTTS.NS"
        }
    },
    "NIFTY FMCG Index": {
        "ticker": "^CNXFMCG",
        "individual_stocks": {
            "Hindustan Unilever": "HINDUNILVR.NS",
            "ITC": "ITC.NS",
            "Nestle India": "NESTLEIND.NS",
            "Britannia Industries": "BRITANNIA.NS",
            "Dabur India": "DABUR.NS",
            "Marico": "MARICO.NS",
            "Godrej Consumer Products": "GODREJCP.NS",
            "Colgate-Palmolive": "COLPAL.NS",
            "United Spirits": "UBL.NS",
            "Tata Consumer Products": "TATACONSUM.NS",
            "Emami": "EMAMILTD.NS",
            "P&G Hygiene": "PGHH.NS",
            "VBL": "VBL.NS"
        }
    }
}
# ==============================================================================
# === GLOBAL HELPER FUNCTIONS (Defined before they are called) =================
# ==============================================================================
@st.cache_data(ttl=3600) # Cache the results for 1 hour
def run_pre_market_screener():
    """
    Downloads all NSE stock symbols, fetches their previous day's data,
    and filters them based on price and volume criteria.
    """
    st.write("Running Pre-Market Screener...")
    try:
        # Step 1: Fetch the list of all NSE-listed equity symbols
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df_all_stocks = pd.read_csv(url)
        # We only need the 'SYMBOL' column and we'll append '.NS' for yfinance
        nse_symbols = [f"{symbol}.NS" for symbol in df_all_stocks['SYMBOL']]
        
        # Limit to the first 100 stocks for speed during development, remove later
        symbols_to_scan = nse_symbols[:100] 
        st.write(f"Found {len(nse_symbols)} total stocks. Scanning the first {len(symbols_to_scan)} for this run...")

        # Step 2: Fetch previous day's data for all stocks at once
        # Using a 5-day period to ensure we get the last trading day's data
        tickers_str = " ".join(symbols_to_scan)
        data = yf.download(tickers_str, period="5d", group_by='ticker', auto_adjust=True)

        screened_list = {}
        progress_bar = st.progress(0)
        
        # Step 3: Apply filtering criteria
        for i, ticker in enumerate(symbols_to_scan):
            try:
                stock_data = data[ticker]
                if not stock_data.empty:
                    # Get the last available trading day's data
                    last_day = stock_data.iloc[-1]
                    price = last_day['Close']
                    volume = last_day['Volume']
                    
                    # Your screening criteria from the notes
                    if price > 100 and volume > 100000:
                        screened_list[ticker] = {'price': price, 'volume': volume}
            except Exception:
                # Ignore tickers that fail to download, often due to delisting
                continue
            progress_bar.progress((i + 1) / len(symbols_to_scan))

        st.write(f"Screening complete. Found {len(screened_list)} stocks meeting your criteria.")
        return screened_list

    except Exception as e:
        st.error(f"An error occurred during the pre-market scan: {e}")
        return {}

@st.cache_data
def search_for_ticker(query: str, asset_type: str = "EQUITY") -> dict:
    """Searches Yahoo Finance for a given query, filtered by asset type."""
    asset_type_map = {
        "Equities (Stocks)": "EQUITY", "Cryptocurrencies": "CRYPTOCURRENCY", "ETFs": "ETF",
        "Mutual Funds": "MUTUALFUND", "Indices": "INDEX", "Commodities": "COMMODITY", "Currencies / Forex": "CURRENCY"
    }
    api_quote_type = asset_type_map.get(asset_type, "EQUITY")
    base_url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {
        'q': query, 'quotesCount': 10, 'newsCount': 0, 'listsCount': 0, 'enableFuzzyQuery': 'false',
        'quotesQueryId': 'tss_match_phrase_query', 'multiQuoteQueryId': 'multi_quote_single_token_query',
        'newsQueryId': 'news_cie_vespa', 'enableCb': 'true', 'enableNavLinks': 'true',
        'enableEnhancedTrivialQuery': 'true', 'quoteType': api_quote_type
    }
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        results = data.get('quotes', [])
        if not results: return {}
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
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info.get('longName') and not info.get('shortName'):
            st.error(f"Ticker '{ticker}' not found or is invalid.")
            return None
        hist = stock.history(period=period)
        if hist.empty:
            st.error(f"No historical data found for ticker: {ticker}.")
            return None
        return hist
    except Exception as e:
        st.error(f"Error fetching data for '{ticker}': {e}")
        return None

def create_plotly_charts(data, ticker_name):
    """Creates a focused trading chart with Price/EMAs, RSI, and color-coded Volume."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=('Price & EMAs', 'RSI', 'Volume'), row_heights=[0.6, 0.2, 0.2]
    )
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price', line=dict(color='white', width=2)), row=1, col=1)
    for span, color in zip([20, 50, 200], ['#1f77b4', '#ff7f0e', '#d62728']):
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].ewm(span=span, adjust=False).mean(), mode='lines', name=f'EMA {span}', line=dict(width=1.5, color=color)), row=1, col=1)
    rsi_series = 100 - (100 / (1 + (data['Close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / -data['Close'].diff().where(lambda x: x < 0, 0).rolling(14).mean())))
    fig.add_trace(go.Scatter(x=data.index, y=rsi_series, mode='lines', name='RSI', line=dict(color='#9467bd')), row=2, col=1)
    rsi_lines = [
        {'y': 70, 'dash': 'dash', 'text': 'Overbought'}, {'y': 60, 'dash': 'dot', 'text': ''},
        {'y': 40, 'dash': 'dot', 'text': ''}, {'y': 30, 'dash': 'dash', 'text': 'Oversold'}
    ]
    for line in rsi_lines:
        fig.add_hline(y=line['y'], line_dash=line['dash'], line_color="grey", row=2, col=1,
                      annotation_text=line['text'], annotation_position="right")
    colors = ['#2ca02c' if row['Close'] >= row['Open'] else '#d62728' for index, row in data.iterrows()]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors), row=3, col=1)
    fig.update_layout(height=800, title_text=f'Technical Analysis for {ticker_name}', showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_rangeslider_visible=False, template='plotly_dark')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    return fig

def embed_tradingview_widget(ticker):
    """Generates and embeds a TradingView Advanced Real-Time Chart widget."""
    tv_ticker = f"NSE:{ticker.replace('.NS', '')}" if ".NS" in ticker else f"BSE:{ticker.replace('.BO', '')}" if ".BO" in ticker else ticker.replace('-', '')
    html_code = f"""
    <div class="tradingview-widget-container" style="height:500px;width:100%;">
      <div id="tradingview_chart" style="height:100%;width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%", "height": 500, "symbol": "{tv_ticker}", "interval": "D",
        "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
        "enable_publishing": false, "allow_symbol_change": true, "container_id": "tradingview_chart"
      }});
      </script>
    </div>"""
    return html_code

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

# ==============================================================================
# === PRIMARY ANALYSIS CLASS ===================================================
# ==============================================================================
class StockAnalyzer:
# Add these new methods INSIDE the StockAnalyzer class

    def check_candlestick_pattern(self, five_min_data):
        """Identifies key 3-candle reversal patterns like Morning Star."""
        if len(five_min_data) < 3:
            return "Not enough data"
    
        last3 = five_min_data.tail(3)
        c1, c2, c3 = last3.iloc[0], last3.iloc[1], last3.iloc[2]
    
        # Morning Star (Bullish Reversal)
        is_morning_star = (c1['close'] < c1['open'] and  # 1st is bearish
                           abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) and # 2nd is small body
                           c3['close'] > c3['open'] and  # 3rd is bullish
                           c3['close'] > c1['open'])     # 3rd closes above 1st open
        if is_morning_star:
            return "Morning Star (Bullish)"
    
        # Evening Star (Bearish Reversal)
        is_evening_star = (c1['close'] > c1['open'] and  # 1st is bullish
                           abs(c2['close'] - c2['open']) < abs(c1['close'] - c1['open']) and # 2nd is small body
                           c3['close'] < c3['open'] and  # 3rd is bearish
                           c3['close'] < c1['open'])     # 3rd closes below 1st open
        if is_evening_star:
            return "Evening Star (Bearish)"
    
        return "No significant pattern"
    
    def run_confirmation_checklist(self, analysis_results):
        """
        Runs the full 5-point confirmation checklist to generate a final trade signal.
        """
        checklist = {
            "1. At Key S/R Level": "⚠️ PENDING",
            "2. Price Rejection": "⚠️ PENDING",
            "3. Chart Pattern Confirmed": "⚠️ PENDING",
            "4. Candlestick Signal": "⚠️ PENDING",
            "5. Indicator Alignment": "⚠️ PENDING",
            "FINAL_SIGNAL": "HOLD"
        }
    
        five_min_df = analysis_results['5m_data']
        if five_min_df.empty:
            return checklist
    
        resistance = analysis_results['resistance']
        support = analysis_results['support']
        latest_price = analysis_results['latest_price']
    
        # Check 1 & 2: At a key level with price rejection (simplified check)
        at_resistance = abs(latest_price - resistance) / resistance < 0.005 # Within 0.5% of resistance
        at_support = abs(latest_price - support) / support < 0.005 # Within 0.5% of support
        
        if at_support:
            checklist["1. At Key S/R Level"] = "✅ At Support"
            # Check for a bullish rejection candle (hammer-like)
            last_candle = five_min_df.iloc[-1]
            if (last_candle['low'] < support) and (last_candle['close'] > support):
                checklist["2. Price Rejection"] = "✅ Bullish Rejection"
        elif at_resistance:
            checklist["1. At Key S/R Level"] = "✅ At Resistance"
            # Check for a bearish rejection candle (shooting star-like)
            last_candle = five_min_df.iloc[-1]
            if (last_candle['high'] > resistance) and (last_candle['close'] < resistance):
                checklist["2. Price Rejection"] = "✅ Bearish Rejection"
        else:
            checklist["1. At Key S/R Level"] = "❌ Not at a key level"
            checklist["2. Price Rejection"] = "❌ No Rejection"
    
        # Check 3: Chart Pattern (using our existing function)
        pattern_status = self.detect_breakout_retest(five_min_df, resistance)
        if "✅ Retest Confirmed" in pattern_status:
            checklist["3. Chart Pattern Confirmed"] = "✅ Breakout/Retest"
        else:
            checklist["3. Chart Pattern Confirmed"] = "❌ No Confirmed Pattern"
            
        # Check 4: Candlestick Pattern
        candle_pattern = self.check_candlestick_pattern(five_min_df)
        checklist["4. Candlestick Signal"] = f"✅ {candle_pattern}" if "No significant" not in candle_pattern else "❌ No Signal"
    
        # Check 5: Indicator Alignment
        rsi = self.compute_rsi(five_min_df.rename(columns={'close': 'Close'}))
        five_min_df = self.compute_vwap(five_min_df)
        vwap = five_min_df['vwap'].iloc[-1]
        
        # Bullish Alignment Check
        if (checklist["1. At Key S/R Level"] == "✅ At Support" and 
            rsi < 70 and latest_price > vwap):
            checklist["5. Indicator Alignment"] = "✅ Bullish Alignment"
        # Bearish Alignment Check
        elif (checklist["1. At Key S/R Level"] == "✅ At Resistance" and
              rsi > 30 and latest_price < vwap):
            checklist["5. Indicator Alignment"] = "✅ Bearish Alignment"
        else:
            checklist["5. Indicator Alignment"] = "❌ No Alignment"
            
        # Final Signal Generation
        bullish_checks = sum(1 for v in checklist.values() if "✅ Bullish" in str(v) or ("✅ Breakout/Retest" in str(v)))
        if bullish_checks >= 3:
            checklist["FINAL_SIGNAL"] = "BUY"
        
        return checklist
    
    # Add this new method INSIDE the StockAnalyzer class
    
    def detect_breakout_retest(self, five_min_data, resistance):
        """
        Analyzes 5-minute data to detect a breakout, retest, and confirmation.
        Returns a status string indicating the current pattern phase.
        """
        if five_min_data.empty or resistance == 0:
            return "Not Analyzed"
    
        # We'll analyze the last 20 candles for this pattern
        recent_data = five_min_data.tail(20)
        
        breakout_candle_index = -1
        retest_candle_index = -1
    
        # 1. Detect the Breakout
        for i in range(1, len(recent_data)):
            prev_high = recent_data['high'].iloc[i-1]
            current_high = recent_data['high'].iloc[i]
            
            # A breakout happens when price crosses the resistance level
            if current_high > resistance and prev_high <= resistance:
                breakout_candle_index = i
                break # Found the most recent breakout
    
        if breakout_candle_index == -1:
            return "No Breakout Detected"
    
        # 2. Detect the Retest (after the breakout)
        for i in range(breakout_candle_index + 1, len(recent_data)):
            current_low = recent_data['low'].iloc[i]
            
            # A retest happens when the price comes back down to touch the old resistance
            if current_low <= resistance:
                retest_candle_index = i
                break # Found the retest
    
        if retest_candle_index == -1:
            return f"Breakout Occurred at {recent_data.index[breakout_candle_index].strftime('%H:%M')}. Awaiting Retest."
    
        # 3. Check for Confirmation (1-2 candles after retest)
        if retest_candle_index < len(recent_data) - 1:
            # Check the candle immediately following the retest
            confirmation_candle = recent_data.iloc[retest_candle_index + 1]
            
            # Confirmation is when the price bounces back up above the resistance level
            if confirmation_candle['close'] > resistance:
                return f"✅ Retest Confirmed at {recent_data.index[retest_candle_index + 1].strftime('%H:%M')}. Potential Entry."
    
        return f"Retest in Progress at {recent_data.index[retest_candle_index].strftime('%H:%M')}. Awaiting Confirmation."
    
# Add this new method INSIDE the StockAnalyzer class

    def run_multi_timeframe_analysis(self, ticker):
        """
        Performs a multi-timeframe analysis on a given ticker.
        1. Daily chart for trend.
        2. 15-min chart for support & resistance.
        3. Returns the 5-min data for execution analysis.
        """
        analysis_results = {}
    
        # 1. Daily Chart Analysis for Overall Trend
        daily_data = fetch_stock_data(ticker, period="6mo")
        if daily_data is not None and not daily_data.empty:
            # Use a 50-day moving average to determine the primary trend
            daily_data['SMA50'] = daily_data['Close'].rolling(window=50).mean()
            last_price = daily_data['Close'].iloc[-1]
            last_sma50 = daily_data['SMA50'].iloc[-1]
            
            if last_price > last_sma50:
                analysis_results['trend'] = "Uptrend"
            else:
                analysis_results['trend'] = "Downtrend"
        else:
            analysis_results['trend'] = "Unknown"
    
        # 2. 15-Minute Chart for Support & Resistance
        fifteen_min_data = fetch_intraday_data(ticker, interval="15minute")
        if fifteen_min_data is not None and not fifteen_min_data.empty:
            # Calculate support (day's low) and resistance (day's high)
            analysis_results['support'] = fifteen_min_data['low'].min()
            analysis_results['resistance'] = fifteen_min_data['high'].max()
        else:
            analysis_results['support'] = 0
            analysis_results['resistance'] = 0
    
        # 3. Get 5-Minute Data for Execution
        five_min_data = fetch_intraday_data(ticker, interval="5minute")
        if five_min_data is not None and not five_min_data.empty:
            analysis_results['5m_data'] = five_min_data
            analysis_results['latest_price'] = five_min_data['close'].iloc[-1]
        else:
            analysis_results['5m_data'] = pd.DataFrame() # Return empty DataFrame
            analysis_results['latest_price'] = daily_data['Close'].iloc[-1] if daily_data is not None else 0
    
        return analysis_results
    
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
        """Calculate Moving Averages including 25-day MA"""
        try:
            ma_20 = data['Close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else np.nan
            ma_25 = data['Close'].rolling(window=25).mean().iloc[-1] if len(data) >= 25 else np.nan
            ma_50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else np.nan
            ma_200 = data['Close'].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else np.nan

            return {
                'MA_20': ma_20 if not pd.isna(ma_20) else data['Close'].iloc[-1],
                'MA_25': ma_25 if not pd.isna(ma_25) else data['Close'].iloc[-1],
                'MA_50': ma_50 if not pd.isna(ma_50) else data['Close'].iloc[-1],
                'MA_200': ma_200 if not pd.isna(ma_200) else data['Close'].iloc[-1]
            }
        except:
            current_price = data['Close'].iloc[-1]
            return {
                'MA_20': current_price,
                'MA_25': current_price,
                'MA_50': current_price,
                'MA_200': current_price
            }

    def scrape_news_headlines(self, ticker_name, days=1): # Defaulting to 1 day for recent news
        """Scrape news headlines from NewsAPI.org"""
        try:
            # It's good practice to store API keys in environment variables or a config file
            # For this example, I'm using the one you provided.
            # Consider moving this to a more secure location like an environment variable.
            api_key = "e205d77d7bc14acc8744d3ea10568f50"

            # Use the full company name for the query if available, otherwise use the ticker
            # This might require a mapping from ticker to full company name
            # For now, we'll use the ticker_name directly, cleaned up.
            search_query = ticker_name.replace("^", "").replace(".NS", "").replace("NSE", "")

            # Construct the NewsAPI URL
            # sortBy=publishedAt will give the most recent articles first
            # You can adjust pageSize if you need more or fewer headlines
            url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=publishedAt&apiKey={api_key}&pageSize=5"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            news_data = response.json()
            headlines = []

            if news_data.get("status") == "ok" and news_data.get("articles"):
                for article in news_data["articles"]:
                    title = article.get("title")
                    if title and len(title) > 15: # Basic filter for relevance
                        headlines.append(title)
                    if len(headlines) >= 5: # Limit to 5 headlines
                        break

            return headlines if headlines else [f"Recent market news for {search_query} unavailable via NewsAPI"]

        except requests.exceptions.RequestException as e:
            st.warning(f"Could not fetch news from NewsAPI: {e}")
            return [f"Market news for {ticker_name} unavailable via NewsAPI"]
        except Exception as e:
            st.warning(f"An unexpected error occurred while fetching news: {e}")
            return [f"Market news for {ticker_name} unavailable via NewsAPI"]

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
                            # FinBERT format
                            sentiment_scores = {item['label']: item['score'] for item in result[0]}
                            if 'positive' in sentiment_scores:
                                sentiments.append(sentiment_scores['positive'] - sentiment_scores.get('negative', 0))
                            else:
                                sentiments.append(0)
                        else:
                            # Standard format
                            label = result[0]['label'].upper()
                            score = result[0]['score']
                            if label in ['POSITIVE', 'POS']:
                                sentiments.append(score)
                            elif label in ['NEGATIVE', 'NEG']:
                                sentiments.append(-score)
                            else:
                                sentiments.append(0)

            avg_sentiment = np.mean(sentiments) if sentiments else 0

            if avg_sentiment > 0.1:
                return "Positive", avg_sentiment
            elif avg_sentiment < -0.1:
                return "Negative", avg_sentiment
            else:
                return "Neutral", avg_sentiment

        except Exception as e:
            st.warning(f"Sentiment analysis error: {e}")
            return "Neutral", 0

    def generate_signal(self, rsi, macd, ma_data, sentiment_score):
        """Generate trading signal based on technical indicators with updated RSI levels (40:60)"""
        signal = "Hold"
        confidence = "Low"

        # Signal generation logic
        bullish_signals = 0
        bearish_signals = 0

        # RSI signals with new levels (40:60 instead of 30:70)
        if rsi < 40:
            bullish_signals += 1
        elif rsi > 60:
            bearish_signals += 1

        # MACD signals
        if macd['line'] > macd['signal'] and macd['histogram'] > 0:
            bullish_signals += 1
        elif macd['line'] < macd['signal'] and macd['histogram'] < 0:
            bearish_signals += 1

        # Moving Average signals (using 25-day MA as well)
        current_price = ma_data.get('current_price', ma_data['MA_20'])
        if (current_price > ma_data['MA_25'] > ma_data['MA_50'] and
                ma_data['MA_25'] > ma_data['MA_200']):
            bullish_signals += 1
        elif (current_price < ma_data['MA_25'] < ma_data['MA_50'] and
              ma_data['MA_25'] < ma_data['MA_200']):
            bearish_signals += 1

        # Sentiment boost
        if sentiment_score > 0.2:
            bullish_signals += 0.5
        elif sentiment_score < -0.2:
            bearish_signals += 0.5

        # Determine signal and confidence
        if bullish_signals >= 2:
            signal = "Buy"
            confidence = "High" if bullish_signals >= 2.5 else "Medium"
        elif bearish_signals >= 2:
            signal = "Sell"
            confidence = "High" if bearish_signals >= 2.5 else "Medium"
        else:
            signal = "Hold"
            confidence = "Medium" if abs(bullish_signals - bearish_signals) > 0.5 else "Low"

        return signal, confidence

    def get_ai_summary(self, ticker_name, rsi, macd, ma_data, signal, confidence, sentiment_label, headlines):
        """Generate AI summary using OpenRouter DeepSeek/Google Gemini Pro"""
        try:
            # Prepare the prompt
            prompt = f"""
            You are a professional stock market analyst. Analyze the following data for {ticker_name}:

            Technical Indicators:
            - RSI: {rsi:.2f} (A value below 40 suggests oversold, above 60 suggests overbought)
            - MACD Line: {macd['line']:.2f}
            - MACD Signal: {macd['signal']:.2f}
            - MACD Histogram: {macd['histogram']:.2f}
            - 20-day MA: {ma_data['MA_20']:.2f}
            - 25-day MA: {ma_data['MA_25']:.2f}
            - 50-day MA: {ma_data['MA_50']:.2f}
            - 200-day MA: {ma_data['MA_200']:.2f}

            Generated Signal: {signal}
            Confidence: {confidence}
            News Sentiment: {sentiment_label}

            Recent Headlines (Last 3 days):
            {chr(10).join(headlines[:3])}

            Provide a concise trading strategy summary in 2-3 sentences. Explain why this signal was generated, mention key risk factors, and suggest a potential entry/exit strategy if applicable. The tone should be professional and actionable for a retail trader.
            """

            # Google Gemini Pro API call
            if GOOGLE_API_KEY:
                model = genai.GenerativeModel('models/gemini-2.5-pro')
                response = model.generate_content(prompt)
                # It's good practice to check if the response has text.
                if response.text:
                    return response.text.strip()
                else:
                    # Handle cases where the model might return an empty response due to safety settings or other issues.
                    raise Exception("Model returned an empty response.")
            else:
                raise Exception("Google API key not configured")

        except Exception as e:
            st.warning(f"AI summary unavailable: {e}")
            # Fallback summary if the API fails
            rsi_status = "oversold" if rsi < 40 else "overbought" if rsi > 60 else "neutral"
            macd_trend = "bullish" if macd['line'] > macd['signal'] else "bearish"
            return (f"Technical analysis suggests a '{signal}' signal with {confidence} confidence. "
                    f"The RSI at {rsi:.1f} indicates {rsi_status} conditions, and the MACD shows {macd_trend} momentum. "
                    f"Always consider overall market sentiment and employ proper risk management.")

def setup_google_sheets():
    """Setup Google Sheets connection"""
    try:
        # Define the scope
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        # Load credentials from service account file
        creds = Credentials.from_service_account_file(
            "service_account.json",
            scopes=scope
        )

        client = gspread.authorize(creds)

        # Try to open existing sheet or create new one
        try:
            sheet = client.open("StockAnalyzerLog").sheet1
        except gspread.SpreadsheetNotFound:
            # Create new spreadsheet
            spreadsheet = client.create("StockAnalyzerLog")
            sheet = spreadsheet.sheet1

            # Add headers
            headers = [
                "Timestamp", "Ticker", "RSI", "MACD_Line", "MACD_Signal",
                "MA_20", "MA_25", "MA_50", "MA_200", "Signal", "Confidence",
                "AI_Summary", "News_Sentiment"
            ]
            sheet.append_row(headers)

        return sheet
    except Exception as e:
        st.error(f"Google Sheets setup failed: {e}")
        return None

def log_to_sheets(sheet, data):
    """Log analysis results to Google Sheets"""
    if sheet:
        try:
            sheet.append_row(data)
            return True
        except Exception as e:
            st.warning(f"Failed to log to sheets: {e}")
            return False
    return False

def send_email_alert(ticker_name, signal, confidence, ai_summary, sentiment_label):
    """Send email alert"""
    try:
        if not GMAIL_EMAIL or not GMAIL_APP_PASSWORD:
            st.warning("Email credentials not configured")
            return False

        msg = MIMEMultipart()
        msg['From'] = GMAIL_EMAIL
        msg['To'] = GMAIL_EMAIL  # Sending to self for demo
        msg['Subject'] = f"{confidence}-Confidence {signal.upper()} Alert: {ticker_name}"

        body = f"""
        Stock Analysis Alert

        Index/Stock: {ticker_name}
        Signal: {signal.upper()}
        Confidence: {confidence}

        AI Strategy Summary:
        {ai_summary}

        News Sentiment: {sentiment_label}

        Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        This is an automated alert from your Stock Analyzer AI Agent.
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()

        return True
    except Exception as e:
        st.warning(f"Email sending failed: {e}")
        return False

def create_indicator_charts(data, rsi, macd, ma_data):
    """Create indicator charts with improved MA display"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Price and Moving Averages
    axes[0].plot(data.index[-30:], data['Close'][-30:], label='Close Price', linewidth=2, color='black')

    # Calculate and plot moving averages for the chart
    if len(data) >= 20:
        ma_20_series = data['Close'].rolling(window=20).mean()
        axes[0].plot(data.index[-30:], ma_20_series[-30:], color='orange', linestyle='--', label='MA 20', alpha=0.8)

    if len(data) >= 25:
        ma_25_series = data['Close'].rolling(window=25).mean()
        axes[0].plot(data.index[-30:], ma_25_series[-30:], color='purple', linestyle='--', label='MA 25', alpha=0.8)

    if len(data) >= 50:
        ma_50_series = data['Close'].rolling(window=50).mean()
        axes[0].plot(data.index[-30:], ma_50_series[-30:], color='blue', linestyle='--', label='MA 50', alpha=0.8)

    if len(data) >= 200:
        ma_200_series = data['Close'].rolling(window=200).mean()
        axes[0].plot(data.index[-30:], ma_200_series[-30:], color='red', linestyle='--', label='MA 200', alpha=0.8)

    axes[0].set_title('Price & Moving Averages')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RSI with updated levels (40:60)
    rsi_series = []
    for i in range(len(data)):
        if i >= 13:  # Need at least 14 points for RSI
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
    axes[1].axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Neutral (50)')
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

    axes[2].plot(data.index[-30:], macd_line[-30:], label='MACD Line', color='blue')
    axes[2].plot(data.index[-30:], signal_line[-30:], label='Signal Line', color='red')
    axes[2].bar(data.index[-30:], histogram[-30:], label='Histogram', alpha=0.3, color='gray')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_title('MACD')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    st.set_page_config(
        page_title="📈 Stock Market Analyzer AI",
        page_icon="📈",
        layout="wide"
    )

    # Title and description
    st.title("📈 Stock Market Analyzer AI Agent")
    st.markdown("*Intelligent analysis for Any stock indices and individual stocks with AI-powered insights*")

    # --- NEW DYNAMIC STOCK SELECTION LOGIC ---
    
    # Initialize session state to hold our screened stocks
    if 'screened_stocks' not in st.session_state:
        st.session_state.screened_stocks = None
    
    st.sidebar.header("Pre-Market Analysis")
    if st.sidebar.button("Run Daily Stock Screener"):
        # When button is clicked, run the screener and store results
        st.session_state.screened_stocks = run_pre_market_screener()
    
    st.sidebar.header("Stock Selection")
    
    # Let the user choose between analyzing an Index or a stock from the screened list
    analysis_target = st.sidebar.radio(
        "What do you want to analyze?",
        ("Index", "Individual Stock from Screener")
    )
    
    ticker = None
    ticker_name = None
    
    if analysis_target == "Index":
        selected_category = st.sidebar.selectbox(
            "Choose Index:",
            list(STOCK_CATEGORIES.keys()),
            index=0
        )
        ticker = STOCK_CATEGORIES[selected_category]["ticker"]
        ticker_name = selected_category
    
    else: # This handles "Individual Stock from Screener"
        if st.session_state.screened_stocks:
            # If the screener has been run, populate the dropdown with its results
            # The screener returns tickers as keys, so we use them directly
            screened_tickers = list(st.session_state.screened_stocks.keys())
            
            selected_ticker = st.sidebar.selectbox(
                "Choose Screened Stock:",
                screened_tickers
            )
            ticker = selected_ticker
            # For display, remove the ".NS" from the ticker name
            ticker_name = selected_ticker.replace(".NS", "")
            
        else:
            # If screener hasn't been run, show a message
            st.sidebar.warning("Please run the daily stock screener first to see a list of stocks.")
    
    # --- END OF NEW LOGIC ---

    # Display current selection
    st.sidebar.success(f"Selected: {ticker_name}")
    st.sidebar.info(f"Ticker: {ticker}")

    # Analysis settings
    st.sidebar.header("⚙️ Settings")
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
        # --- SWING ANALYSIS MODE ---
        if analysis_mode == "Swing Analysis":
            with st.spinner(f"Running Swing Analysis for {ticker_name}..."):
                try:
                    # This is all your original, working code for swing analysis
                    data = st.session_state.analyzer.fetch_stock_data(ticker, period)
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
                    
                    # Get news and sentiment
                    headlines = st.session_state.analyzer.scrape_news_headlines(ticker_name, days=1)
                    sentiment_label, sentiment_score = st.session_state.analyzer.analyze_sentiment(headlines)
                    
                    # Generate signal
                    signal, confidence = st.session_state.analyzer.generate_signal(
                        rsi, macd, ma_data, sentiment_score
                    )
                    
                    # Get AI summary
                    ai_summary = st.session_state.analyzer.get_ai_summary(
                        ticker_name, rsi, macd, ma_data, signal, confidence,
                        sentiment_label, headlines
                    )
                    
                    # Display all results (your original UI code)
                    st.success("Swing Analysis Complete!")
                    
                    # (Your existing metrics, charts, and display code would go here)
                    # For brevity, I'll just put the chart call
                    st.subheader("📈 Technical Analysis Charts")
                    fig = create_indicator_charts(data, rsi, macd, ma_data)
                    st.pyplot(fig)
    
                except Exception as e:
                    st.error(f"❌ Swing Analysis failed: {e}")
                    st.exception(e)
    
        # --- INTRADAY ANALYSIS MODE ---
        elif analysis_mode == "Intraday Analysis":
            with st.spinner(f"Running Multi-Timeframe Analysis for {ticker_name}..."):
                try:
                    # Run our new comprehensive analysis function
                    analysis = st.session_state.analyzer.run_multi_timeframe_analysis(ticker)
    
                    if analysis:
                        st.success("Multi-Timeframe Analysis Complete!")
    
                        # Display the high-level findings
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Selected Stock", ticker_name)
                        col2.metric("Overall Trend (Daily)", analysis['trend'])
                        col3.metric("Support (15-min)", f"₹{analysis['support']:.2f}")
                        col4.metric("Resistance (15-min)", f"₹{analysis['resistance']:.2f}")

                        # --- START NEW CODE ---
                        # Run Pattern Recognition
                        resistance_level = analysis['resistance']
                        pattern_status = st.session_state.analyzer.detect_breakout_retest(five_min_df, resistance_level)
                        
                        st.subheader("📈 Pattern Recognition Status")
                        st.info(pattern_status)
                        # --- END NEW CODE ---

    # --- START NEW CODE ---
                        # Run the full confirmation checklist
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
                        elif final_signal == "SELL":
                            st.error(f"**{final_signal}** - High probability bearish setup detected.")
                        else:
                            st.warning(f"**{final_signal}** - Conditions not met for a high-probability trade.")
                        # --- END NEW CODE ---

                        # Now, we use the 5-minute data for detailed indicator calculation
                        five_min_df = analysis['5m_data']
                        if not five_min_df.empty:
                            # Calculate Intraday Indicators like VWAP on the 5-min data
                            five_min_df = st.session_state.analyzer.compute_vwap(five_min_df)
                            
                            # Display the Intraday Chart
                            st.subheader("5-Minute Execution Chart")
                            intraday_fig = create_intraday_chart(five_min_df, ticker_name)
                            st.plotly_chart(intraday_fig, use_container_width=True)
                        else:
                            st.warning("Could not retrieve 5-minute data for detailed analysis.")
                except Exception as e:
                    st.error(f"❌ Intraday Analysis failed: {e}")
                    st.exception(e)
    
    elif analyze_button and not ticker:
        st.sidebar.error("Please select a stock or index to analyze.")

    # Information panel
    with st.expander("ℹ️ About This Analyzer"):
        st.write("""
        **Enhanced Features:**
        - ✅ Updated RSI levels (40:60 instead of 30:70)
        - ✅ Added 25-day Moving Average
        - ✅ Fixed Moving Average calculations
        - ✅ 3-day news headlines
        - ✅ Complete Nifty stock coverage
        - ✅ Individual stock analysis
        - ✅ Enhanced volume analysis
        - ✅ Support/Resistance levels

        **Technical Indicators:**
        - **RSI (40:60)**: Oversold below 40, Overbought above 60
        - **MACD**: Momentum indicator with signal line crossovers
        - **Moving Averages**: 20, 25, 50, and 200-day periods
        - **Volume**: Compared to 20-day average

        **Signal Generation:**
        Combines technical indicators with news sentiment for comprehensive analysis.
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "*Enhanced Stock Analyzer powered by yfinance, FinBERT, OpenRouter DeepSeek/Google Gemini Pro, and Google Sheets API*"
    )
    st.markdown("*Updated with RSI (40:60), 25-day MA, 3-day news, and complete Nifty stock coverage*")

if __name__ == "__main__":
    main()
