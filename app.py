# --- START OF FILE app.py (Final Merged Version) ---

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import openai
import gspread
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import time
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# --- Static Data & Mappings ---
STOCK_CATEGORIES = {
    "NIFTY 50 Index": {
        "ticker": "^NSEI",
        "individual_stocks": {
            "Reliance Industries": "RELIANCE.NS", "Tata Consultancy Services": "TCS.NS", "HDFC Bank": "HDFCBANK.NS", "Infosys": "INFY.NS",
            "ICICI Bank": "ICICIBANK.NS", "Hindustan Unilever": "HINDUNILVR.NS", "State Bank of India": "SBIN.NS", "ITC": "ITC.NS",
            "Bharti Airtel": "BHARTIARTL.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS", "Larsen & Toubro": "LT.NS", "Axis Bank": "AXISBANK.NS",
            "Maruti Suzuki": "MARUTI.NS", "Asian Paints": "ASIANPAINT.NS", "Nestle India": "NESTLEIND.NS", "HCL Technologies": "HCLTECH.NS",
            "Bajaj Finance": "BAJFINANCE.NS", "Wipro": "WIPRO.NS", "Ultratech Cement": "ULTRACEMCO.NS", "Titan Company": "TITAN.NS",
            "Tata Motors": "TATAMOTORS.NS", "Sun Pharmaceutical": "SUNPHARMA.NS", "NTPC": "NTPC.NS", "Power Grid Corporation": "POWERGRID.NS",
            "Bajaj Finserv": "BAJAJFINSV.NS", "Dr. Reddy's Laboratories": "DRREDDY.NS", "Tech Mahindra": "TECHM.NS", "Oil & Natural Gas Corp": "ONGC.NS",
            "Tata Steel": "TATASTEEL.NS", "IndusInd Bank": "INDUSINDBK.NS", "Mahindra & Mahindra": "M&M.NS", "Adani Enterprises": "ADANIENT.NS",
            "Coal India": "COALINDIA.NS", "JSW Steel": "JSWSTEEL.NS", "Cipla": "CIPLA.NS", "Grasim Industries": "GRASIM.NS",
            "Hero MotoCorp": "HEROMOTOCO.NS", "UPL": "UPL.NS", "Britannia Industries": "BRITANNIA.NS", "Eicher Motors": "EICHERMOT.NS",
            "HDFC Life Insurance": "HDFCLIFE.NS", "SBI Life Insurance": "SBILIFE.NS", "Divis Laboratories": "DIVISLAB.NS",
            "Hindalco Industries": "HINDALCO.NS", "Bajaj Auto": "BAJAJ-AUTO.NS", "Shree Cement": "SHREECEM.NS", "Apollo Hospitals": "APOLLOHOSP.NS",
            "HDFC Asset Management": "HDFCAMC.NS", "Adani Ports": "ADANIPORTS.NS", "BPCL": "BPCL.NS"
        }
    },
    "BANK NIFTY Index": {
        "ticker": "^NSEBANK",
        "individual_stocks": {
            "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "State Bank of India": "SBIN.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS",
            "Axis Bank": "AXISBANK.NS", "IndusInd Bank": "INDUSINDBK.NS", "Punjab National Bank": "PNB.NS", "Bank of Baroda": "BANKBARODA.NS",
            "Federal Bank": "FEDERALBNK.NS", "IDFC First Bank": "IDFCFIRSTB.NS", "AU Small Finance Bank": "AUBANK.NS", "Bandhan Bank": "BANDHANBNK.NS"
        }
    },
    "NIFTY AUTO Index": { "ticker": "^CNXAUTO", "individual_stocks": { "Maruti Suzuki": "MARUTI.NS", "Tata Motors": "TATAMOTORS.NS", "Mahindra & Mahindra": "M&M.NS", "Hero MotoCorp": "HEROMOTOCO.NS", "Eicher Motors": "EICHERMOT.NS", "Bajaj Auto": "BAJAJ-AUTO.NS", "TVS Motor Company": "TVSMOTOR.NS", "Ashok Leyland": "ASHOKLEY.NS", "Force Motors": "FORCEMOT.NS", "MRF": "MRF.NS", "Balkrishna Industries": "BALKRISIND.NS", "Ceat": "CEATLTD.NS", "Apollo Tyres": "APOLLOTYRE.NS", "JK Tyre": "JKTYRE.NS", "Bharat Forge": "BHARATFORG.NS" } },
    "NIFTY PHARMA Index": { "ticker": "^CNXPHARMA", "individual_stocks": { "Sun Pharmaceutical": "SUNPHARMA.NS", "Dr. Reddy's Laboratories": "DRREDDY.NS", "Cipla": "CIPLA.NS", "Divis Laboratories": "DIVISLAB.NS", "Lupin": "LUPIN.NS", "Aurobindo Pharma": "AUROPHARMA.NS", "Biocon": "BIOCON.NS", "Cadila Healthcare": "ZYDUSLIFE.NS", "Alkem Laboratories": "ALKEM.NS", "Glenmark Pharmaceuticals": "GLENMARK.NS", "Torrent Pharmaceuticals": "TORNTPHARM.NS", "Ipca Laboratories": "IPCALAB.NS", "Abbott India": "ABBOTINDIA.NS", "Pfizer": "PFIZER.NS", "GSK Pharma": "GSK.NS" } },
    "NIFTY METAL Index": { "ticker": "^CNXMETAL", "individual_stocks": { "Tata Steel": "TATASTEEL.NS", "JSW Steel": "JSWSTEEL.NS", "Hindalco Industries": "HINDALCO.NS", "Vedanta": "VEDL.NS", "Coal India": "COALINDIA.NS", "Steel Authority of India": "SAIL.NS", "NMDC": "NMDC.NS", "Jindal Steel & Power": "JINDALSTEL.NS", "National Aluminium Company": "NATIONALUM.NS", "APL Apollo Tubes": "APLAPOLLO.NS", "Hindustan Zinc": "HINDZINC.NS", "Ratnamani Metals": "RATNAMANI.NS", "Welspun Corp": "WELCORP.NS", "MOIL": "MOIL.NS" } },
    "NIFTY IT Index": { "ticker": "^CNXIT", "individual_stocks": { "Tata Consultancy Services": "TCS.NS", "Infosys": "INFY.NS", "HCL Technologies": "HCLTECH.NS", "Wipro": "WIPRO.NS", "Tech Mahindra": "TECHM.NS", "LTI Mindtree": "LTIM.NS", "Mphasis": "MPHASIS.NS", "Persistent Systems": "PERSISTENT.NS", "Coforge": "COFORGE.NS", "L&T Technology Services": "LTTS.NS" } },
    "NIFTY FMCG Index": { "ticker": "^CNXFMCG", "individual_stocks": { "Hindustan Unilever": "HINDUNILVR.NS", "ITC": "ITC.NS", "Nestle India": "NESTLEIND.NS", "Britannia Industries": "BRITANNIA.NS", "Dabur India": "DABUR.NS", "Marico": "MARICO.NS", "Godrej Consumer Products": "GODREJCP.NS", "Colgate-Palmolive": "COLPAL.NS", "United Spirits": "UBL.NS", "Tata Consumer Products": "TATACONSUM.NS", "Emami": "EMAMILTD.NS", "P&G Hygiene": "PGHH.NS", "VBL": "VBL.NS" } }
}

# ==============================================================================
# === GLOBAL HELPER FUNCTIONS (Defined before they are called) =================
# ==============================================================================

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
            sheet = client.open("EliteTradingAnalyzerLog").sheet1
        except gspread.SpreadsheetNotFound:
            sheet = client.create("EliteTradingAnalyzerLog").sheet1
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
    def __init__(self):
        self.sentiment_analyzer = None
        self.setup_sentiment_analyzer()
        
    def setup_sentiment_analyzer(self):
        try: self.sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True)
        except: self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def compute_rsi(self, data, window=14):
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except: return pd.Series([50.0] * len(data), index=data.index)

    def compute_macd(self, data):
        """Computes MACD values for the entire dataset."""
        try:
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            return {'line': macd_line, 'signal': signal_line, 'histogram': histogram}
        except: return None

    def compute_moving_averages(self, data):
        """Calculate EXPONENTIAL Moving Averages (EMA) for the entire dataset."""
        try:
            return {
                'EMA_20': data['Close'].ewm(span=20, adjust=False).mean(),
                'EMA_50': data['Close'].ewm(span=50, adjust=False).mean(),
                'EMA_200': data['Close'].ewm(span=200, adjust=False).mean()
            }
        except: return None

    def generate_signal(self, rsi_series, macd_data, ema_data, price_series):
        """Generates a series of signals based on the original logic."""
        bullish = pd.Series(0, index=price_series.index)
        bearish = pd.Series(0, index=price_series.index)
        
        bullish[rsi_series < 40] += 1
        bearish[rsi_series > 60] += 1
        
        bullish[macd_data['histogram'] > 0] += 1
        bearish[macd_data['histogram'] < 0] += 1
        
        bullish[(price_series > ema_data['EMA_20']) & (ema_data['EMA_20'] > ema_data['EMA_50'])] += 1
        bearish[(price_series < ema_data['EMA_20']) & (ema_data['EMA_20'] < ema_data['EMA_50'])] += 1

        signals = pd.Series("Hold", index=price_series.index)
        signals[bullish >= 2] = "Buy"
        signals[bearish >= 2] = "Sell"
        
        confidence = pd.Series("Low", index=price_series.index)
        confidence[abs(bullish - bearish) > 0.5] = "Medium"
        confidence[abs(bullish - bearish) >= 1.5] = "High"
        
        return signals.iloc[-1], confidence.iloc[-1]

    def scrape_news_headlines(self, company_name: str, ticker: str):
        if not NEWSAPI_KEY: return ["News analysis skipped: API key not configured."]
        try:
            cleaned_name = company_name.replace(" Ltd.", "").replace(" Inc.", "").replace(" Limited", "")
            search_query = f'"{cleaned_name}" OR "{ticker}"'
            url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=relevancy&apiKey={NEWSAPI_KEY}&pageSize=10"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()
            news_data = response.json()
            headlines = [article.get("title") for article in news_data.get("articles", []) if article.get("title") and article.get("title") != "[Removed]"]
            return headlines[:5] if headlines else [f"No recent news found for {company_name}"]
        except Exception as e:
            return [f"News for {company_name} unavailable: {e}"]

    def analyze_sentiment(self, headlines):
        if not headlines or all("unavailable" in h.lower() for h in headlines): return "Neutral", 0
        try:
            sentiments = []
            for headline in headlines:
                result = self.sentiment_analyzer(headline)
                if isinstance(result[0], list):
                    scores = {item['label']: item['score'] for item in result[0]}
                    sentiments.append(scores.get('positive', 0) - scores.get('negative', 0))
                else:
                    score = result[0]['score'] * (1 if result[0]['label'].upper() in ['POSITIVE', 'POS'] else -1)
                    sentiments.append(score)
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            if avg_sentiment > 0.1: return "Positive", avg_sentiment
            elif avg_sentiment < -0.1: return "Negative", avg_sentiment
            else: return "Neutral", avg_sentiment
        except: return "Neutral", 0
    
    def get_ai_summary(self, ticker_name, rsi, macd, signal, confidence, sentiment_label, headlines):
        if not OPENROUTER_API_KEY: return "AI summary unavailable: OpenRouter API key not configured."
        prompt = f"""
        As a trading analyst, summarize the state of {ticker_name}.
        - Technical Signal: {signal} (Confidence: {confidence})
        - Reasoning: Based on RSI({rsi:.2f}), MACD Histogram({macd['histogram']:.2f}), and EMA crossovers.
        - News Sentiment: {sentiment_label}
        Provide a 3-sentence, actionable summary based on these combined factors. Explain the 'why' behind the current signal.
        """
        try:
            client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
            response = client.chat.completions.create(model="deepseek/deepseek-chat", messages=[{"role": "user", "content": prompt}], max_tokens=250, temperature=0.7)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"AI summary unavailable: {e}"

# ==============================================================================
# === MAIN APPLICATION LOGIC ===================================================
# ==============================================================================

def main():
    st.set_page_config(page_title="📈 Elite Trading Analyzer", page_icon="📈", layout="wide")
    st.title("📈 Elite Trading Analyzer")
    st.markdown("*A focused trading tool using an EMA/RSI strategy with advanced charting.*")

    st.sidebar.header("🔎 Asset Selection")
    asset_type = st.sidebar.selectbox("1. Select Asset Class", ["Equities (Stocks)", "Cryptocurrencies", "ETFs", "Indices", "Currencies / Forex", "Commodities", "Browse Curated Indian Lists"])
    
    ticker, ticker_name = None, None
    if asset_type == "Browse Curated Indian Lists":
        st.sidebar.markdown("---")
        selected_category = st.sidebar.selectbox("Choose Category:", list(STOCK_CATEGORIES.keys()))
        stock_type = st.sidebar.radio("Analysis Type:", ["Index", "Individual Stock"])
        if stock_type == "Index":
            ticker, ticker_name = STOCK_CATEGORIES[selected_category]["ticker"], selected_category
        else:
            stocks = STOCK_CATEGORIES[selected_category]["individual_stocks"]
            selected_stock = st.sidebar.selectbox("Choose Stock:", list(stocks.keys()))
            ticker, ticker_name = stocks[selected_stock], selected_stock
    else:
        search_query = st.sidebar.text_input(f"2. Search for {asset_type}", key="search_query")
        if 'search_results' not in st.session_state: st.session_state.search_results = {}
        if st.sidebar.button("Search", key="search_button"):
            with st.spinner(f"Searching for {search_query}..."):
                st.session_state.search_results = search_for_ticker(search_query, asset_type) if search_query else {}
        if st.session_state.search_results:
            selected_display_name = st.sidebar.selectbox("3. Select from results:", options=list(st.session_state.search_results.keys()))
            if selected_display_name:
                ticker = st.session_state.search_results[selected_display_name]
                ticker_name = selected_display_name.split('(')[0].strip()

    if not ticker:
        st.info("Please select or search for an asset in the sidebar to begin analysis.")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.success(f"**Selected:** {ticker_name}")
    st.sidebar.info(f"**Ticker:** `{ticker}`")
    st.sidebar.header("⚙️ Settings")
    period_options = ['5d', '1mo', '3mo', '6mo', 'ytd', '1y', '2y', '5y', 'max']
    period = st.sidebar.selectbox("Data Period:", period_options, index=5)

    if 'analyzer' not in st.session_state: st.session_state.analyzer = StockAnalyzer()
    if 'sheets_client' not in st.session_state: st.session_state.sheets_client = setup_google_sheets()

    if st.button("📈 Run Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Running analysis for {ticker_name} ({ticker})..."):
            try:
                data = fetch_stock_data(ticker, period)
                if data is None or data.empty: st.error("Could not fetch data for this asset. Please try another."); st.stop()
                
                official_company_name = yf.Ticker(ticker).info.get('longName', ticker_name)
                st.header(f"Analysis for: {official_company_name} ({ticker})")

                # --- Run Vectorized Analysis on the whole dataset ---
                rsi_series = st.session_state.analyzer.compute_rsi(data)
                macd_data = st.session_state.analyzer.compute_macd(data)
                ema_data = st.session_state.analyzer.compute_moving_averages(data)

                headlines = st.session_state.analyzer.scrape_news_headlines(official_company_name, ticker)
                sentiment_label, sentiment_score = st.session_state.analyzer.analyze_sentiment(headlines)
                
                # Get the latest signal by passing the entire data series to the signal generator
                signal, confidence = st.session_state.analyzer.generate_signal(rsi_series, macd_data, ema_data, data['Close'])
                
                ai_summary = st.session_state.analyzer.get_ai_summary(
                    official_company_name, rsi_series.iloc[-1], 
                    {k: v.iloc[-1] for k, v in macd_data.items()}, 
                    signal, confidence, sentiment_label, headlines
                )
                
                # --- Display UI ---
                st.subheader("📊 Key Metrics")
                price = data['Close'].iloc[-1]
                delta = price - data['Close'].iloc[-2] if len(data) > 1 else 0
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"{price:,.2f}", f"{delta:,.2f} ({delta/price*100:.2f}%)" if price > 0 else "N/A")
                col2.metric("Trading Signal", f"{signal}", f"Confidence: {confidence}")
                col3.metric("RSI (14)", f"{rsi_series.iloc[-1]:.2f}", "Oversold" if rsi_series.iloc[-1] < 30 else "Overbought" if rsi_series.iloc[-1] > 70 else "Normal")
                col4.metric("News Sentiment", f"{sentiment_label}", f"Score: {sentiment_score:.2f}")

                st.subheader("📈 Custom Analysis Chart")
                st.plotly_chart(create_plotly_charts(data, official_company_name), use_container_width=True)

                st.subheader("💡 Indicator Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Exponential Moving Averages:**")
                    st.write(f"• 20-period EMA: {ema_data['EMA_20'].iloc[-1]:.2f}")
                    st.write(f"• 50-period EMA: {ema_data['EMA_50'].iloc[-1]:.2f}")
                    st.write(f"• 200-period EMA: {ema_data['EMA_200'].iloc[-1]:.2f}")
                with col2:
                    st.write("**MACD:**")
                    st.write(f"• MACD Line: {macd_data['line'].iloc[-1]:.4f}")
                    st.write(f"• Signal Line: {macd_data['signal'].iloc[-1]:.4f}")
                    st.write(f"• Histogram: {macd_data['histogram'].iloc[-1]:.4f}")

                st.subheader("🤖 AI Strategy Summary"); st.info(ai_summary)
                
                st.subheader("📰 Recent News Headlines")
                if headlines and "unavailable" not in headlines[0].lower():
                    for headline in headlines: st.markdown(f"- {headline}")
                else:
                    st.markdown(f"- {headlines[0]}")

                st.subheader("🚀 Live Professional Chart (for Discretionary Analysis)"); st.info("Use this chart for your own drawing and advanced indicator analysis.")
                components.html(embed_tradingview_widget(ticker), height=520)
                
                # --- Action Buttons ---
                st.subheader("Actions")
                
                
                log_data = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, signal, confidence, f"{rsi_series.iloc[-1]:.2f}", sentiment_label, ai_summary[:500]]
                if log_to_sheets(st.session_state.sheets_client, log_data): st.success("✅ Results logged to Google Sheets.")
                
                report_data = {
                    "Asset": official_company_name, "Ticker": ticker, "Analysis_Date": datetime.now(),
                    "Current_Price": price, "Signal": signal, "Confidence": confidence,
                    "RSI": rsi_series.iloc[-1], "MACD": {k: v.iloc[-1] for k, v in macd_data.items()},
                    "EMAs": {k: v.iloc[-1] for k, v in ema_data.items()},
                    "Sentiment": {"label": sentiment_label, "score": sentiment_score}, "AI_Summary": ai_summary
                }
                st.download_button("📥 Download Full Analysis (JSON)", data=json.dumps(report_data, indent=4, default=str),file_name=f"{ticker}_analysis.json", mime="application/json", use_container_width=True)

            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()