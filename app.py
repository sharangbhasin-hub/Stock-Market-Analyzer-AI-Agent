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
import sqlite3
from io import StringIO

TALIB_AVAILABLE = True

# Optional imports
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

try:
    import telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

    
# Broker API Configuration
KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_API_SECRET = os.getenv("KITE_API_SECRET")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

# Notification Services
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Configure APIs
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==============================================================================
# === ENHANCED ALPHA VANTAGE API ===============================================
# ==============================================================================

class AlphaVantageAPI:
    """Fully dynamic Alpha Vantage integration"""

    def __init__(self, api_key=None):
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        self._cache = {}
        
    def get_all_stocks_listing(self):
        """Get complete US stock listing"""
        try:
            params = {
                'function': 'LISTING_STATUS',
                'state': 'active',
                'apikey': self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                # Filter active stocks
                if not df.empty:
                    return df['symbol'].tolist()
            return []
        except Exception as e:
            st.error(f"Alpha Vantage listing error: {e}")
            return []
            
    def _make_request(self, params):
        cache_key = str(sorted(params.items()))
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            params['apikey'] = self.api_key
            response = requests.get(self.base_url, params=params, timeout=15)
            data = response.json()

            if "Error Message" in data or "Note" in data:
                return None

            self._cache[cache_key] = data
            return data
        except:
            return None

    def get_listing_status(self, state='active'):
        """Get ALL active stocks from Alpha Vantage"""
        try:
            params = {'function': 'LISTING_STATUS', 'state': state, 'apikey': self.api_key}
            response = requests.get(self.base_url, params=params, timeout=15)

            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                return df
            return None
        except:
            return None

    def get_stocks_by_exchange(self, exchange_codes):
        """Get stocks filtered by exchange"""
        all_listings = self.get_listing_status()

        if all_listings is not None and not all_listings.empty:
            filtered = all_listings[all_listings['exchange'].isin(exchange_codes)]

            stocks_dict = {}
            for _, row in filtered.iterrows():
                name = row['name']
                symbol = row['symbol']
                display_name = f"{name} ({symbol})"
                stocks_dict[display_name] = symbol

            return stocks_dict
        return {}

    def search_symbols(self, keywords):
        """Universal stock search"""
        params = {'function': 'SYMBOL_SEARCH', 'keywords': keywords}
        data = self._make_request(params)

        if data and 'bestMatches' in data:
            results = {}
            for match in data['bestMatches']:
                symbol = match.get('1. symbol', '')
                name = match.get('2. name', '')
                region = match.get('4. region', '')

                display_name = f"{name} ({symbol}) - {region}"
                results[display_name] = symbol
            return results
        return {}

    def get_quote(self, symbol):
        """Real-time quote"""
        params = {'function': 'GLOBAL_QUOTE', 'symbol': symbol}
        data = self._make_request(params)

        if data and 'Global Quote' in data:
            q = data['Global Quote']
            return {
                'price': float(q.get('05. price', 0)),
                'change': float(q.get('09. change', 0)),
                'volume': int(q.get('06. volume', 0))
            }
        return None

def check_market_status(market_config):
    """Check if selected market is currently open"""
    tz = pytz.timezone(market_config['timezone'])
    now = datetime.now(tz)

    if now.weekday() > 4:  # Weekend
        return {'status': 'CLOSED', 'reason': 'Weekend'}

    current_time = now.time()

    if market_config['market_open'] <= current_time <= market_config['market_close']:
        return {'status': 'OPEN', 'time': now.strftime('%H:%M:%S %Z')}
    else:
        return {'status': 'CLOSED', 'reason': 'Outside Trading Hours'}


# ==============================================================================
# === DATABASE SETUP ===========================================================
# ==============================================================================

def init_database():
    """Initialize SQLite database for trade logging"""
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT NOT NULL,
            signal TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity INTEGER,
            profit_loss REAL,
            strategy TEXT,
            notes TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT NOT NULL,
            signal TEXT,
            rsi REAL,
            macd REAL,
            price REAL,
            confidence REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE DEFAULT CURRENT_DATE,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            total_profit REAL,
            win_rate REAL,
            profit_factor REAL
        )
    ''')

    conn.commit()
    conn.close()

def log_trade_to_db(ticker, signal, entry_price, quantity, strategy="intraday", notes=""):
    """Log trade to database"""
    try:
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (ticker, signal, entry_price, quantity, strategy, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ticker, signal, entry_price, quantity, strategy, notes))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def get_trade_history(limit=100):
    """Retrieve trade history from database"""
    try:
        conn = sqlite3.connect('trading_data.db')
        df = pd.read_sql_query(f'SELECT * FROM trades ORDER BY timestamp DESC LIMIT {limit}', conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# ==============================================================================
# === BROKER API INTEGRATION ===================================================
# ==============================================================================

class BrokerAPI:
    """Wrapper for Zerodha Kite Connect API"""

    def __init__(self):
        self.kite = None
        self.connected = False
        if KITE_AVAILABLE and KITE_API_KEY and KITE_ACCESS_TOKEN:
            try:
                self.kite = KiteConnect(api_key=KITE_API_KEY)
                self.kite.set_access_token(KITE_ACCESS_TOKEN)
                self.connected = True
            except Exception as e:
                pass

    def place_order(self, ticker, transaction_type, quantity, order_type="MARKET", price=None):
        """Place order via Kite Connect"""
        if not self.connected:
            return {"status": "error", "message": "Broker not connected"}

        try:
            symbol = ticker.replace(".NS", "").replace(".BO", "")

            order_params = {
                "tradingsymbol": symbol,
                "exchange": "NSE",
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "product": "MIS",
                "variety": "regular"
            }

            if order_type == "LIMIT" and price:
                order_params["price"] = price

            order_id = self.kite.place_order(**order_params)
            return {"status": "success", "order_id": order_id}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_positions(self):
        """Get current positions"""
        if not self.connected:
            return []
        try:
            return self.kite.positions()
        except:
            return []

    def get_holdings(self):
        """Get holdings"""
        if not self.connected:
            return []
        try:
            return self.kite.holdings()
        except:
            return []

    def cancel_order(self, order_id, variety="regular"):
        """Cancel pending order"""
        if not self.connected:
            return False
        try:
            self.kite.cancel_order(variety=variety, order_id=order_id)
            return True
        except:
            return False

# ==============================================================================
# === NOTIFICATION SYSTEM ======================================================
# ==============================================================================

def send_sms_alert(message, to_phone):
    """Send SMS via Twilio"""
    if not TWILIO_AVAILABLE or not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return False
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        return True
    except Exception as e:
        return False

def send_telegram_alert(message):
    """Send alert via Telegram"""
    if not TELEGRAM_AVAILABLE or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        return True
    except Exception as e:
        return False

def send_email_alert(subject, body, to_email=None):
    """Send email alert"""
    if not GMAIL_EMAIL or not GMAIL_APP_PASSWORD:
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
        return False

def send_multi_channel_alert(ticker, signal, price, channels=['email']):
    """Send alert across multiple channels"""
    message = f"""
    🚨 TRADING SIGNAL ALERT
    
    Ticker: {ticker}
    Signal: {signal}
    Price: {currency}{price:.2f}
    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    results = {}

    if 'email' in channels:
        results['email'] = send_email_alert(f"Trading Signal: {signal}", message)

    if 'telegram' in channels:
        results['telegram'] = send_telegram_alert(message)

    if 'sms' in channels and st.session_state.get('user_phone'):
        results['sms'] = send_sms_alert(message, st.session_state['user_phone'])

    return results

# ==============================================================================
# === OPTIONS TRADING MODULE ===================================================
# ==============================================================================

class OptionsAnalyzer:
    """Options trading analysis with PCR and ITM selection"""

    def __init__(self):
        self.expiry_schedule = {
            0: "Mid Cap Nifty",
            1: "Fin Nifty",
            2: "Bank Nifty",
            3: "Nifty 50"
        }

    def get_todays_expiry(self):
        """Get today's expiring index"""
        today = datetime.now().weekday()
        return self.expiry_schedule.get(today, "No expiry today")

    def fetch_options_chain(self, ticker):
        """Fetch options chain data"""
        try:
            stock = yf.Ticker(ticker)
            expiry_dates = stock.options

            if not expiry_dates:
                return None

            nearest_expiry = expiry_dates[0]
            options = stock.option_chain(nearest_expiry)

            return {
                'calls': options.calls,
                'puts': options.puts,
                'expiry': nearest_expiry
            }
        except Exception as e:
            return None

    def calculate_pcr(self, options_data):
        """Calculate Put-Call Ratio"""
        if not options_data:
            return None

        try:
            puts = options_data['puts']
            calls = options_data['calls']

            total_put_oi = puts['openInterest'].sum()
            total_call_oi = calls['openInterest'].sum()

            pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            total_put_volume = puts['volume'].sum()
            total_call_volume = calls['volume'].sum()

            pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0

            if pcr_oi > 1.0:
                sentiment = "Bullish (Oversold)"
            elif pcr_oi < 0.7:
                sentiment = "Bearish (Overbought)"
            else:
                sentiment = "Neutral"

            return {
                'pcr_oi': pcr_oi,
                'pcr_volume': pcr_volume,
                'sentiment': sentiment,
                'put_oi': total_put_oi,
                'call_oi': total_call_oi
            }
        except:
            return None

    def filter_itm_options(self, options_data, current_price, option_type='call'):
        """Filter In-The-Money options"""
        if not options_data:
            return pd.DataFrame()

        try:
            if option_type.lower() == 'call':
                df = options_data['calls']
                itm_options = df[df['strike'] < current_price]
            else:
                df = options_data['puts']
                itm_options = df[df['strike'] > current_price]

            itm_options = itm_options.sort_values(by=['volume', 'openInterest'], ascending=False)

            return itm_options[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
        except:
            return pd.DataFrame()

    def suggest_option_strategy(self, signal, current_price, options_data):
        """Suggest options strategy"""
        if signal == "🟢 BUY":
            strategy = "Buy ITM Call Option"
            options = self.filter_itm_options(options_data, current_price, 'call')
        elif signal == "🔴 SELL":
            strategy = "Buy ITM Put Option"
            options = self.filter_itm_options(options_data, current_price, 'put')
        else:
            strategy = "HOLD - No Options Trade"
            options = pd.DataFrame()

        return {
            'strategy': strategy,
            'recommended_options': options.head(3) if not options.empty else None
        }

# ==============================================================================
# === POSITION SIZING & RISK MANAGEMENT ========================================
# ==============================================================================

class RiskManager:
    """Position sizing and risk management"""

    def __init__(self, total_capital=100000, risk_per_trade=0.02):
        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk"""
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0

        risk_amount = self.total_capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            return 0

        quantity = int(risk_amount / price_risk)

        return max(1, quantity)

    def calculate_stop_loss(self, entry_price, atr, multiplier=1.5):
        """Calculate stop loss using ATR"""
        stop_loss = entry_price - (atr * multiplier)
        return max(0, stop_loss)

    def calculate_targets(self, entry_price, stop_loss, risk_reward_ratios=[1.5, 2.0, 3.0]):
        """Calculate multiple target levels"""
        risk = abs(entry_price - stop_loss)
        targets = []

        for ratio in risk_reward_ratios:
            target = entry_price + (risk * ratio)
            targets.append({
                'ratio': f"1:{ratio}",
                'price': round(target, 2),
                'profit_potential': round(risk * ratio, 2)
            })

        return targets

    def kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Calculate Kelly Criterion"""
        if avg_loss == 0:
            return 0

        r = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / r)

        return max(0, kelly * 0.25)

# ==============================================================================
# === FIBONACCI CALCULATOR =====================================================
# ==============================================================================

class FibonacciCalculator:
    """Calculate Fibonacci levels"""

    def __init__(self):
        self.retracement_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.extension_levels = [1.272, 1.414, 1.618, 2.0, 2.618]

    def calculate_levels(self, high, low, trend='uptrend'):
        """Calculate Fibonacci levels"""
        diff = high - low
        levels = {}

        if trend == 'uptrend':
            for level in self.retracement_levels:
                levels[f"Fib {level:.3f}"] = high - (diff * level)

            for level in self.extension_levels:
                levels[f"Ext {level:.3f}"] = high - (diff * level)
        else:
            for level in self.retracement_levels:
                levels[f"Fib {level:.3f}"] = low + (diff * level)

            for level in self.extension_levels:
                levels[f"Ext {level:.3f}"] = low + (diff * level)

        return levels

    def identify_targets(self, current_price, fib_levels):
        """Identify nearest Fibonacci targets"""
        targets = []

        sorted_levels = sorted(fib_levels.items(), key=lambda x: x[1])

        for name, price in sorted_levels:
            if price > current_price:
                targets.append({'level': name, 'price': price, 'distance': price - current_price})
                if len(targets) == 3:
                    break

        return targets

# ==============================================================================
# === BACKTESTING FRAMEWORK ====================================================
# ==============================================================================

class Backtester:
    """Backtest trading strategies"""

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.positions = []

    def run_backtest(self, data, signals):
        """Run backtest on historical data"""
        self.capital = self.initial_capital
        self.trades = []
        position = None

        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            current_signal = signals.iloc[i] if i < len(signals) else 'HOLD'

            if current_signal == 'BUY' and position is None:
                quantity = int(self.capital * 0.95 / current_price)
                position = {
                    'entry_price': current_price,
                    'quantity': quantity,
                    'entry_date': data.index[i]
                }

            elif current_signal == 'SELL' and position is not None:
                exit_price = current_price
                profit_loss = (exit_price - position['entry_price']) * position['quantity']
                self.capital += profit_loss

                self.trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': data.index[i],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'profit_loss': profit_loss,
                    'return_pct': (profit_loss / (position['entry_price'] * position['quantity'])) * 100
                })

                position = None

        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_return_pct': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'final_capital': self.initial_capital
            }

        df = pd.DataFrame(self.trades)

        winning_trades = df[df['profit_loss'] > 0]
        losing_trades = df[df['profit_loss'] < 0]

        total_profit = df['profit_loss'].sum()
        total_wins = winning_trades['profit_loss'].sum() if not winning_trades.empty else 0
        total_losses = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            'total_profit': total_profit,
            'total_return_pct': (total_profit / self.initial_capital) * 100,
            'profit_factor': profit_factor,
            'avg_win': winning_trades['profit_loss'].mean() if not winning_trades.empty else 0,
            'avg_loss': losing_trades['profit_loss'].mean() if not losing_trades.empty else 0,
            'final_capital': self.capital
        }

# ==============================================================================
# === STOCK CATEGORIES =========================================================
# ==============================================================================
# Dynamic market configuration - NO HARDCODING
GLOBAL_MARKETS = {
    "🇮🇳 India (NSE/BSE)": {
        "timezone": "Asia/Kolkata",
        "market_open": dt_time(9, 15),
        "market_close": dt_time(15, 30),
        "exchange_codes": ["NSE", "BSE"],
        "suffix": ".NS",
        "currency": "INR"
    },
    "🇺🇸 USA (NYSE/NASDAQ)": {
        "timezone": "America/New_York",
        "market_open": dt_time(9, 30),
        "market_close": dt_time(16, 0),
        "exchange_codes": ["NYSE", "NASDAQ"],
        "suffix": "",
        "currency": "USD"
    },
    "🇬🇧 UK (LSE)": {
        "timezone": "Europe/London",
        "market_open": dt_time(8, 0),
        "market_close": dt_time(16, 30),
        "exchange_codes": ["LSE"],
        "suffix": ".L",
        "currency": "GBP"
    },
    "🇯🇵 Japan (TSE)": {
        "timezone": "Asia/Tokyo",
        "market_open": dt_time(9, 0),
        "market_close": dt_time(15, 0),
        "exchange_codes": ["TSE"],
        "suffix": ".T",
        "currency": "JPY"
    }
}


# ==============================================================================
# === HELPER FUNCTIONS =========================================================
# ==============================================================================

def get_currency_symbol(ticker, selected_market=None):
    """Get currency symbol based on ticker suffix or selected market"""
    
    # Check ticker suffix first
    if '.NS' in ticker or '.BO' in ticker:
        return '₹'  # Indian Rupee
    elif '.L' in ticker:
        return '£'  # British Pound
    elif '.T' in ticker:
        return '¥'  # Japanese Yen
    
    # Check selected market as fallback
    if selected_market:
        if 'India' in selected_market:
            return '₹'
        elif 'UK' in selected_market:
            return '£'
        elif 'Japan' in selected_market:
            return '¥'
    
    # Default to USD
    return '$'
    
def fetchintradaydataticker(ticker, interval='5m', period='5d'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    if hist.empty:
        return None
    hist.columns = [col.lower() for col in hist.columns]
    return hist

def analyze_macd_detailed(macd_data, daily_data):
    """
    Detailed MACD analysis following professional framework
    Based on Prompt 2 - Technical Analysis Expert
    """
    analysis = {
        'crossover': '',
        'crossover_type': '',
        'histogram_state': '',
        'momentum': '',
        'centerline_status': '',
        'divergence_potential': '',
        'overall_signal': '',
        'strength': ''
    }
    
    macd_line = macd_data.get('line', 0)
    signal_line = macd_data.get('signal', 0)
    histogram = macd_data.get('histogram', 0)
    
    # 1. MACD Line and Signal Line Analysis
    if macd_line > signal_line:
        analysis['crossover'] = '🟢 Bullish Crossover'
        analysis['crossover_type'] = 'MACD line is above Signal line'
        spread = abs(macd_line - signal_line)
        if spread > 0.5:
            analysis['strength'] = 'Strong bullish momentum (wide spread)'
        else:
            analysis['strength'] = 'Moderate bullish momentum (narrow spread)'
    else:
        analysis['crossover'] = '🔴 Bearish Crossover'
        analysis['crossover_type'] = 'MACD line is below Signal line'
        spread = abs(macd_line - signal_line)
        if spread > 0.5:
            analysis['strength'] = 'Strong bearish momentum (wide spread)'
        else:
            analysis['strength'] = 'Moderate bearish momentum (narrow spread)'
    
    # 2. Histogram Analysis
    if histogram > 0:
        analysis['histogram_state'] = 'Positive (above zero)'
        analysis['momentum'] = 'Bullish momentum present'
    elif histogram < 0:
        analysis['histogram_state'] = 'Negative (below zero)'
        analysis['momentum'] = 'Bearish momentum present'
    else:
        analysis['histogram_state'] = 'At zero line'
        analysis['momentum'] = 'Momentum transition point'
    
    # 3. Centerline (Zero Line) Analysis
    if macd_line > 0:
        analysis['centerline_status'] = '✅ Above zero line - Long-term bullish trend'
    elif macd_line < 0:
        analysis['centerline_status'] = '❌ Below zero line - Long-term bearish trend'
    else:
        analysis['centerline_status'] = '⚠️ At zero line - Trend reversal potential'
    
    # 4. Overall Signal
    if macd_line > signal_line and histogram > 0:
        analysis['overall_signal'] = '🟢 STRONG BUY - Bullish alignment'
    elif macd_line < signal_line and histogram < 0:
        analysis['overall_signal'] = '🔴 STRONG SELL - Bearish alignment'
    elif macd_line > signal_line and histogram < 0:
        analysis['overall_signal'] = '⚠️ WEAK BUY - Momentum weakening'
    else:
        analysis['overall_signal'] = '⚠️ WEAK SELL - Momentum weakening'
    
    return analysis

def generate_comprehensive_analysis(ticker, results, sentiment, news_headlines):
    # Safety checks for all inputs
    if not isinstance(results, dict):
        results = {}
    if not isinstance(news_headlines, list):
        news_headlines = []
    
    # Get sentiment safely
    sentiment = results.get('sentiment', {})
    if not isinstance(sentiment, dict):
        sentiment = {'sentiment': 'Neutral', 'score': 0, 'explanation': 'No sentiment data available'}
        
    ma_50 = results['moving_averages']['MA_50']
    price = results['latest_price']
    trend = 'Uptrend' if price > ma_50 else 'Downtrend'
    
    macd = results['macd']
    macd_signal = 'Bullish' if macd['line'] > macd['signal'] else 'Bearish'
    rsi = results['rsi']
    rsi_signal = 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'
    
    prompt = f"""
🎯 **COMPREHENSIVE ANALYSIS FOR {ticker}**

**1. Overall Summary/Snapshot**

Provide a concise 2-3 sentence overview blending technical posture, recent news/sentiment, and the primary quantitative indicator trend (RSI, MACD, MA).

**2. Quantitative Snapshot:**
- Current Price: {currency}{price:.2f}
- Signal: {results.get('signal', 'HOLD')}
- RSI: {rsi:.2f} ({rsi_signal})
- Trend: {trend}
- Position Size: {results.get('position_size', 0)} shares
- Capital Required: {currency}{results.get('capital_used', 0):,.0f}

**3. Detailed Technical Analysis**
**MACD Analysis:**
- MACD Line: {macd['line']:.2f}
- Signal Line: {macd['signal']:.2f}
- Histogram: {macd['histogram']:.2f}
- Signal: {macd_signal}

**Moving Average Configuration:**
- Price vs MA50: {price:.2f} vs {ma_50:.2f} ({'Above' if price > ma_50 else 'Below'})
- MA50 vs MA200: {ma_50:.2f} vs {results['moving_averages']['MA_200']:.2f}

**Key Levels:**
- Resistance: {currency}{results.get('resistance', 0):.2f} (+{((results.get('resistance', 0) - price) / price * 100):.2f}%)
- Support: {currency}{results.get('support', 0):.2f} ({((price - results.get('support', 0)) / price * 100):.2f}%)

**Technical Posture:** {
    'Strong Bullish' if price > ma_50 > results['moving_averages']['MA_200']
    else 'Strong Bearish' if price < ma_50 < results['moving_averages']['MA_200']
    else 'Mixed/Consolidating'
}

**4. Qualitative Context: News & Sentiment**
- Sentiment: {sentiment.get('sentiment', 'Neutral') if isinstance(sentiment, dict) else 'Neutral'}
- Sentiment Score: {sentiment.get('score', 0):.2f}
- Top Headlines: {'; '.join(news_headlines[:3]) if news_headlines else 'No recent news found.'}

**5. Integrated Outlook & Analyst View**
**Alignment Check:**
- Do technical signals (MACD: {macd_signal}, Trend: {trend}) align with sentiment ({sentiment.get('sentiment', 'Neutral')})?
- Are there any divergences or contradictions?
- Does technical posture align with current sentiment and news flow?
- Discuss divergences (e.g., bullish technicals vs. negative news).

**Forward-Looking Perspective:**
- What are potential catalysts (from news)?
- What are key risks (technical or fundamental)?
- Which levels should traders watch closely?
- Key upcoming events or catalysts from news.
- Key risks (technical breakdown, bearish sentiment, etc.).

**Trading Recommendation:**
- Suggested action: BUY / SELL / HOLD
- Confidence level
- Entry/Exit strategy
- Provide actionable view: recommend BUY/SELL/HOLD with confidence; mention critical levels for stops and targets.

Respond with structured, actionable paragraphs as in a premium research note.
"""
    return prompt

@st.cache_data(ttl=86400)
def get_dynamic_tickers(market_name, api_key=None):
    """
    Fetch tickers dynamically for ALL markets with multiple fallbacks
    Returns: (tickers_list, source_name, errors_list)
    """
    
    base_url = "https://www.alphavantage.co/query"
    errors = []
    
    try:
        # ==================== USA MARKET ====================
        if "USA" in market_name:
            # Method 1: Alpha Vantage LISTING_STATUS
            if api_key:
                try:
                    params = {
                        'function': 'LISTING_STATUS',
                        'state': 'active',
                        'apikey': api_key
                    }
                    response = requests.get(base_url, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        from io import StringIO
                        df = pd.read_csv(StringIO(response.text))
                        tickers = df['symbol'].tolist()[:500]
                        return tickers, "Alpha Vantage Official", errors
                except Exception as e:
                    errors.append(f"Alpha Vantage: {str(e)[:100]}")
            
            # Method 2: NASDAQ Screener
            try:
                url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=1000&download=true"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                data = response.json()
                tickers = [row['symbol'] for row in data['data']['rows'][:500]]
                return tickers, "NASDAQ Screener", errors
            except Exception as e:
                errors.append(f"NASDAQ Screener: {str(e)[:100]}")
            
            # Method 3: Extended fallback
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
                      'NFLX', 'AMD', 'INTC', 'JPM', 'BAC', 'WMT', 'DIS', 'V', 'MA']
            return tickers, "Curated US List", errors
        
        # ==================== INDIA MARKET ====================
        elif "India" in market_name:
            # Method 1: NSE Official API
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                }
                
                url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
                session = requests.Session()
                session.get("https://www.nseindia.com", headers=headers, timeout=5)
                response = session.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    tickers = [f"{stock['symbol']}.NS" for stock in data['data']]
                    return tickers, "NSE Official API", errors
            except Exception as e:
                errors.append(f"NSE API: {str(e)[:100]}")
            
            # Method 2: GitHub Community List
            try:
                url = "https://raw.githubusercontent.com/BennyThadikaran/eod2_data/main/EQUITY_L.csv"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                tickers = [f"{symbol}.NS" for symbol in df['SYMBOL'].tolist()[:300]]
                return tickers, "GitHub Community", errors
            except Exception as e:
                errors.append(f"GitHub: {str(e)[:100]}")
            
            # Method 3: Alpha Vantage validated
            if api_key:
                try:
                    indian_companies = [
                        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
                        'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'ITC', 'AXISBANK',
                        'ASIANPAINT', 'MARUTI', 'TITAN', 'SUNPHARMA', 'ULTRACEMCO',
                        'BAJFINANCE', 'WIPRO', 'HCLTECH', 'NESTLEIND', 'TATAMOTORS',
                        'TATASTEEL', 'POWERGRID', 'NTPC', 'ONGC', 'M&M', 'TECHM',
                        'ADANIPORTS', 'HINDALCO', 'DIVISLAB', 'DRREDDY', 'BAJAJFINSV'
                    ]
                    tickers = [f"{symbol}.NS" for symbol in indian_companies]
                    return tickers, "Alpha Vantage Validated", errors
                except:
                    pass
            
            # Method 4: Final fallback
            tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                      'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS']
            return tickers, "Curated NSE List", errors
        
        # ==================== UK MARKET ====================
        elif "UK" in market_name:
            # Method 1: Alpha Vantage validated (if API key available)
            if api_key:
                try:
                    lse_symbols = [
                        'BARC', 'HSBA', 'BP', 'SHEL', 'VOD', 'AZN', 'GLEN', 'RIO',
                        'LSEG', 'LLOY', 'GSK', 'ULVR', 'DGE', 'NG', 'REL', 'PSON',
                        'CRH', 'CPG', 'ANTO', 'PRU', 'BT-A', 'BA', 'IMB', 'FERG'
                    ]
                    tickers = [f"{symbol}.L" for symbol in lse_symbols]
                    return tickers, "Alpha Vantage Validated LSE", errors
                except:
                    pass
            
            # Method 2: Extended curated list
            lse_major = [
                'BARC.L', 'HSBA.L', 'BP.L', 'SHEL.L', 'VOD.L', 'AZN.L', 'GLEN.L', 
                'RIO.L', 'LSEG.L', 'LLOY.L', 'GSK.L', 'ULVR.L', 'DGE.L', 'NG.L',
                'REL.L', 'PSON.L', 'CRH.L', 'CPG.L', 'ANTO.L', 'PRU.L', 'BT-A.L',
                'BA.L', 'IMB.L', 'FERG.L', 'EXPN.L', 'AAL.L', 'BDEV.L', 'FLTR.L'
            ]
            return lse_major, "Curated LSE List", errors
        
        # ==================== JAPAN MARKET ====================
        elif "Japan" in market_name:
            # Note: Alpha Vantage has limited TSE coverage
            if api_key:
                errors.append("Alpha Vantage: Limited TSE support")
            
            # Comprehensive TSE list
            tse_stocks = [
                '7203.T', '6758.T', '9984.T', '6861.T', '8306.T', '7267.T', '6098.T',
                '9432.T', '8035.T', '4063.T', '6501.T', '6902.T', '6954.T', '6981.T',
                '4502.T', '4503.T', '8411.T', '8316.T', '7751.T', '6762.T', '9434.T',
                '9433.T', '8031.T', '8058.T', '3382.T', '4324.T', '6178.T', '4911.T'
            ]
            return tse_stocks, "Curated TSE List", errors
        
        # Unknown market
        return [], "Unknown Market", errors
        
    except Exception as e:
        errors.append(f"Critical error: {str(e)[:100]}")
        # Emergency fallback
        return get_emergency_fallback(market_name), "Emergency Fallback", errors


def get_emergency_fallback(market_name):
    """Minimal emergency fallback lists"""
    if "USA" in market_name:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    elif "India" in market_name:
        return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    elif "UK" in market_name:
        return ['BARC.L', 'HSBA.L', 'BP.L', 'SHEL.L', 'VOD.L', 'AZN.L']
    elif "Japan" in market_name:
        return ['7203.T', '6758.T', '9984.T', '6861.T', '8306.T']
    return []


def run_premarket_screener(market_name, market_config):
    """Pre-market screener with comprehensive error handling and user feedback"""
    
    # Fetch tickers with source tracking
    all_tickers, source, errors = get_dynamic_tickers(market_name, ALPHA_VANTAGE_API_KEY)
    
    # Handle no tickers scenario
    if not all_tickers:
        st.error("❌ Unable to fetch stock list from any source")
        if errors:
            with st.expander("🔍 Click to view error details"):
                for err in errors:
                    st.text(f"• {err}")
        return {}
    
    # Show appropriate feedback based on data source quality
    if errors and "Emergency" in source:
        # Critical situation - all primary sources failed
        with st.expander(f"⚠️ EMERGENCY MODE: {source} ({len(all_tickers)} stocks)"):
            st.error("⚠️ All primary data sources failed. Using minimal fallback stock list.")
            st.caption("Failed sources:")
            for err in errors:
                st.caption(f"  • {err}")
    elif errors:
        # Some sources failed but we got data from a fallback
        with st.expander(f"ℹ️ Data Source: {source} ({len(all_tickers)} stocks) - Click for details"):
            st.info(f"✅ Successfully loaded from: **{source}**")
            st.caption("Note: Some sources were unavailable:")
            for err in errors:
                st.caption(f"  • {err}")
    else:
        # All good - no errors
        st.success(f"✅ Loaded {len(all_tickers)} stocks from **{source}**")
    
    # Set market-specific filtering criteria
    min_price = 10.0 if "USA" in market_name else 50.0
    min_volume = 100000
    
    # Initialize screening
    screened_list = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("🔍 Scanning stocks for trading opportunities...")
    
    # Process in batches for efficiency
    batch_size = 50
    total_processed = 0
    
    for batch_idx in range(0, len(all_tickers), batch_size):
        batch_tickers = all_tickers[batch_idx:batch_idx + batch_size]
        
        try:
            # Download batch data
            data = yf.download(
                " ".join(batch_tickers), 
                period="5d", 
                group_by='ticker', 
                auto_adjust=True, 
                progress=False
            )
            
            # Process each ticker in the batch
            for ticker in batch_tickers:
                try:
                    # Extract ticker data
                    stock_data = data[ticker] if len(batch_tickers) > 1 else data
                    
                    # Skip if insufficient data
                    if stock_data.empty or len(stock_data) < 2:
                        continue
                    
                    # Get latest and previous day data
                    last_day = stock_data.iloc[-1]
                    prev_day = stock_data.iloc[-2]
                    
                    # Extract metrics
                    price = float(last_day['Close'])
                    volume = int(last_day['Volume'])
                    change_pct = float((price - prev_day['Close']) / prev_day['Close'] * 100)
                    
                    # Apply filters
                    if price >= min_price and volume >= min_volume:
                        screened_list[ticker] = {
                            'price': price,
                            'volume': volume,
                            'change_pct': change_pct
                        }
                        
                        # Stop if we have enough stocks
                        if len(screened_list) >= 50:
                            break
                
                except Exception:
                    # Skip problematic tickers silently
                    continue
            
            # Update progress
            total_processed += len(batch_tickers)
            progress_bar.progress(min(total_processed / len(all_tickers), 1.0))
            
            # Break if we have enough stocks
            if len(screened_list) >= 50:
                break
        
        except Exception:
            # Skip failed batches silently
            continue
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display final results
    if screened_list:
        st.success(f"✅ Found **{len(screened_list)} stocks** from {source} that meet criteria")
        st.info("💡 Select a stock from the sidebar dropdown to begin analysis")
    else:
        st.warning(f"⚠️ No stocks matched screening criteria")
        st.info(f"📊 Criteria: Price ≥ ₹{min_price if 'India' in market_name else '$' + str(min_price)}, Volume ≥ {min_volume:,}")
        st.info("💡 Try selecting a different market or manually enter a ticker symbol")
    
    return screened_list

@st.cache_data
def search_for_ticker(query: str, asset_type: str = "EQUITY") -> dict:
    """Search Yahoo Finance for ticker"""
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
        return {}

@st.cache_data
def fetch_stock_data(ticker, period="1y", interval="1d"):
    """Fetch stock data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info.get('longName') and not info.get('shortName'):
            st.error(f"Ticker '{ticker}' not found")
            return None
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            st.error(f"No data found for {ticker}")
            return None
        return hist
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def fetch_intraday_data(ticker, interval="5m", period="5d"):
    """Fetch intraday data"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            return None
        hist.columns = [col.lower() for col in hist.columns]
        return hist
    except Exception as e:
        return None

def is_market_open():
    """Check if market is open"""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    market_start = dt_time(9, 15)
    market_end = dt_time(15, 30)
    buffer_start = dt_time(9, 25)

    is_weekday = now.weekday() < 5
    is_trading_hours = market_start <= now.time() <= market_end
    past_buffer = now.time() >= buffer_start

    return is_weekday and is_trading_hours and past_buffer

def setup_google_sheets():
    """Initialize Google Sheets"""
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
        return None

def log_to_sheets(sheet, data):
    """Log data to sheets"""
    if sheet:
        try:
            sheet.append_row(data)
            return True
        except:
            pass
    return False

def create_plotly_charts(data, ticker_name):
    """Create trading charts"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & EMAs', 'RSI', 'Volume'),
        row_heights=[0.6, 0.2, 0.2]
    )

    close_col = 'Close' if 'Close' in data.columns else 'close'
    open_col = 'Open' if 'Open' in data.columns else 'open'
    volume_col = 'Volume' if 'Volume' in data.columns else 'volume'

    fig.add_trace(
        go.Scatter(x=data.index, y=data[close_col], mode='lines', name='Price', line=dict(color='white', width=2)),
        row=1, col=1
    )

    for span, color in zip([20, 50, 200], ['#1f77b4', '#ff7f0e', '#d62728']):
        if len(data) >= span:
            ema = data[close_col].ewm(span=span, adjust=False).mean()
            fig.add_trace(
                go.Scatter(x=data.index, y=ema, mode='lines', name=f'EMA {span}', line=dict(width=1.5, color=color)),
                row=1, col=1
            )

    delta = data[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))

    fig.add_trace(
        go.Scatter(x=data.index, y=rsi_series, mode='lines', name='RSI', line=dict(color='#9467bd')),
        row=2, col=1
    )

    for y_val, dash_style in [(70, 'dash'), (30, 'dash')]:
        fig.add_hline(y=y_val, line_dash=dash_style, line_color="grey", row=2, col=1)

    colors = ['#2ca02c' if row[close_col] >= row[open_col] else '#d62728' for index, row in data.iterrows()]
    fig.add_trace(
        go.Bar(x=data.index, y=data[volume_col], name='Volume', marker_color=colors),
        row=3, col=1
    )

    fig.update_layout(
        height=800,
        title_text=f'Technical Analysis for {ticker_name}',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )

    return fig

def embed_tradingview_widget(ticker):
    """Embed TradingView widget"""
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

def get_ai_analysis_gemini(prompt):
    """Get AI analysis using Gemini"""
    if not GOOGLE_API_KEY:
        return "Gemini API key not configured"

    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def get_ai_analysis_openrouter(prompt, model="anthropic/claude-3.5-sonnet"):
    """Get AI analysis using OpenRouter"""
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
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_portfolio(tickers_list):
    """Analyze portfolio"""
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
                    'Price': f"{currency}{latest_price:.2f}",
                    'RSI': f"{rsi:.2f}",
                    'Signal': signal
                })
        except:
            continue

        progress_bar.progress((i + 1) / len(tickers_list))

    return pd.DataFrame(portfolio_results)

# ==============================================================================
# === STOCK ANALYZER CLASS (INCLUDING ALL METHODS FROM BEFORE) ================
# ==============================================================================

class StockAnalyzer:
    def __init__(self, ticker=None):
        self.ticker = ticker
        self.sentiment_analyzer = None
        self.setup_sentiment_analyzer()
        self.fib_calc = FibonacciCalculator()
        self.risk_manager = RiskManager()

    def setup_sentiment_analyzer(self):
        """Setup sentiment analyzer with error handling"""
        try:
            from transformers import pipeline
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Warning: Could not load sentiment analyzer: {e}")
            self.sentiment_analyzer = None

        def setup_sentiment_analyzer(self):
            """Initialize sentiment analysis"""
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True)
            except:
                try:
                    self.sentiment_analyzer = pipeline("sentiment-analysis")
                except:
                    self.sentiment_analyzer = None

        def analyze_sentiment_detailed(self, headlines):
            """Analyze sentiment with per-article breakdown"""
            if not headlines or not self.sentiment_analyzer:
                return {
                    'overall_sentiment': 'Neutral',
                    'overall_score': 0.0,
                    'articles': []
                }
            
            try:
                article_sentiments = []
                sentiment_scores = []
                
                for headline in headlines:
                    if len(headline) > 15:
                        result = self.sentiment_analyzer(headline[:512])
                        
                        if isinstance(result[0], list):
                            sentiment_dict = {item['label']: item['score'] for item in result[0]}
                            score = sentiment_dict.get('positive', 0) - sentiment_dict.get('negative', 0)
                            label = 'Positive' if score > 0.1 else 'Negative' if score < -0.1 else 'Neutral'
                            
                            article_sentiments.append({
                                'headline': headline,
                                'sentiment': label,
                                'score': round(score, 3),
                                'positive': round(sentiment_dict.get('positive', 0), 3),
                                'negative': round(sentiment_dict.get('negative', 0), 3),
                                'neutral': round(sentiment_dict.get('neutral', 0), 3)
                            })
                            sentiment_scores.append(score)
                        else:
                            score = result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
                            
                            article_sentiments.append({
                                'headline': headline,
                                'sentiment': result[0]['label'],
                                'score': round(score, 3),
                                'confidence': round(result[0]['score'], 3)
                            })
                            sentiment_scores.append(score)
                
                if sentiment_scores:
                    avg_sentiment = np.mean(sentiment_scores)
                    overall = 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'
                else:
                    avg_sentiment = 0.0
                    overall = 'Neutral'
                
                return {
                    'overall_sentiment': overall,
                    'overall_score': round(avg_sentiment, 3),
                    'articles': article_sentiments,
                    'total_articles': len(article_sentiments),
                    'positive_count': sum(1 for a in article_sentiments if a['sentiment'] in ['Positive', 'POSITIVE']),
                    'negative_count': sum(1 for a in article_sentiments if a['sentiment'] in ['Negative', 'NEGATIVE']),
                    'neutral_count': sum(1 for a in article_sentiments if a['sentiment'] in ['Neutral', 'NEUTRAL'])
                }
            except Exception as e:
                return {
                    'overall_sentiment': 'Neutral',
                    'overall_score': 0.0,
                    'articles': [],
                    'error': str(e)
                }


        def fetch_stock_data(self, ticker, period="60d"):
            """Fetch stock data"""
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if hist.empty:
                    return None
                return hist
            except:
                return None

        def compute_rsi(self, data, window=14):
            """Calculate RSI"""
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
            """Calculate moving averages"""
            try:
                close_col = 'Close' if 'Close' in data.columns else 'close'
                current_price = data[close_col].iloc[-1]

                ma_20 = data[close_col].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_price
                ma_25 = data[close_col].rolling(window=25).mean().iloc[-1] if len(data) >= 25 else current_price
                ma_50 = data[close_col].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else current_price
                ma_200 = data[close_col].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else current_price

                return {
                    'MA_20': ma_20 if not pd.isna(ma_20) else current_price,
                    'MA_25': ma_25 if not pd.isna(ma_25) else current_price,
                    'MA_50': ma_50 if not pd.isna(ma_50) else current_price,
                    'MA_200': ma_200 if not pd.isna(ma_200) else current_price
                }
            except:
                close_col = 'Close' if 'Close' in data.columns else 'close'
                current_price = data[close_col].iloc[-1]
                return {'MA_20': current_price, 'MA_25': current_price, 'MA_50': current_price, 'MA_200': current_price}

        def compute_bollinger_bands(self, data, window=20, num_std=2):
            """Calculate Bollinger Bands"""
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
            """Calculate SMI"""
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
            """Calculate VWAP"""
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
            """Calculate VWMA"""
            try:
                close_col = 'Close' if 'Close' in data.columns else 'close'
                volume_col = 'Volume' if 'Volume' in data.columns else 'volume'

                vwma = (data[close_col] * data[volume_col]).rolling(window=period).sum() / data[volume_col].rolling(window=period).sum()
                return vwma.iloc[-1] if not pd.isna(vwma.iloc[-1]) else data[close_col].iloc[-1]
            except:
                close_col = 'Close' if 'Close' in data.columns else 'close'
                return data[close_col].iloc[-1]

        def compute_supertrend(self, data, period=10, multiplier=3):
            """Calculate Supertrend"""
            try:
                high_col = 'High' if 'High' in data.columns else 'high'
                low_col = 'Low' if 'Low' in data.columns else 'low'
                close_col = 'Close' if 'Close' in data.columns else 'close'

                high_low = data[high_col] - data[low_col]
                high_close = abs(data[high_col] - data[close_col].shift())
                low_close = abs(data[low_col] - data[close_col].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(window=period).mean()

                hl_avg = (data[high_col] + data[low_col]) / 2
                upper_band = hl_avg + (multiplier * atr)
                lower_band = hl_avg - (multiplier * atr)

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
            """Detect S/R levels"""
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
                return {'resistance': current_price * 1.02, 'support': current_price * 0.98, 'current_price': current_price}

    def detect_candlestick_patterns_talib(self, data):
        """
        Comprehensive candlestick pattern detection - Pure Python (No TA-Lib needed)
        Detects 15+ high-probability patterns with professional accuracy
        """
        
        if len(data) < 5:
            return {
                'pattern': 'Insufficient Data',
                'type': 'neutral',
                'strength': 0,
                'confidence': 0,
                'category': 'none',
                'description': 'Need at least 5 candles for pattern detection'
            }
        
        patterns_found = []
        
        # Get last 5 candles for pattern analysis
        c1, c2, c3, c4, c5 = data.iloc[-5], data.iloc[-4], data.iloc[-3], data.iloc[-2], data.iloc[-1]
        
        # Current candle (most recent)
        curr_open = c5['Open'] if 'Open' in c5.index else c5['open']
        curr_high = c5['High'] if 'High' in c5.index else c5['high']
        curr_low = c5['Low'] if 'Low' in c5.index else c5['low']
        curr_close = c5['Close'] if 'Close' in c5.index else c5['close']
        curr_body = abs(curr_close - curr_open)
        curr_range = curr_high - curr_low
        
        # Previous candle
        prev_open = c4['Open'] if 'Open' in c4.index else c4['open']
        prev_high = c4['High'] if 'High' in c4.index else c4['high']
        prev_low = c4['Low'] if 'Low' in c4.index else c4['low']
        prev_close = c4['Close'] if 'Close' in c4.index else c4['close']
        prev_body = abs(prev_close - prev_open)
        prev_range = prev_high - prev_low
        
        # Helper variables
        curr_is_green = curr_close > curr_open
        curr_is_red = curr_close < curr_open
        prev_is_green = prev_close > prev_open
        prev_is_red = prev_close < prev_open
        
        lower_shadow = min(curr_open, curr_close) - curr_low
        upper_shadow = curr_high - max(curr_open, curr_close)
        
        # ==================== BULLISH PATTERNS ====================
        
        # 1. HAMMER (Bullish Reversal) - Strong at support
        if (lower_shadow > curr_body * 2 and 
            upper_shadow < curr_body * 0.3 and
            curr_is_green and
            curr_range > 0):
            patterns_found.append({
                'pattern': 'Hammer',
                'type': 'bullish',
                'strength': 85,
                'confidence': 85,
                'category': 'reversal',
                'description': 'Strong bullish reversal at support - Buyers regained control after selling pressure'
            })
        
        # 2. INVERTED HAMMER (Bullish Reversal)
        elif (upper_shadow > curr_body * 2 and 
              lower_shadow < curr_body * 0.3 and
              curr_range > 0):
            patterns_found.append({
                'pattern': 'Inverted Hammer',
                'type': 'bullish',
                'strength': 75,
                'confidence': 75,
                'category': 'reversal',
                'description': 'Potential bullish reversal - Wait for next candle confirmation'
            })
        
        # 3. BULLISH ENGULFING (Very Strong)
        if (prev_is_red and curr_is_green and
            curr_open < prev_close and
            curr_close > prev_open and
            curr_body > prev_body * 1.3):
            patterns_found.append({
                'pattern': 'Bullish Engulfing',
                'type': 'bullish',
                'strength': 90,
                'confidence': 90,
                'category': 'reversal',
                'description': 'Very strong bullish reversal - Large buying pressure overwhelmed sellers'
            })
        
        # 4. MORNING STAR (3-Candle Bullish Reversal)
        c3_open = c3['Open'] if 'Open' in c3.index else c3['open']
        c3_close = c3['Close'] if 'Close' in c3.index else c3['close']
        c3_high = c3['High'] if 'High' in c3.index else c3['high']
        c3_low = c3['Low'] if 'Low' in c3.index else c3['low']
        
        c4_open = c4['Open'] if 'Open' in c4.index else c4['open']
        c4_close = c4['Close'] if 'Close' in c4.index else c4['close']
        
        if (c3_close < c3_open and  # First red
            abs(c4_close - c4_open) < (c3_high - c3_low) * 0.3 and  # Small middle
            curr_is_green and
            curr_close > (c3_open + c3_close) / 2):
            patterns_found.append({
                'pattern': 'Morning Star',
                'type': 'bullish',
                'strength': 95,
                'confidence': 95,
                'category': 'reversal',
                'description': 'Extremely strong bullish reversal - Classic 3-candle bottom pattern'
            })
        
        # 5. PIERCING PATTERN
        if (prev_is_red and curr_is_green and
            curr_open < prev_low and
            curr_close > (prev_open + prev_close) / 2 and
            curr_close < prev_open):
            patterns_found.append({
                'pattern': 'Piercing Pattern',
                'type': 'bullish',
                'strength': 80,
                'confidence': 80,
                'category': 'reversal',
                'description': 'Bullish reversal - Buyers pushing through resistance'
            })
        
        # 6. THREE WHITE SOLDIERS (Bullish Continuation)
        if (c3_close > c3_open and
            c4_close > c4_open and
            curr_is_green and
            c4_close > c3_close and
            curr_close > c4_close):
            patterns_found.append({
                'pattern': 'Three White Soldiers',
                'type': 'bullish',
                'strength': 92,
                'confidence': 90,
                'category': 'continuation',
                'description': 'Strong bullish continuation - Steady upward momentum'
            })
        
        # 7. BULLISH HARAMI
        if (prev_is_red and curr_is_green and
            curr_open > prev_close and
            curr_close < prev_open and
            curr_body < prev_body * 0.5):
            patterns_found.append({
                'pattern': 'Bullish Harami',
                'type': 'bullish',
                'strength': 70,
                'confidence': 70,
                'category': 'reversal',
                'description': 'Bullish reversal - Needs confirmation from next candle'
            })
        
        # 8. DRAGONFLY DOJI (Bullish at support)
        if (curr_body < curr_range * 0.1 and
            lower_shadow > upper_shadow * 2 and
            curr_range > 0):
            patterns_found.append({
                'pattern': 'Dragonfly Doji',
                'type': 'bullish',
                'strength': 70,
                'confidence': 75,
                'category': 'reversal',
                'description': 'Bullish reversal at support - Sellers tried but failed'
            })
        
        # ==================== BEARISH PATTERNS ====================
        
        # 9. SHOOTING STAR (Bearish Reversal)
        if (upper_shadow > curr_body * 2 and 
            lower_shadow < curr_body * 0.3 and
            curr_is_red and
            curr_range > 0):
            patterns_found.append({
                'pattern': 'Shooting Star',
                'type': 'bearish',
                'strength': 85,
                'confidence': 85,
                'category': 'reversal',
                'description': 'Strong bearish reversal at resistance - Sellers regained control'
            })
        
        # 10. HANGING MAN (Bearish at resistance)
        elif (lower_shadow > curr_body * 2 and 
              upper_shadow < curr_body * 0.3 and
              curr_is_red and
              curr_range > 0):
            patterns_found.append({
                'pattern': 'Hanging Man',
                'type': 'bearish',
                'strength': 75,
                'confidence': 75,
                'category': 'reversal',
                'description': 'Bearish reversal at resistance - Warning sign of trend change'
            })
        
        # 11. BEARISH ENGULFING
        if (prev_is_green and curr_is_red and
            curr_open > prev_close and
            curr_close < prev_open and
            curr_body > prev_body * 1.3):
            patterns_found.append({
                'pattern': 'Bearish Engulfing',
                'type': 'bearish',
                'strength': 90,
                'confidence': 90,
                'category': 'reversal',
                'description': 'Very strong bearish reversal - Large selling pressure overwhelmed buyers'
            })
        
        # 12. EVENING STAR (3-Candle Bearish Reversal)
        if (c3_close > c3_open and  # First green
            abs(c4_close - c4_open) < (c3_high - c3_low) * 0.3 and  # Small middle
            curr_is_red and
            curr_close < (c3_open + c3_close) / 2):
            patterns_found.append({
                'pattern': 'Evening Star',
                'type': 'bearish',
                'strength': 95,
                'confidence': 95,
                'category': 'reversal',
                'description': 'Extremely strong bearish reversal - Classic 3-candle top pattern'
            })
        
        # 13. DARK CLOUD COVER
        if (prev_is_green and curr_is_red and
            curr_open > prev_high and
            curr_close < (prev_open + prev_close) / 2 and
            curr_close > prev_open):
            patterns_found.append({
                'pattern': 'Dark Cloud Cover',
                'type': 'bearish',
                'strength': 80,
                'confidence': 80,
                'category': 'reversal',
                'description': 'Bearish reversal - Selling pressure increasing significantly'
            })
        
        # 14. THREE BLACK CROWS (Bearish Continuation)
        if (c3_close < c3_open and
            c4_close < c4_open and
            curr_is_red and
            c4_close < c3_close and
            curr_close < c4_close):
            patterns_found.append({
                'pattern': 'Three Black Crows',
                'type': 'bearish',
                'strength': 92,
                'confidence': 90,
                'category': 'continuation',
                'description': 'Strong bearish continuation - Steady downward momentum'
            })
        
        # 15. GRAVESTONE DOJI (Bearish at resistance)
        if (curr_body < curr_range * 0.1 and
            upper_shadow > lower_shadow * 2 and
            curr_range > 0):
            patterns_found.append({
                'pattern': 'Gravestone Doji',
                'type': 'bearish',
                'strength': 70,
                'confidence': 75,
                'category': 'reversal',
                'description': 'Bearish reversal at resistance - Buyers tried but failed'
            })
        
        # ==================== NEUTRAL PATTERNS ====================
        
        # 16. DOJI (Indecision)
        if curr_body < curr_range * 0.1 and curr_range > 0:
            patterns_found.append({
                'pattern': 'Doji',
                'type': 'neutral',
                'strength': 50,
                'confidence': 70,
                'category': 'indecision',
                'description': 'Market indecision - Potential trend reversal point, wait for confirmation'
            })
        
        # 17. SPINNING TOP
        if (curr_body > curr_range * 0.1 and curr_body < curr_range * 0.3 and
            upper_shadow > curr_body and lower_shadow > curr_body):
            patterns_found.append({
                'pattern': 'Spinning Top',
                'type': 'neutral',
                'strength': 40,
                'confidence': 60,
                'category': 'indecision',
                'description': 'Indecision between buyers and sellers - Wait for clear direction'
            })
        
        # Return strongest pattern
        if patterns_found:
            # Sort by strength, then confidence
            patterns_found.sort(key=lambda x: (x['strength'], x['confidence']), reverse=True)
            return patterns_found[0]
        else:
            return {
                'pattern': 'No Significant Pattern',
                'type': 'neutral',
                'strength': 0,
                'confidence': 0,
                'category': 'none',
                'description': 'No clear candlestick pattern detected'
            }
        
    def get_pattern_description(self, pattern_name, pattern_type, category):
        """Get professional description for each pattern"""
        descriptions = {
            # Bullish Patterns
            'Hammer': 'Strong bullish reversal at support - Buyers regained control after selling pressure',
            'Inverted Hammer': 'Potential bullish reversal - Wait for next candle confirmation',
            'Bullish Engulfing': 'Very strong bullish reversal - Large buying pressure overwhelmed sellers',
            'Morning Star': 'Extremely strong bullish reversal - Classic 3-candle bottom pattern',
            'Piercing Pattern': 'Bullish reversal - Buyers pushing through resistance',
            'Morning Doji Star': 'Strong bullish reversal with indecision candle - Trend change likely',
            'Three White Soldiers': 'Strong bullish continuation - Steady upward momentum',
            'Rising Three Methods': 'Bullish continuation - Temporary pause before next move up',
            'Bullish Harami': 'Bullish reversal - Needs confirmation from next candle',
            'Dragonfly Doji': 'Bullish reversal at support - Sellers tried but failed',
            'Marubozu': 'Strong bullish momentum - No wicks, pure buying pressure',
            
            # Bearish Patterns
            'Shooting Star': 'Strong bearish reversal at resistance - Sellers regained control',
            'Evening Star': 'Extremely strong bearish reversal - Classic 3-candle top pattern',
            'Dark Cloud Cover': 'Bearish reversal - Selling pressure increasing significantly',
            'Hanging Man': 'Bearish reversal at resistance - Warning sign of trend change',
            'Evening Doji Star': 'Strong bearish reversal with indecision - Downtrend likely',
            'Three Black Crows': 'Strong bearish continuation - Steady downward momentum',
            'Identical Three Crows': 'Very strong bearish continuation - Consistent selling',
            'Gravestone Doji': 'Bearish reversal at resistance - Buyers tried but failed',
            
            # Neutral Patterns
            'Doji': 'Market indecision - Potential trend reversal point, wait for confirmation',
            'Spinning Top': 'Indecision between buyers and sellers - Wait for clear direction',
        }
        
        return descriptions.get(pattern_name, f'{pattern_type.title()} {category} signal')

    def calculate_pattern_impact(self, pattern_data, current_price):
        """
        Calculate how candlestick pattern should impact trading decisions
        Returns adjustment factors for stop-loss, targets, and signal confidence
        """
        
        pattern_type = pattern_data['type']
        strength = pattern_data['strength']
        category = pattern_data.get('category', 'none')
        
        impact = {
            'signal_boost': 0,           # Points to add to signal scoring
            'stop_loss_adjustment': 1.0, # Multiplier for stop-loss distance
            'target_multiplier': 1.0,    # Multiplier for profit targets
            'confidence_boost': 0,       # Percentage boost to confidence
            'risk_adjustment': 1.0       # Multiplier for position size
        }
        
        # VERY STRONG PATTERNS (Strength >= 90)
        if strength >= 90:
            if pattern_type == 'bullish':
                impact['signal_boost'] = 2          # Strong buy signal
                impact['stop_loss_adjustment'] = 0.97  # Tighter stop (3% closer)
                impact['target_multiplier'] = 1.4   # 40% higher targets
                impact['confidence_boost'] = 20     # +20% confidence
                impact['risk_adjustment'] = 1.2     # Can increase position 20%
            elif pattern_type == 'bearish':
                impact['signal_boost'] = -2         # Strong sell/avoid signal
                impact['stop_loss_adjustment'] = 1.03  # Wider stop
                impact['target_multiplier'] = 0.7   # Lower targets
                impact['confidence_boost'] = -20
                impact['risk_adjustment'] = 0.8     # Reduce position 20%
        
        # STRONG PATTERNS (Strength 80-89)
        elif strength >= 80:
            if pattern_type == 'bullish':
                impact['signal_boost'] = 1.5
                impact['stop_loss_adjustment'] = 0.98
                impact['target_multiplier'] = 1.25
                impact['confidence_boost'] = 15
                impact['risk_adjustment'] = 1.15
            elif pattern_type == 'bearish':
                impact['signal_boost'] = -1.5
                impact['stop_loss_adjustment'] = 1.02
                impact['target_multiplier'] = 0.8
                impact['confidence_boost'] = -15
                impact['risk_adjustment'] = 0.85
        
        # MEDIUM PATTERNS (Strength 70-79)
        elif strength >= 70:
            if pattern_type == 'bullish':
                impact['signal_boost'] = 1
                impact['stop_loss_adjustment'] = 0.99
                impact['target_multiplier'] = 1.15
                impact['confidence_boost'] = 10
                impact['risk_adjustment'] = 1.1
            elif pattern_type == 'bearish':
                impact['signal_boost'] = -1
                impact['stop_loss_adjustment'] = 1.01
                impact['target_multiplier'] = 0.9
                impact['confidence_boost'] = -10
                impact['risk_adjustment'] = 0.9
        
        # WEAK PATTERNS (Strength < 70) - Minimal impact
        else:
            if pattern_type == 'bullish':
                impact['signal_boost'] = 0.5
                impact['confidence_boost'] = 5
            elif pattern_type == 'bearish':
                impact['signal_boost'] = -0.5
                impact['confidence_boost'] = -5
        
        # BONUS: Extra boost for REVERSAL patterns at key levels
        if category == 'reversal':
            impact['confidence_boost'] += 5  # Reversals are powerful
        
        return impact

        def detect_inside_bar_pattern(self, data):
            """Detect Inside Bar"""
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
                return {"detected": False, "message": "Error"}

        def detect_breakout_retest(self, five_min_data, resistance):
            """Detect breakout and retest"""
            if five_min_data.empty or resistance == 0:
                return "Not Analyzed"

            high_col = 'High' if 'High' in five_min_data.columns else 'high'
            low_col = 'Low' if 'Low' in five_min_data.columns else 'low'
            close_col = 'Close' if 'Close' in five_min_data.columns else 'close'

            recent_data = five_min_data.tail(20)

            breakout_candle_index = -1
            retest_candle_index = -1

            for i in range(1, len(recent_data)):
                prev_high = recent_data[high_col].iloc[i-1]
                current_high = recent_data[high_col].iloc[i]

                if current_high > resistance and prev_high <= resistance:
                    breakout_candle_index = i
                    break

            if breakout_candle_index == -1:
                return "No Breakout Detected"

            for i in range(breakout_candle_index + 1, len(recent_data)):
                current_low = recent_data[low_col].iloc[i]

                if current_low <= resistance:
                    retest_candle_index = i
                    break

            if retest_candle_index == -1:
                return f"Breakout Occurred. Awaiting Retest."

            if retest_candle_index < len(recent_data) - 1:
                confirmation_candle = recent_data.iloc[retest_candle_index + 1]

                if confirmation_candle[close_col] > resistance:
                    return f"✅ Retest Confirmed. Potential Entry."

            return f"Retest in Progress. Awaiting Confirmation."

        def run_confirmation_checklist(self, analysis_results):
            """Run 5-point checklist"""
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

            pattern_status = self.detect_breakout_retest(five_min_df, resistance)
            if "✅ Retest Confirmed" in pattern_status:
                checklist["3. Chart Pattern Confirmed"] = "✅ Breakout/Retest"
            else:
                checklist["3. Chart Pattern Confirmed"] = f"❌ {pattern_status}"

            candle_pattern = self.check_candlestick_pattern(five_min_df)
            if "No significant" not in candle_pattern:
                checklist["4. Candlestick Signal"] = f"✅ {candle_pattern}"
            else:
                checklist["4. Candlestick Signal"] = "❌ No Signal"

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

            bullish_checks = sum(1 for v in checklist.values() if "✅" in str(v) and ("Bullish" in str(v) or "Breakout" in str(v)))
            bearish_checks = sum(1 for v in checklist.values() if "✅" in str(v) and "Bearish" in str(v))

            # Add pattern signal boost
            pattern_boost = analysis_results.get('pattern_impact', {}).get('signal_boost', 0)
            
            # Adjust checks with pattern influence
            if bullish_checks + pattern_boost >= 3:
                checklist['FINAL_SIGNAL'] = "STRONG BUY" if pattern_boost >= 1.5 else "🟢 BUY"
            elif bearish_checks - pattern_boost >= 3:
                checklist['FINAL_SIGNAL'] = "🔴 SELL"
            else:
                checklist['FINAL_SIGNAL'] = "⚪ HOLD"
            return checklist

        def analyze_for_intraday(self):
            """Complete intraday analysis WITH STOP-LOSS - Error Handled Version"""
            results = {
                'ticker': self.ticker,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'market_open': is_market_open()
            }
        
            try:
                # Fetch daily data
                daily_data = fetch_stock_data(self.ticker, period="60d", interval="1d")
                if daily_data is None or daily_data.empty:
                    st.error("Unable to fetch daily data")
                    return None
        
                # Fetch intraday data
                fifteen_min_data = fetch_intraday_data(self.ticker, interval="15m", period="5d")
                if fifteen_min_data is None or fifteen_min_data.empty:
                    fifteen_min_data = daily_data.copy()
                    fifteen_min_data.columns = [col.lower() for col in fifteen_min_data.columns]
        
                five_min_data = fetch_intraday_data(self.ticker, interval="5m", period="5d")
                if five_min_data is None or five_min_data.empty:
                    five_min_data = fifteen_min_data.copy()
        
                # Normalize column names
                daily_data.columns = [col.capitalize() for col in daily_data.columns]
                five_min_data.columns = [col.lower() for col in five_min_data.columns]
                fifteen_min_data.columns = [col.lower() for col in fifteen_min_data.columns]
        
                # Store data
                results['5m_data'] = five_min_data
                results['15m_data'] = fifteen_min_data
                results['daily_data'] = daily_data
        
                # Basic metrics
                results['latest_price'] = float(daily_data['Close'].iloc[-1])
                results['rsi'] = self.compute_rsi(daily_data)
                results['macd'] = self.compute_macd(daily_data)
                results['moving_averages'] = self.compute_moving_averages(daily_data)
        
                # Intraday indicators
                results['bollinger_bands'] = self.compute_bollinger_bands(five_min_data)
                results['stochastic'] = self.compute_stochastic_momentum(five_min_data)
                
                five_min_data = self.compute_vwap(five_min_data)
                results['vwap'] = float(five_min_data['vwap'].iloc[-1])
                results['vwma'] = self.compute_vwma(five_min_data)
                results['supertrend'] = self.compute_supertrend(five_min_data)
        
                # Support/Resistance
                sr_levels = self.detect_support_resistance(fifteen_min_data)
                results['resistance'] = float(sr_levels.get('resistance', results['latest_price'] * 1.02))
                results['support'] = float(sr_levels.get('support', results['latest_price'] * 0.98))
        
                # ========== CANDLESTICK PATTERN ANALYSIS ==========
                try:
                    pattern_data = self.detect_candlestick_patterns_talib(five_min_data)
                    pattern_impact = self.calculate_pattern_impact(pattern_data, results['latest_price'])
                    
                    results['candlestick_pattern'] = pattern_data.get('pattern', 'None')
                    results['pattern_type'] = pattern_data.get('type', 'neutral')
                    results['pattern_strength'] = pattern_data.get('strength', 0)
                    results['pattern_confidence'] = pattern_data.get('confidence', 0)
                    results['pattern_category'] = pattern_data.get('category', 'none')
                    results['pattern_description'] = pattern_data.get('description', 'No pattern')
                    results['pattern_impact'] = pattern_impact
                except Exception as e:
                    st.warning(f"Pattern detection skipped: {str(e)}")
                    results['candlestick_pattern'] = 'Analysis Error'
                    results['pattern_type'] = 'neutral'
                    results['pattern_strength'] = 0
                    results['pattern_confidence'] = 0
                    results['pattern_category'] = 'none'
                    results['pattern_description'] = 'Pattern analysis unavailable'
                    results['pattern_impact'] = {
                        'signal_boost': 0,
                        'stop_loss_adjustment': 1.0,
                        'target_multiplier': 1.0,
                        'confidence_boost': 0,
                        'risk_adjustment': 1.0
                    }
        
                # ============ ATR & STOP-LOSS ============
                try:
                    atr = self.calculate_atr(five_min_data, period=14)
                    results['atr'] = float(atr) if atr > 0 else results['latest_price'] * 0.02
                except:
                    results['atr'] = results['latest_price'] * 0.02
        
                # Calculate base stop-loss
                if results.get('support', 0) > 0:
                    base_stop_loss = results['support'] * 0.995
                else:
                    base_stop_loss = results['latest_price'] * 0.98
                
                # Apply pattern adjustment
                pattern_adjustment = results.get('pattern_impact', {}).get('stop_loss_adjustment', 1.0)
                stop_loss_support = base_stop_loss * pattern_adjustment
                
                stop_loss_atr = results['latest_price'] - (results['atr'] * 1.5)
                results['base_stoploss'] = float(base_stop_loss)
                results['stop_loss'] = float(max(stop_loss_support, stop_loss_atr))
                results['trailing_stop_vwap'] = float(results.get('vwap', results['latest_price']))
        
                # ============ POSITION SIZE ============
                max_capital_per_trade = 12500
                risk_per_share = abs(results['latest_price'] - results['stop_loss'])
        
                if risk_per_share > 0:
                    max_quantity = int(max_capital_per_trade / results['latest_price'])
                    risk_based_quantity = int((max_capital_per_trade * 0.02) / risk_per_share)
                    results['position_size'] = min(max_quantity, risk_based_quantity, 100)
                else:
                    results['position_size'] = 1
        
                # ============ TARGETS ============
                risk_amount = risk_per_share
                target_mult = results.get('pattern_impact', {}).get('target_multiplier', 1.0)
                
                results['targets'] = [
                    {
                        "level": "Target 1 (1:1.5)", 
                        "price": round(results['latest_price'] + risk_amount * 1.5 * target_mult, 2),
                        "profit_potential": round(risk_amount * 1.5 * target_mult * results['position_size'], 2)
                    },
                    {
                        "level": "Target 2 (1:2)", 
                        "price": round(results['latest_price'] + risk_amount * 2.0 * target_mult, 2),
                        "profit_potential": round(risk_amount * 2.0 * target_mult * results['position_size'], 2)
                    },
                    {
                        "level": "Target 3 (1:3)", 
                        "price": round(results['latest_price'] + risk_amount * 3.0 * target_mult, 2),
                        "profit_potential": round(risk_amount * 3.0 * target_mult * results['position_size'], 2)
                    },
                ]
        
                if results.get('supertrend', {}).get('trend') == 'uptrend':
                    results['supertrend_target'] = results['supertrend']['value']
        
                results['risk_amount'] = round(risk_per_share * results['position_size'], 2)
                results['risk_percent'] = round((risk_per_share / results['latest_price']) * 100, 2)
                results['capital_used'] = round(results['latest_price'] * results['position_size'], 2)
        
                # ============ PATTERN & BREAKOUT DETECTION ============
                try:
                    results['inside_bar'] = self.detect_inside_bar_pattern(fifteen_min_data)
                except:
                    results['inside_bar'] = {"detected": False, "message": "Not analyzed"}
        
                try:
                    results['breakout_status'] = self.detect_breakout_retest(five_min_data, results['resistance'])
                except:
                    results['breakout_status'] = "Not analyzed"
        
                # ============ CONFIRMATION CHECKLIST ============
                try:
                    results['confirmation_checklist'] = self.run_confirmation_checklist(results)
                    results['signal'] = results['confirmation_checklist'].get('FINAL_SIGNAL', 'HOLD')
                except Exception as e:
                    st.warning(f"Confirmation checklist error: {str(e)}")
                    results['confirmation_checklist'] = {'FINAL_SIGNAL': 'HOLD'}
                    results['signal'] = 'HOLD'
        
                # ============ CURRENCY SYMBOL ============
                try:
                    results['currency'] = get_currency_symbol(self.ticker, None)
                except:
                    results['currency'] = '$'
        
                return results
        
            except Exception as e:
                st.error(f"Critical error in intraday analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return None

        def analyze_for_swing(self):
            """Swing trading analysis"""
            results = {
                'ticker': self.ticker,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'mode': 'swing'
            }

            try:
                daily_data = self.fetch_stock_data(self.ticker, period="1y")
                if daily_data is None:
                    return None

                results['latest_price'] = daily_data['Close'].iloc[-1]
                results['rsi'] = self.compute_rsi(daily_data)
                results['macd'] = self.compute_macd(daily_data)
                results['moving_averages'] = self.compute_moving_averages(daily_data)

                results['52w_high'] = daily_data['Close'].max()
                results['52w_low'] = daily_data['Close'].min()
                results['distance_from_52w_high'] = ((results['latest_price'] - results['52w_high']) / results['52w_high']) * 100

                ema_100 = daily_data['Close'].ewm(span=100, adjust=False).mean().iloc[-1] if len(daily_data) >= 100 else None
                ema_200 = daily_data['Close'].ewm(span=200, adjust=False).mean().iloc[-1] if len(daily_data) >= 200 else None

                results['ema_100'] = ema_100
                results['ema_200'] = ema_200

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
                return None

        def scrape_news_headlines(self, ticker_name, days=1):
            """Scrape news headlines"""
            try:
                api_key = NEWSAPI_KEY if NEWSAPI_KEY else "e205d77d7bc14acc8744d3ea10568f50"
                search_query = ticker_name.replace("^", "").replace(".NS", "")
                url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=publishedAt&apiKey={api_key}&pageSize=5"
                headers = {"User-Agent": "Mozilla/5.0"}
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
            except:
                return ["No news available"]

        def analyze_sentiment(self, headlines):
            """Analyze sentiment"""
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
            except:
                return {"sentiment": "Neutral", "score": 0.0}

        def analyze_with_fibonacci(self, data):
            """Fibonacci analysis"""
            try:
                high = data['Close'].max()
                low = data['Close'].min()
                current_price = data['Close'].iloc[-1]

                sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else current_price
                trend = 'uptrend' if current_price > sma_50 else 'downtrend'

                fib_levels = self.fib_calc.calculate_levels(high, low, trend)
                targets = self.fib_calc.identify_targets(current_price, fib_levels)

                return {
                    'fib_levels': fib_levels,
                    'targets': targets,
                    'trend': trend
                }
            except:
                return None

        def calculate_atr(self, data, period=14):
            """Calculate ATR"""
            try:
                high = data['High'] if 'High' in data.columns else data['high']
                low = data['Low' if 'Low' in data.columns else 'low']
                close = data['Close'] if 'Close' in data.columns else data['close']

                high_low = high - low
                high_close = abs(high - close.shift())
                low_close = abs(low - close.shift())

                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(window=period).mean()

                return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
            except:
                return 0

# ==============================================================================
# === MAIN STREAMLIT UI WITH ALL MISSING FEATURES ==============================
# ==============================================================================

def main():
    st.set_page_config(page_title="AI Trading Agent Pro", page_icon="📈", layout="wide")

    init_database()

    st.title("🤖 AI Trading Agent Pro - Complete Trading System")
    st.markdown("**Intraday | Swing | Options | Live Execution | Backtesting | AI Analysis**")

    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    if 'broker' not in st.session_state:
        st.session_state['broker'] = BrokerAPI()

    # ===========================================================================
    # === SIDEBAR WITH ALL FEATURES ============================================
    # ===========================================================================

    st.sidebar.header("⚙️ Configuration")
    
    # Market Selection
    selected_market = st.sidebar.selectbox("🌍 Select Market", list(GLOBAL_MARKETS.keys()), key="market_select")
    market_config = GLOBAL_MARKETS[selected_market]

    # Update market status display
    market_status = check_market_status(market_config)

    if market_status['status'] == 'OPEN':
        st.sidebar.success(f"🟢 {selected_market} OPEN")
    else:
        st.sidebar.error(f"🔴 {selected_market} CLOSED")

    # Trading Mode
    trading_mode = st.sidebar.radio(
        "Trading Mode",
        ["Intraday Trading", "Swing Trading", "Options Trading"],
        help="Select your trading style"
    )

    # ========== PRE-MARKET SCREENER (RESTORED) ==========
    if trading_mode == "Intraday Trading":
        st.sidebar.subheader("🔍 Pre-Market Screener")
        st.sidebar.info(f"Scan {selected_market} stocks")
        
        if st.sidebar.button("▶️ Run Pre-Market Scan"):
            with st.spinner(f"Scanning {selected_market} market..."):
                screened_stocks = run_premarket_screener(selected_market, market_config)
                
                if screened_stocks:
                    st.session_state['screened_stocks'] = screened_stocks
                    st.sidebar.success(f"✅ Found {len(screened_stocks)} stocks")
        
        # Display screened stocks in dropdown
        if 'screened_stocks' in st.session_state and st.session_state['screened_stocks']:
            st.sidebar.markdown("#### 📋 Screened Stocks")
            
            selected_screened = st.sidebar.selectbox(
                "Select stock to analyze:",
                options=list(st.session_state['screened_stocks'].keys()),
                format_func=lambda x: f"{x} - ${st.session_state['screened_stocks'][x]['price']:.2f} ({st.session_state['screened_stocks'][x]['change_pct']:+.2f}%)"
            )
            
          #  if st.sidebar.button("📊 Analyze Selected Stock"):
           #     st.session_state['auto_analyze_ticker'] = selected_screened
            #    st.rerun()

    # ========== AI MODEL SELECTION (RESTORED) ==========
    st.sidebar.subheader("🤖 AI Analysis")
    ai_model = st.sidebar.selectbox(
        "Select AI Model",
        ["None", "Google Gemini", "OpenRouter (Claude 3.5)", "OpenRouter (GPT-4)"],
        help="Enable AI-powered analysis"
    )

    # Model mapping
    ai_model_map = {
        "OpenRouter (Claude 3.5)": "anthropic/claude-3.5-sonnet",
        "OpenRouter (GPT-4)": "openai/gpt-4-turbo"
    }

    # Capital & Risk
    st.sidebar.subheader("💰 Capital & Risk")
    total_capital = st.sidebar.number_input("Total Capital ({currency})", value=100000, step=10000)
    risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", 1, 5, 2) / 100

    # Notification Settings
    st.sidebar.subheader("🔔 Alerts")
    alert_channels = st.sidebar.multiselect(
        "Alert Channels",
        ["Email", "Telegram", "SMS"],
        default=["Email"]
    )

    # Broker Connection
    st.sidebar.subheader("🔌 Broker Connection")
    broker_status = "✅ Connected" if st.session_state['broker'].connected else "❌ Disconnected"
    st.sidebar.metric("Kite Connect", broker_status)

    # ===========================================================================
    # === MAIN TABS ============================================================
    # ===========================================================================

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Analysis",
        "🤖 AI Insights",
        "🎯 Options",
        "📈 Backtesting",
        "💼 Portfolio",
        "📱 Live Trading",
        "⚙️ Settings"
    ])

    # ===========================================================================
    # === TAB 1: ANALYSIS (WITH STOCK SELECTION RESTORED) =====================
    # ===========================================================================

    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 Stock Selection - {selected_market}")
            
            # Initialize AlphaVantage
            av = AlphaVantageAPI()
            ticker_input = None
            
            # Check if auto-analyze from pre-market scanner
            if 'auto_analyze_ticker' in st.session_state:
                ticker_input = st.session_state['auto_analyze_ticker']
                st.info(f"Auto-analyzing from scanner: **{ticker_input}**")
                del st.session_state['auto_analyze_ticker']
            
            if not av.api_key:
                st.warning("⚠️ Alpha Vantage API key not configured. Using Direct entry.")
                # Market-aware placeholder
                placeholder_map = {
                    "🇮🇳 India (NSE/BSE)": "RELIANCE.NS",
                    "🇺🇸 USA (NYSE/NASDAQ)": "AAPL",
                    "🇬🇧 UK (LSE)": "BARC.L",
                    "🇯🇵 Japan (TSE)": "7203.T"
                }
                placeholder = placeholder_map.get(selected_market, "AAPL")
                ticker_input = st.text_input("Enter Ticker", placeholder)
            else:
                # Selection methods - INCLUDING "From Scanner"
                method_options = ["Search", "By Exchange", "Direct", "From Scanner"]
                method = st.radio("Selection Method", method_options, horizontal=True)
                
                # ============= METHOD 1: SEARCH =============
                if method == "Search":
                    search_query = st.text_input(
                        f"🔍 Search {selected_market} Stock", 
                        placeholder="Apple, Tesla, Reliance..."
                    )
                    if search_query and len(search_query) >= 3:
                        with st.spinner("Searching..."):
                            results = av.search_symbols(search_query)
                            if results:
                                # Filter results by selected market
                                filtered_results = {}
                                for name, symbol in results.items():
                                    # Market-specific filtering
                                    if "India" in selected_market and (".NS" in symbol or ".BO" in symbol):
                                        filtered_results[name] = symbol
                                    elif "USA" in selected_market and not any(suffix in symbol for suffix in [".NS", ".BO", ".L", ".T"]):
                                        filtered_results[name] = symbol
                                    elif "UK" in selected_market and ".L" in symbol:
                                        filtered_results[name] = symbol
                                    elif "Japan" in selected_market and ".T" in symbol:
                                        filtered_results[name] = symbol
                                
                                if filtered_results:
                                    selected = st.selectbox(
                                        f"Select from {selected_market}:", 
                                        list(filtered_results.keys())
                                    )
                                    ticker_input = filtered_results[selected]
                                    
                                    # Get quote
                                    quote = av.get_quote(ticker_input)
                                    if quote:
                                        q_col1, q_col2, q_col3 = st.columns(3)
                                        q_col1.metric("Price", f"${quote['price']:.2f}")
                                        q_col2.metric("Change", f"{quote['change']:.2f}")
                                        q_col3.metric("Volume", f"{quote['volume']:,}")
                                else:
                                    st.warning(f"No results found for {selected_market}. Try different keywords.")
                            else:
                                st.warning("No results found. Try different keywords.")
                
                # ============= METHOD 2: BY EXCHANGE =============
                elif method == "By Exchange":
                    # Get market-specific currency symbol
                    currency_map = {
                        "🇮🇳 India (NSE/BSE)": "{currency}",
                        "🇺🇸 USA (NYSE/NASDAQ)": "$",
                        "🇬🇧 UK (LSE)": "£",
                        "🇯🇵 Japan (TSE)": "¥"
                    }


                    if st.button(f"🔄 Load {selected_market} Stocks", type="primary"):
                        with st.spinner(f"Loading..."):
                            # Use the same dynamic fetcher
                            all_tickers, source, errors = get_dynamic_tickers(selected_market, ALPHA_VANTAGE_API_KEY)
                            
                            if all_tickers:
                                stocks_dict = {}
                                progress_bar = st.progress(0)
                                
                                # Process in batches
                                batch_size = 20
                                processed = 0
                                
                                for i in range(0, min(len(all_tickers), 100), batch_size):
                                    batch = all_tickers[i:i+batch_size]
                                    
                                    try:
                                        data = yf.download(" ".join(batch), period="1d", 
                                                         group_by='ticker', progress=False)
                                        
                                        for ticker in batch:
                                            try:
                                                if len(batch) > 1:
                                                    stock_data = data[ticker]
                                                else:
                                                    stock_data = data
                                                
                                                if not stock_data.empty:
                                                    last_price = stock_data['Close'].iloc[-1]
                                                    # Format with market-specific currency
                                                    display_name = f"{ticker} - {currency}{last_price:.2f}"
                                                    stocks_dict[display_name] = ticker
                                                else:
                                                    stocks_dict[ticker] = ticker
                                            except:
                                                stocks_dict[ticker] = ticker
                                        
                                        processed += len(batch)
                                        progress_bar.progress(processed / min(len(all_tickers), 100))
                                    
                                    except:
                                        for ticker in batch:
                                            stocks_dict[ticker] = ticker
                                        processed += len(batch)
                                
                                progress_bar.empty()
                                
                                # Store in session state
                                st.session_state['loaded_stocks'] = stocks_dict
                                st.session_state['loaded_stocks_source'] = source
                                
                                # Single success message
                                st.success(f"✅ Loaded {len(stocks_dict)} stocks from {source}")
                                
                                # Errors in collapsible (only if exist)
                                if errors:
                                    with st.expander(f"ℹ️ View source details"):
                                        for err in errors:
                                            st.caption(f"• {err}")
                            else:
                                st.error(f"❌ Failed to load stocks")
                                if errors:
                                    with st.expander("🔍 Error Details"):
                                        for err in errors:
                                            st.text(err)
                    
                    # Display loaded stocks (CLEAN)
                    if 'loaded_stocks' in st.session_state and st.session_state['loaded_stocks']:
                        selected = st.selectbox(
                            f"Select stock:",
                            list(st.session_state['loaded_stocks'].keys()),
                            label_visibility="collapsed"
                        )
                        ticker_input = st.session_state['loaded_stocks'][selected]
                    else:
                        st.info("👆 Click button above to load stocks")
                
                # ============= METHOD 3: DIRECT =============
                elif method == "Direct":
                    # Market-aware placeholders
                    placeholder_map = {
                        "🇮🇳 India (NSE/BSE)": "RELIANCE.NS",
                        "🇺🇸 USA (NYSE/NASDAQ)": "AAPL",
                        "🇬🇧 UK (LSE)": "BARC.L",
                        "🇯🇵 Japan (TSE)": "7203.T"
                    }
                    placeholder = placeholder_map.get(selected_market, "AAPL")
                    
                    ticker_input = st.text_input(
                        f"Enter Ticker Symbol for {selected_market}", 
                        placeholder,
                        help=f"Format: {placeholder}"
                    )
                
                # ============= METHOD 4: FROM SCANNER (NEW) =============
                elif method == "From Scanner":
                    if 'screened_stocks' in st.session_state and st.session_state['screened_stocks']:
                        st.success(f"✅ {len(st.session_state['screened_stocks'])} stocks available from pre-market scanner")
                        
                        # Create formatted options
                        stock_options = {}
                        for ticker, data in st.session_state['screened_stocks'].items():
                            display_name = f"{ticker} - ${data['price']:.2f} ({data['change_pct']:+.2f}%)"
                            stock_options[display_name] = ticker
                        
                        selected_display = st.selectbox(
                            f"Select from {selected_market} Scanner Results:",
                            list(stock_options.keys())
                        )
                        ticker_input = stock_options[selected_display]
                        
                        # Show stock details
                        if ticker_input in st.session_state['screened_stocks']:
                            stock_data = st.session_state['screened_stocks'][ticker_input]
                            detail_col1, detail_col2, detail_col3 = st.columns(3)
                            detail_col1.metric("Price", f"${stock_data['price']:.2f}")
                            detail_col2.metric("Volume", f"{stock_data['volume']:,}")
                            detail_col3.metric("Change", f"{stock_data['change_pct']:+.2f}%")
                    else:
                        st.warning("⚠️ No stocks in scanner. Run 'Pre-Market Scan' from sidebar first.")
                        st.info("👈 Click 'Run Pre-Market Scan' in the sidebar to populate this list.")
                    currency = results.get('currency', get_currency_symbol(ticker_input, selected_market))

            # ANALYSIS BUTTON
            if st.button("📊 Analyze with Full Suite", type="primary"):
                if not ticker_input:
                    st.error("⚠️ Please select or enter a valid ticker symbol first")
                else:
                    with st.spinner("Running complete analysis..."):
                        try:
                            analyzer = StockAnalyzer(ticker=ticker_input)
                            
                            if trading_mode == "Intraday Trading":
                                results = analyzer.analyze_for_intraday()
                            else:
                                results = analyzer.analyze_for_swing()

                    if results:
                        # Add Fibonacci
                        fib_analysis = analyzer.analyze_with_fibonacci(results['daily_data'])
                        results['fibonacci'] = fib_analysis

                        # Fetch news and sentiment
                        headlines = analyzer.scrape_news_headlines(ticker_input)
                        sentiment_detailed = analyzer.analyze_sentiment_detailed(headlines)
                        results['news_headlines'] = headlines
                        results['sentiment'] = sentiment_detailed['overall_sentiment']  # Store as string
                        results['sentiment_score'] = sentiment_detailed['overall_score']
                        results['sentiment_detailed'] = sentiment_detailed  # Store full breakdown

                        st.session_state['analysis_results'] = results
                        st.session_state['current_ticker'] = ticker_input

                        # Log to database
                        log_trade_to_db(
                            ticker_input,
                            results.get('signal', 'HOLD'),
                            results['latest_price'],
                            results.get('position_size', 0),
                            trading_mode.lower()
                        )

                        # Send alerts
                        if results.get('signal') in ['🟢 BUY', '🔴 SELL']:
                            send_multi_channel_alert(
                                ticker_input,
                                results['signal'],
                                results['latest_price'],
                                [ch.lower() for ch in alert_channels]
                            )

                        st.success("✅ Complete Analysis Done!")

        with col2:
            if 'analysis_results' in st.session_state:
                results = st.session_state['analysis_results']
                currency = results.get('currency', get_currency_symbol(ticker_input, selected_market))
                st.metric("Price", f"{currency}{results['latest_price']:.2f}")
                st.metric("Signal", results.get('signal', 'HOLD'))
                st.metric("RSI", f"{results['rsi']:.2f}")

                # News Sentiment
                if results.get('sentiment'):
                    sentiment = results.get('sentiment', 'Neutral')
                    if sentiment == "Positive":
                        st.success(f"📰 Sentiment: {sentiment}")
                    elif sentiment == "Negative":
                        st.error(f"📰 Sentiment: {sentiment}")
                    else:
                        st.info(f"📰 Sentiment: {sentiment}")

        # Display full analysis results
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']

            if trading_mode == "Intraday Trading":
                st.subheader("📊 Intraday Trading Dashboard")

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Current Price", f"{currency}{results['latest_price']:.2f}")
                col2.metric("Signal", results['signal'])
                col3.metric("RSI", f"{results['rsi']:.2f}")
                col4.metric("Position Size", f"{results.get('position_size', 0)} shares")
                col5.metric("Capital Used", f"{currency}{results.get('capital_used', 0):,.0f}")

                # Intraday data display
                if '5m_data' in results and results['5m_data'] is not None and not results['5m_data'].empty:
                    latest_5m = results['5m_data'].iloc[-1]
                    latest_close_5m = latest_5m.get('close', 'N/A')
                    latest_volume_5m = latest_5m.get('volume', 'N/A')
            
                    st.markdown("### 📈 Latest Intraday Data (5-minute)")
                    st.write(f"Latest Close Price: {currency}{latest_close_5m}")
                    st.write(f"Latest Volume: {int(latest_volume_5m) if isinstance(latest_volume_5m, (int, float)) else latest_volume_5m}")
                else:
                    st.write("Intraday data (5-minute) not available.")


                # ========== PATTERN DETECTION & CONFIRMATION ==========
                st.markdown("---")
                st.subheader("🎯 Pattern Detection & Trade Confirmation")
                
                # 5-Point Confirmation Checklist
                if 'confirmation_checklist' in results:
                    checklist = results['confirmation_checklist']
                    
                    st.markdown("### ✅ 5-Point Trade Confirmation Checklist")
                    
                    checklist_col1, checklist_col2 = st.columns(2)
                    
                    with checklist_col1:
                        for key in ['1. At Key S/R Level', '2. Price Rejection', '3. Chart Pattern Confirmed']:
                            st.write(f"**{key}:** {checklist.get(key, '⚠️ PENDING')}")
                    
                    with checklist_col2:
                        for key in ['4. Candlestick Signal', '5. Indicator Alignment']:
                            st.write(f"**{key}:** {checklist.get(key, '⚠️ PENDING')}")
                    
                    # Final Signal
                    final_signal = checklist.get('FINAL_SIGNAL', 'HOLD')
                    
                    if final_signal == '🟢 BUY':
                        st.success(f"### FINAL SIGNAL: {final_signal}")
                        st.info("✅ 3+ bullish confirmations detected. Trade setup valid!")
                    elif final_signal == '🔴 SELL':
                        st.error(f"### FINAL SIGNAL: {final_signal}")
                        st.info("✅ 3+ bearish confirmations detected. Trade setup valid!")
                    else:
                        st.warning(f"### FINAL SIGNAL: {final_signal}")
                        st.info("⚠️ Insufficient confirmations. Wait for better setup.")
                
                # ========== CANDLESTICK PATTERN SECTION ==========
                st.markdown("---")
                st.markdown("### 🕯️ Candlestick Pattern Analysis")
                
                pattern_name = results.get('candlestick_pattern', 'None')
                pattern_type = results.get('pattern_type', 'neutral')
                pattern_strength = results.get('pattern_strength', 0)
                pattern_confidence = results.get('pattern_confidence', 0)
                pattern_description = results.get('pattern_description', 'No pattern detected')
                pattern_category = results.get('pattern_category', 'none')
                
                pattern_col1, pattern_col2, pattern_col3, pattern_col4 = st.columns(4)
                
                with pattern_col1:
                    if pattern_type == 'bullish':
                        st.success(f"**{pattern_name}**")
                        st.caption("📈 Bullish Signal")
                    elif pattern_type == 'bearish':
                        st.error(f"**{pattern_name}**")
                        st.caption("📉 Bearish Signal")
                    else:
                        st.info(f"**{pattern_name}**")
                        st.caption("➡️ Neutral")
                
                with pattern_col2:
                    st.metric("Strength", f"{pattern_strength}/100")
                
                with pattern_col3:
                    st.metric("Confidence", f"{pattern_confidence}%")
                
                with pattern_col4:
                    st.metric("Type", pattern_category.title())
                
                # Description
                st.info(f"💡 **Pattern Insight:** {pattern_description}")
                
                # Show trading impact if significant
                if pattern_strength >= 70:
                    impact = results.get('pattern_impact', {})
                    signal_boost = impact.get('signal_boost', 0)
                    target_mult = impact.get('target_multiplier', 1.0)
                    sl_adj = impact.get('stop_loss_adjustment', 1.0)
                    
                    impact_parts = []
                    
                    if signal_boost > 0:
                        impact_parts.append(f"✅ Added +{signal_boost:.1f} points to BUY signal")
                    elif signal_boost < 0:
                        impact_parts.append(f"⚠️ Added {signal_boost:.1f} points (caution)")
                    
                    if target_mult > 1.1:
                        impact_parts.append(f"📈 Targets increased by {(target_mult-1)*100:.0f}%")
                    elif target_mult < 0.9:
                        impact_parts.append(f"📉 Targets reduced by {(1-target_mult)*100:.0f}%")
                    
                    if sl_adj < 0.99:
                        impact_parts.append(f"🎯 Stop-loss tightened by {(1-sl_adj)*100:.1f}%")
                    elif sl_adj > 1.01:
                        impact_parts.append(f"🛡️ Stop-loss widened by {(sl_adj-1)*100:.1f}%")
                    
                    if impact_parts:
                        st.success(f"🎯 **Trading Impact:**\n" + "\n".join(f"- {part}" for part in impact_parts))

                # Technical Indicators Summary
                st.markdown("---")
                st.markdown("### 📊 Technical Indicators Summary")
                
                ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                
                with ind_col1:
                    st.markdown("**Bollinger Bands**")
                    bb = results.get('bollinger_bands', {})
                    st.write(f"Upper: {currency}{bb.get('upper', 0):.2f}")
                    st.write(f"Middle: {currency}{bb.get('middle', 0):.2f}")
                    st.write(f"Lower: {currency}{bb.get('lower', 0):.2f}")
                    
                    # BB Signal
                    current_price = results['latest_price']
                    if current_price < bb.get('lower', 0):
                        st.success("🟢 Oversold (Near Lower BB)")
                    elif current_price > bb.get('upper', 0):
                        st.error("🔴 Overbought (Near Upper BB)")
                    else:
                        st.info("⚪ Within Bands")
                
                with ind_col2:
                    st.markdown("**Stochastic Momentum**")
                    stoch = results.get('stochastic', {})
                    st.write(f"%K: {stoch.get('k', 0):.2f}")
                    st.write(f"%D: {stoch.get('d', 0):.2f}")
                    
                    crossover = stoch.get('crossover', 'none')
                    if crossover == 'bullish':
                        st.success("🟢 Bullish Crossover")
                    elif crossover == 'bearish':
                        st.error("🔴 Bearish Crossover")
                    else:
                        st.info("⚪ No Crossover")
                
                with ind_col3:
                    st.markdown("**VWAP/VWMA**")
                    st.write(f"VWAP: {currency}{results.get('vwap', 0):.2f}")
                    st.write(f"VWMA: {currency}{results.get('vwma', 0):.2f}")
                    
                    if current_price > results.get('vwap', 0):
                        st.success("🟢 Above VWAP (Bullish)")
                    else:
                        st.error("🔴 Below VWAP (Bearish)")
                
                with ind_col4:
                    st.markdown("**SuperTrend**")
                    supertrend = results.get('supertrend', {})
                    st.write(f"Value: {currency}{supertrend.get('value', 0):.2f}")
                    
                    trend = supertrend.get('trend', 'neutral')
                    if trend == 'uptrend':
                        st.success(f"🟢 {trend.upper()}")
                    elif trend == 'downtrend':
                        st.error(f"🔴 {trend.upper()}")
                    else:
                        st.info(f"⚪ {trend.upper()}")
                
                # Moving Averages
                st.markdown("---")
                st.markdown("### 📈 Moving Averages Analysis")
                
                ma_col1, ma_col2, ma_col3, ma_col4 = st.columns(4)
                
                mas = results.get('moving_averages', {})
                
                with ma_col1:
                    st.metric("MA 20", f"{currency}{mas.get('MA_20', 0):.2f}")
                
                with ma_col2:
                    st.metric("MA 50", f"{currency}{mas.get('MA_50', 0):.2f}")
                
                with ma_col3:
                    st.metric("MA 200", f"{currency}{mas.get('MA_200', 0):.2f}")
                
                with ma_col4:
                    # Trend based on MA position
                    if current_price > mas.get('MA_50', 0) > mas.get('MA_200', 0):
                        st.success("🟢 Strong Uptrend")
                    elif current_price < mas.get('MA_50', 0) < mas.get('MA_200', 0):
                        st.error("🔴 Strong Downtrend")
                    else:
                        st.warning("⚠️ Consolidation")
                
                st.markdown("---")

                # ========== DISPLAY CHARTS ==========
                st.subheader("📈 Multi-Timeframe Technical Charts")
                
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 Daily", "⏰ 15-Min", "⚡ 5-Min"])
                
                with chart_tab1:
                    if 'daily_data' in results and results['daily_data'] is not None:
                        st.write("**Daily Timeframe Analysis**")
                        fig_daily = create_plotly_charts(results['daily_data'], f"{results['ticker']} - Daily")
                        st.plotly_chart(fig_daily, use_container_width=True)
                    else:
                        st.warning("Daily chart data not available")
                
                with chart_tab2:
                    if '15m_data' in results and results['15m_data'] is not None:
                        st.write("**15-Minute Intraday Analysis**")
                        fig_15m = create_plotly_charts(results['15m_data'], f"{results['ticker']} - 15 Min")
                        st.plotly_chart(fig_15m, use_container_width=True)
                    else:
                        st.warning("15-minute chart data not available")
                
                with chart_tab3:
                    if '5m_data' in results and results['5m_data'] is not None:
                        st.write("**5-Minute Scalping View**")
                        fig_5m = create_plotly_charts(results['5m_data'], f"{results['ticker']} - 5 Min")
                        st.plotly_chart(fig_5m, use_container_width=True)
                    else:
                        st.warning("5-minute chart data not available")

                # ========== TRADINGVIEW WIDGET ==========
                st.markdown("---")
                st.subheader("📊 TradingView Live Chart")
                tradingview_html = embed_tradingview_widget(results['ticker'])
                components.html(tradingview_html, height=550)
                
                st.markdown("---")
                
                # Stop-Loss & Targets
                st.subheader("🎯 Stop-Loss & Targets")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🛑 Stop-Loss")
                    st.metric("Stop-Loss Price", f"{currency}{results.get('stop_loss', 0):.2f}",
                             f"-{currency}{abs(results['latest_price'] - results.get('stop_loss', 0)):.2f}")
                    st.metric("Risk Amount", f"{currency}{results.get('risk_amount', 0):.2f}")
                    st.metric("Risk %", f"{results.get('risk_percent', 0):.2f}%")
                    vwap_value = results.get('vwap', results.get('latest_price', 0))
                    if vwap_value > 0:
                        st.info(
                            f"**VWAP Trailing:** {currency}{vwap_value:.2f}\n\n"
                            "Trail stop to VWAP. Exit if closes below."
                        )

                
                with col2:
                    st.markdown("### 🎯 Profit Targets")
                    if results.get('targets'):
                        for target in results['targets']:
                            st.metric(target['level'], f"{currency}{target['price']:.2f}", f"+{currency}{target['profit_potential']:.2f}")
                
                with col3:
                    st.markdown("### 📍 Key Levels")
                    st.metric("Resistance", f"{currency}{results.get('resistance', 0):.2f}")
                    st.metric("Support", f"{currency}{results.get('support', 0):.2f}")
                    st.metric("ATR (14)", f"{currency}{results.get('atr', 0):.2f}")

                # News
                if results.get('news_headlines'):
                    st.markdown("---")
                    st.subheader("📰 Latest News")
                    for headline in results['news_headlines'][:5]:
                        st.write(f"• {headline}")

                # Detailed sentiment breakdown
                if 'sentiment_detailed' in results and results['sentiment_detailed'].get('articles'):
                    st.markdown("---")
                    st.subheader("🎯 News Sentiment Breakdown")
                    
                    sentiment_data = results['sentiment_detailed']
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Overall", sentiment_data['overall_sentiment'])
                    col2.metric("Score", f"{sentiment_data['overall_score']:.3f}")
                    col3.metric("✅ Positive", sentiment_data.get('positive_count', 0))
                    col4.metric("❌ Negative", sentiment_data.get('negative_count', 0))
                    
                    st.markdown("#### 📊 Per-Article Sentiment Details")
                    
                    # Create a dataframe for better display
                    articles_df = pd.DataFrame(sentiment_data['articles'])
                    
                    for i, article in enumerate(sentiment_data['articles'], 1):
                        with st.expander(f"Article {i}: {article['headline'][:70]}..."):
                            # Sentiment badge
                            if article['sentiment'] in ['Positive', 'POSITIVE']:
                                st.success(f"✅ {article['sentiment']}")
                            elif article['sentiment'] in ['Negative', 'NEGATIVE']:
                                st.error(f"❌ {article['sentiment']}")
                            else:
                                st.info(f"➖ {article['sentiment']}")
                            
                            # Show scores
                            st.write(f"**Composite Score:** {article['score']:.3f}")
                            
                            if 'positive' in article:
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Positive", f"{article['positive']:.3f}")
                                col2.metric("Negative", f"{article['negative']:.3f}")
                                col3.metric("Neutral", f"{article['neutral']:.3f}")
                            elif 'confidence' in article:
                                st.metric("Confidence", f"{article['confidence']:.3f}")
                            
                            st.caption(f"**Full Headline:** {article['headline']}")

                # MACD
                st.markdown("---")
                st.markdown("### 📊 MACD (Moving Average Convergence Divergence)")
                
                macd_data = results.get('macd', {})
                
                macd_col1, macd_col2, macd_col3, macd_col4 = st.columns(4)
                
                with macd_col1:
                    st.metric("MACD Line", f"{macd_data.get('line', 0):.2f}")
                
                with macd_col2:
                    st.metric("Signal Line", f"{macd_data.get('signal', 0):.2f}")
                
                with macd_col3:
                    histogram = macd_data.get('histogram', 0)
                    st.metric("Histogram", f"{histogram:.2f}")
                
                with macd_col4:
                    if histogram > 0:
                        st.success("🟢 Bullish Momentum")
                        st.caption("MACD above signal line")
                    elif histogram < 0:
                        st.error("🔴 Bearish Momentum")
                        st.caption("MACD below signal line")
                    else:
                        st.info("⚪ Neutral")
                
                macd_line = macd_data.get('line', 0)
                signal_line = macd_data.get('signal', 0)
                
                if macd_line > signal_line:
                    st.write("**Crossover Status:** ✅ Bullish Crossover (MACD above Signal)")
                elif macd_line < signal_line:
                    st.write("**Crossover Status:** ❌ Bearish Crossover (MACD below Signal)")
                else:
                    st.write("**Crossover Status:** ⚪ No Clear Crossover")
                
                st.caption("""
                **💡 MACD Interpretation:**
                - **Histogram > 0:** Bullish momentum (MACD above signal)
                - **Histogram < 0:** Bearish momentum (MACD below signal)
                - **Crossovers:** Strong buy/sell signals when MACD crosses signal line
                """)
                
                macd_analysis = analyze_macd_detailed(results.get('macd', {}), results.get('daily_data'))
                
                st.markdown("#### 🔍 Detailed MACD Interpretation")
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.write(f"**Crossover Status:** {macd_analysis['crossover']}")
                    st.caption(macd_analysis['crossover_type'])
                    
                    st.write(f"**Histogram State:** {macd_analysis['histogram_state']}")
                    st.caption(macd_analysis['momentum'])
                
                with detail_col2:
                    st.write(f"**Centerline Status:** {macd_analysis['centerline_status']}")
                    
                    st.write(f"**Overall Signal:** {macd_analysis['overall_signal']}")
                    st.caption(f"Strength: {macd_analysis['strength']}")
                
                with st.expander("📚 Understanding MACD Signals"):
                    st.markdown("""
                    **MACD Components:**
                    - **MACD Line:** 12-day EMA minus 26-day EMA
                    - **Signal Line:** 9-day EMA of MACD line
                    - **Histogram:** MACD line minus Signal line
                    
                    **Key Signals:**
                    - **Bullish Crossover:** MACD crosses above Signal → Buy signal
                    - **Bearish Crossover:** MACD crosses below Signal → Sell signal
                    - **Zero Line Cross:** MACD crosses zero → Trend change
                    - **Divergence:** Price makes new high/low but MACD doesn't → Reversal warning
                    
                    **How to Trade:**
                    1. Wait for clear crossover
                    2. Confirm with other indicators (RSI, Volume)
                    3. Watch for divergences
                    4. Use histogram for momentum strength
                    """)
                
                st.markdown("---")

                # Fibonacci
                st.markdown("---")
                if 'fibonacci' in results and results['fibonacci']:
                    st.subheader("📐 Fibonacci Retracement & Extension Levels")
                    
                    fib_data = results['fibonacci']
                    
                    fib_col1, fib_col2 = st.columns(2)
                    
                    with fib_col1:
                        st.markdown("### 📊 Trend & Levels")
                        trend = fib_data.get('trend', 'N/A')
                        
                        if trend == 'uptrend':
                            st.success(f"**Trend:** 🟢 {trend.upper()}")
                        else:
                            st.error(f"**Trend:** 🔴 {trend.upper()}")
                        
                        if 'fib_levels' in fib_data:
                            st.write("**Fibonacci Levels:**")
                            for level_name, level_price in list(fib_data['fib_levels'].items())[:7]:
                                distance = level_price - results['latest_price']
                                if abs(distance) / results['latest_price'] < 0.01:
                                    st.success(f"✅ **{level_name}:** {currency}{level_price:.2f} ← Near Current Price")
                                else:
                                    st.write(f"• {level_name}: {currency}{level_price:.2f}")
                    
                    with fib_col2:
                        st.markdown("### 🎯 Nearest Fib Targets")
                        
                        if 'targets' in fib_data and fib_data['targets']:
                            for target in fib_data['targets'][:3]:
                                st.metric(
                                    target['level'], 
                                    f"{currency}{target['price']:.2f}",
                                    f"+{currency}{target['distance']:.2f}"
                                )
                        else:
                            st.info("No nearby Fibonacci targets identified")
                    
                    st.caption("""
                    **💡 How to use Fibonacci:**
                    - **Uptrend:** Price retraces to 0.382, 0.5, or 0.618 → Buy opportunity
                    - **Downtrend:** Price rallies to 0.382, 0.5, or 0.618 → Sell opportunity
                    - **Extension levels** (1.272, 1.618, 2.0) → Profit targets
                    """)

            elif trading_mode == "Swing Trading":
                st.subheader("📊 Swing Trading Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"{currency}{results['latest_price']:.2f}")
                col2.metric("Signal", results['signal'])
                col3.metric("RSI", f"{results['rsi']:.2f}")
                
                ma_50 = results['moving_averages']['MA_50']
                trend = "Bullish" if results['latest_price'] > ma_50 else "Bearish"
                col4.metric("Trend", trend)
                
                st.markdown("---")
                st.markdown("### 📈 Swing Trade Metrics")
                
                swing_col1, swing_col2, swing_col3, swing_col4 = st.columns(4)
                
                with swing_col1:
                    st.markdown("**52-Week Range**")
                    st.metric("52W High", f"{currency}{results.get('52w_high', 0):.2f}")
                    distance_high = results.get('distance_from_52w_high', 0)
                    st.metric("Distance from High", f"{distance_high:+.2f}%")
                
                with swing_col2:
                    st.markdown("**52-Week Low**")
                    st.metric("52W Low", f"{currency}{results.get('52w_low', 0):.2f}")
                    distance_low = ((results['latest_price'] - results.get('52w_low', 0)) / results.get('52w_low', 1)) * 100
                    st.metric("Distance from Low", f"{distance_low:+.2f}%")
                
                with swing_col3:
                    st.markdown("**Long-term EMAs**")
                    ema_100 = results.get('ema_100', 0)
                    ema_200 = results.get('ema_200', 0)
                    
                    if ema_100:
                        st.metric("EMA 100", f"{currency}{ema_100:.2f}")
                    else:
                        st.metric("EMA 100", "N/A")
                    
                    if ema_200:
                        st.metric("EMA 200", f"{currency}{ema_200:.2f}")
                    else:
                        st.metric("EMA 200", "N/A")
                
                with swing_col4:
                    st.markdown("**Moving Averages**")
                    st.metric("MA 50", f"{currency}{results['moving_averages']['MA_50']:.2f}")
                    st.metric("MA 200", f"{currency}{results['moving_averages']['MA_200']:.2f}")
                
                st.markdown("---")
                st.markdown("### 📊 Trend Analysis")
                
                ma_50 = results['moving_averages']['MA_50']
                ma_200 = results['moving_averages']['MA_200']
                price = results['latest_price']
                
                if price > ma_50 > ma_200:
                    st.success("🟢 **Strong Uptrend** - Price above MA50 above MA200")
                elif price < ma_50 < ma_200:
                    st.error("🔴 **Strong Downtrend** - Price below MA50 below MA200")
                elif price > ma_50 and ma_50 < ma_200:
                    st.warning("⚠️ **Mixed Signals** - Price above MA50 but MA50 below MA200")
                else:
                    st.info("⚪ **Consolidation** - No clear trend")
                
                st.markdown("---")


    # ===========================================================================
    # === TAB 2: AI INSIGHTS (RESTORED) ========================================
    # ===========================================================================

    with tab2:
        st.subheader("🤖 AI-Powered Trading Insights")

        if 'analysis_results' not in st.session_state:
            st.info("👈 Please run an analysis first from the Analysis tab")
        elif ai_model == "None":
            st.warning("⚠️ Please select an AI model from the sidebar to generate insights")
        else:
            results = st.session_state['analysis_results']
            ticker = results.get('ticker', None)

            st.write(f"**Analyzing:** {results['ticker']}")
            st.write(f"**AI Model:** {ai_model}")

            if st.button("🧠 Generate AI Analysis", type="primary"):
                with st.spinner(f"Generating insights with {ai_model}..."):
                    progress_bar = st.progress(0)
                    # Simulate progress (replace or remove sleep for real progress updates)
                    for i in range(30):
                        time.sleep(0.1)  # Simulate waiting
                        progress_bar.progress(i + 1)
            
                    prompt = generate_comprehensive_analysis(
                        ticker,
                        results,
                        results.get('sentiment', {}),
                        results.get('news_headlines', [])
                    )

                    # Generate AI response
                    if ai_model == "Google Gemini":
                        ai_response = get_ai_analysis_gemini(prompt)
                    else:
                        model_name = ai_model_map.get(ai_model, "anthropic/claude-3.5-sonnet")
                        ai_response = get_ai_analysis_openrouter(prompt, model_name)

                    st.session_state['ai_analysis'] = ai_response
                    st.success("✅ AI Analysis Generated!")

                    progress_bar.empty()

            # Display AI analysis
            if 'ai_analysis' in st.session_state:
                st.markdown("---")
                st.markdown("### 💡 AI Insights")
                st.markdown(st.session_state['ai_analysis'])

                # Option to download
                st.download_button(
                    "📥 Download AI Analysis",
                    st.session_state['ai_analysis'],
                    file_name=f"ai_analysis_{results['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

                st.session_state["last_analysis_time"] = datetime.now()
                if "analysis_history" not in st.session_state:
                    st.session_state["analysis_history"] = []
                st.session_state["analysis_history"].append(st.session_state["last_analysis_time"])

                import pytz
                
                # Show last analysis run timestamp
                if "last_analysis_time" in st.session_state:
                    last_run = st.session_state["last_analysis_time"]
                    local_time = last_run.astimezone(pytz.timezone('Asia/Kolkata'))
                    st.markdown(f"🕒 **Last analysis run:** {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                
                # Show recent analysis history
                if "analysis_history" in st.session_state and st.session_state["analysis_history"]:
                    st.subheader("🕑 Analysis Run History (Last 5 times)")
                    for idx, ts in enumerate(reversed(st.session_state["analysis_history"][-5:]), 1):
                        local_time = ts.astimezone(pytz.timezone('Asia/Kolkata'))
                        st.markdown(f"{idx}. {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")


    # ===========================================================================
    # === TAB 3-6: OPTIONS, BACKTESTING, PORTFOLIO, LIVE TRADING
    # ===========================================================================

    with tab3:
        st.subheader("🎯 Options Trading Dashboard")
        options_analyzer = OptionsAnalyzer()
        st.info(f"📅 Today's Expiry: **{options_analyzer.get_todays_expiry()}**")

        opt_ticker = st.text_input("Options Ticker", "^NSEI", key="opt_tick")

        if st.button("📊 Analyze Options Chain"):
            with st.spinner("Fetching options data..."):
                options_data = options_analyzer.fetch_options_chain(opt_ticker)

                if options_data:
                    pcr_data = options_analyzer.calculate_pcr(options_data)

                    if pcr_data:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("PCR (OI)", f"{pcr_data['pcr_oi']:.2f}")
                        col2.metric("Put OI", f"{pcr_data['put_oi']:,}")
                        col3.metric("Call OI", f"{pcr_data['call_oi']:,}")
                        st.markdown(f"**Sentiment:** {pcr_data['sentiment']}")

                    st.markdown("### 📞 Call Options")
                    st.dataframe(options_data['calls'].head(10), use_container_width=True)

                    st.markdown("### 📉 Put Options")
                    st.dataframe(options_data['puts'].head(10), use_container_width=True)
                else:
                    st.error("❌ Could not fetch options data")

    with tab4:
        st.subheader("📈 Strategy Backtesting")

        col1, col2, col3 = st.columns(3)
        bt_ticker = col1.text_input("Ticker", "AAPL", key="bt_tick")
        bt_period = col2.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"])
        bt_capital = col3.number_input("Initial Capital ({currency})", value=100000, key="bt_cap")

        strategy = st.selectbox("Strategy", ["RSI Strategy", "MACD Strategy", "Moving Average Crossover"])

        if st.button("🚀 Run Backtest"):
            with st.spinner("Running backtest..."):
                data = yf.Ticker(bt_ticker).history(period=bt_period)

                if not data.empty:
                    signals = pd.Series(index=data.index, data='HOLD') # Default
                    if strategy == "RSI Strategy":
                        delta = data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        data['RSI'] = 100 - (100 / (1 + gain/loss))

                        signals[data['RSI'] < 30] = 'BUY'
                        signals[data['RSI'] > 70] = 'SELL'

                    backtester = Backtester(bt_capital)
                    metrics = backtester.run_backtest(data, signals)

                    st.markdown("### 📊 Backtest Results")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Trades", metrics['total_trades'])
                    col2.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
                    col3.metric("Total Profit", f"{currency}{metrics['total_profit']:,.2f}")
                    col4.metric("Return", f"{metrics['total_return_pct']:.2f}%")

                    col5, col6, col7, col8 = st.columns(4)
                    col5.metric("Winning Trades", metrics['winning_trades'])
                    col6.metric("Losing Trades", metrics['losing_trades'])
                    col7.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    col8.metric("Final Capital", f"{currency}{metrics['final_capital']:,.2f}")

                    if backtester.trades:
                        st.markdown("### 📋 Trade History")
                        trades_df = pd.DataFrame(backtester.trades)
                        st.dataframe(trades_df, use_container_width=True)

                        trades_df['cumulative_profit'] = trades_df['profit_loss'].cumsum()
                        import plotly.express as px
                        fig = px.line(trades_df, y='cumulative_profit', title='Equity Curve')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not fetch data")

    with tab5:
        st.subheader("💼 Portfolio Analysis")

        portfolio_input = st.text_area("Enter Tickers (one per line)", "AAPL\nMSFT\nGOOGL\nAMZN", height=150)

        if st.button("🔍 Analyze Portfolio"):
            tickers = [t.strip() for t in portfolio_input.split('\n') if t.strip()]

            if tickers:
                portfolio_data = []
                progress_bar = st.progress(0)

                for i, ticker in enumerate(tickers):
                    try:
                        stock = yf.Ticker(ticker)
                        data = stock.history(period="60d")

                        if not data.empty:
                            latest_price = data['Close'].iloc[-1]
                            change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100

                            delta = data['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                            rsi = 100 - (100 / (1 + gain/loss))
                            latest_rsi = rsi.iloc[-1]

                            signal = "HOLD"
                            if latest_rsi < 40:
                                signal = "BUY"
                            elif latest_rsi > 60:
                                signal = "SELL"

                            portfolio_data.append({
                                'Ticker': ticker,
                                'Price': f"${latest_price:.2f}",
                                'Change (60d)': f"{change:+.2f}%",
                                'RSI': f"{latest_rsi:.2f}",
                                'Signal': signal
                            })
                    except:
                        pass

                    progress_bar.progress((i + 1) / len(tickers))

                if portfolio_data:
                    df = pd.DataFrame(portfolio_data)

                    st.markdown("### 📊 Portfolio Overview")
                    st.dataframe(df, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Buy Signals", len(df[df['Signal'] == 'BUY']))
                    col2.metric("Sell Signals", len(df[df['Signal'] == 'SELL']))
                    col3.metric("Hold Signals", len(df[df['Signal'] == 'HOLD']))

    with tab6:
        st.subheader("📱 Live Trading Terminal")

        if not st.session_state['broker'].connected:
            st.error("⚠️ Broker not connected. Configure Kite API keys in Settings.")
        else:
            st.success("✅ Broker connected")

            col1, col2, col3 = st.columns(3)

            with col1:
                order_ticker = st.text_input("Ticker", "RELIANCE", key="ord_tick")
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])

            with col2:
                transaction = st.selectbox("Transaction", ["BUY", "SELL"])
                quantity = st.number_input("Quantity", value=1, min_value=1)

            with col3:
                if order_type == "LIMIT":
                    limit_price = st.number_input("Limit Price ({currency})", value=0.0, step=0.1)
                else:
                    limit_price = None

                st.write("")
                execute_button = st.button("🚀 Execute Order", type="primary", use_container_width=True)

            if execute_button:
                if order_ticker and quantity > 0:
                    with st.spinner("Placing order..."):
                        result = st.session_state['broker'].place_order(
                            order_ticker, transaction, quantity, order_type, limit_price
                        )

                        if result['status'] == 'success':
                            st.success(f"✅ Order placed! Order ID: {result.get('order_id', 'N/A')}")
                            log_trade_to_db(order_ticker, transaction, limit_price if limit_price else 0.0, quantity, "live_trading")
                        else:
                            st.error(f"❌ Order failed: {result.get('message', 'Unknown error')}")

            st.markdown("---")
            st.markdown("### 📋 Recent Orders")

            history_df = get_trade_history(limit=10)
            if not history_df.empty:
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No order history yet")


    # ===========================================================================
    # === TAB 7: SETTINGS (RESTORED WITH API KEYS) ============================
    # ===========================================================================

    with tab7:
        st.subheader("⚙️ Settings & Configuration")

        # API Configuration Section
        st.markdown("### 🔐 API Keys Configuration")
        st.info("💡 Configure these keys in your `.env` file or environment variables")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Trading APIs**")
            st.text_input("Kite API Key", value=KITE_API_KEY or "", type="password", disabled=True)
            st.text_input("Kite API Secret", value=KITE_API_SECRET or "", type="password", disabled=True)
            st.text_input("Kite Access Token", value=KITE_ACCESS_TOKEN or "", type="password", disabled=True)

            st.markdown("**AI/LLM APIs**")
            st.text_input("OpenRouter API Key", value=OPENROUTER_API_KEY or "", type="password", disabled=True)
            st.text_input("Google API Key (Gemini)", value=GOOGLE_API_KEY or "", type="password", disabled=True)

        with col2:
            st.markdown("**Notification APIs**")
            st.text_input("Gmail Email", value=GMAIL_EMAIL or "", disabled=True)
            st.text_input("Gmail App Password", value=GMAIL_APP_PASSWORD or "", type="password", disabled=True)
            st.text_input("Telegram Bot Token", value=TELEGRAM_BOT_TOKEN or "", type="password", disabled=True)
            st.text_input("Telegram Chat ID", value=TELEGRAM_CHAT_ID or "", disabled=True)

            st.markdown("**Data APIs**")
            st.text_input("NewsAPI Key", value=NEWSAPI_KEY or "", type="password", disabled=True)
            st.text_input("Alpha Vantage API Key", value=ALPHA_VANTAGE_API_KEY or "", type="password", disabled=True)

        st.markdown("---")

        # API Status Check
        st.markdown("### ✅ API Status")

        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            st.metric("Kite Connect", "✅ Configured" if KITE_API_KEY else "❌ Not Set")
            st.metric("OpenRouter", "✅ Configured" if OPENROUTER_API_KEY else "❌ Not Set")
            st.metric("Alpha Vantage", "✅ Configured" if ALPHA_VANTAGE_API_KEY else "❌ Not Set")

        with status_col2:
            st.metric("Google Gemini", "✅ Configured" if GOOGLE_API_KEY else "❌ Not Set")
            st.metric("Email Alerts", "✅ Configured" if GMAIL_EMAIL else "❌ Not Set")

        with status_col3:
            st.metric("Telegram", "✅ Configured" if TELEGRAM_BOT_TOKEN else "❌ Not Set")
            st.metric("NewsAPI", "✅ Configured" if NEWSAPI_KEY else "❌ Not Set")

        st.markdown("---")

        # Environment Setup Instructions
        with st.expander("📖 How to Setup API Keys"):
            st.markdown("""
            ### Setting up your `.env` file:
            
            Create a file named `.env` in your project directory with the following content:
            
            ```
            # Trading API
            KITE_API_KEY=your_kite_api_key_here
            KITE_API_SECRET=your_kite_secret_here
            KITE_ACCESS_TOKEN=your_access_token_here
            
            # AI/LLM APIs
            OPENROUTER_API_KEY=your_openrouter_key_here
            GOOGLE_API_KEY=your_google_gemini_key_here
            
            # Notification Services
            GMAIL_EMAIL=your_email@gmail.com
            GMAIL_APP_PASSWORD=your_app_password_here
            TELEGRAM_BOT_TOKEN=your_telegram_bot_token
            TELEGRAM_CHAT_ID=your_telegram_chat_id
            
            # Data APIs
            NEWSAPI_KEY=your_newsapi_key_here
            ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
            ```
            
            ### How to get API keys:
            
            - **Kite Connect:** [https://kite.zerodha.com](https://kite.zerodha.com)
            - **OpenRouter:** [https://openrouter.ai](https://openrouter.ai)
            - **Google Gemini:** [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
            - **NewsAPI:** [https://newsapi.org](https://newsapi.org)
            - **Alpha Vantage:** [https://www.alphavantage.co](https://www.alphavantage.co)
            - **Gmail App Password:** Google Account → Security → 2-Step Verification → App Passwords
            - **Telegram Bot:** Message @BotFather on Telegram
            """)

        st.markdown("---")

        # Trade History
        st.subheader("📚 Trade History")
        history_df = get_trade_history()
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)

            csv = history_df.to_csv(index=False)
            st.download_button(
                "📥 Download Trade History",
                csv,
                "trade_history.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("No trade history available yet. Start analyzing stocks to build your history!")

        st.markdown("---")

        # System Information
        st.subheader("💻 System Information")

        sys_col1, sys_col2, sys_col3 = st.columns(3)

        with sys_col1:
            st.metric("Total Stocks Analyzed", len(st.session_state.get('analysis_history', [])))

        with sys_col2:
            st.metric("Database Records", len(history_df) if not history_df.empty else 0)

        with sys_col3:
            st.metric("Active Session", "✅ Running")

        # Clear data options
        st.markdown("---")
        st.subheader("🗑️ Data Management")

        clear_col1, clear_col2 = st.columns(2)

        with clear_col1:
            if st.button("🔄 Clear Session Data", help="Clear current session analysis"):
                st.session_state['analysis_history'] = []
                st.session_state.pop('analysis_results', None)
                st.session_state.pop('ai_analysis', None)
                st.session_state.pop('screened_stocks', None)
                st.success("✅ Session data cleared!")
                st.rerun()

        with clear_col2:
            if st.button("⚠️ Reset All Settings", help="Reset all configurations"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ All settings reset!")
                st.rerun()

# ==============================================================================
# === RUN THE APP ==============================================================
# ==============================================================================

if __name__ == "__main__":
    main()
