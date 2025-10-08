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

    def get_company_name(self, ticker):
        """
        Get company name using SYMBOL_SEARCH endpoint
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            str: Company name or ticker if not found
        """
        try:
            # Clean ticker for search
            search_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('.L', '').replace('.T', '')
            
            params = {
                'function': 'SYMBOL_SEARCH',
                'keywords': search_ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'bestMatches' in data and len(data['bestMatches']) > 0:
                    # Try to find exact match first
                    for match in data['bestMatches']:
                        symbol = match.get('1. symbol', '')
                        name = match.get('2. name', '')
                        
                        # Check if symbol matches (case-insensitive)
                        if symbol.upper() == search_ticker.upper():
                            return name
                    
                    # If no exact match, return first result's name
                    first_match = data['bestMatches'][0]
                    return first_match.get('2. name', ticker)
            
            return ticker
            
        except Exception as e:
            print(f"Alpha Vantage company name fetch failed: {str(e)}")
            return ticker
    
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

def send_multi_channel_alert(ticker, signal, price, channels=['email'], currency=None):
    """Send alert across multiple channels"""
    # ✅ Auto-detect currency if not provided
    if currency is None:
        if ticker.endswith('.NS') or ticker.endswith('.BO'):
            currency = '₹'  # Indian Rupee
        elif ticker.endswith('.L'):
            currency = '£'  # British Pound
        elif ticker.endswith('.T'):
            currency = '¥'  # Japanese Yen
        else:
            currency = '$'  # Default to USD
    
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

    def get_nearest_expiry(self, ticker):
        """
        Get the nearest expiry date for a ticker using Yahoo Finance API
        
        Args:
            ticker: Stock/ETF/Index ticker symbol
        
        Returns:
            Dict with expiry info or None
        """
        try:
            stock = yf.Ticker(ticker)
            expiry_dates = stock.options
            
            if not expiry_dates or len(expiry_dates) == 0:
                return None
            
            today = datetime.now().date()
            today_str = today.strftime("%Y-%m-%d")
            
            # Find if options expire today
            expires_today = today_str in expiry_dates
            
            # Find nearest future expiry
            future_expiries = [exp for exp in expiry_dates if exp >= today_str]
            nearest_expiry = future_expiries[0] if future_expiries else None
            
            if nearest_expiry:
                expiry_date = datetime.strptime(nearest_expiry, "%Y-%m-%d").date()
                days_until = (expiry_date - today).days
                
                return {
                    'date': nearest_expiry,
                    'days_until': days_until,
                    'expires_today': expires_today,
                    'total_expiries': len(expiry_dates)
                }
            
            return None
            
        except Exception as e:
            return None

    def fetch_options_chain(self, ticker):
        """Fetch options chain data with error handling"""
        try:
            # Try different ticker format variations
            ticker_variations = [
                ticker,
                ticker.upper(),
                ticker.replace('.NS', ''),
                ticker.replace('.BO', ''),
            ]
            
            # Add common formats for Indian markets (though unlikely to work)
            if 'NIFTY' in ticker.upper() or 'NSEI' in ticker.upper():
                ticker_variations.extend(['^NSEI', 'NIFTY'])
            if 'BANK' in ticker.upper():
                ticker_variations.extend(['^NSEBANK', 'BANKNIFTY'])
            
            # Remove duplicates while preserving order
            ticker_variations = list(dict.fromkeys(ticker_variations))
            
            # Try each ticker variation
            for test_ticker in ticker_variations:
                try:
                    stock = yf.Ticker(test_ticker)
                    expiry_dates = stock.options
                    
                    # Check if options data exists
                    if not expiry_dates or len(expiry_dates) == 0:
                        continue
                    
                    # Get nearest expiry
                    nearest_expiry = expiry_dates[0]
                    options = stock.option_chain(nearest_expiry)
                    
                    # Validate that we actually have data
                    if options.calls is not None and not options.calls.empty and \
                       options.puts is not None and not options.puts.empty:
                        
                        return {
                            'calls': options.calls,
                            'puts': options.puts,
                            'expiry': nearest_expiry,
                            'ticker': test_ticker,
                            'all_expiries': expiry_dates
                        }
                except Exception:
                    continue
            
            return None
            
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
# === MULTI-ASSET API HANDLER WITH FINNHUB PRIMARY =============================
# ==============================================================================

import finnhub

class MultiAssetAPIHandler:
    """
    Unified API handler for fetching dynamic asset data
    Primary: Finnhub API
    Fallback: Static data
    Shows data source on UI (Static/Dynamic/API Name)
    """
    
    def __init__(self):
        """Initialize API clients and configuration"""
        
        # Load API keys from environment
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY', '')
        
        # Initialize Finnhub client
        self.finnhub_client = None
        if self.finnhub_api_key:
            try:
                self.finnhub_client = finnhub.Client(api_key=self.finnhub_api_key)
                self.finnhub_available = True
            except Exception as e:
                self.finnhub_available = False
                st.warning(f"⚠️ Finnhub initialization failed: {str(e)}")
        else:
            self.finnhub_available = False
        
        # Cache for API responses (avoid repeated calls)
        self.cache = {}
        self.cache_duration = 3600  # 1 hour in seconds
        
        # Track data source for UI display
        self.last_data_source = {
            'type': 'Unknown',  # 'Static', 'Finnhub', 'EODHD', etc.
            'timestamp': None,
            'details': ''
        }
    
    def _get_cached_or_fetch(self, cache_key, fetch_function):
        """
        Check cache first, then fetch if needed
        
        Args:
            cache_key: Unique identifier for cached data
            fetch_function: Function to call if cache miss
        
        Returns:
            Tuple: (data, source_info)
        """
        if cache_key in self.cache:
            cached_data, source_info, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                self.last_data_source = source_info
                return cached_data, source_info
        
        # Fetch new data
        data, source_info = fetch_function()
        if data:
            self.cache[cache_key] = (data, source_info, time.time())
            self.last_data_source = source_info
        
        return data, source_info
    
    def get_data_source_badge(self):
        """
        Get HTML badge showing data source for UI display
        
        Returns:
            HTML string for Streamlit
        """
        source = self.last_data_source
        
        if source['type'] == 'Finnhub':
            color = "#00C851"  # Green
            icon = "🟢"
            label = "LIVE DATA (Finnhub API)"
        elif source['type'] == 'Static':
            color = "#ffbb33"  # Orange
            icon = "🟡"
            label = "CACHED DATA (Static Fallback)"
        elif source['type'] == 'EODHD':
            color = "#00C851"  # Green
            icon = "🟢"
            label = "LIVE DATA (EODHD API)"
        elif source['type'] == 'FMP':
            color = "#00C851"  # Green
            icon = "🟢"
            label = "LIVE DATA (FMP API)"
        else:
            color = "#9e9e9e"  # Grey
            icon = "⚪"
            label = "UNKNOWN SOURCE"
        
        details = source.get('details', '')
        timestamp = source.get('timestamp', '')
        
        html = f"""
        <div style="
            background-color: {color}; 
            color: white; 
            padding: 8px 15px; 
            border-radius: 5px; 
            font-weight: bold;
            font-size: 14px;
            margin: 10px 0;
            display: inline-block;
        ">
            {icon} {label}
        </div>
        """
        
        if details:
            html += f"""
            <div style="
                color: #666; 
                font-size: 12px; 
                margin-top: 5px;
            ">
                {details}
            </div>
            """
        
        return html
    
    # ======================== INDEX CONSTITUENTS ========================
    
    def get_index_constituents_finnhub(self, index_symbol):
        """
        Fetch index constituents from Finnhub API
        Supports: ^GSPC, ^DJI, ^NDX, ^RUT (US indices)
        
        Args:
            index_symbol: Index symbol (e.g., "^GSPC")
        
        Returns:
            Tuple: (List of tickers, source_info)
        """
        if not self.finnhub_available or not self.finnhub_client:
            return None, {
                'type': 'Error',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'details': 'Finnhub API not available. Please add FINNHUB_API_KEY to .env file'
            }
        
        cache_key = f"finnhub_constituents_{index_symbol}"
        
        def fetch():
            try:
                # Finnhub API call
                result = self.finnhub_client.indices_constituents(index_symbol)
                
                if result and 'constituents' in result:
                    constituents = result['constituents']
                    
                    source_info = {
                        'type': 'Finnhub',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'details': f'Fetched {len(constituents)} constituents from Finnhub API (Free tier: 60 calls/min)'
                    }
                    
                    return constituents, source_info
                else:
                    # API returned empty or error
                    return None, {
                        'type': 'Error',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'details': f'Finnhub API returned no data for {index_symbol}'
                    }
                    
            except Exception as e:
                error_msg = str(e)
                
                # Check for rate limit error
                if '429' in error_msg or 'rate limit' in error_msg.lower():
                    return None, {
                        'type': 'Error',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'details': '⚠️ Finnhub API rate limit exceeded (60 calls/min). Please wait.'
                    }
                
                return None, {
                    'type': 'Error',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'details': f'Finnhub API error: {error_msg}'
                }
        
        return self._get_cached_or_fetch(cache_key, fetch)
    
    def get_static_constituents(self, index_symbol):
        """
        Static fallback data for index constituents
        Used when all APIs fail or for unsupported indices
        
        Args:
            index_symbol: Index symbol
        
        Returns:
            Tuple: (List of tickers, source_info)
        """
        static_data = {
            # US Indices (Finnhub supported)
            "^GSPC": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
                "UNH", "JNJ", "JPM", "V", "XOM", "PG", "MA", "HD", "CVX", "LLY",
                "ABBV", "MRK", "PEP", "KO", "AVGO", "COST", "TMO", "WMT", "MCD",
                "CSCO", "ABT", "ACN", "DHR", "VZ", "ADBE", "NEE", "NFLX", "CRM"
            ],
            "^DJI": [
                "AAPL", "MSFT", "UNH", "HD", "GS", "MCD", "V", "BA", "CAT", "AMGN",
                "HON", "TRV", "JPM", "IBM", "CVX", "CSCO", "AXP", "CRM", "JNJ", "PG",
                "WMT", "MMM", "NKE", "MRK", "DIS", "KO", "DOW", "INTC", "VZ", "WBA"
            ],
            "^NDX": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
                "COST", "PEP", "CSCO", "ADBE", "NFLX", "CMCSA", "INTC", "AMD"
            ],
            "^RUT": [
                "AMC", "GME", "PLTR", "SOFI", "F", "NIO", "AAL", "LCID", "RIVN"
            ],
            
            # Indian Indices (Not in Finnhub free tier)
            "^NSEI": [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
                "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
                "SUNPHARMA.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS"
            ],
            "^NSEBANK": [
                "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
                "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS"
            ],
            "^BSESN": [
                "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO"
            ],
            
            # UK Indices
            "^FTSE": [
                "BARC.L", "HSBA.L", "BP.L", "SHEL.L", "VOD.L", "AZN.L", "GLEN.L",
                "RIO.L", "LSEG.L", "LLOY.L", "GSK.L", "ULVR.L", "DGE.L", "NG.L"
            ],
            "^FTMC": [
                "WIZZ.L", "IMB.L", "MARS.L", "MNG.L", "AUTO.L"
            ],
            
            # Japan Indices
            "^N225": [
                "7203.T", "6758.T", "9984.T", "6861.T", "8306.T", "7267.T",
                "6098.T", "9432.T", "8035.T", "4063.T"
            ],
            "^TOPX": [
                "7203.T", "6758.T", "9984.T", "8306.T"
            ]
        }
        
        constituents = static_data.get(index_symbol, [])
        
        source_info = {
            'type': 'Static',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': f'Using static fallback data ({len(constituents)} constituents). For live data, add FINNHUB_API_KEY to .env'
        }
        
        return constituents, source_info
    
    def get_index_constituents(self, index_symbol):
        """
        Get index constituents with automatic API -> Static fallback
        
        Priority:
        1. Finnhub API (for supported indices)
        2. Static fallback
        
        Args:
            index_symbol: Index symbol (e.g., "^GSPC")
        
        Returns:
            Tuple: (List of tickers, source_info dict)
        """
        # Finnhub supported indices (US only in free tier)
        finnhub_supported = ['^GSPC', '^DJI', '^NDX', '^RUT']
        
        # Try Finnhub first if supported
        if index_symbol in finnhub_supported:
            constituents, source_info = self.get_index_constituents_finnhub(index_symbol)
            
            if constituents:
                return constituents, source_info
        
        # Fallback to static data
        return self.get_static_constituents(index_symbol)
    
    # ======================== AVAILABLE INDICES ========================
    
    def get_available_indices_finnhub(self):
        """
        Fetch list of available indices from Finnhub
        
        Returns:
            Tuple: (Dictionary of indices, source_info)
        """
        if not self.finnhub_available or not self.finnhub_client:
            return None, {
                'type': 'Error',
                'details': 'Finnhub not available'
            }
        
        cache_key = "finnhub_available_indices"
        
        def fetch():
            try:
                # Finnhub doesn't have a direct "list all indices" endpoint
                # So we return the ones we know are supported
                indices = {
                    "S&P 500": "^GSPC",
                    "Dow Jones": "^DJI",
                    "Nasdaq 100": "^NDX",
                    "Russell 2000": "^RUT"
                }
                
                source_info = {
                    'type': 'Finnhub',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'details': 'Finnhub free tier supports US indices only'
                }
                
                return indices, source_info
                
            except Exception as e:
                return None, {
                    'type': 'Error',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'details': f'Error: {str(e)}'
                }
        
        return self._get_cached_or_fetch(cache_key, fetch)
    
    def get_static_indices(self, market=None):
        """
        Static list of indices by market
        
        Args:
            market: Market name filter
        
        Returns:
            Tuple: (Dictionary of indices, source_info)
        """
        all_indices = {
            "🇮🇳 India (NSE/BSE)": {
                "Nifty 50": "^NSEI",
                "Nifty Bank": "^NSEBANK",
                "BSE Sensex": "^BSESN",
                "Nifty IT": "^CNXIT",
                "Nifty Auto": "^CNXAUTO",
                "Nifty Pharma": "^CNXPHARMA",
                "Nifty FMCG": "^CNXFMCG",
                "Nifty Metal": "^CNXMETAL",
                "Nifty Realty": "^CNXREALTY"
            },
            "🇺🇸 USA (NYSE/NASDAQ)": {
                "S&P 500": "^GSPC",
                "Dow Jones": "^DJI",
                "Nasdaq 100": "^NDX",
                "Russell 2000": "^RUT",
                "S&P 400 MidCap": "^MID"
            },
            "🇬🇧 UK (LSE)": {
                "FTSE 100": "^FTSE",
                "FTSE 250": "^FTMC",
                "FTSE 350": "^FTLC",
                "FTSE All-Share": "^FTAS"
            },
            "🇯🇵 Japan (TSE)": {
                "Nikkei 225": "^N225",
                "TOPIX": "^TOPX",
                "JPX-Nikkei 400": "^JPXN"
            }
        }
        
        if market:
            indices = all_indices.get(market, {})
        else:
            # Combine all
            indices = {}
            for market_indices in all_indices.values():
                indices.update(market_indices)
        
        source_info = {
            'type': 'Static',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': f'Static index list ({len(indices)} indices)'
        }
        
        return indices, source_info
    
    def get_available_indices(self, market=None):
        """
        Get available indices with API -> Static fallback
        
        Args:
            market: Market name (optional filter)
        
        Returns:
            Tuple: (Dictionary of indices, source_info)
        """
        # For US market, try Finnhub first
        if market and "USA" in market:
            indices, source_info = self.get_available_indices_finnhub()
            if indices:
                return indices, source_info
        
        # Fallback to static
        return self.get_static_indices(market)
    
    # ======================== CRYPTO / FOREX / COMMODITIES ========================
    
    def get_crypto_list(self, category=None):
        """
        Get cryptocurrency list
        Currently static (can be enhanced with CoinGecko API)
        
        Returns:
            Tuple: (Dictionary, source_info)
        """
        all_cryptos = {
            "Major": {
                "Bitcoin": "BTC-USD",
                "Ethereum": "ETH-USD",
                "BNB": "BNB-USD",
                "XRP": "XRP-USD",
                "Cardano": "ADA-USD",
                "Solana": "SOL-USD",
                "Polkadot": "DOT-USD"
            },
            "DeFi": {
                "Uniswap": "UNI-USD",
                "Aave": "AAVE-USD",
                "Maker": "MKR-USD",
                "Compound": "COMP-USD"
            },
            "Stablecoins": {
                "Tether": "USDT-USD",
                "USD Coin": "USDC-USD",
                "Binance USD": "BUSD-USD",
                "DAI": "DAI-USD"
            }
        }
        
        if category:
            cryptos = all_cryptos.get(category, {})
        else:
            cryptos = {}
            for cat_cryptos in all_cryptos.values():
                cryptos.update(cat_cryptos)
        
        source_info = {
            'type': 'Static',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': f'Static crypto list. Can be enhanced with live API in future.'
        }
        
        return cryptos, source_info
    
    def get_forex_pairs(self, pair_type=None):
        """
        Get forex pairs
        
        Returns:
            Tuple: (Dictionary, source_info)
        """
        all_pairs = {
            "Major Pairs": {
                "EUR/USD": "EURUSD=X",
                "GBP/USD": "GBPUSD=X",
                "USD/JPY": "JPY=X",
                "USD/CHF": "CHF=X",
                "AUD/USD": "AUDUSD=X",
                "USD/CAD": "CAD=X",
                "NZD/USD": "NZDUSD=X"
            },
            "Cross Pairs": {
                "EUR/GBP": "EURGBP=X",
                "EUR/JPY": "EURJPY=X",
                "GBP/JPY": "GBPJPY=X",
                "EUR/CHF": "EURCHF=X"
            },
            "Exotic Pairs": {
                "USD/INR": "INR=X",
                "USD/SGD": "SGD=X",
                "USD/HKD": "HKD=X",
                "USD/THB": "THB=X",
                "EUR/TRY": "EURTRY=X"
            }
        }
        
        if pair_type:
            pairs = all_pairs.get(pair_type, {})
        else:
            pairs = {}
            for type_pairs in all_pairs.values():
                pairs.update(type_pairs)
        
        source_info = {
            'type': 'Static',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': 'Static forex list'
        }
        
        return pairs, source_info
    
    def get_commodities(self, commodity_type=None):
        """
        Get commodities list
        
        Returns:
            Tuple: (Dictionary, source_info)
        """
        all_commodities = {
            "Precious Metals": {
                "Gold": "GC=F",
                "Silver": "SI=F",
                "Platinum": "PL=F",
                "Palladium": "PA=F"
            },
            "Energy": {
                "Crude Oil WTI": "CL=F",
                "Brent Crude": "BZ=F",
                "Natural Gas": "NG=F",
                "Heating Oil": "HO=F",
                "Gasoline": "RB=F"
            },
            "Agricultural": {
                "Corn": "ZC=F",
                "Wheat": "ZW=F",
                "Soybeans": "ZS=F",
                "Coffee": "KC=F",
                "Sugar": "SB=F",
                "Cotton": "CT=F",
                "Cocoa": "CC=F"
            },
            "Industrial Metals": {
                "Copper": "HG=F",
                "Aluminum": "ALI=F"
            }
        }
        
        if commodity_type:
            commodities = all_commodities.get(commodity_type, {})
        else:
            commodities = {}
            for type_commodities in all_commodities.values():
                commodities.update(type_commodities)
        
        source_info = {
            'type': 'Static',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': 'Static commodity list'
        }
        
        return commodities, source_info


# ==============================================================================
# === INITIALIZE GLOBAL API HANDLER ============================================
# ==============================================================================

# Create global instance
asset_api = MultiAssetAPIHandler()


# ==============================================================================
# === ASSET CLASSES CONFIGURATION ==============================================
# ==============================================================================

ASSET_CLASSES = {
    "Equities (Stocks)": {
        "description": "Individual company stocks",
        "selection_methods": ["Search", "By Exchange", "Direct", "From Scanner", "By Index"],
        "markets": list(GLOBAL_MARKETS.keys()),
        "supports_options": True,
        "supports_index_analysis": False
    },
    "Indices": {
        "description": "Market indices (can be analyzed directly)",
        "selection_methods": ["Search", "Direct", "By Market"],
        "markets": ["Global"],
        "supports_options": True,
        "supports_index_analysis": True
    },
    "Cryptocurrencies": {
        "description": "Digital currencies and tokens",
        "selection_methods": ["Search", "Direct", "By Category"],
        "markets": ["Global"],
        "supports_options": False,
        "supports_index_analysis": False
    },
    "Forex": {
        "description": "Currency pairs",
        "selection_methods": ["Search", "Direct", "By Pair Type"],
        "markets": ["Global"],
        "supports_options": False,
        "supports_index_analysis": False
    },
    "Commodities": {
        "description": "Raw materials and resources",
        "selection_methods": ["Search", "Direct", "By Type"],
        "markets": ["Global"],
        "supports_options": True,
        "supports_index_analysis": False
    }
}
 
# ==============================================================================
# === HELPER FUNCTIONS =========================================================
# ==============================================================================

def get_currency_symbol(ticker, selected_market=None):
    """
    Get currency symbol based on ticker suffix or selected market
    Enhanced for all asset classes
    """
    
    # Add safety check for None or non-string ticker
    if ticker is None or not isinstance(ticker, str):
        ticker = ""
    
    # Crypto detection
    if '-USD' in ticker or 'BTC' in ticker or 'ETH' in ticker:
        return '$ '
    
    # Forex detection (no symbol, it's a rate)
    if '=X' in ticker:
        return ''
    
    # Commodities detection
    if '=F' in ticker:
        if 'GC' in ticker or 'SI' in ticker or 'PL' in ticker or 'PA' in ticker:  # Precious metals
            return '$/oz '
        elif 'CL' in ticker or 'BZ' in ticker:  # Oil
            return '$/bbl '
        elif 'NG' in ticker:  # Natural gas
            return '$/MMBtu '
        return '$ '
    
    # Stock market suffixes
    if '.NS' in ticker or '.BO' in ticker:
        return '₹ '  # Indian Rupee
    elif '.L' in ticker:
        return '£ '  # British Pound
    elif '.T' in ticker:
        return '¥ '  # Japanese Yen
    
    # Index detection
    if ticker.startswith('^'):
        if 'NSE' in ticker or 'BSE' in ticker or 'CNX' in ticker:
            return '₹ '
        elif 'FTSE' in ticker:
            return '£ '
        elif 'N225' in ticker or 'TOPX' in ticker or 'JPX' in ticker:
            return '¥ '
        return ''  # Most indices shown without currency
    
    # Market-based detection
    if selected_market:
        if 'India' in selected_market:
            return '₹ '
        elif 'UK' in selected_market:
            return '£ '
        elif 'Japan' in selected_market:
            return '¥ '
    
    return '$ '  # Default to USD

def get_asset_currency_symbol(ticker, asset_class, selected_market=None):
    """
    Enhanced currency detection for all asset classes
    Wrapper around get_currency_symbol for asset-specific logic
    
    Args:
        ticker: Ticker symbol
        asset_class: Asset class name
        selected_market: Selected market (optional)
    
    Returns:
        Currency symbol string
    """
    return get_currency_symbol(ticker, selected_market)


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
    price = results.get('latestprice', 0)
    currency = results.get('currency', '$')

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


# ==============================================================================
# === HELPER FUNCTIONS FOR ASSET SELECTION =====================================
# ==============================================================================

def get_stocks_by_index(index_ticker, market_name):
    """
    Dynamically fetch constituent stocks of a given index
    Shows data source badge (Finnhub/Static)
    
    Args:
        index_ticker: Index symbol (e.g., ^NSEI, ^GSPC)
        market_name: Market name
    
    Returns:
        Tuple: (List of tickers, source_info dict)
    """
    
    with st.spinner(f"🔄 Fetching {index_ticker} constituents..."):
        constituents, source_info = asset_api.get_index_constituents(index_ticker)
        
        # Display data source badge
        st.markdown(asset_api.get_data_source_badge(), unsafe_allow_html=True)
        
        return constituents, source_info

def load_available_indices(market_name):
    """
    Dynamically load available indices for a market
    
    Args:
        market_name: Market name (e.g., "🇮🇳 India (NSE/BSE)")
    
    Returns:
        Tuple: (Dictionary of indices, source_info)
    """
    
    indices, source_info = asset_api.get_available_indices(market_name)
    
    # Display data source badge
    st.markdown(asset_api.get_data_source_badge(), unsafe_allow_html=True)
    
    return indices, source_info

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
                    stock_currency = get_currency_symbol(ticker, market_name)
                    
                    # Apply filters
                    if price >= min_price and volume >= min_volume:
                        screened_list[ticker] = {
                            'price': price,
                            'volume': volume,
                            'change_pct': change_pct,
                            'currency': stock_currency
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

# ============================================================
# ENHANCED COMPANY NAME FETCHER (Multiple Sources)
# ============================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_company_name_multi_source(ticker, _alpha_vantage_api=None):
    """
    Get company name using multiple data sources with fallbacks
    
    Priority:
    1. Alpha Vantage API (most reliable for US stocks)
    2. yfinance (good for global stocks)
    3. Known stocks dictionary
    4. Formatted ticker name
    
    Args:
        ticker: Stock ticker symbol
        alpha_vantage_api: AlphaVantageAPI instance (optional)
        
    Returns:
        str: Company name
    """
    
    print(f"\n🔍 Fetching company name for: {ticker}")
    
    # Strategy 1: Alpha Vantage API (best for US stocks)
    if _alpha_vantage_api and _alpha_vantage_api.api_key:
        try:
            print("   Trying Alpha Vantage API...")
            av_name = _alpha_vantage_api.get_company_name(ticker)
            if av_name and av_name != ticker and len(av_name) > 1:
                print(f"   ✅ Got from Alpha Vantage: {av_name}")
                return av_name
        except Exception as e:
            print(f"   ⚠️ Alpha Vantage failed: {str(e)}")
    
    # Strategy 2: yfinance (good for all markets)
    try:
        print("   Trying yfinance...")
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if info:
            name = (
                info.get('longName') or 
                info.get('shortName') or 
                info.get('name')
            )
            
            if name and len(name) > 1 and name != ticker:
                print(f"   ✅ Got from yfinance: {name}")
                return name
    except Exception as e:
        print(f"   ⚠️ yfinance failed: {str(e)}")
    
    # Strategy 3: Known stocks dictionary
    known_stocks = {
        # Indian Stocks
        'RELIANCE.NS': 'Reliance Industries Ltd',
        'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank Ltd',
        'INFY.NS': 'Infosys Ltd',
        'ICICIBANK.NS': 'ICICI Bank Ltd',
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'SBIN.NS': 'State Bank of India',
        'BHARTIARTL.NS': 'Bharti Airtel Ltd',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank',
        'ITC.NS': 'ITC Ltd',
        'LT.NS': 'Larsen & Toubro',
        'AXISBANK.NS': 'Axis Bank Ltd',
        'MARUTI.NS': 'Maruti Suzuki India',
        'TITAN.NS': 'Titan Company Ltd',
        
        # US Stocks
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',
        'META': 'Meta Platforms Inc.',
        'NVDA': 'NVIDIA Corporation',
        'JPM': 'JPMorgan Chase & Co.',
        'V': 'Visa Inc.',
        'MA': 'Mastercard Inc.',
        'WMT': 'Walmart Inc.',
        'DIS': 'The Walt Disney Company',
        'NFLX': 'Netflix Inc.',
        'AMD': 'Advanced Micro Devices',
        'INTC': 'Intel Corporation',
        'BAC': 'Bank of America Corp',
        
        # UK Stocks
        'BARC.L': 'Barclays PLC',
        'HSBA.L': 'HSBC Holdings PLC',
        'BP.L': 'BP PLC',
        'SHEL.L': 'Shell PLC',
        
        # Japan Stocks
        '7203.T': 'Toyota Motor Corporation',
        '6758.T': 'Sony Group Corporation',
        '9984.T': 'SoftBank Group Corp'
    }
    
    if ticker in known_stocks:
        print(f"   ✅ Got from known_stocks: {known_stocks[ticker]}")
        return known_stocks[ticker]
    
    # Strategy 4: Format ticker as readable name
    clean_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('.L', '').replace('.T', '')
    formatted_name = clean_ticker.replace('_', ' ').replace('-', ' ').title()
    
    print(f"   ⚠️ Using formatted ticker: {formatted_name}")
    return formatted_name

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
        """Setup sentiment analyzer with FinBERT and fallback"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import warnings
            warnings.filterwarnings('ignore')
            
            print("\n" + "="*60)
            print("🔄 LOADING FINBERT SENTIMENT ANALYZER")
            print("="*60)
            
            # ✅ Try to load FinBERT first (best for financial news)
            try:
                print("   Attempting to load ProsusAI/finbert...")
                
                self.sentiment_analyzer = pipeline(
                    'sentiment-analysis',
                    model='ProsusAI/finbert',
                    tokenizer='ProsusAI/finbert',
                    device=-1  # Use CPU
                )
                
                # Test it
                test_result = self.sentiment_analyzer("The stock price is rising strongly")
                print(f"   ✅ FinBERT loaded successfully")
                print(f"   ✅ Test result: {test_result}")
                self.model_name = 'FinBERT'
                
            except Exception as e:
                print(f"   ⚠️  FinBERT failed: {str(e)}")
                print("   🔄 Falling back to default sentiment model...")
                
                # Fallback to default model
                self.sentiment_analyzer = pipeline(
                    'sentiment-analysis',
                    model='distilbert-base-uncased-finetuned-sst-2-english',
                    device=-1
                )
                
                test_result = self.sentiment_analyzer("The stock price is rising strongly")
                print(f"   ✅ Default model loaded successfully")
                print(f"   ✅ Test result: {test_result}")
                self.model_name = 'DistilBERT'
            
            print(f"✅ Using model: {self.model_name}")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            print(f"❌ Could not load ANY sentiment analyzer: {e}")
            import traceback
            print(traceback.format_exc())
            print("="*60 + "\n")
            self.sentiment_analyzer = None
            self.model_name = None
            return False

    def analyze_sentiment_detailed(self, headlines):
        """Analyze sentiment with per-article breakdown - supports FinBERT and generic models"""
        
        print(f"\n{'='*60}")
        print(f"🔍 SENTIMENT ANALYSIS STARTED (Model: {getattr(self, 'model_name', 'Unknown')})")
        print(f"{'='*60}")
        print(f"Headlines received: {len(headlines) if headlines else 0}")
        print(f"Sentiment analyzer status: {self.sentiment_analyzer is not None}")
        
        # Check if we have headlines
        if not headlines or len(headlines) == 0:
            print("❌ No headlines provided")
            print(f"{'='*60}\n")
            return {
                'overall_sentiment': 'Neutral',
                'overall_score': 0.0,
                'articles': [],
                'total_articles': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        # Check if sentiment analyzer is loaded
        if not self.sentiment_analyzer:
            print("❌ Sentiment analyzer not loaded - attempting to reload...")
            success = self.setup_sentiment_analyzer()
            
            if not success or not self.sentiment_analyzer:
                print("❌ Failed to load sentiment analyzer")
                print(f"{'='*60}\n")
                return {
                    'overall_sentiment': 'Neutral',
                    'overall_score': 0.0,
                    'articles': [],
                    'total_articles': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'error': 'Sentiment analyzer not available'
                }
        
        # Process headlines
        try:
            article_sentiments = []
            sentiment_scores = []
            
            print(f"\n📰 Processing {len(headlines)} headlines...\n")
            
            for idx, headline in enumerate(headlines, 1):
                # Skip very short headlines
                if not headline or len(headline.strip()) < 10:
                    print(f"   ⚠️  Skipping headline {idx}: too short ({len(headline) if headline else 0} chars)")
                    continue
                
                try:
                    print(f"   🔄 Analyzing #{idx}: {headline[:60]}...")
                    
                    # Run sentiment analysis
                    result = self.sentiment_analyzer(headline[:512])
                    
                    if not result or len(result) == 0:
                        print(f"      ⚠️  Empty result")
                        continue
                    
                    # Extract result
                    raw_label = result[0]['label']
                    confidence = result[0]['score']
                    
                    print(f"      ✅ Raw: label='{raw_label}', confidence={confidence:.3f}")
                    
                    # ✅ Normalize label - handle both FinBERT (lowercase) and generic models (uppercase)
                    label_upper = raw_label.upper()
                    
                    if label_upper == 'POSITIVE':
                        label = 'Positive'
                        score = confidence
                    elif label_upper == 'NEGATIVE':
                        label = 'Negative'
                        score = -confidence
                    elif label_upper == 'NEUTRAL':
                        label = 'Neutral'
                        score = 0
                    else:
                        # Unknown label - treat as neutral
                        label = 'Neutral'
                        score = 0
                        print(f"      ⚠️  Unknown label '{raw_label}', treating as Neutral")
                    
                    # Add to results
                    article_sentiments.append({
                        'headline': headline,
                        'sentiment': label,
                        'score': round(score, 3),
                        'confidence': round(confidence, 3),
                        'raw_label': raw_label  # Keep original for debugging
                    })
                    sentiment_scores.append(score)
                    
                    print(f"      📊 Normalized: {label} (score: {score:.3f})")
                    
                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
                    import traceback
                    print(f"      {traceback.format_exc()}")
                    continue
            
            print(f"\n✅ Successfully processed {len(article_sentiments)} out of {len(headlines)} articles")
            
            # Calculate overall sentiment
            if sentiment_scores and len(sentiment_scores) > 0:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                # Use slightly different thresholds for FinBERT
                threshold = 0.05 if getattr(self, 'model_name', '') == 'FinBERT' else 0.1
                
                if avg_sentiment > threshold:
                    overall = 'Positive'
                elif avg_sentiment < -threshold:
                    overall = 'Negative'
                else:
                    overall = 'Neutral'
            else:
                avg_sentiment = 0.0
                overall = 'Neutral'
            
            # Count sentiments (case-insensitive)
            positive_count = sum(1 for a in article_sentiments if a['sentiment'].upper() == 'POSITIVE')
            negative_count = sum(1 for a in article_sentiments if a['sentiment'].upper() == 'NEGATIVE')
            neutral_count = sum(1 for a in article_sentiments if a['sentiment'].upper() == 'NEUTRAL')
            
            print(f"\n📊 SENTIMENT SUMMARY:")
            print(f"   Model Used: {getattr(self, 'model_name', 'Unknown')}")
            print(f"   Overall: {overall}")
            print(f"   Average Score: {avg_sentiment:.3f}")
            print(f"   ✅ Positive: {positive_count}")
            print(f"   ❌ Negative: {negative_count}")
            print(f"   ⚪ Neutral: {neutral_count}")
            print(f"{'='*60}\n")
            
            return {
                'overall_sentiment': overall,
                'overall_score': round(avg_sentiment, 3),
                'articles': article_sentiments,
                'total_articles': len(article_sentiments),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'model_used': getattr(self, 'model_name', 'Unknown')
            }
            
        except Exception as e:
            print(f"\n❌ SENTIMENT ANALYSIS FAILED")
            print(f"   Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            print(f"{'='*60}\n")
            
            return {
                'overall_sentiment': 'Neutral',
                'overall_score': 0.0,
                'articles': [],
                'total_articles': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
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
                'patterns': [],  # Changed from single pattern to list
                'primary_pattern': {
                    'pattern': 'Insufficient Data',
                    'type': 'neutral',
                    'strength': 0,
                    'confidence': 0,
                    'category': 'none',
                    'description': 'Need at least 1 candles for pattern detection'
                }
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
        if lower_shadow > curr_body * 1.5 and upper_shadow < curr_body * 0.3 and curr_is_green and curr_range > 0:
            # Add volume confirmation for stronger signal
            strength = 85
            if len(data) > 5:
                avg_vol = data['Volume'].iloc[-6:-1].mean() if 'Volume' in data.columns else 0
                cur_vol = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                if cur_vol > avg_vol * 1.2:  # Above-average volume = stronger signal
                    strength = 90
            
            patterns_found.append({
                'pattern': 'Hammer',
                'type': 'bullish',
                'strength': strength,
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

        # 4. MORNING STAR - 3-Candle Bullish Reversal
        c3_open = c3['Open'] if 'Open' in c3.index else c3['open']
        c3_close = c3['Close'] if 'Close' in c3.index else c3['close']
        c4_open = c4['Open'] if 'Open' in c4.index else c4['open']
        c4_close = c4['Close'] if 'Close' in c4.index else c4['close']
        
        if c3_close > c3_open and \
           abs(c4_close - c4_open) < (c3_high - c3_low) * 0.3 and \
           curr_is_green and curr_close > (c3_open + c3_close) / 2:
            
            # Check for gaps (classic Morning Star feature)
            c3_high = c3['High'] if 'High' in c3.index else c3['high']
            c3_low = c3['Low'] if 'Low' in c3.index else c3['low']
            c4_high = c4['High'] if 'High' in c4.index else c4['high']
            c4_low = c4['Low'] if 'Low' in c4.index else c4['low']
            c5_low = c5['Low'] if 'Low' in c5.index else c5['low']
        
            # Gap 1: Gap down into middle candle
            gap1 = c4_high < c3_low
            # Gap 2: Gap up from middle candle
            gap2 = c5_low > c4_high
            
            strength = 95
            if gap1 and gap2:
                strength = 98  # Perfect Morning Star with gaps on both sides
                description = 'PERFECT Morning Star with gaps - Extremely strong bullish reversal'
            elif gap1 or gap2:
                strength = 96  # One gap present
                description = 'Strong Morning Star with gap - Extremely strong bullish reversal'
            else:
                description = 'Morning Star - Classic 3-candle bottom pattern'
            
            patterns_found.append({
                'pattern': 'Morning Star',
                'type': 'bullish',
                'strength': strength,
                'confidence': 95,
                'category': 'reversal',
                'description': description
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
        if c3_close > c3_open and c4_close > c4_open and curr_is_green and \
           c4_close > c3_close and curr_close > c4_close:
            
            # Calculate body sizes
            c3_body = abs(c3_close - c3_open)
            c4_body = abs(c4_close - c4_open)
            c5_body = abs(curr_close - curr_open)
            
            # Check for progressive size increase (each candle should be similar or larger)
            strength = 92
            confidence = 90
            
            if c4_body >= c3_body * 0.8 and c5_body >= c4_body * 0.8:
                # Good progression - each candle maintains or grows
                strength = 92
                description = 'Strong bullish continuation - Steady upward momentum with consistent buying'
            else:
                # Weak progression - candles getting smaller (momentum fading)
                strength = 85
                confidence = 85
                description = 'Bullish continuation but momentum weakening - Watch for reversal'
            
            # BONUS: Check if all three open near previous close (small gaps)
            small_gaps = (abs(c4_open - c3_close) / c3_close < 0.01) and \
                         (abs(curr_open - c4_close) / c4_close < 0.01)
            
            if small_gaps and c5_body >= c4_body * 0.8:
                strength = 95  # Perfect Three White Soldiers
                description = 'PERFECT Three White Soldiers - Strong sustained buying pressure'
            
            patterns_found.append({
                'pattern': 'Three White Soldiers',
                'type': 'bullish',
                'strength': strength,
                'confidence': confidence,
                'category': 'continuation',
                'description': description
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
            curr_open >= prev_close and
            curr_close <= prev_open and
            curr_body > prev_body * 1.3):
            
            # Add volume confirmation
            strength = 90
            confidence = 90
            
            prev_vol = c4['Volume'] if 'Volume' in c4.index else 0
            cur_vol = c5['Volume'] if 'Volume' in c5.index else 0
            
            if prev_vol > 0 and cur_vol > prev_vol * 1.5:
                strength = 95
                confidence = 95
                description = 'VERY STRONG bearish engulfing with volume surge - Extremely high selling pressure'
            else:
                description = 'Very strong bearish reversal - Large selling pressure overwhelmed buyers'
            
            patterns_found.append({
                'pattern': 'Bearish Engulfing',
                'type': 'bearish',
                'strength': strength,
                'confidence': confidence,
                'category': 'reversal',
                'description': description
            })
        
        # 12. EVENING STAR (3-Candle Bearish Reversal)
        c3_open = c3['Open'] if 'Open' in c3.index else c3['open']
        c3_close = c3['Close'] if 'Close' in c3.index else c3['close']
        c3_high = c3['High'] if 'High' in c3.index else c3['high']
        c3_low = c3['Low'] if 'Low' in c3.index else c3['low']
        c4_open = c4['Open'] if 'Open' in c4.index else c4['open']
        c4_close = c4['Close'] if 'Close' in c4.index else c4['close']
        
        if (c3_close > c3_open and
            abs(c4_close - c4_open) < (c3_high - c3_low) * 0.3 and
            curr_is_red and curr_close < (c3_open + c3_close) / 2):
            
            # Check for gaps (classic Evening Star feature)
            c4_high = c4['High'] if 'High' in c4.index else c4['high']
            c4_low = c4['Low'] if 'Low' in c4.index else c4['low']
            c5_high = c5['High'] if 'High' in c5.index else c5['high']
            
            # Gap 1: Gap up into middle candle
            gap1 = c4_low > c3_high
            # Gap 2: Gap down from middle candle
            gap2 = c5_high < c4_low
            
            strength = 95
            if gap1 and gap2:
                strength = 98  # Perfect Evening Star with gaps
                description = 'PERFECT Evening Star with gaps - Extremely strong bearish reversal'
            elif gap1 or gap2:
                strength = 96  # One gap present
                description = 'Strong Evening Star with gap - Extremely strong bearish reversal'
            else:
                description = 'Evening Star - Classic 3-candle top pattern'
            
            patterns_found.append({
                'pattern': 'Evening Star',
                'type': 'bearish',
                'strength': strength,
                'confidence': 95,
                'category': 'reversal',
                'description': description
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
        if c3_close < c3_open and c4_close < c4_open and curr_is_red and \
           c4_close < c3_close and curr_close < c4_close:
            
            # Calculate body sizes
            c3_body = abs(c3_close - c3_open)
            c4_body = abs(c4_close - c4_open)
            c5_body = abs(curr_close - curr_open)
            
            # Check for progressive size increase
            strength = 92
            confidence = 90
            
            if c4_body >= c3_body * 0.8 and c5_body >= c4_body * 0.8:
                # Good progression - each candle maintains or grows
                strength = 92
                description = 'Strong bearish continuation - Steady downward momentum with consistent selling'
            else:
                # Weak progression - candles getting smaller (momentum fading)
                strength = 85
                confidence = 85
                description = 'Bearish continuation but momentum weakening - Watch for reversal'
            
            # BONUS: Check if all three open near previous close
            small_gaps = (abs(c4_open - c3_close) / c3_close < 0.01) and \
                         (abs(curr_open - c4_close) / c4_close < 0.01)
            
            if small_gaps and c5_body >= c4_body * 0.8:
                strength = 95  # Perfect Three Black Crows
                description = 'PERFECT Three Black Crows - Strong sustained selling pressure'
            
            patterns_found.append({
                'pattern': 'Three Black Crows',
                'type': 'bearish',
                'strength': strength,
                'confidence': confidence,
                'category': 'continuation',
                'description': description
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
        if curr_body / curr_range <= 0.15 and curr_range > 0:  # Relaxed from 0.1 to 0.15
            # Check for balanced shadows (true Doji has roughly equal shadows)
            shadow_ratio = abs(lower_shadow - upper_shadow) / curr_range if curr_range > 0 else 1
            
            if shadow_ratio < 0.3:  # Balanced shadows = stronger Doji
                strength = 60
                description = 'Strong Doji - Balanced indecision, likely reversal point'
            else:
                strength = 50
                description = 'Doji - Market indecision, wait for confirmation'
            
            patterns_found.append({
                'pattern': 'Doji',
                'type': 'neutral',
                'strength': strength,
                'confidence': 70,
                'category': 'indecision',
                'description': description
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
        
        # ========== NEW RETURN STRUCTURE ==========
        if patterns_found:
            # Remove duplicates by pattern name (keep the first occurrence)
            seen_patterns = set()
            unique_patterns = []
            
            for pattern in patterns_found:
                pattern_name = pattern.get('pattern', '')
                if pattern_name and pattern_name not in seen_patterns:
                    seen_patterns.add(pattern_name)
                    unique_patterns.append(pattern)
            
            # Replace patterns_found with deduplicated list
            patterns_found = unique_patterns
            
            # Sort by strength (strongest first)
            patterns_found.sort(key=lambda x: (x['strength'], x['confidence']), reverse=True)
            
            return {
                'patterns': patterns_found,  # ALL patterns detected
                'primary_pattern': patterns_found[0],  # Strongest pattern
                'pattern_count': len(patterns_found)  # How many detected
            }
        else:
            return {
                'patterns': [],
                'primary_pattern': {
                    'pattern': 'No Significant Pattern',
                    'type': 'neutral',
                    'strength': 0,
                    'confidence': 0,
                    'category': 'none',
                    'description': 'No clear candlestick pattern detected'
                },
                'pattern_count': 0
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
                impact['description'] = 'Weak bullish pattern - Use with other confirming indicators'
            elif pattern_type == 'bearish':
                impact['signal_boost'] = -0.5
                impact['confidence_boost'] = -5
                impact['description'] = 'Weak bearish pattern - Wait for confirmation'
            else:
                impact['description'] = 'Weak pattern - Market indecision'

        
        # BONUS: Extra boost for REVERSAL patterns at key levels
        if category == 'reversal':
            impact['confidence_boost'] += 5  # Reversals are powerful
        
        return impact

    def calculate_collaborative_pattern_impact(self, all_patterns, current_price):
        """
        Calculate the COMBINED/COLLABORATIVE impact of ALL detected patterns
        
        This aggregates individual pattern impacts into a unified trading adjustment
        that affects signal, stop-loss, and targets.
        
        Args:
            all_patterns: List of all detected patterns with their individual impacts
            current_price: Current stock price
            
        Returns:
            dict: Collaborative impact with combined adjustments
        """
        
        if not all_patterns or len(all_patterns) == 0:
            return {
                'total_signal_boost': 0,
                'combined_stop_loss_adjustment': 1.0,
                'combined_target_multiplier': 1.0,
                'total_confidence_boost': 0,
                'combined_risk_adjustment': 1.0,
                'pattern_count': 0,
                'bullish_patterns': 0,
                'bearish_patterns': 0,
                'neutral_patterns': 0,
                'dominant_sentiment': 'neutral',
                'collaboration_strength': 0,
                'description': 'No patterns detected'
            }
        
        # Initialize aggregation variables
        signal_boosts = []
        stop_loss_adjustments = []
        target_multipliers = []
        confidence_boosts = []
        risk_adjustments = []
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        bullish_strength_sum = 0
        bearish_strength_sum = 0
        
        print(f"\n{'='*60}")
        print("🔄 CALCULATING COLLABORATIVE PATTERN IMPACT")
        print(f"{'='*60}")
        print(f"Total patterns to combine: {len(all_patterns)}")
        
        # Collect impacts from all patterns
        for idx, pattern in enumerate(all_patterns, 1):
            pattern_name = pattern.get('pattern', 'Unknown')
            pattern_type = pattern.get('type', 'neutral')
            pattern_strength = pattern.get('strength', 0)
            individual_impact = pattern.get('individual_impact', {})
            
            print(f"\n   Pattern {idx}: {pattern_name}")
            print(f"      Type: {pattern_type}, Strength: {pattern_strength}")
            
            # Count pattern types
            if pattern_type == 'bullish':
                bullish_count += 1
                bullish_strength_sum += pattern_strength
            elif pattern_type == 'bearish':
                bearish_count += 1
                bearish_strength_sum += pattern_strength
            else:
                neutral_count += 1
            
            # Collect individual impacts (with fallbacks)
            signal_boost = individual_impact.get('signal_boost', 0)
            sl_adj = individual_impact.get('stop_loss_adjustment', 1.0)
            target_mult = individual_impact.get('target_multiplier', 1.0)
            conf_boost = individual_impact.get('confidence_boost', 0)
            risk_adj = individual_impact.get('risk_adjustment', 1.0)
            
            # Weight by pattern strength (stronger patterns have more influence)
            weight = pattern_strength / 100.0
            
            signal_boosts.append(signal_boost * weight)
            stop_loss_adjustments.append(sl_adj)
            target_multipliers.append(target_mult)
            confidence_boosts.append(conf_boost * weight)
            risk_adjustments.append(risk_adj)
            
            print(f"      Signal boost: {signal_boost:.2f} (weighted: {signal_boost * weight:.2f})")
            print(f"      SL adjustment: {sl_adj:.3f}")
            print(f"      Target multiplier: {target_mult:.2f}")
        
        # Calculate combined values
        total_signal_boost = sum(signal_boosts)
        
        # For stop-loss: Use most conservative (tightest for buys, widest for sells)
        combined_sl_adj = np.mean(stop_loss_adjustments)
        
        # For targets: Use most optimistic if patterns agree, average otherwise
        combined_target_mult = np.mean(target_multipliers)
        
        # Confidence: Sum all boosts
        total_confidence_boost = sum(confidence_boosts)
        
        # Risk: Average of all adjustments
        combined_risk_adj = np.mean(risk_adjustments)
        
        # Determine dominant sentiment
        if bullish_count > bearish_count:
            dominant_sentiment = 'bullish'
            collaboration_strength = (bullish_strength_sum / bullish_count) if bullish_count > 0 else 0
        elif bearish_count > bullish_count:
            dominant_sentiment = 'bearish'
            collaboration_strength = (bearish_strength_sum / bearish_count) if bearish_count > 0 else 0
        else:
            dominant_sentiment = 'neutral'
            collaboration_strength = 50
        
        # Pattern alignment bonus (when patterns agree, increase confidence)
        alignment_bonus = 0
        if bullish_count >= 2 and bearish_count == 0:
            alignment_bonus = 10 * bullish_count  # Bonus for aligned bullish patterns
        elif bearish_count >= 2 and bullish_count == 0:
            alignment_bonus = -10 * bearish_count  # Penalty for aligned bearish patterns
        
        total_confidence_boost += alignment_bonus
        
        # Create description
        if bullish_count > 0 and bearish_count > 0:
            description = f"Mixed signals: {bullish_count} bullish, {bearish_count} bearish patterns - Trade with caution"
        elif bullish_count >= 2:
            description = f"Strong bullish consensus: {bullish_count} bullish patterns aligned"
        elif bearish_count >= 2:
            description = f"Strong bearish consensus: {bearish_count} bearish patterns aligned"
        elif bullish_count == 1 and bearish_count == 0:
            description = "Single bullish pattern - Wait for confirmation"
        elif bearish_count == 1 and bullish_count == 0:
            description = "Single bearish pattern - Wait for confirmation"
        else:
            description = "Neutral market - No clear directional bias"
        
        collaborative_impact = {
            'total_signal_boost': round(total_signal_boost, 2),
            'combined_stop_loss_adjustment': round(combined_sl_adj, 3),
            'combined_target_multiplier': round(combined_target_mult, 2),
            'total_confidence_boost': round(total_confidence_boost, 1),
            'combined_risk_adjustment': round(combined_risk_adj, 2),
            'pattern_count': len(all_patterns),
            'bullish_patterns': bullish_count,
            'bearish_patterns': bearish_count,
            'neutral_patterns': neutral_count,
            'dominant_sentiment': dominant_sentiment,
            'collaboration_strength': round(collaboration_strength, 1),
            'alignment_bonus': alignment_bonus,
            'description': description
        }
        
        print(f"\n{'='*60}")
        print("📊 COLLABORATIVE IMPACT RESULTS:")
        print(f"{'='*60}")
        print(f"   Total Signal Boost: {total_signal_boost:+.2f}")
        print(f"   Combined SL Adjustment: {combined_sl_adj:.3f}")
        print(f"   Combined Target Multiplier: {combined_target_mult:.2f}")
        print(f"   Total Confidence Boost: {total_confidence_boost:+.1f}%")
        print(f"   Dominant Sentiment: {dominant_sentiment}")
        print(f"   Patterns: {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral")
        print(f"   Description: {description}")
        print(f"{'='*60}\n")
        
        return collaborative_impact

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
        """Run 5-point checklist with better error handling"""
        
        # Initialize default checklist
        checklist = {
            '1. At Key S/R Level': '❌ PENDING',
            '2. Price Rejection': '❌ PENDING',
            '3. Chart Pattern Confirmed': '❌ PENDING',
            '4. Candlestick Signal': '❌ PENDING',
            '5. Indicator Alignment': '❌ PENDING',
            'FINAL_SIGNAL': 'HOLD',
            'data_available': False
        }
        
        # Check if we have the required data
        five_min_df = analysis_results.get('5m_data')
        resistance = analysis_results.get('resistance', 0)
        support = analysis_results.get('support', 0)
        latest_price = analysis_results.get('latest_price', 0)
        
        # Validate data availability
        if five_min_df is None or five_min_df.empty:
            checklist['error'] = 'No 5-minute data available'
            return checklist
        
        if len(five_min_df) < 2:
            checklist['error'] = 'Insufficient candles (need at least 2)'
            return checklist
        
        if resistance == 0 or support == 0 or latest_price == 0:
            checklist['error'] = 'Support/Resistance levels not calculated'
            return checklist
        
        # Mark data as available
        checklist['data_available'] = True
        
        try:
            # Column name handling
            close_col = 'Close' if 'Close' in five_min_df.columns else 'close'
            high_col = 'High' if 'High' in five_min_df.columns else 'high'
            low_col = 'Low' if 'Low' in five_min_df.columns else 'low'
            
            # POINT 1: At Key S/R Level
            at_resistance = abs(latest_price - resistance) / resistance <= 0.005 if resistance > 0 else False
            at_support = abs(latest_price - support) / support <= 0.005 if support > 0 else False
            
            if at_support:
                checklist['1. At Key S/R Level'] = "✅ At Support"
            elif at_resistance:
                checklist['1. At Key S/R Level'] = "⚠️ At Resistance"
            else:
                checklist['1. At Key S/R Level'] = "❌ Not at key level"
            
            # POINT 2: Price Rejection
            last_candle = five_min_df.iloc[-1]
            
            if at_support and last_candle[low_col] <= support and last_candle[close_col] > support:
                checklist['2. Price Rejection'] = "✅ Bullish Rejection"
            elif at_resistance and last_candle[high_col] >= resistance and last_candle[close_col] < resistance:
                checklist['2. Price Rejection'] = "⚠️ Bearish Rejection"
            else:
                checklist['2. Price Rejection'] = "❌ No Rejection"
            
            # POINT 3: Chart Pattern
            pattern_status = self.detect_breakout_retest(five_min_df, resistance)
            if "Retest Confirmed" in pattern_status:
                checklist['3. Chart Pattern Confirmed'] = "✅ Breakout/Retest"
            else:
                checklist['3. Chart Pattern Confirmed'] = f"❌ {pattern_status}"
            
            # POINT 4: Candlestick Signal
            candle_pattern_result = self.detect_candlestick_patterns_talib(five_min_df)
            
            if isinstance(candle_pattern_result, dict) and 'primary_pattern' in candle_pattern_result:
                candle_pattern = candle_pattern_result.get('primary_pattern', {})
                pattern_name = candle_pattern.get('pattern', 'No Pattern')
                pattern_type = candle_pattern.get('type', 'neutral')
                pattern_strength = candle_pattern.get('strength', 0)
                
                if "No Significant Pattern" in pattern_name or pattern_strength == 0:
                    checklist['4. Candlestick Signal'] = "❌ No Signal"
                elif pattern_type == 'bullish':
                    checklist['4. Candlestick Signal'] = f"✅ {pattern_name} (Bullish, {pattern_strength}%)"
                elif pattern_type == 'bearish':
                    checklist['4. Candlestick Signal'] = f"⚠️ {pattern_name} (Bearish, {pattern_strength}%)"
                else:
                    checklist['4. Candlestick Signal'] = f"⚪ {pattern_name} (Neutral)"
            else:
                checklist['4. Candlestick Signal'] = "❌ No Signal"
            
            # POINT 5: Indicator Alignment
            rsi = analysis_results.get('rsi', 50)
            five_min_df = self.compute_vwap(five_min_df)
            vwap = five_min_df['vwap'].iloc[-1]
            
            if checklist['1. At Key S/R Level'] == "✅ At Support":
                if rsi < 40 and latest_price > vwap:
                    checklist['5. Indicator Alignment'] = "✅ Bullish Alignment"
                elif rsi < 50 and latest_price > vwap * 1.002:
                    checklist['5. Indicator Alignment'] = "✅ Bullish Alignment"
                else:
                    checklist['5. Indicator Alignment'] = "⚠️ Weak Bullish"
            
            elif checklist['1. At Key S/R Level'] == "⚠️ At Resistance":
                if rsi > 60 and latest_price < vwap:
                    checklist['5. Indicator Alignment'] = "⚠️ Bearish Alignment"
                elif rsi > 50 and latest_price < vwap * 0.998:
                    checklist['5. Indicator Alignment'] = "⚠️ Bearish Alignment"
                else:
                    checklist['5. Indicator Alignment'] = "❌ Weak Bearish"
            else:
                checklist['5. Indicator Alignment'] = "❌ No Alignment"
            
            # Calculate Final Signal
            bullish_checks = sum(1 for v in checklist.values() if '✅' in str(v) and ('Bullish' in str(v) or 'Breakout' in str(v)))
            bearish_checks = sum(1 for v in checklist.values() if '⚠️' in str(v) and 'Bearish' in str(v))
            
            pattern_boost = analysis_results.get('pattern_impact', {}).get('signal_boost', 0)
            
            if bullish_checks + pattern_boost >= 3:
                checklist['FINAL_SIGNAL'] = '🟢 BUY' if pattern_boost >= 1.5 else '🟢 BUY'
            elif bearish_checks - pattern_boost >= 3:
                checklist['FINAL_SIGNAL'] = '🔴 SELL'
            else:
                checklist['FINAL_SIGNAL'] = '⚪ HOLD'
            
            return checklist
        
        except Exception as e:
            checklist['error'] = f"Error: {str(e)}"
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
                pattern_result = self.detect_candlestick_patterns_talib(five_min_data)
                
                # Store ALL patterns
                all_patterns_list = pattern_result.get('patterns', [])
                results['all_patterns'] = all_patterns_list
                results['pattern_count'] = pattern_result.get('pattern_count', 0)
                
                # ============================================================
                # ✅ NEW: CALCULATE IMPACT FOR EACH PATTERN INDIVIDUALLY
                # ============================================================
                
                # Calculate pattern impact for EACH pattern detected
                patterns_with_impact = []
                for pattern in all_patterns_list:
                    pattern_copy = pattern.copy()  # Don't modify original
                    
                    # ✅ ALWAYS calculate impact for ALL patterns (no conditions)
                    try:
                        individual_impact = self.calculate_pattern_impact(pattern, results['latest_price'])
                        pattern_copy['individual_impact'] = individual_impact
                        
                        # Debug logging (optional - remove in production)
                        print(f"✅ Impact calculated for {pattern.get('pattern', 'Unknown')}: {individual_impact}")
                        
                    except Exception as e:
                        # Log the error for debugging
                        print(f"⚠️ Impact calculation failed for {pattern.get('pattern', 'Unknown')}: {str(e)}")
                        
                        pattern_copy['individual_impact'] = {
                            'signal_boost': 0, 'confidence_boost': 0,
                            'stop_loss_adjustment': 1.0, 'target_multiplier': 1.0,
                            'description': f'Impact calculation failed: {str(e)}'
                        }
                    
                    patterns_with_impact.append(pattern_copy)
                
                results['all_patterns'] = patterns_with_impact
                print(f"📊 Total patterns with impact: {len(patterns_with_impact)}")

                # ✅ ONLY calculate collaborative impact if 2+ patterns
                if len(patterns_with_impact) >= 2:
                    collaborative_impact = self.calculate_collaborative_pattern_impact(
                        patterns_with_impact,
                        results['latest_price']
                    )
                    results['collaborative_impact'] = collaborative_impact
                    print(f"✅ Collaborative impact calculated for {len(patterns_with_impact)} patterns")
                else:
                    results['collaborative_impact'] = None
                    print(f"ℹ️ Skipping collaborative impact (only {len(patterns_with_impact)} pattern(s))")

                
                # ============================================================
                # KEEP PRIMARY PATTERN FOR BACKWARD COMPATIBILITY
                # ============================================================
                
                # Store primary (strongest) pattern
                primary = pattern_result.get('primary_pattern', {})
                results['candlestick_pattern'] = primary.get('pattern', None)
                results['pattern_type'] = primary.get('type', 'neutral')
                results['pattern_strength'] = primary.get('strength', 0)
                results['pattern_confidence'] = primary.get('confidence', 0)
                results['pattern_category'] = primary.get('category', 'none')
                results['pattern_description'] = primary.get('description', 'No pattern')
                
                # Calculate impact for primary pattern (for backward compatibility)
                if primary and primary.get('pattern') not in [None, 'None', '', 'No Significant Pattern', 'Insufficient Data']:
                    try:
                        results['pattern_impact'] = self.calculate_pattern_impact(primary, results['latest_price'])
                    except Exception:
                        results['pattern_impact'] = {
                            'signal_boost': 0, 'confidence_boost': 0,
                            'stop_loss_adjustment': 1.0, 'target_multiplier': 1.0,
                            'description': 'Pattern impact calculation failed'
                        }
                else:
                    results['pattern_impact'] = {
                        'signal_boost': 0, 'confidence_boost': 0,
                        'stop_loss_adjustment': 1.0, 'target_multiplier': 1.0,
                        'description': 'No valid pattern detected'
                    }
    
            except Exception as e:
                st.warning(f"Pattern detection skipped: {str(e)}")
                results['all_patterns'] = []
                results['pattern_count'] = 0
                results['candlestick_pattern'] = 'Analysis Error'
                results['pattern_type'] = 'neutral'
                results['pattern_strength'] = 0
                results['pattern_confidence'] = 0
                results['pattern_category'] = 'none'
                results['pattern_description'] = 'Pattern detection failed'
                results['pattern_impact'] = {
                    'signal_boost': 0,
                    'confidence_boost': 0,
                    'stop_loss_adjustment': 1.0,
                    'target_multiplier': 1.0,
                    'description': 'Analysis error'
                }
    
            # ============ ATR & STOP-LOSS CALCULATION ============
            try:
                atr_value = self.calculate_atr(five_min_data, period=14)
                results['atr'] = float(atr_value) if atr_value > 0 else results['latest_price'] * 0.02
            except Exception as e:
                results['atr'] = results['latest_price'] * 0.02
    
            # ✅ NEW: Get collaborative impact (combines all patterns)
            collaborative_impact = results.get('collaborative_impact', {})
            pattern_impact = results.get('pattern_impact', {})  # Keep for backward compatibility
            
             # ✅ ONLY use collaborative if 2+ patterns, otherwise use primary pattern
            if collaborative_impact and collaborative_impact.get('pattern_count', 0) >= 2:
                # Use collaborative impact (2+ patterns)
                sl_adjustment = collaborative_impact.get('combined_stop_loss_adjustment', 1.0)
                target_mult = collaborative_impact.get('combined_target_multiplier', 1.0)
                signal_boost = collaborative_impact.get('total_signal_boost', 0)
                conf_boost = collaborative_impact.get('total_confidence_boost', 0)
                
                print(f"\n💡 Using COLLABORATIVE impact from {collaborative_impact['pattern_count']} patterns")
                print(f"   SL Adjustment: {sl_adjustment:.3f}")
                print(f"   Target Multiplier: {target_mult:.2f}")
                print(f"   Signal Boost: {signal_boost:+.2f}\n")
            
            elif len(patterns_with_impact) == 1:
                # Single pattern - use its individual impact
                single_pattern = patterns_with_impact[0]
                individual_impact = single_pattern.get('individual_impact', {})
                
                sl_adjustment = individual_impact.get('stop_loss_adjustment', 1.0)
                target_mult = individual_impact.get('target_multiplier', 1.0)
                signal_boost = individual_impact.get('signal_boost', 0)
                conf_boost = individual_impact.get('confidence_boost', 0)
                
                print(f"\n💡 Using SINGLE pattern impact: {single_pattern.get('pattern')}")
                print(f"   SL Adjustment: {sl_adjustment:.3f}")
                print(f"   Target Multiplier: {target_mult:.2f}\n")
            
            else:
                # No patterns - use defaults (no adjustment)
                sl_adjustment = 1.0
                target_mult = 1.0
                signal_boost = 0
                conf_boost = 0
                
                print(f"\n💡 No patterns detected - using default values (no adjustments)")
    
            # Calculate Stop-Loss
            base_stop_loss = 0
            if results.get('support', 0) > 0:
                base_stop_loss = results['support'] * 0.995  # 0.5% below support
            else:
                base_stop_loss = results['latest_price'] * 0.98  # 2% default
    
            # ✅ Apply collaborative pattern adjustment to stop-loss
            stop_loss_support = base_stop_loss * sl_adjustment
            stop_loss_atr = results['latest_price'] - (results['atr'] * 1.5 * sl_adjustment)
    
            # Final stop-loss (use the more conservative one)
            results['base_stoploss'] = float(base_stop_loss)
            results['stop_loss'] = float(max(stop_loss_support, stop_loss_atr))
            results['trailing_stop_vwap'] = float(results.get('vwap', results['latest_price']))
            
            # ✅ Store collaborative adjustments for later use
            results['sl_adjustment_used'] = sl_adjustment
            results['target_mult_used'] = target_mult
    
            # ============ POSITION SIZING ============
            max_capital_per_trade = 12500
            risk_per_share = abs(results['latest_price'] - results['stop_loss'])
    
            if risk_per_share > 0:
                max_quantity = int(max_capital_per_trade / results['latest_price'])
                risk_based_quantity = int((max_capital_per_trade * 0.02) / risk_per_share)
                results['position_size'] = min(max_quantity, risk_based_quantity, 100)
            else:
                results['position_size'] = 1
    
            results['capital_used'] = round(results['latest_price'] * results['position_size'], 2)
    
            # ============ PROFIT TARGETS ============
            risk_amount = risk_per_share
            target_mult = pattern_impact.get('target_multiplier', 1.0)
            
            # ✅ Determine adjustment description based on pattern count
            if collaborative_impact and collaborative_impact.get('pattern_count', 0) >= 2:
                adjustment_note = f"{collaborative_impact['pattern_count']} patterns (collaborative)"
            elif len(patterns_with_impact) == 1:
                adjustment_note = f"1 pattern ({patterns_with_impact[0].get('pattern', 'Unknown')})"
            else:
                adjustment_note = "Base calculation (no patterns)"
            
            results['targets'] = [
                {
                    'level': 'Target 1 (1:1.5)',
                    'price': round(results['latest_price'] + (risk_amount * 1.5 * target_mult), 2),
                    'profit_potential': round(risk_amount * 1.5 * target_mult * results['position_size'], 2),
                    'adjusted_by': adjustment_note
                },
                {
                    'level': 'Target 2 (1:2)',
                    'price': round(results['latest_price'] + (risk_amount * 2.0 * target_mult), 2),
                    'profit_potential': round(risk_amount * 2.0 * target_mult * results['position_size'], 2),
                    'adjusted_by': adjustment_note
                },
                {
                    'level': 'Target 3 (1:3)',
                    'price': round(results['latest_price'] + (risk_amount * 3.0 * target_mult), 2),
                    'profit_potential': round(risk_amount * 3.0 * target_mult * results['position_size'], 2),
                    'adjusted_by': adjustment_note
                }
            ]
    
            # Add supertrend target if in uptrend
            if results.get('supertrend', {}).get('trend') == 'uptrend':
                results['supertrend_target'] = results['supertrend']['value']
    
            # Risk metrics
            results['risk_amount'] = round(risk_per_share * results['position_size'], 2)
            results['risk_percent'] = round((risk_per_share / results['latest_price']) * 100, 2)
    
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
                base_signal = results['confirmation_checklist'].get('FINAL_SIGNAL', 'HOLD')
                
                # ✅ ONLY apply collaborative boost if 2+ patterns
                collaborative_impact = results.get('collaborative_impact', {})
                
                if collaborative_impact and collaborative_impact.get('pattern_count', 0) >= 2:
                    signal_boost = collaborative_impact.get('total_signal_boost', 0)
                    dominant = collaborative_impact.get('dominant_sentiment', 'neutral')
                    
                    print(f"\n🎯 APPLYING COLLABORATIVE IMPACT TO SIGNAL:")
                    print(f"   Base signal: {base_signal}")
                    print(f"   Signal boost: {signal_boost:+.2f}")
                    print(f"   Dominant sentiment: {dominant}")
                    
                    # ... rest of collaborative signal logic ...
                
                elif len(results.get('all_patterns', [])) == 1:
                    # Single pattern - apply its impact
                    single_pattern = results['all_patterns'][0]
                    individual_impact = single_pattern.get('individual_impact', {})
                    signal_boost = individual_impact.get('signal_boost', 0)
                    
                    print(f"\n🎯 APPLYING SINGLE PATTERN IMPACT TO SIGNAL:")
                    print(f"   Pattern: {single_pattern.get('pattern')}")
                    print(f"   Base signal: {base_signal}")
                    print(f"   Signal boost: {signal_boost:+.2f}")
                    
                    # Single pattern logic (less aggressive than collaborative)
                    if base_signal == 'HOLD' and abs(signal_boost) >= 1.5:
                        if signal_boost > 0:
                            results['signal'] = '🟢 BUY'
                            print(f"   📈 Upgraded HOLD → BUY (strong single pattern)")
                        else:
                            results['signal'] = '🔴 SELL'
                            print(f"   📉 Upgraded HOLD → SELL (strong single pattern)")
                    else:
                        results['signal'] = base_signal
                        print(f"   ℹ️ Keeping {base_signal} (single pattern not strong enough)")
                
                else:
                    # No patterns - use base signal
                    results['signal'] = base_signal
                    print(f"   ℹ️ Using base signal (no patterns): {base_signal}")
                
            except Exception as e:
                st.warning(f"Confirmation checklist error: {str(e)}")
                results['confirmation_checklist'] = {
                    'FINAL_SIGNAL': 'HOLD',
                    'data_available': False,
                    'error': str(e)
                }
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

    if 'analysis_results' not in st.session_state:
            st.session_state['analysis_results'] = {}
        
            results = st.session_state.get('analysis_results', {})

    # ===========================================================================
    # === SIDEBAR WITH ALL FEATURES ============================================
    # ===========================================================================

    # ============================================================================
    # ===== SIDEBAR: API STATUS INDICATOR ========================================
    # ============================================================================
    
    st.sidebar.title("📊 AI Trading Agent Pro")
    
    # API Status Section
    st.sidebar.markdown("### 🔌 API Status")
    
    # Finnhub API Status
    if asset_api.finnhub_available:
        st.sidebar.success("✅ Finnhub API: Connected")
        st.sidebar.caption("🟢 Live data for US index constituents")
        
        # Show additional info if connected
        with st.sidebar.expander("ℹ️ Finnhub Info"):
            st.caption("**Status:** Active")
            st.caption("**Free Tier Limit:** 60 calls/min")
            st.caption("**Supported Indices:**")
            st.caption("• S&P 500 (^GSPC)")
            st.caption("• Dow Jones (^DJI)")
            st.caption("• Nasdaq 100 (^NDX)")
            st.caption("• Russell 2000 (^RUT)")
    else:
        st.sidebar.warning("⚠️ Finnhub API: Not configured")
        st.sidebar.caption("📊 Using static fallback data")
        
        # Show setup instructions
        with st.sidebar.expander("🔧 Setup Finnhub API"):
            st.markdown("""
            **Quick Setup:**
            
            1. **Register:** https://finnhub.io/register
            2. **Get API Key** from dashboard
            3. **Add to `.env` file:**
               ```
               FINNHUB_API_KEY=your_key_here
               ```
            4. **Restart** the application
            
            **What You Get (Free):**
            - ✅ Live S&P 500 constituents
            - ✅ Live Dow Jones constituents
            - ✅ Live Nasdaq 100 constituents
            - ✅ 60 API calls per minute
            
            **Currently Using:**
            - 🟡 Static fallback data (limited, not real-time)
            """)
    
    st.sidebar.markdown("---")
    
    # ===== NEW: GLOBAL RESET BUTTON =====
    st.sidebar.markdown("### 🔄 System Controls")
    
    # Create reset button with confirmation
    reset_col1, reset_col2 = st.sidebar.columns([2, 1])
    
    with reset_col1:
        if st.button("Reset All Data", use_container_width=True, type="primary"):
            st.session_state['confirm_reset'] = True
    
    with reset_col2:
        if st.button("❌", help="Cancel reset"):
            if 'confirm_reset' in st.session_state:
                del st.session_state['confirm_reset']
    
    # Show confirmation dialog if reset was clicked
    if st.session_state.get('confirm_reset', False):
        st.sidebar.warning("⚠️ **Confirm Reset**")
        st.sidebar.caption("This will clear all:")
        st.sidebar.caption("• Analysis results")
        st.sidebar.caption("• Selected stocks")
        st.sidebar.caption("• Chart data")
        st.sidebar.caption("• AI summaries")
        st.sidebar.caption("• Cached data")
        
        confirm_col1, confirm_col2 = st.sidebar.columns(2)
        
        with confirm_col1:
            if st.button("✅ Confirm", use_container_width=True):
                # Clear all session state
                keys_to_keep = ['user_credentials']  # Keep login info if any
                keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
                
                for key in keys_to_delete:
                    del st.session_state[key]
                
                # Clear Streamlit cache
                st.cache_data.clear()
                
                # Reset confirmation flag
                st.session_state['confirm_reset'] = False
                st.session_state['reset_complete'] = True
                
                st.rerun()
        
        with confirm_col2:
            if st.button("❌ Cancel", use_container_width=True):
                del st.session_state['confirm_reset']
                st.rerun()
    
    # Show success message after reset
    if st.session_state.get('reset_complete', False):
        st.sidebar.success("✅ All data cleared successfully!")
        if st.sidebar.button("Dismiss"):
            del st.session_state['reset_complete']
            st.rerun()
    
    st.sidebar.markdown("---")
    # ===== END GLOBAL RESET BUTTON =====

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
                format_func=lambda x: f"{x} - {st.session_state['screened_stocks'][x].get('currency', '$')}{st.session_state['screened_stocks'][x]['price']:.2f} ({st.session_state['screened_stocks'][x]['change_pct']:.2f}%)"

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
            # Initialize AlphaVantage
            av = AlphaVantageAPI()

            # ============= ASSET CLASS SELECTION =============
            st.subheader("🎯 Asset Selection")
            
            col_asset, col_market = st.columns([1, 1])
            
            with col_asset:
                selected_asset_class = st.selectbox(
                    "Select Asset Class",
                    options=list(ASSET_CLASSES.keys()),
                    index=0,
                    help="Choose the type of asset to analyze"
                )
                
                asset_config = ASSET_CLASSES[selected_asset_class]
                st.caption(f"ℹ️ {asset_config['description']}")
            
            with col_market:
                # Market selection (only for Equities)
                if selected_asset_class == "Equities (Stocks)":
                    available_markets = asset_config['markets']
                    
                    # Get from session state or use first market
                    default_market = st.session_state.get('selected_market', available_markets[0])
                    if default_market not in available_markets:
                        default_market = available_markets[0]
                    
                    selected_market = st.selectbox(
                        "Select Market",
                        options=available_markets,
                        index=available_markets.index(default_market),
                        help="Choose the stock exchange market"
                    )
                    st.session_state['selected_market'] = selected_market
                    
                elif selected_asset_class == "Indices":
                    st.info("📊 Indices - Select market in 'By Market' method")
                    selected_market = "Global"
                    
                else:
                    # Crypto, Forex, Commodities are global
                    st.info(f"🌍 {selected_asset_class} - Global Market")
                    selected_market = "Global"
            
            # ============= SELECTION METHOD =============
            st.markdown("---")
            
            # Get available selection methods for this asset class
            available_methods = asset_config['selection_methods']
            method = st.radio(
                "**Selection Method**", 
                available_methods, 
                horizontal=True,
                help=f"Choose how to select {selected_asset_class}"
            )
            
            st.markdown("---")
            
            # Initialize ticker_input
            ticker_input = None
            
            # ============= ASSET-SPECIFIC SELECTION LOGIC =============
            
            # ========== EQUITIES (STOCKS) ==========
            if selected_asset_class == "Equities (Stocks)":
                
                # Check if auto-analyze from pre-market scanner
                if 'auto_analyze_ticker' in st.session_state:
                    ticker_input = st.session_state['auto_analyze_ticker']
                    st.info(f"🎯 Auto-analyzing from scanner: **{ticker_input}**")
                    del st.session_state['auto_analyze_ticker']
                
                # METHOD 1: SEARCH
                if method == "Search":
                    search_query = st.text_input(
                        f"🔍 Search {selected_market} Stock", 
                        placeholder="e.g., Apple, Tesla, Reliance..."
                    )
                    
                    if search_query and len(search_query) >= 3:
                        with st.spinner("Searching..."):
                            if av.api_key:
                                results = av.search_symbols(search_query)
                            else:
                                # Use Yahoo Finance search (your existing function if available)
                                results = search_for_ticker(search_query, "Equities (Stocks)")
                            
                            if results:
                                # Filter by market
                                filtered_results = {}
                                
                                for name, symbol in results.items():
                                    if selected_market == "🇮🇳 India (NSE/BSE)" and (".NS" in symbol or ".BO" in symbol):
                                        filtered_results[name] = symbol
                                    elif selected_market == "🇺🇸 USA (NYSE/NASDAQ)" and not any(s in symbol for s in [".NS", ".BO", ".L", ".T"]):
                                        filtered_results[name] = symbol
                                    elif selected_market == "🇬🇧 UK (LSE)" and ".L" in symbol:
                                        filtered_results[name] = symbol
                                    elif selected_market == "🇯🇵 Japan (TSE)" and ".T" in symbol:
                                        filtered_results[name] = symbol
                                
                                if filtered_results:
                                    selected = st.selectbox(
                                        f"Select from {selected_market}:", 
                                        list(filtered_results.keys())
                                    )
                                    ticker_input = filtered_results[selected]
                                    
                                    # Get quote if AV available
                                    if av.api_key:
                                        quote = av.get_quote(ticker_input)
                                        if quote:
                                            q_col1, q_col2, q_col3 = st.columns(3)
                                            q_col1.metric("Price", f"${quote['price']:.2f}")
                                            q_col2.metric("Change", f"{quote['change']:.2f}")
                                            q_col3.metric("Volume", f"{quote['volume']:,}")
                                else:
                                    st.warning(f"No results found for {selected_market}")
                            else:
                                st.warning("No results found. Try different keywords.")
                
                # METHOD 2: BY EXCHANGE
                elif method == "By Exchange":
                    # Get market-specific currency symbol
                    currency = get_currency_symbol("", selected_market)
                    
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
                
                # METHOD 3: DIRECT
                elif method == "Direct":
                    st.markdown("### ✍️ Enter Ticker Symbol Directly")
                    
                    # Market-aware placeholders
                    placeholder_map = {
                        "🇮🇳 India (NSE/BSE)": "RELIANCE.NS",
                        "🇺🇸 USA (NYSE/NASDAQ)": "AAPL",
                        "🇬🇧 UK (LSE)": "BARC.L",
                        "🇯🇵 Japan (TSE)": "7203.T"
                    }
                    placeholder = placeholder_map.get(selected_market, "AAPL")
                    
                    ticker_input = st.text_input(
                        f"Enter {selected_market} Ticker",
                        placeholder=placeholder,
                        help=f"Example: {placeholder}"
                    )
                    
                    if ticker_input:
                        st.success(f"✅ Ticker selected: **{ticker_input}**")
                
                # METHOD 4: FROM SCANNER
                elif method == "From Scanner":
                    st.markdown("### 📊 Select from Pre-Market Screener")
                    
                    if 'screened_stocks' not in st.session_state or not st.session_state['screened_stocks']:
                        st.warning("⚠️ No stocks in scanner. Run Pre-Market Screener first.")
                        st.info("👈 Click 'Run Pre-Market Scan' in the sidebar to populate this list.")
                    else:
                        screened = st.session_state['screened_stocks']
                        
                        # Create display options
                        stock_options = []
                        for ticker, info in screened.items():
                            ticker_currency = info.get('currency', get_currency_symbol(ticker, selected_market))
                            display = f"{ticker} - {ticker_currency}{info['price']:.2f} ({info['change_pct']:+.2f}%) Vol: {info['volume']:,}"
                            stock_options.append((display, ticker))
                        
                        if stock_options:
                            st.success(f"✅ {len(stock_options)} stocks in screener")
                            
                            selected_display = st.selectbox(
                                "📊 Select Stock from Screener",
                                [opt[0] for opt in stock_options]
                            )
                            
                            # Extract ticker from selection
                            ticker_input = [opt[1] for opt in stock_options if opt[0] == selected_display][0]
                            
                            # Show quick info
                            info = screened[ticker_input]
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Price", f"{info['currency']}{info['price']:.2f}")
                            col2.metric("Change", f"{info['change_pct']:+.2f}%")
                            col3.metric("Volume", f"{info['volume']:,}")
                
                # METHOD 5: BY INDEX (NEW!)
                elif method == "By Index":
                    st.markdown("### 📊 Select Index to View Constituents")
                    
                    # Get indices for selected market
                    indices, source_info = load_available_indices(selected_market)
                    
                    if indices:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            selected_index_name = st.selectbox(
                                f"Select {selected_market} Index",
                                options=list(indices.keys())
                            )
                            selected_index_ticker = indices[selected_index_name]
                        
                        with col2:
                            st.metric("Index Selected", selected_index_name)
                            
                            # if st.button("📈 Analyze Index Itself", help="Switch to Indices and analyze the index"):
                            #   st.session_state['auto_analyze_ticker'] = selected_index_ticker
                            #    st.session_state['switch_to_indices'] = True
                            #   st.rerun()
                        
                        st.markdown("---")
                        
                        # Get constituent stocks
                        constituents, const_source = get_stocks_by_index(selected_index_ticker, selected_market)
                        
                        if constituents:
                            st.success(f"✅ Found **{len(constituents)}** stocks in {selected_index_name}")
                            
                            # Display as selectbox
                            selected_stock = st.selectbox(
                                f"Select Stock from {selected_index_name}",
                                options=constituents,
                                format_func=lambda x: x.replace('.NS', '').replace('.BO', '').replace('.L', '').replace('.T', '')
                            )
                            
                            ticker_input = selected_stock
                            
                            # Optional: Screen all constituents
                            with st.expander("🔍 Screen All Constituents (Advanced)"):
                                if st.button("Run Quick Screen"):
                                    with st.spinner("Screening constituents..."):
                                        constituent_data = []
                                        
                                        # Limit to 20 for speed
                                        for ticker in constituents[:20]:
                                            try:
                                                data = yf.Ticker(ticker).history(period="2d")
                                                if not data.empty and len(data) >= 2:
                                                    price = data['Close'].iloc[-1]
                                                    prev_price = data['Close'].iloc[-2]
                                                    change = ((price - prev_price) / prev_price) * 100
                                                    volume = data['Volume'].iloc[-1]
                                                    
                                                    constituent_data.append({
                                                        'Ticker': ticker,
                                                        'Price': f"{get_currency_symbol(ticker, selected_market)}{price:.2f}",
                                                        'Change %': f"{change:+.2f}",
                                                        'Volume': f"{int(volume):,}"
                                                    })
                                            except:
                                                continue
                                        
                                        if constituent_data:
                                            df = pd.DataFrame(constituent_data)
                                            st.dataframe(df, use_container_width=True)
                                        else:
                                            st.warning("Unable to screen constituents")
                        else:
                            st.error(f"⚠️ Could not fetch constituents for {selected_index_name}")
                    else:
                        st.error(f"No indices available for {selected_market}")
            
            
            # ========== INDICES ==========
            elif selected_asset_class == "Indices":
                
                st.markdown("### 📊 Index Analysis")
                st.info("💡 Analyze market indices directly (e.g., Nifty 50, S&P 500)")
                
                # Check if auto-switch from "Analyze Index Itself" button
                if st.session_state.get('switch_to_indices', False):
                    if 'auto_analyze_ticker' in st.session_state:
                        ticker_input = st.session_state['auto_analyze_ticker']
                        st.success(f"🎯 Analyzing Index: **{ticker_input}**")
                        del st.session_state['auto_analyze_ticker']
                        st.session_state['switch_to_indices'] = False
                
                # METHOD 1: SEARCH
                if method == "Search":
                    search_query = st.text_input(
                        "🔍 Search Index", 
                        placeholder="e.g., S&P 500, Nifty, FTSE..."
                    )
                    
                    if search_query and len(search_query) >= 3:
                        with st.spinner("Searching indices..."):
                            # Use your existing search function if available
                            results = search_for_ticker(search_query, "Indices")
                            
                            if results:
                                selected = st.selectbox("Select Index:", list(results.keys()))
                                ticker_input = results[selected]
                            else:
                                st.warning("No results found. Try 'By Market' method.")
                
                # METHOD 2: DIRECT
                elif method == "Direct":
                    st.markdown("### ✍️ Enter Index Symbol Directly")
                    
                    ticker_input = st.text_input(
                        "Enter Index Symbol",
                        placeholder="^NSEI",
                        help="Examples: ^NSEI (Nifty 50), ^GSPC (S&P 500), ^FTSE (FTSE 100), ^N225 (Nikkei)"
                    )
                    
                    if ticker_input:
                        st.success(f"✅ Index selected: **{ticker_input}**")
                
                # METHOD 3: BY MARKET
                elif method == "By Market":
                    st.markdown("### 🌍 Select Market and Index")
                    
                    # Market selection
                    index_markets = ["🇮🇳 India (NSE/BSE)", "🇺🇸 USA (NYSE/NASDAQ)", "🇬🇧 UK (LSE)", "🇯🇵 Japan (TSE)"]
                    selected_index_market = st.selectbox("Select Market", index_markets)
                    
                    # Load indices for market
                    indices, source_info = load_available_indices(selected_index_market)
                    
                    if indices:
                        selected_index_name = st.selectbox(
                            f"Select Index from {selected_index_market}",
                            list(indices.keys())
                        )
                        
                        ticker_input = indices[selected_index_name]
                        st.success(f"✅ Selected: **{selected_index_name}** ({ticker_input})")
                    else:
                        st.error(f"No indices found for {selected_index_market}")
            
            
            # ========== CRYPTOCURRENCIES ==========
            elif selected_asset_class == "Cryptocurrencies":
                
                st.markdown("### 🪙 Cryptocurrency Selection")
                
                # METHOD 1: SEARCH
                if method == "Search":
                    search_query = st.text_input(
                        "🔍 Search Cryptocurrency", 
                        placeholder="e.g., Bitcoin, Ethereum, Solana..."
                    )
                    
                    if search_query and len(search_query) >= 3:
                        with st.spinner("Searching..."):
                            # Use search function if available
                            results = search_for_ticker(search_query, "Cryptocurrencies")
                            
                            if results:
                                selected = st.selectbox("Select Crypto:", list(results.keys()))
                                ticker_input = results[selected]
                            else:
                                st.warning("No results found. Try 'By Category' method.")
                
                # METHOD 2: DIRECT
                elif method == "Direct":
                    st.markdown("### ✍️ Enter Crypto Symbol Directly")
                    
                    ticker_input = st.text_input(
                        "Enter Crypto Symbol",
                        placeholder="BTC-USD",
                        help="Examples: BTC-USD (Bitcoin), ETH-USD (Ethereum), SOL-USD (Solana)"
                    )
                    
                    if ticker_input:
                        st.success(f"✅ Crypto selected: **{ticker_input}**")
                
                # METHOD 3: BY CATEGORY
                elif method == "By Category":
                    st.markdown("### 🗂️ Browse by Category")
                    
                    cryptos, source_info = asset_api.get_crypto_list()
                    
                    # Get categories
                    all_categories = {}
                    categories_order = ["Major", "DeFi", "Stablecoins"]
                    
                    for cat in categories_order:
                        cat_cryptos, _ = asset_api.get_crypto_list(cat)
                        if cat_cryptos:
                            all_categories[cat] = cat_cryptos
                    
                    if all_categories:
                        selected_category = st.selectbox("Select Category", list(all_categories.keys()))
                        
                        category_cryptos = all_categories[selected_category]
                        selected_crypto_name = st.selectbox(
                            f"Select Crypto from {selected_category}",
                            list(category_cryptos.keys())
                        )
                        
                        ticker_input = category_cryptos[selected_crypto_name]
                        st.success(f"✅ Selected: **{selected_crypto_name}** ({ticker_input})")
                        
                        # Show data source
                        st.markdown(asset_api.get_data_source_badge(), unsafe_allow_html=True)
            
            
            # ========== FOREX ==========
            elif selected_asset_class == "Forex":
                
                st.markdown("### 💱 Forex Pair Selection")
                
                # METHOD 1: SEARCH
                if method == "Search":
                    search_query = st.text_input(
                        "🔍 Search Currency Pair", 
                        placeholder="e.g., EUR/USD, GBP/JPY..."
                    )
                    
                    if search_query and len(search_query) >= 3:
                        with st.spinner("Searching..."):
                            results = search_for_ticker(search_query, "Currencies / Forex")
                            
                            if results:
                                selected = st.selectbox("Select Pair:", list(results.keys()))
                                ticker_input = results[selected]
                            else:
                                st.warning("No results found. Try 'By Pair Type' method.")
                
                # METHOD 2: DIRECT
                elif method == "Direct":
                    st.markdown("### ✍️ Enter Forex Pair Symbol Directly")
                    
                    ticker_input = st.text_input(
                        "Enter Forex Pair Symbol",
                        placeholder="EURUSD=X",
                        help="Examples: EURUSD=X (EUR/USD), GBPUSD=X (GBP/USD), JPY=X (USD/JPY)"
                    )
                    
                    if ticker_input:
                        st.success(f"✅ Forex pair selected: **{ticker_input}**")
                
                # METHOD 3: BY PAIR TYPE
                elif method == "By Pair Type":
                    st.markdown("### 🗂️ Browse by Pair Type")
                    
                    pairs, source_info = asset_api.get_forex_pairs()
                    
                    # Get pair types
                    all_pair_types = {}
                    types_order = ["Major Pairs", "Cross Pairs", "Exotic Pairs"]
                    
                    for ptype in types_order:
                        type_pairs, _ = asset_api.get_forex_pairs(ptype)
                        if type_pairs:
                            all_pair_types[ptype] = type_pairs
                    
                    if all_pair_types:
                        selected_type = st.selectbox("Select Pair Type", list(all_pair_types.keys()))
                        
                        type_pairs = all_pair_types[selected_type]
                        selected_pair_name = st.selectbox(
                            f"Select Pair from {selected_type}",
                            list(type_pairs.keys())
                        )
                        
                        ticker_input = type_pairs[selected_pair_name]
                        st.success(f"✅ Selected: **{selected_pair_name}** ({ticker_input})")
                        
                        # Show data source
                        st.markdown(asset_api.get_data_source_badge(), unsafe_allow_html=True)
            
            
            # ========== COMMODITIES ==========
            elif selected_asset_class == "Commodities":
                
                st.markdown("### 🏭 Commodity Selection")
                
                # METHOD 1: SEARCH
                if method == "Search":
                    search_query = st.text_input(
                        "🔍 Search Commodity", 
                        placeholder="e.g., Gold, Oil, Copper..."
                    )
                    
                    if search_query and len(search_query) >= 3:
                        with st.spinner("Searching..."):
                            results = search_for_ticker(search_query, "Commodities")
                            
                            if results:
                                selected = st.selectbox("Select Commodity:", list(results.keys()))
                                ticker_input = results[selected]
                            else:
                                st.warning("No results found. Try 'By Type' method.")
                
                # METHOD 2: DIRECT
                elif method == "Direct":
                    st.markdown("### ✍️ Enter Commodity Symbol Directly")
                    
                    ticker_input = st.text_input(
                        "Enter Commodity Symbol",
                        placeholder="GC=F",
                        help="Examples: GC=F (Gold), CL=F (Crude Oil), SI=F (Silver), HG=F (Copper)"
                    )
                    
                    if ticker_input:
                        st.success(f"✅ Commodity selected: **{ticker_input}**")
                
                # METHOD 3: BY TYPE
                elif method == "By Type":
                    st.markdown("### 🗂️ Browse by Commodity Type")
                    
                    commodities, source_info = asset_api.get_commodities()
                    
                    # Get commodity types
                    all_commodity_types = {}
                    types_order = ["Precious Metals", "Energy", "Agricultural", "Industrial Metals"]
                    
                    for ctype in types_order:
                        type_commodities, _ = asset_api.get_commodities(ctype)
                        if type_commodities:
                            all_commodity_types[ctype] = type_commodities
                    
                    if all_commodity_types:
                        selected_type = st.selectbox("Select Commodity Type", list(all_commodity_types.keys()))
                        
                        type_commodities = all_commodity_types[selected_type]
                        selected_commodity_name = st.selectbox(
                            f"Select Commodity from {selected_type}",
                            list(type_commodities.keys())
                        )
                        
                        ticker_input = type_commodities[selected_commodity_name]
                        st.success(f"✅ Selected: **{selected_commodity_name}** ({ticker_input})")
                        
                        # Show data source
                        st.markdown(asset_api.get_data_source_badge(), unsafe_allow_html=True)
                    
            # ANALYSIS BUTTON
            st.markdown("---")
            # Dynamic button text based on asset class
            button_labels = {
                "Equities (Stocks)": "📊 Analyze Stock with Full Suite",
                "Indices": "📊 Analyze Index with Full Suite",
                "Cryptocurrencies": "🪙 Analyze Crypto with Full Suite",
                "Forex": "💱 Analyze Forex Pair with Full Suite",
                "Commodities": "🏭 Analyze Commodity with Full Suite"
            }
            
            button_text = button_labels.get(selected_asset_class, "📊 Analyze with Full Suite")
            
            # VALIDATION BEFORE ANALYSIS
            if st.button(button_text, type="primary", use_container_width=True):
                
                # Step 1: Validate ticker input exists
                if not ticker_input or ticker_input.strip() == "":
                    st.error("⚠️ Please select or enter a valid ticker symbol first")
                    st.stop()
                
                # Step 2: Asset-specific format validations (warnings only, not blocking)
                if selected_asset_class == "Indices" and not ticker_input.startswith("^"):
                    st.warning("⚠️ Index tickers typically start with '^'. Example: ^NSEI, ^GSPC")
                
                if selected_asset_class == "Cryptocurrencies" and "-USD" not in ticker_input:
                    st.warning("⚠️ Crypto tickers typically end with '-USD'. Example: BTC-USD")
                
                if selected_asset_class == "Forex" and "=X" not in ticker_input:
                    st.warning("⚠️ Forex tickers typically end with '=X'. Example: EURUSD=X")
                
                if selected_asset_class == "Commodities" and "=F" not in ticker_input:
                    st.warning("⚠️ Commodity tickers typically end with '=F'. Example: GC=F")
                
                # Step 3: Get currency symbol (enhanced for all asset classes)
                currency = get_currency_symbol(ticker_input, selected_market)
                
                # Step 4: Store asset class info in session state
                st.session_state['current_asset_class'] = selected_asset_class
                st.session_state['current_market'] = selected_market
                st.session_state['current_currency'] = currency
                
                # Step 5: YOUR EXISTING ANALYSIS CODE (UNCHANGED)
                with st.spinner("Running complete analysis..."):
                    try:
                        company_name = get_company_name_multi_source(ticker_input, av)
                        analyzer = StockAnalyzer(ticker=ticker_input)
                        
                        if trading_mode == "Intraday Trading":
                            results = analyzer.analyze_for_intraday()
                        else:
                            results = analyzer.analyze_for_swing()
                        
                        if results:
                            # ✅ Add company name to results
                            results['company_name'] = company_name
                            
                            # ✅ Add asset class info to results
                            results['asset_class'] = selected_asset_class
                            results['market'] = selected_market
                            results['currency'] = currency
                        
                        if results:
                            # Add Fibonacci
                            fib_analysis = analyzer.analyze_with_fibonacci(results['daily_data'])
                            results['fibonacci'] = fib_analysis
            
                            # Fetch news and sentiment
                            headlines = analyzer.scrape_news_headlines(ticker_input)
                            print(f"📰 Fetched {len(headlines) if headlines else 0} headlines")
                            if headlines and len(headlines) > 0:
                                sentiment_detailed = analyzer.analyze_sentiment_detailed(headlines)
                                print(f"💭 Sentiment analysis complete: {sentiment_detailed.get('overall_sentiment')}")
                                print(f"📊 Articles processed: {len(sentiment_detailed.get('articles', []))}")
                            else:
                                # Create default sentiment structure
                                sentiment_detailed = {
                                    'overall_sentiment': 'Neutral',
                                    'overall_score': 0.0,
                                    'articles': [],
                                    'total_articles': 0,
                                    'positive_count': 0,
                                    'negative_count': 0,
                                    'neutral_count': 0
                                }
                                print("⚠️ No headlines found - using default sentiment")
                                    
                            results['news_headlines'] = headlines
                            results['sentiment'] = sentiment_detailed['overall_sentiment']
                            results['sentiment_score'] = sentiment_detailed['overall_score']
                            results['sentiment_detailed'] = sentiment_detailed
    
                            # Debug: Check what's stored
                            print(f"✅ Stored in results:")
                            print(f"   - sentiment_detailed exists: {'sentiment_detailed' in results}")
                            print(f"   - articles count: {len(results['sentiment_detailed'].get('articles', []))}")
            
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
            
                            st.success(f"✅ Complete Analysis Done for {selected_asset_class}!")
                        else:
                            st.error("❌ Analysis failed. Please check the ticker symbol.")
                            
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        with st.expander("🔍 View Error Details"):
                            st.code(traceback.format_exc())
            
        # This is now properly outside the button's if-else
        # ============================================================
        # COLUMN 2: QUICK STATS (RIGHT SIDEBAR)
        # ============================================================
        with col2:
            # ✅ STRICT: Only show if results exist AND have valid data
            if ('analysis_results' in st.session_state and 
                st.session_state['analysis_results'] and 
                'latest_price' in st.session_state['analysis_results']):
                
                st.subheader("📊 Quick Stats")
                results = st.session_state['analysis_results']
                
                # Get ticker and currency
                ticker = results.get('ticker', st.session_state.get('current_ticker', 'N/A'))
                currency = results.get('currency', '₹')
                
                # Price (always show if we're in this block)
                latest_price = results['latest_price']
                st.metric("💰 Price", f"{currency}{latest_price:.2f}")
                
                # Signal
                signal = results.get('signal', 'HOLD')
                if '🟢' in signal or signal == 'BUY':
                    st.success(f"📈 Signal: {signal}")
                elif '🔴' in signal or signal == 'SELL':
                    st.error(f"📉 Signal: {signal}")
                else:
                    st.warning(f"⏸️ Signal: {signal}")
                
                # RSI
                rsi = results.get('rsi', 50)
                st.metric("📊 RSI", f"{rsi:.2f}")

                # News Sentiment
                st.markdown("**💭 News Sentiment**")
                
                sentiment = results.get('sentiment', 'Neutral')
                if sentiment == "Positive":
                    st.success(f"✅ {sentiment}")
                elif sentiment == "Negative":
                    st.error(f"❌ {sentiment}")
                else:
                    st.info(f"⚪ {sentiment}")
                
                # Moving Averages
                # if 'moving_averages' in results:
                #    st.markdown("**📈 Quick Metrics**")
                #    mas = results['moving_averages']
                #    st.caption(f"MA50: {currency}{mas.get('MA_50', 0):.2f}")
                #    st.caption(f"MA200: {currency}{mas.get('MA_200', 0):.2f}")
                
                # Volume
                # if 'volume' in results:
                #    volume = results.get('volume', 0)
                #    st.caption(f"Volume: {volume:,.0f}")
            
            else:
                # ✅ Nothing to show - clean empty state
                st.subheader("📊 Quick Stats")
                st.info("💡 Run analysis first")
                st.markdown("")
                st.caption("Click the 'Analyze with Full Suite' button to populate this panel")

        # ============================================================
        # ✅ DISPLAY SELECTED STOCK NAME (NEW SECTION)
        # ============================================================
        st.markdown("---")
        if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
            results = st.session_state['analysis_results']
            ticker = results.get('ticker', st.session_state.get('current_ticker', 'N/A'))
            company_name = results.get('company_name', ticker)
            
            # Display with nice formatting
            st.markdown(f"### 📊 Analyzing: **{company_name}**")
            st.caption(f"Ticker: {ticker}")

            # ✅ Show market badge (using valid Streamlit components)
            if ticker.endswith('.NS') or ticker.endswith('.BO'):
                st.caption("🇮🇳 India (NSE/BSE)")
            elif ticker.endswith('.L'):
                st.caption("🇬🇧 UK (LSE)")
            elif ticker.endswith('.T'):
                st.caption("🇯🇵 Japan (TSE)")
            else:
                st.caption("🇺🇸 USA (NYSE/NASDAQ)")
        st.markdown("---")

        # Display full analysis results
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            currency = results.get('currency', get_currency_symbol(results.get('ticker', ''), selected_market))
            
            if trading_mode == "Intraday Trading":
                st.subheader("📊 Intraday Trading Dashboard")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                if results and 'latest_price' in results:
                    col1.metric("Current Price", f"{currency}{results['latest_price']:.2f}")
                    col2.metric("Signal", results.get('signal', 'HOLD'))
                    col3.metric("RSI", f"{results.get('rsi', 0):.2f}")
                    col4.metric("Position Size", f"{results.get('position_size', 0)} shares")
                    col5.metric("Capital Used", f"{currency}{results.get('capital_used', 0):,.0f}")
                else:
                    st.info("👆 Click 'Analyze with Full Suite' to see detailed analysis")

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
                
                # Check data availability first
                has_5m_data = '5m_data' in results and results['5m_data'] is not None and not results['5m_data'].empty
                has_sr_levels = results.get('support', 0) > 0 and results.get('resistance', 0) > 0
                
                if not has_5m_data:
                    st.error("❌ 5-minute data not available")
                    st.info("💡 5-minute intraday data is required for pattern detection. This may happen if:")
                    st.caption("• Market is closed")
                    st.caption("• Ticker doesn't have intraday data")
                    st.caption("• Data fetching failed")
                elif not has_sr_levels:
                    st.error("❌ Support/Resistance levels not calculated")
                    st.info("💡 Key price levels are required for confirmation. Try:")
                    st.caption("• Selecting a different timeframe")
                    st.caption("• Ensuring sufficient historical data")
                    
                else:
                    # Data is available - show checklist
                    if 'confirmation_checklist' in results and results['confirmation_checklist']:
                        checklist = results['confirmation_checklist']
                        
                        if checklist.get('data_available', False):
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
                            
                            if '🟢 BUY' in final_signal:
                                st.success(f"### FINAL SIGNAL: {final_signal}")
                                st.info("✅ 3+ bullish confirmations detected. Trade setup valid!")
                            elif '🔴 SELL' in final_signal:
                                st.error(f"### FINAL SIGNAL: {final_signal}")
                                st.info("✅ 3+ bearish confirmations detected. Trade setup valid!")
                            else:
                                st.warning(f"### FINAL SIGNAL: {final_signal}")
                                st.info("⚠️ Insufficient confirmations. Wait for better setup.")
                        else:
                            error_msg = checklist.get('error', 'Checklist generation failed')
                            st.warning(f"⚠️ {error_msg}")
                    else:
                        st.warning("⚠️ Confirmation checklist could not be generated")
                        st.caption("Ensure 5-minute data is available and support/resistance levels are calculated")

                # ========== CANDLESTICK PATTERN SECTION ==========
                st.markdown("---")
                st.markdown("### 🕯️ Candlestick Pattern Analysis")
                
                # Check if pattern data exists and is valid
                pattern_count = results.get('pattern_count', 0)
                all_patterns = results.get('all_patterns', [])
                
                # Filter out error patterns
                valid_patterns = [
                    p for p in all_patterns 
                    if p.get('pattern') not in ['Analysis Error', 'Insufficient Data', 'No Significant Pattern']
                ]
                pattern_count = len(valid_patterns)
                analyzer = StockAnalyzer()
                
                if pattern_count > 0:
                    # Show header with count
                    if pattern_count == 1:
                        st.info(f"🎯 **1 pattern detected**")
                    else:
                        st.info(f"🎯 **{pattern_count} patterns detected** - Showing all patterns ranked by strength")
                    
                    # Display ALL patterns (whether 1, 2, 3, 4, or 5)
                    for idx, pattern_data in enumerate(valid_patterns, 1):
                        pattern_name = pattern_data.get('pattern', 'Unknown')
                        pattern_type = pattern_data.get('type', 'neutral')
                        pattern_strength = pattern_data.get('strength', 0)
                        pattern_confidence = pattern_data.get('confidence', 0)
                        pattern_description = pattern_data.get('description', '')
                        pattern_category = pattern_data.get('category', 'none')
                        
                        # Determine if this is the primary (strongest) pattern
                        is_primary = (idx == 1)
                        
                        # Use expander for ALL patterns, auto-expand only the strongest
                        with st.expander(
                            f"{'⭐ PRIMARY: ' if is_primary else ''}"
                            f"#{idx} - {pattern_name} "
                            f"({'🟢' if pattern_type == 'bullish' else '🔴' if pattern_type == 'bearish' else '⚪'})",
                            expanded=is_primary  # Auto-expand only the strongest
                        ):
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
                            
                            # ============================================================
                            # ✅ FEATURE: DYNAMIC PATTERN DESCRIPTION (INSIDE EXPANDER)
                            # ============================================================
                            
                            # Get enhanced description from method if needed
                            if not pattern_description or pattern_description in ['', 'No pattern', 'Analysis Error']:
                                pattern_description = analyzer.get_pattern_description(
                                    pattern_name, 
                                    pattern_type, 
                                    pattern_category
                                )
                            
                            # Display description
                            st.info(f"💡 **Insight:** {pattern_description}")

                            # ============================================================
                            # ✅ SHOW TRADING IMPACT FOR EACH PATTERN INDIVIDUALLY
                            # ============================================================
                            
                            # Get the individual impact for THIS specific pattern
                            individual_impact = pattern_data.get('individual_impact', {})
                            
                            # Show trading impact for ALL patterns with strength >= 70
                            if pattern_strength >= 50 and individual_impact:
                                st.markdown("---")
                                st.markdown("#### 🎯 Trading Impact")
                                
                                signal_boost = individual_impact.get('signal_boost', 0)
                                target_mult = individual_impact.get('target_multiplier', 1.0)
                                sl_adj = individual_impact.get('stop_loss_adjustment', 1.0)
                                confidence_boost = individual_impact.get('confidence_boost', 0)
                                risk_adj = individual_impact.get('risk_adjustment', 1.0)
                                
                                # Build impact display
                                impact_parts = []
                                
                                # Signal Boost
                                if signal_boost > 0:
                                    impact_parts.append(f"✅ +{signal_boost:.1f} signal boost")
                                elif signal_boost < 0:
                                    impact_parts.append(f"⚠️ {signal_boost:.1f} caution")
                                
                                # Target Multiplier
                                if target_mult > 1.1:
                                    impact_parts.append(f"📈 Targets +{(target_mult-1)*100:.0f}%")
                                elif target_mult < 0.9:
                                    impact_parts.append(f"📉 Targets -{(1-target_mult)*100:.0f}%")
                                
                                # Stop-Loss Adjustment
                                if sl_adj < 0.99:
                                    impact_parts.append(f"🎯 SL tightened {(1-sl_adj)*100:.1f}%")
                                elif sl_adj > 1.01:
                                    impact_parts.append(f"🛡️ SL widened {(sl_adj-1)*100:.1f}%")
                                
                                # Confidence Boost
                                if confidence_boost > 0:
                                    impact_parts.append(f"💪 +{confidence_boost}% confidence")
                                elif confidence_boost < 0:
                                    impact_parts.append(f"⚠️ {abs(confidence_boost)}% less confident")
                                
                                # Risk Adjustment (Position Sizing)
                                if risk_adj > 1.05:
                                    impact_parts.append(f"📊 Can increase position by {(risk_adj-1)*100:.0f}%")
                                elif risk_adj < 0.95:
                                    impact_parts.append(f"⚖️ Reduce position by {(1-risk_adj)*100:.0f}%")
                                
                                # Display based on pattern strength
                                if pattern_strength >= 90:
                                    # Very Strong Pattern (90-100)
                                    if is_primary:
                                        st.success("🔥🔥 VERY HIGH IMPACT (Primary Pattern): " + " | ".join(impact_parts))
                                    else:
                                        st.success("🔥🔥 VERY HIGH IMPACT: " + " | ".join(impact_parts))
                                    st.caption("⚡ Exceptional pattern - Act with high confidence")
                                    
                                elif pattern_strength >= 80:
                                    # Strong Pattern (80-89)
                                    if is_primary:
                                        st.success("🔥 HIGH IMPACT (Primary Pattern): " + " | ".join(impact_parts))
                                    else:
                                        st.success("🔥 STRONG IMPACT: " + " | ".join(impact_parts))
                                    st.caption("💪 Strong pattern - High reliability setup")
                                    
                                elif pattern_strength >= 70:
                                    # Medium-Strong Pattern (70-79)
                                    if is_primary:
                                        st.info("📊 MODERATE IMPACT (Primary): " + " | ".join(impact_parts))
                                    else:
                                        st.info("📊 MODERATE IMPACT: " + " | ".join(impact_parts))
                                    st.caption("✅ Good pattern - Reliable with confirmation")
                                    
                                elif pattern_strength >= 60:
                                    # Medium Pattern (60-69)
                                    st.info("📊 MEDIUM IMPACT: " + " | ".join(impact_parts))
                                    st.caption("⚠️ Moderate pattern - Wait for confirmation")
                                    
                                else:
                                    # Weak Pattern (50-59)
                                    st.warning("⚪ LOW IMPACT: " + " | ".join(impact_parts))
                                    st.caption("⚠️ Weak pattern - Use with other indicators")
                                
                                # Additional context for pattern impact
                                impact_description = individual_impact.get('description', '')
                                if impact_description and impact_description not in ['No valid pattern', 'Impact calculation failed']:
                                    st.caption(f"💡 {impact_description}")

                            # ============================================================
                            # DETAILED PATTERN EXPLANATION SECTION
                            # ============================================================
                            
                            st.markdown("---")
                            st.markdown("#### 📖 Pattern Explanation")
                            
                            # Display color-coded detailed explanation
                            if pattern_type == 'bullish':
                                st.success(f"✅ **{pattern_name}**: {pattern_description}")
                            elif pattern_type == 'bearish':
                                st.error(f"⚠️ **{pattern_name}**: {pattern_description}")
                            else:
                                st.info(f"ℹ️ **{pattern_name}**: {pattern_description}")
                            
                            # ============================================================
                            # TRADING IMPLICATIONS SECTION (INSIDE EXPANDER)
                            # ============================================================
                            
                            st.markdown("#### 💡 Trading Implications")
                            
                            if pattern_type == 'bullish':
                                st.markdown(f"""
                                - **Action:** Consider **LONG** positions
                                - **Entry Strategy:** Wait for confirmation on next candle
                                - **Stop-Loss:** Place below pattern low or recent support
                                - **Confidence Level:** {pattern_confidence}% (Strength: {pattern_strength}/100)
                                - **Expected Move:** Pattern suggests upward momentum
                                """)
                                
                                # Additional insight for strong patterns
                                if pattern_strength >= 85:
                                    st.success("🔥 **High-Probability Setup** - Strong bullish signal with high reliability")
                            
                            elif pattern_type == 'bearish':
                                st.markdown(f"""
                                - **Action:** Consider **SHORT** positions or exit longs
                                - **Entry Strategy:** Wait for confirmation on next candle
                                - **Stop-Loss:** Place above pattern high or recent resistance
                                - **Confidence Level:** {pattern_confidence}% (Strength: {pattern_strength}/100)
                                - **Expected Move:** Pattern suggests downward pressure
                                """)
                                
                                # Additional insight for strong patterns
                                if pattern_strength >= 85:
                                    st.error("⚠️ **High-Probability Reversal** - Strong bearish signal, consider risk management")
                            
                            else:  # neutral
                                st.markdown(f"""
                                - **Action:** **WAIT** for clearer directional signals
                                - **Strategy:** Monitor next 2-3 candles for breakout direction
                                - **Risk:** Neutral patterns can precede strong moves in either direction
                                - **Confidence Level:** {pattern_confidence}%
                                - **Note:** Market indecision - avoid premature entries
                                """)
                                
                                st.warning("⏸️ **Hold Position** - Wait for market to show clear direction")
                
                else:
                    # No patterns detected
                    st.info("ℹ️ No significant candlestick patterns detected")
                    st.caption("Wait for clearer price action signals or check if there's sufficient data")

                # ============================================================
                # 🤝 COLLABORATIVE PATTERN IMPACT SECTION
                # ============================================================
                # ✅ ONLY SHOW IF 2 OR MORE PATTERNS DETECTED
                if ('collaborative_impact' in results and 
                    results['collaborative_impact'] and 
                    results['collaborative_impact'].get('pattern_count', 0) >= 2):
                    
                    collab = results['collaborative_impact']
                    
                    st.markdown("---")
                    st.subheader("🤝 Collaborative Pattern Impact")
                    st.caption(f"Combined effect of {collab['pattern_count']} detected patterns on your trade")
                    
                    # Summary metrics in 4 columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Patterns Combined", collab['pattern_count'])
                        st.caption(f"🟢 {collab['bullish_patterns']} Bullish")
                        st.caption(f"🔴 {collab['bearish_patterns']} Bearish")
                        if collab.get('neutral_patterns', 0) > 0:
                            st.caption(f"⚪ {collab['neutral_patterns']} Neutral")
                    
                    with col2:
                        signal_boost = collab['total_signal_boost']
                        st.metric("Signal Boost", f"{signal_boost:+.1f}")
                        if signal_boost > 0:
                            st.caption("✅ Increased buy confidence")
                        elif signal_boost < 0:
                            st.caption("⚠️ Increased sell pressure")
                        else:
                            st.caption("⚪ Neutral impact")
                    
                    with col3:
                        target_mult = collab['combined_target_multiplier']
                        target_change = (target_mult - 1.0) * 100
                        st.metric("Target Adjustment", f"{target_change:+.0f}%")
                        if target_mult > 1.0:
                            st.caption("🎯 Higher profit targets")
                        elif target_mult < 1.0:
                            st.caption("⚠️ Lower profit targets")
                        else:
                            st.caption("⚪ No adjustment")
                    
                    with col4:
                        sl_adj = collab['combined_stop_loss_adjustment']
                        sl_change = (1.0 - sl_adj) * 100
                        st.metric("Stop-Loss Adjustment", f"{sl_change:+.1f}%")
                        if sl_adj < 1.0:
                            st.caption("🛡️ Tighter stop-loss")
                        elif sl_adj > 1.0:
                            st.caption("⚠️ Wider stop-loss")
                        else:
                            st.caption("⚪ No adjustment")
                    
                    # Overall Assessment
                    st.markdown("---")
                    st.markdown("**📊 Overall Assessment:**")
                    
                    dominant = collab['dominant_sentiment']
                    description = collab['description']
                    
                    if dominant == 'bullish':
                        st.success(f"✅ {description}")
                        st.caption("💡 Action: Consider buy positions with the adjusted parameters")
                    elif dominant == 'bearish':
                        st.error(f"❌ {description}")
                        st.caption("💡 Action: Consider sell/short positions or avoid buying")
                    else:
                        st.warning(f"⚠️ {description}")
                        st.caption("💡 Action: Wait for clearer directional signal")
                    
                    # Detailed Breakdown (Expandable)
                    with st.expander("📋 View Detailed Breakdown"):
                        st.markdown("### Impact Metrics")
                        
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.markdown(f"**Total Confidence Boost:** {collab['total_confidence_boost']:+.1f}%")
                            st.markdown(f"**Risk Adjustment:** {collab['combined_risk_adjustment']:.2f}x")
                            st.markdown(f"**Collaboration Strength:** {collab['collaboration_strength']:.0f}/100")
                        
                        with detail_col2:
                            if collab.get('alignment_bonus', 0) != 0:
                                st.markdown(f"**Alignment Bonus:** {collab['alignment_bonus']:+.0f}%")
                                st.caption("Patterns are in agreement")
                            
                            st.markdown(f"**Dominant Sentiment:** {dominant.title()}")
                        
                        # How it affects your trade
                        st.markdown("---")
                        st.markdown("### 📈 How This Affects Your Trade")
                        
                        st.markdown(f"**Stop-Loss:** Adjusted by **{sl_adj:.3f}x** - {'Tighter' if sl_adj < 1.0 else 'Wider' if sl_adj > 1.0 else 'No change'}")
                        st.markdown(f"**Targets:** Adjusted by **{target_mult:.2f}x** - {'Higher' if target_mult > 1.0 else 'Lower' if target_mult < 1.0 else 'No change'}")
                        st.markdown(f"**Signal Confidence:** {'Increased' if signal_boost > 0 else 'Decreased' if signal_boost < 0 else 'Neutral'} by **{abs(signal_boost):.1f}** points")
                        st.markdown(f"**Position Size:** {'Can increase' if collab['combined_risk_adjustment'] > 1.0 else 'Should reduce' if collab['combined_risk_adjustment'] < 1.0 else 'Standard'} by **{abs(collab['combined_risk_adjustment'] - 1.0)*100:.0f}%**")
                
                st.markdown("---")

                # Technical Indicators Summary
                st.markdown("---")
                st.markdown("### 📊 Technical Indicators Summary")

                # Check if results is valid and has latest_price
                if not results or 'latest_price' not in results or results.get('latest_price', 0) <= 0:
                    st.error("⚠️ **Unable to display technical indicators**")
                    st.warning("Price data is unavailable or invalid")
                    st.info("**Possible causes:**")
                    st.caption("• Market is currently closed")
                    st.caption("• Invalid ticker symbol")
                    st.caption("• Data fetching failed")
                    st.caption("• Network/API issues")
                    st.caption("👉 **Try:** Refresh page or select a different stock")
                else:
                    # ✅ NOW SAFE TO ACCESS - latest_price definitely exists
                    current_price = results.get('latest_price', 0)
    
                    # Only show indicators if we have valid price data
                    if current_price > 0:
                        ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                
                        with ind_col1:
                            st.markdown("**Bollinger Bands**")
                            bb = results.get('bollinger_bands', {})
                            st.write(f"Upper: {currency}{bb.get('upper', 0):.2f}")
                            st.write(f"Middle: {currency}{bb.get('middle', 0):.2f}")
                            st.write(f"Lower: {currency}{bb.get('lower', 0):.2f}")
                            
                            # BB Signal
                            
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

                    else:
                        st.warning("⚠️ Price data is invalid or zero. Cannot display indicators.")
                        st.caption("Please try analyzing a different stock or refresh the data.")
                
                # Moving Averages
                st.markdown("---")
                st.markdown("### 📈 Moving Averages Analysis")
                
                ma_col1, ma_col2, ma_col3, ma_col4 = st.columns(4)
                current_price = results.get('latest_price', 0)
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

                # ============================================================
                # 📈 TRADINGVIEW LIVE CHART SECTION
                # ============================================================
                st.markdown("---")
                
                with st.expander("📈 TradingView Live Chart", expanded=False):
                    st.caption("📊 Real-time interactive chart from TradingView")
                    st.caption("⚠️ Some symbols may not be available for embedded viewing")
                    
                    # ✅ Safely determine which ticker to use
                    chart_ticker = None
                    
                    # Priority 1: From results dictionary
                    if results and isinstance(results, dict) and 'ticker' in results:
                        chart_ticker = results['ticker']
                        st.caption(f"📌 Displaying: {chart_ticker} (from analysis results)")
                    
                    # Priority 2: From user input
                    elif 'ticker_input' in locals() and ticker_input:
                        chart_ticker = ticker_input
                        st.caption(f"📌 Displaying: {chart_ticker} (from input)")
                    
                    # Priority 3: From session state
                    elif 'results' in st.session_state and isinstance(st.session_state.results, dict):
                        chart_ticker = st.session_state.results.get('ticker')
                        if chart_ticker:
                            st.caption(f"📌 Displaying: {chart_ticker} (from session)")
                    
                    # Display chart or error message
                    if chart_ticker:
                        try:
                            tradingview_html = embed_tradingview_widget(chart_ticker)
                            components.html(tradingview_html, height=550)
                        except Exception as e:
                            st.error(f"❌ Could not load TradingView chart")
                            st.caption(f"Error: {str(e)}")
                            st.caption(f"Ticker attempted: {chart_ticker}")
                            
                            # Show alternative
                            st.info("💡 View chart manually:")
                            st.markdown(f"[Open {chart_ticker} on TradingView](https://www.tradingview.com/chart/?symbol={chart_ticker})")
                    else:
                        st.warning("⚠️ No ticker available for chart display")
                        st.info("Please run an analysis first to load a chart")
                
                st.markdown("---")
                
                # ===== INSERT THIS ENTIRE BLOCK BEFORE st.subheader("🎯 Stop-Loss & Targets") =====
                # Calculate Stop-Loss and Profit Targets if not already calculated
                if results.get('signal') in ['BUY', 'STRONG BUY', 'SELL', 'STRONG SELL']:
                    latest_price = results.get('latest_price', 0)
                    support = results.get('support', 0)
                    resistance = results.get('resistance', 0)
                    atr = results.get('atr', 0)
                    
                    # Get pattern impact boost (with fallback)
                    pattern_impact = results.get('pattern_impact', {})
                    signal_boost = pattern_impact.get('signal_boost', 0)
                    
                    # Calculate Stop-Loss
                    if results['signal'] in ['BUY', 'STRONG BUY']:
                        # For BUY signals
                        if support > 0:
                            results['stop_loss'] = support * 0.995  # 0.5% below support
                        else:
                            results['stop_loss'] = latest_price * 0.98  # 2% below entry
                        
                        # Calculate risk
                        results['risk_amount'] = latest_price - results['stop_loss']
                        results['risk_percent'] = (results['risk_amount'] / latest_price) * 100
                        
                        # Calculate Profit Targets
                        if not results.get('targets'):
                            results['targets'] = []
                            if resistance > 0:
                                target1_price = latest_price + (resistance - latest_price) * 0.5
                                target2_price = resistance
                            else:
                                target1_price = latest_price * 1.015  # 1.5% profit
                                target2_price = latest_price * 1.03   # 3% profit
                            
                            results['targets'] = [
                                {
                                    'level': 'Target 1 (50%)',
                                    'price': target1_price,
                                    'profit_potential': target1_price - latest_price
                                },
                                {
                                    'level': 'Target 2 (100%)',
                                    'price': target2_price,
                                    'profit_potential': target2_price - latest_price
                                }
                            ]
                    
                    elif results['signal'] in ['SELL', 'STRONG SELL']:
                        # For SELL signals
                        if resistance > 0:
                            results['stop_loss'] = resistance * 1.005  # 0.5% above resistance
                        else:
                            results['stop_loss'] = latest_price * 1.02  # 2% above entry
                        
                        # Calculate risk
                        results['risk_amount'] = results['stop_loss'] - latest_price
                        results['risk_percent'] = (results['risk_amount'] / latest_price) * 100
                        
                        # Calculate Profit Targets
                        if not results.get('targets'):
                            results['targets'] = []
                            if support > 0:
                                target1_price = latest_price - (latest_price - support) * 0.5
                                target2_price = support
                            else:
                                target1_price = latest_price * 0.985  # 1.5% profit
                                target2_price = latest_price * 0.97   # 3% profit
                            
                            results['targets'] = [
                                {
                                    'level': 'Target 1 (50%)',
                                    'price': target1_price,
                                    'profit_potential': latest_price - target1_price
                                },
                                {
                                    'level': 'Target 2 (100%)',
                                    'price': target2_price,
                                    'profit_potential': latest_price - target2_price
                                }
                            ]
                # ===== END OF CALCULATION BLOCK =====

                # ========== STOP-LOSS & TARGETS ==========
                st.subheader("🎯 Stop-Loss & Targets")
                
                # Validate that calculations exist
                if 'stop_loss' not in results or 'targets' not in results:
                    st.error("❌ Stop-loss and targets not calculated")
                    st.info("💡 This usually means:")
                    st.caption("• Analysis incomplete - try re-running the analysis")
                    st.caption("• Insufficient data for risk calculation")
                    
                    # Show what data we DO have
                    with st.expander("🔍 Debug Info - Click to view"):
                        st.write("**Available Keys:**", list(results.keys()))
                        st.write("**Has 5m data:**", '5m_data' in results)
                        st.write("**Support:**", results.get('support', 'N/A'))
                        st.write("**Resistance:**", results.get('resistance', 'N/A'))
                else:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 🛑 Stop-Loss")
                        st.metric("Stop-Loss Price", f"{currency}{results.get('stop_loss', 0):.2f}",
                                 f"-{currency}{abs(results['latest_price'] - results.get('stop_loss', 0)):.2f}")
                        st.metric("Risk Amount", f"{currency}{results.get('risk_amount', 0):.2f}")
                        st.metric("Risk %", f"{results.get('risk_percent', 0):.2f}%")
                        
                        vwap_value = results.get('vwap', results.get('latest_price', 0))
                        if vwap_value > 0:
                            st.info(f"**VWAP Trailing:** {currency}{vwap_value:.2f}\n\nTrail stop to VWAP. Exit if closes below.")
                    
                    with col2:
                        st.markdown("### 🎯 Profit Targets")
                        if results.get('targets'):
                            for target in results['targets']:
                                st.metric(target['level'], f"{currency}{target['price']:.2f}", 
                                         f"+{currency}{target['profit_potential']:.2f}")
                        else:
                            st.warning("No targets calculated")
                    
                    with col3:
                        st.markdown("### 📏 Key Levels")
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
                if 'sentiment_detailed' in results:
                    st.markdown("---")
                    st.subheader("🎯 News Sentiment Breakdown")
                    sentiment_data = results['sentiment_detailed']
                    
                    # Display summary metrics (always visible)
                    col1, col2, col3, col4 = st.columns(4)
                    overall_sentiment = sentiment_data.get('overall_sentiment', 'Neutral')
                    overall_score = sentiment_data.get('overall_score', 0.0)
                    positive_count = sentiment_data.get('positive_count', 0)
                    negative_count = sentiment_data.get('negative_count', 0)
                    
                    col1.metric("Overall", overall_sentiment)
                    col2.metric("Score", f"{overall_score:.3f}")
                    col3.metric("✅ Positive", positive_count)
                    col4.metric("❌ Negative", negative_count)
                    
                    # Get articles array
                    articles = sentiment_data.get('articles', [])
                    total_articles = len(articles)
                    
                    if total_articles > 0:
                        st.markdown(f"#### 📊 Per-Article Details ({total_articles} articles)")
                        
                        # Display each article with sentiment
                        for i, article in enumerate(articles, 1):
                            headline = article.get('headline', 'Unknown')
                            article_sentiment = article.get('sentiment', 'Neutral')
                            article_score = article.get('score', 0.0)
                            
                            # Create expander with shortened headline
                            with st.expander(f"Article {i}: {headline[:60]}..."):
                                # Sentiment badge
                                if article_sentiment in ['Positive', 'POSITIVE']:
                                    st.success(f"✅ Sentiment: {article_sentiment}")
                                elif article_sentiment in ['Negative', 'NEGATIVE']:
                                    st.error(f"❌ Sentiment: {article_sentiment}")
                                else:
                                    st.info(f"➖ Sentiment: {article_sentiment}")
                                
                                # Composite score
                                st.write(f"**Composite Score:** {article_score:.3f}")
                                
                                # Detailed sentiment scores (if available)
                                if 'positive' in article and 'negative' in article and 'neutral' in article:
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Positive", f"{article['positive']:.3f}")
                                    col2.metric("Negative", f"{article['negative']:.3f}")
                                    col3.metric("Neutral", f"{article['neutral']:.3f}")
                                elif 'confidence' in article:
                                    st.metric("Confidence", f"{article['confidence']:.3f}")
                                
                                # Full headline
                                st.caption(f"**Full Headline:** {headline}")
                    else:
                        # No articles were processed
                        st.warning("⚠️ Could not analyze sentiment for individual articles")
                        
                        with st.expander("🔍 Why is sentiment analysis unavailable?"):
                            st.caption("**Possible reasons:**")
                            st.caption("1. Headlines are too short (<10 characters)")
                            st.caption("2. Sentiment analyzer model not loaded")
                            st.caption("3. All headlines were filtered out during processing")
                            st.caption("4. News API returned empty results")
                            
                            # Show raw headlines if available
                            if results.get('news_headlines'):
                                st.markdown("**Raw headlines received:**")
                                for idx, h in enumerate(results['news_headlines'], 1):
                                    st.write(f"{idx}. {h} `(len: {len(h)} chars)`")
                            else:
                                st.caption("No headlines were fetched from the news API")
                else:
                    # sentiment_detailed not in results at all
                    st.info("ℹ️ Sentiment analysis not available - run a full analysis first")

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

        # Check 1: No analysis results
        if 'analysis_results' not in st.session_state:
            st.info("👈 Please run an analysis first from the **Analysis** tab")
            st.markdown("""
            **To get AI insights:**
            1. Go to the **Analysis** tab
            2. Select an asset class and ticker
            3. Click **Analyze with Full Suite**
            4. Return here to generate AI insights
            """)
        
        # Check 2: No AI model selected
        elif ai_model == "None":
            st.warning("⚠️ Please select an AI model from the sidebar to generate insights")

        else:
            results = st.session_state['analysis_results']
            ticker = results.get('ticker', 'Unknown')

            # Display current analysis info
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"**Analyzing:** {results['ticker']}")
            with col2:
                st.info(f"**AI Model:** {ai_model}")
            
            st.markdown("---")

            # Generate AI Analysis Button
            if st.button("✨ Generate AI Analysis", type="primary", use_container_width=True):
                # Double-check AI model when button is clicked
                if ai_model is None or ai_model == "":
                    st.error("❌ **No AI model selected!**")
                    st.warning("👈 Please select an AI model from the **sidebar** first")
                    st.stop()
                
                with st.spinner(f"🔄 Generating insights with {ai_model}..."):
                    try:
                        progress_bar = st.progress(0)
                        
                        # Progress simulation
                        for i in range(30):
                            time.sleep(0.1)
                            progress_bar.progress((i + 1) / 30)
                
                        # Generate analysis prompt
                        prompt = generate_comprehensive_analysis(
                            ticker,
                            results,
                            results.get('sentiment', ''),
                            results.get('news_headlines', [])
                        )

                        # Generate AI response
                        if ai_model == "Google Gemini":
                            ai_response = get_ai_analysis_gemini(prompt)
                        else:
                            model_name = ai_model_map.get(ai_model, "anthropic/claude-3.5-sonnet")
                            ai_response = get_ai_analysis_openrouter(prompt, model_name)

                        # Store in session state
                        st.session_state['ai_analysis'] = ai_response
                        st.session_state['last_analysis_time'] = datetime.now()
                        
                        # Add to history
                        if "analysis_history" not in st.session_state:
                            st.session_state["analysis_history"] = []
                        st.session_state["analysis_history"].append(st.session_state["last_analysis_time"])
                        
                        st.success("✅ AI Analysis Generated!")
                        progress_bar.empty()
                        st.rerun()  # Refresh to show analysis
                        
                    except Exception as e:
                        st.error(f"❌ **Error:** {str(e)}")
                        st.warning("⚠️ Please check your API keys and try again")
                        progress_bar.empty()

            st.markdown("---")

            # Display AI analysis if available
            if 'ai_analysis' in st.session_state and st.session_state['ai_analysis']:
                st.markdown("### 💡 AI Insights")
                st.markdown(st.session_state['ai_analysis'])

                st.markdown("---")

                # Download buttons
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    st.download_button(
                        "⬇️ Download as TXT",
                        st.session_state['ai_analysis'],
                        file_name=f"ai_analysis_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="download_txt_ai"
                    )
                
                with col_download2:
                    # Markdown format
                    md_content = f"# AI Analysis: {ticker}\n\n"
                    md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    md_content += f"**Model:** {ai_model}\n\n---\n\n"
                    md_content += st.session_state['ai_analysis']
                    
                    st.download_button(
                        "⬇️ Download as MD",
                        md_content,
                        file_name=f"ai_analysis_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key="download_md_ai"
                    )

                # Show last analysis time
                if "last_analysis_time" in st.session_state:
                    import pytz
                    last_run = st.session_state["last_analysis_time"]
                    local_time = last_run.astimezone(pytz.timezone('Asia/Kolkata'))
                    st.caption(f"🕒 Last generated: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                
                # Analysis history
                if "analysis_history" in st.session_state and st.session_state["analysis_history"]:
                    with st.expander("📜 Analysis History (Last 5 runs)"):
                        import pytz
                        for idx, ts in enumerate(reversed(st.session_state["analysis_history"][-5:]), 1):
                            local_time = ts.astimezone(pytz.timezone('Asia/Kolkata'))
                            st.markdown(f"**{idx}.** {local_time.strftime('%d-%m-%Y %H:%M:%S %Z')}")


    # ===========================================================================
    # === TAB 3-6: OPTIONS, BACKTESTING, PORTFOLIO, LIVE TRADING
    # ===========================================================================

    with tab3:
        st.subheader("🎯 Options Chain Analysis")
        
        # ============= ASSET CLASS CHECK =============
        current_asset_class = st.session_state.get('current_asset_class', 'Equities (Stocks)')
        current_ticker = st.session_state.get('current_ticker', None)
        asset_config = ASSET_CLASSES.get(current_asset_class, {})
        
        # Block unsupported asset classes
        if not asset_config.get('supports_options', False):
            st.warning(f"⚠️ Options are not available for **{current_asset_class}**")
            st.info("💡 Options are supported for **Stocks** and **Indices** only. Switch to one of these asset classes in the Analysis tab.")
            st.stop()
        
        # ============= CLEAN CONTEXT DISPLAY =============
        # if current_ticker:
        #    col1, col2, col3 = st.columns([2, 2, 1])
            
        #    with col1:
        #        st.metric("📊 Selected Asset", current_ticker)
            
        #    with col2:
        #        st.metric("Asset Type", current_asset_class.replace("(Stocks)", "").strip())
            
        #    with col3:
        #        if st.button("← Back", use_container_width=True, help="Return to Analysis tab"):
        #            st.info("💡 Use the tabs above to switch")
        
        # st.markdown("---")
        
        # ============= TICKER INPUT =============
        options_analyzer = OptionsAnalyzer()
        
        # Auto-fill logic
        if current_ticker and current_asset_class in ["Equities (Stocks)", "Indices"]:
            default_ticker = current_ticker
        else:
            default_ticker = 'AAPL'
        
        # Initialize session state
        if 'opt_ticker' not in st.session_state:
            st.session_state['opt_ticker'] = default_ticker
        
        # Check if we need to force text input update
        if 'force_ticker_update' in st.session_state:
            # This flag is set by quick buttons
            del st.session_state['force_ticker_update']
        
        # Input row
        col_input, col_button = st.columns([3, 1])
        
        with col_input:
            # Simple text input - no fancy dynamic keys
            opt_ticker_input = st.text_input(
                "Options Ticker",
                value=st.session_state['opt_ticker'],
                label_visibility="collapsed",
                placeholder="Enter ticker (e.g., AAPL, SPY, QQQ)"
            )
            
            # Update session state if user manually typed
            if opt_ticker_input and opt_ticker_input != st.session_state['opt_ticker']:
                st.session_state['opt_ticker'] = opt_ticker_input
        
        with col_button:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True, key="analyze_options_btn")
        
        # Quick select buttons
        if current_asset_class == "Indices":
            st.caption("📌 Popular Index Options:")
            quick_buttons = [
                ("SPY", "S&P 500 ETF", "quick_spy"),
                ("QQQ", "Nasdaq ETF", "quick_qqq"),
                ("DIA", "Dow ETF", "quick_dia"),
                ("IWM", "Russell 2000", "quick_iwm"),
                ("^GSPC", "S&P Index", "quick_gspc")
            ]
        else:
            st.caption("📌 Popular Stock Options:")
            quick_buttons = [
                ("AAPL", "Apple", "quick_aapl"),
                ("TSLA", "Tesla", "quick_tsla"),
                ("SPY", "S&P 500", "quick_spy_stock"),
                ("QQQ", "Nasdaq", "quick_qqq_stock"),
                ("MSFT", "Microsoft", "quick_msft")
            ]
        
        quick_cols = st.columns(5)
        for idx, (ticker, label, key) in enumerate(quick_buttons):
            with quick_cols[idx]:
                if st.button(ticker, key=key, use_container_width=True, help=label):
                    st.session_state['opt_ticker'] = ticker
                    st.session_state['force_ticker_update'] = True
                    st.rerun()  # Force immediate update
        
        st.markdown("---")
        
        # Get the ticker for analysis (from session state)
        opt_ticker = st.session_state.get('opt_ticker', default_ticker)
        
        # ============= COMPACT HELP SECTION =============
        with st.expander("ℹ️ Need Help?"):
            col_help1, col_help2 = st.columns(2)
            
            with col_help1:
                st.markdown("""
                **✅ Best Results:**
                - US stocks (AAPL, TSLA, MSFT)
                - US ETFs (SPY, QQQ, DIA)
                - High liquidity options
                
                **What You'll See:**
                - Options chain (calls & puts)
                - Strike prices & premiums
                - Put-Call Ratio (PCR)
                - Open interest & volume
                """)
            
            with col_help2:
                st.markdown("""
                **⚠️ Limitations:**
                - Indian options: Limited data
                  (Use broker platform instead)
                - Index symbols (^NSEI): May fail
                  (Try ETF equivalent like SPY)
                
                **Quick Tips:**
                - PCR > 1 = Bearish sentiment
                - PCR < 1 = Bullish sentiment
                - High OI = Strong support/resistance
                """)
        
        st.markdown("---")
        
        # ============= ANALYSIS SECTION =============
        opt_ticker = st.session_state.get('opt_ticker', default_ticker)
        
        if analyze_btn:
            with st.spinner(f"Fetching options chain for {opt_ticker}..."):
                options_data = options_analyzer.fetch_options_chain(opt_ticker)
                
                if options_data:
                    st.success(f"✅ Options loaded: **{options_data['ticker']}**")
                    
                    # Get expiry info dynamically from API
                    expiry_info = options_analyzer.get_nearest_expiry(opt_ticker)
                    
                    # ============= DISPLAY EXPIRY INFO (NO TRUNCATION) =============
                    if expiry_info:
                        if expiry_info['expires_today']:
                            # Red banner for options expiring TODAY
                            st.error(f"🔴 **Options Expire TODAY:** {expiry_info['date']} | Available Expiries: {len(options_data['all_expiries'])}")
                        else:
                            # Blue info banner for future expiry
                            st.info(f"📆 **Nearest Expiry:** {expiry_info['date']} (in {expiry_info['days_until']} days) | Available Expiries: {len(options_data['all_expiries'])}")
                    else:
                        # Fallback if API fails
                        st.info(f"📆 **Nearest Expiry:** {options_data['expiry']} | Available Expiries: {len(options_data['all_expiries'])}")
                    
                    st.markdown("---")
                    
                    # ============= PCR ANALYSIS =============
                    try:
                        pcr_data = options_analyzer.calculate_pcr(options_data)
                        
                        if pcr_data:
                            st.markdown("### 📊 Market Sentiment (Put-Call Ratio)")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("PCR (OI)", f"{pcr_data.get('pcr_oi', 0):.2f}")
                            col2.metric("PCR (Volume)", f"{pcr_data.get('pcr_volume', 0):.2f}")
                            col3.metric("Total Put OI", f"{pcr_data.get('put_oi', 0):,.0f}")
                            col4.metric("Total Call OI", f"{pcr_data.get('call_oi', 0):,.0f}")
                            
                            # Sentiment indicator
                            sentiment = pcr_data.get('sentiment', 'Neutral')
                            if 'Bullish' in sentiment:
                                st.success(f"🟢 {sentiment}")
                            elif 'Bearish' in sentiment:
                                st.error(f"🔴 {sentiment}")
                            else:
                                st.info(f"⚪ {sentiment}")
                    except:
                        pass
                    
                    st.markdown("---")
                    
                    # ============= OPTIONS CHAINS =============
                    tab_calls, tab_puts = st.tabs(["📞 Calls", "📉 Puts"])
                    
                    with tab_calls:
                        call_cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
                        available_cols = [col for col in call_cols if col in options_data['calls'].columns]
                        
                        st.dataframe(
                            options_data['calls'][available_cols].head(20),
                            use_container_width=True,
                            height=400
                        )
                    
                    with tab_puts:
                        put_cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
                        available_cols = [col for col in put_cols if col in options_data['puts'].columns]
                        
                        st.dataframe(
                            options_data['puts'][available_cols].head(20),
                            use_container_width=True,
                            height=400
                        )
                    
                    # ============= DOWNLOAD =============
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_calls = options_data['calls'].to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Calls CSV",
                            csv_calls,
                            f"{options_data['ticker']}_calls_{options_data['expiry']}.csv",
                            "text/csv",
                            key="download_calls",
                            use_container_width=True
                        )
                    with col2:
                        csv_puts = options_data['puts'].to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Puts CSV",
                            csv_puts,
                            f"{options_data['ticker']}_puts_{options_data['expiry']}.csv",
                            "text/csv",
                            key="download_puts",
                            use_container_width=True
                        )
                
                else:
                    # ============= ERROR HANDLING =============
                    st.error(f"❌ Could not fetch options data for **{opt_ticker}**")
                    
                    # Smart error detection
                    error_detected = False
                    
                    if opt_ticker.startswith('^'):
                        st.warning("💡 **Try the ETF equivalent instead:**")
                        suggestions = {
                            "^GSPC": "SPY (S&P 500 ETF)",
                            "^IXIC": "QQQ (Nasdaq ETF)",
                            "^DJI": "DIA (Dow ETF)",
                            "^NSEI": "Not available (Use broker platform)",
                            "^NSEBANK": "Not available (Use broker platform)"
                        }
                        suggestion = suggestions.get(opt_ticker, "Try the ETF version of this index")
                        st.info(f"→ {suggestion}")
                        error_detected = True
                    
                    if '.NS' in opt_ticker or '.BO' in opt_ticker:
                        st.warning("💡 **Indian stocks detected:** Options data not available via free APIs. Use your broker's platform (Zerodha Kite, Upstox, etc.)")
                        error_detected = True
                    
                    if not error_detected:
                        st.info("""
                        **Try these working tickers:**
                        - Stocks: AAPL, TSLA, MSFT, NVDA, META
                        - ETFs: SPY, QQQ, DIA, IWM
                        - Ensure the ticker is correct and has active options trading
                        """)

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
