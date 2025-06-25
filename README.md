# 📈 Elite Trading Analyzer

> **Advanced trading analysis with EMA/RSI strategy, AI insights, and professional charting**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](#contributing)

![Demo](video/0625.gif)

## 🌟 Overview

The **Elite Trading Analyzer** is a sophisticated trading tool built with Streamlit that combines technical analysis, sentiment analysis, and AI-powered insights. Designed for both global markets and Indian stock indices, it provides professional-grade analysis with real-time data, advanced charting, and intelligent trading recommendations.

## 🌟 **Please ⭐ STAR this repository if you find it helpful!**
## 🔄 **Don't forget to FORK this project to contribute!**
## 👍 **LIKE and APPRECIATE this work!**

---

## 🚀 Key Features

### 📊 **Advanced Technical Analysis**
- **EMA-Based Strategy** with 20, 50, and 200-period Exponential Moving Averages
- **Optimized RSI Analysis** with custom thresholds (40 for oversold, 60 for overbought)
- **MACD Analysis** with signal line, histogram, and trend detection
- **Multi-Signal Integration** for high-confidence trading signals
- **Vectorized Calculations** for entire dataset analysis

### 🤖 **AI-Powered Intelligence**
- **DeepSeek AI Integration** via OpenRouter API for strategic summaries
- **FinBERT Sentiment Analysis** for financial news processing
- **Automated Signal Generation** with confidence scoring system
- **Context-Aware Analysis** combining technical and fundamental factors

### 🌍 **Universal Asset Coverage**
- **Global Equities** - Stocks from major exchanges worldwide
- **Cryptocurrencies** - Bitcoin, Ethereum, and altcoins
- **ETFs** - Exchange-traded funds analysis
- **Indices** - Major stock market indices
- **Commodities** - Gold, oil, and commodity futures
- **Forex** - Currency pair analysis
- **Indian Markets** - Comprehensive NIFTY coverage

### 🇮🇳 **Curated Indian Stock Lists**
- **NIFTY 50** - Top 50 Indian companies with individual stock analysis
- **BANK NIFTY** - Complete banking sector coverage
- **NIFTY AUTO** - Automotive sector analysis
- **NIFTY PHARMA** - Pharmaceutical companies
- **NIFTY METAL** - Metals and mining sector
- **NIFTY IT** - Information technology stocks
- **NIFTY FMCG** - Fast-moving consumer goods

### 📈 **Professional Charting**
- **Interactive Plotly Charts** with Price, EMA overlays, RSI, and Volume
- **TradingView Integration** - Embedded professional charts for discretionary analysis
- **Color-Coded Visualization** - Green/red volume bars based on price action
- **Multi-Panel Layout** - Price action, momentum, and volume in one view

### 📰 **News & Sentiment Analysis**
- **Real-Time News Scraping** using NewsAPI
- **Financial Sentiment Analysis** with FinBERT model
- **News Impact Assessment** on trading decisions
- **Sentiment Score Integration** into signal generation

### 🔔 **Logging & Export System**
- **Google Sheets Integration** - Automated analysis logging
- **JSON Report Export** - Complete analysis data download
- **Historical Tracking** - Track signals and performance over time
- **Professional Documentation** - Detailed analysis reports

---

## 🛠️ How This Works

### 🏗️ **Core Architecture**
The application is built around the `StockAnalyzer` class which implements a sophisticated EMA/RSI trading strategy:

#### **Signal Generation Algorithm:**
1. **RSI Component**: Bullish signals when RSI < 40, Bearish when RSI > 60
2. **MACD Component**: Bullish when histogram > 0, Bearish when histogram < 0  
3. **EMA Trend Component**: Bullish when Price > EMA20 > EMA50, Bearish when opposite
4. **Confidence Scoring**: Based on consensus strength across indicators

#### **Technical Implementation:**
- **Vectorized Processing**: Entire dataset analysis for optimal performance
- **Yahoo Finance Integration**: Real-time data via yfinance API
- **Multi-Asset Search**: Dynamic ticker discovery across asset classes
- **Error Handling**: Robust exception management and user feedback

### 🔧 **Advanced Features**

#### **Exponential Moving Averages (EMA)**
- **20-period EMA**: Short-term trend identification
- **50-period EMA**: Medium-term trend confirmation  
- **200-period EMA**: Long-term trend analysis
- **Crossover Analysis**: EMA alignment for trend strength

#### **RSI Optimization**
- **Custom Thresholds**: 40/60 levels optimized for volatile markets
- **14-period Calculation**: Standard momentum analysis
- **Trend Integration**: RSI combined with EMA signals

#### **MACD Analysis**
- **12/26/9 Configuration**: Standard MACD parameters
- **Histogram Focus**: Momentum change detection
- **Signal Line Crossovers**: Entry/exit point identification

---

## 🔧 Installation & Setup

### **Prerequisites**
```bash
Python 3.8+
Streamlit
Internet connection for real-time data
```

### **Step 1: Clone Repository**
```bash
git clone https://github.com/yourusername/elite-trading-analyzer.git
cd elite-trading-analyzer
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Environment Configuration**
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
NEWSAPI_KEY=your_newsapi_key_here
```

### **Step 4: Google Sheets Setup (Optional)**
1. Create a Google Cloud Project
2. Enable Google Sheets API
3. Create Service Account credentials
4. Download `service_account.json` and place in root directory
5. Share your Google Sheet with the service account email

### **Step 5: Run Application**
```bash
streamlit run app.py
```

---

## 📋 Requirements

```txt
streamlit>=1.28.0
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
requests>=2.31.0
beautifulsoup4>=4.12.0
transformers>=4.33.0
openai>=1.3.0
gspread>=5.11.0
google-auth>=2.23.0
python-dotenv>=1.0.0
plotly>=5.15.0
warnings
datetime
json
time
```

---

## 🎯 Usage Guide

### **1. Asset Selection**
Choose from multiple asset classes:
- **Equities**: Search global stocks by company name
- **Cryptocurrencies**: Bitcoin, Ethereum, altcoins
- **ETFs**: Exchange-traded funds
- **Indices**: Major market indices
- **Commodities**: Gold, oil, futures
- **Forex**: Currency pairs
- **Indian Lists**: Curated NIFTY categories

### **2. Search Functionality**
- **Dynamic Search**: Type company names or tickers
- **Auto-Complete**: Yahoo Finance search integration
- **Multi-Exchange**: Global market coverage
- **Asset Filtering**: Results filtered by asset type

### **3. Analysis Configuration**
- **Time Periods**: 5d, 1mo, 3mo, 6mo, ytd, 1y, 2y, 5y, max
- **Real-Time Data**: Latest market prices and volumes
- **Historical Analysis**: Complete dataset technical analysis

### **4. Comprehensive Results**
- **Trading Signal**: Buy/Sell/Hold with confidence levels
- **Key Metrics**: Current price, RSI, sentiment, signal strength
- **Technical Details**: EMA values, MACD components
- **AI Summary**: Strategic recommendations and market analysis
- **News Headlines**: Recent news with sentiment analysis
- **Professional Charts**: Both custom analysis and TradingView integration

---

## 📊 Understanding the Analysis

### **Trading Signals**
- **BUY**: When 2+ bullish conditions are met (RSI < 40, MACD histogram > 0, EMA alignment)
- **SELL**: When 2+ bearish conditions are met (RSI > 60, MACD histogram < 0, EMA misalignment)
- **HOLD**: When conditions are mixed or neutral

### **Confidence Levels**
- **High**: Strong consensus across all indicators
- **Medium**: Moderate agreement between indicators  
- **Low**: Conflicting or weak signals

### **RSI Interpretation**
- **Below 40**: Potentially oversold (bullish signal)
- **40-60**: Normal trading range
- **Above 60**: Potentially overbought (bearish signal)

### **EMA Trend Analysis**
- **Bullish Alignment**: Price > EMA20 > EMA50 > EMA200
- **Bearish Alignment**: Price < EMA20 < EMA50 < EMA200
- **Mixed Signals**: EMAs crossed or conflicting

---

## 🔬 Advanced Features

### **AI Strategy Engine**
Powered by DeepSeek AI via OpenRouter:
- Analyzes all technical indicators simultaneously
- Incorporates market sentiment from news
- Provides actionable trading strategies
- Includes risk assessment and recommendations
- Contextual analysis based on current market conditions

### **News Sentiment Integration**
- **NewsAPI Integration**: Real-time financial news
- **FinBERT Processing**: Financial domain sentiment analysis
- **Sentiment Scoring**: Quantitative sentiment measurement
- **Signal Integration**: Sentiment impact on trading decisions

### **Professional Charting**
- **Custom Plotly Charts**: Price, EMA, RSI, Volume in unified view
- **TradingView Widgets**: Professional charting for advanced analysis
- **Interactive Features**: Zoom, pan, and detailed examination
- **Color Coding**: Visual indicators for trend direction

### **Data Export & Logging**
- **Google Sheets Logging**: Automatic analysis tracking
- **JSON Reports**: Complete analysis data export
- **Historical Records**: Performance tracking over time
- **Professional Documentation**: Detailed analysis summaries

---

## 🇮🇳 Indian Stock Coverage

### **NIFTY 50 Index (50 stocks)**
Major companies including:
- **Technology**: TCS, Infosys, HCL Tech, Wipro, Tech Mahindra
- **Banking**: HDFC Bank, ICICI Bank, SBI, Kotak Mahindra, Axis Bank
- **Energy**: Reliance Industries, ONGC, BPCL, NTPC
- **Consumer**: HUL, ITC, Nestle, Britannia, Asian Paints
- **Automotive**: Maruti Suzuki, Tata Motors, M&M, Hero MotoCorp
- **Pharmaceuticals**: Sun Pharma, Dr. Reddy's, Cipla, Divis Labs

### **Sectoral Indices**
- **BANK NIFTY**: 12 major banking stocks
- **NIFTY AUTO**: 15 automotive companies  
- **NIFTY PHARMA**: 15 pharmaceutical companies
- **NIFTY METAL**: 14 metals & mining stocks
- **NIFTY IT**: 10 information technology companies
- **NIFTY FMCG**: 13 consumer goods companies

---

## 🎨 User Interface

### **Streamlit Dashboard**
- **Clean Design**: Professional financial interface
- **Real-Time Updates**: Live market data integration
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Elements**: Dynamic charts and controls

### **Visual Indicators**
- 🟢 **Green**: Bullish signals, positive sentiment, uptrends
- 🔴 **Red**: Bearish signals, negative sentiment, downtrends  
- 🟡 **Yellow**: Neutral conditions, hold signals
- 📊 **Charts**: Professional-grade visualizations

### **Information Hierarchy**
- **Key Metrics**: Primary trading information at the top
- **Technical Analysis**: Detailed indicator breakdown
- **AI Insights**: Strategic recommendations
- **News & Sentiment**: Market context and sentiment
- **Professional Charts**: Advanced analysis tools

---

## 🚀 Performance & Optimization

### **Analysis Speed**
- **Data Fetching**: 3-5 seconds for real-time data
- **Technical Calculations**: 1-2 seconds for full analysis
- **AI Processing**: 5-10 seconds for strategy summary
- **Chart Generation**: 2-3 seconds for visualizations
- **Total Analysis**: 15-25 seconds end-to-end

### **Accuracy Features**
- **Vectorized Calculations**: Efficient technical analysis
- **Multi-Signal Confirmation**: Reduced false signals
- **Confidence Scoring**: Signal reliability assessment
- **Historical Validation**: Backtested strategy components

---

## 🔐 Security & Privacy

### **API Security**
- **Environment Variables**: Secure credential storage
- **No Hardcoded Keys**: All sensitive data externalized
- **Rate Limiting**: Compliance with API limits
- **Error Handling**: Graceful failure management

### **Data Privacy**
- **No Personal Data**: Only market data processing
- **Local Processing**: Analysis performed locally
- **Optional Logging**: Google Sheets integration is optional
- **No Data Retention**: Session-based analysis only

---

## 🛠️ Customization

### **Strategy Parameters**
The core strategy can be customized by modifying:
- **RSI Thresholds**: Currently 40/60, adjustable in code
- **EMA Periods**: 20/50/200, can be modified
- **MACD Settings**: 12/26/9 configuration
- **Signal Weights**: Adjust component importance

### **UI Customization**
- **Chart Colors**: Modify Plotly color schemes
- **Layout**: Adjust Streamlit components
- **Metrics Display**: Customize key performance indicators
- **Export Formats**: Modify report structures

---

## 🔧 Troubleshooting

### **Common Issues**
1. **No Data Retrieved**: Check ticker symbol validity
2. **API Errors**: Verify internet connection and API keys
3. **Google Sheets**: Confirm service account setup
4. **Slow Performance**: Check network speed for real-time data

### **Error Handling**
- **Graceful Degradation**: App continues with limited features
- **User Feedback**: Clear error messages and suggestions
- **Fallback Options**: Alternative data sources when possible
- **Debug Information**: Detailed logging for troubleshooting

---

## 🚀 Future Enhancements

### **Planned Features**
- **Backtesting Engine**: Historical strategy performance
- **Portfolio Analysis**: Multi-asset portfolio tracking
- **Options Analysis**: Derivatives trading signals
- **Alerts System**: Email/SMS notifications for signals
- **Mobile App**: Native mobile application
- **Machine Learning**: Enhanced prediction models

### **Technical Improvements**
- **WebSocket Integration**: Real-time streaming data
- **Database Integration**: Historical data storage
- **API Rate Optimization**: More efficient data usage
- **Performance Caching**: Faster repeated analysis

---

## 🏆 What Makes This Special

### **Unique Advantages**
1. **Universal Coverage**: Global markets + Indian focus
2. **Professional Integration**: TradingView embedded charts
3. **AI Enhancement**: Context-aware strategy recommendations
4. **EMA-Based Strategy**: Trend-following with momentum confirmation
5. **Multi-Asset Support**: Stocks, crypto, ETFs, indices, forex
6. **Real-Time Analysis**: Live market data integration

### **Technical Excellence**
- **Clean Architecture**: Modular, maintainable code
- **Error Resilience**: Robust exception handling
- **Performance Optimized**: Vectorized calculations
- **User Experience**: Intuitive interface design
- **Documentation**: Comprehensive guides and comments

---

## 📚 Educational Value

### **Learning Technical Analysis**
- **EMA Strategy**: Understanding exponential moving averages
- **RSI Analysis**: Momentum oscillator interpretation
- **MACD Signals**: Trend and momentum combination
- **Multi-Timeframe**: Different period analysis

### **Market Understanding**
- **Sentiment Analysis**: News impact on markets
- **Volume Analysis**: Market participation insights  
- **Trend Recognition**: Identifying market direction
- **Risk Management**: Signal confidence assessment

---

## 💡 Best Practices

### **Using the Analyzer**
1. **Multiple Confirmations**: Don't rely on single indicators
2. **Volume Confirmation**: Check volume with price moves
3. **News Context**: Consider sentiment in decisions
4. **Timeframe Analysis**: Use appropriate data periods
5. **Risk Management**: Always use stop-losses
6. **Paper Trading**: Test strategies before real trading

### **Strategy Optimization**
- **Backtest First**: Validate on historical data
- **Market Conditions**: Adapt to trending vs. ranging markets  
- **Position Sizing**: Use appropriate risk per trade
- **Regular Review**: Monitor and adjust strategy performance

---

## 🎉 **Community & Support**

### **⭐ Star This Repository**
Help others discover this tool by starring the repository!

### **🔄 Fork and Contribute**
- Add new technical indicators
- Improve the UI/UX
- Add new asset classes
- Enhance the AI integration
- Fix bugs and optimize performance

### **📢 Share Your Experience**
- Share screenshots of successful analyses
- Discuss strategy improvements
- Report bugs and suggest features
- Help other traders learn the system

### **💬 Get Support**
- Check the documentation for common issues
- Review the code comments for technical details
- Open issues for bugs or feature requests
- Join discussions about trading strategies

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **yfinance**: Reliable financial data from Yahoo Finance
- **Streamlit**: Amazing web application framework
- **Plotly**: Professional interactive charting
- **TradingView**: Embedded professional charts
- **NewsAPI**: Real-time financial news data
- **OpenRouter**: AI API access for DeepSeek integration
- **FinBERT**: Financial sentiment analysis model
- **Google**: Sheets API for data logging

---

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only. It does not constitute financial advice. Always:**
- Do your own research before making trading decisions
- Consult with qualified financial advisors
- Practice proper risk management
- Never invest more than you can afford to lose
- Understand that past performance doesn't guarantee future results

**Trading and investing involve significant risk of loss.**

---

**Built with ❤️ for the Global Trading Community**

## 🚀 **Don't Forget To:**
- ⭐ **STAR** this repository if you find it useful
- 🔄 **FORK** and contribute improvements  
- 👍 **LIKE** and share with fellow traders
- 📢 **SHARE** on social media and trading communities
- 💬 **PROVIDE** feedback and suggestions
- 🐛 **REPORT** bugs to help improve the tool

**Happy Trading! 📈💰**

---

*Elite Trading Analyzer - Where Technology Meets Trading Excellence*