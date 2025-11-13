# Professional Sentiment APIs Configuration Guide

## 🌟 Enhanced Professional Sentiment Analysis Setup

Your trading bot now includes **professional-grade third-party sentiment APIs** for institutional-level market intelligence!

### 🔑 API Keys Required

To enable all professional sentiment features, you'll need API keys from these providers:

## 1. 📊 Santiment API (On-chain + Social Data)
- **Website**: https://santiment.net/
- **Features**: On-chain metrics, social sentiment, developer activity
- **Environment Variable**: `SANTIMENT_API_KEY`
- **Pricing**: Free tier available, paid plans for advanced features

## 2. 🌙 LunarCrush API (Aggregated Social Sentiment)
- **Website**: https://lunarcrush.com/
- **Features**: Social sentiment aggregation across multiple platforms
- **Environment Variable**: `LUNARCRUSH_API_KEY`
- **Pricing**: Free tier available, premium plans for higher limits

## 3. 🤖 OpenAI API (GPT-4 Contextual Analysis)
- **Website**: https://platform.openai.com/
- **Features**: Advanced natural language processing for market news analysis
- **Environment Variable**: `OPENAI_API_KEY`
- **Pricing**: Pay-per-use, very affordable for trading analysis

## 4. 📈 Messari API (Professional Market Intelligence)
- **Website**: https://messari.io/
- **Features**: Professional crypto market data and intelligence
- **Environment Variable**: `MESSARI_API_KEY`
- **Pricing**: Free tier available, professional plans for advanced data

## 5. 📱 StockGeist API (Multi-platform Sentiment)
- **Website**: https://stockgeist.ai/
- **Features**: Real-time sentiment from multiple social platforms
- **Environment Variable**: `STOCKGEIST_API_KEY`
- **Pricing**: Professional sentiment analysis service

### 🛠️ Setting Up Environment Variables

#### Windows (PowerShell):
```powershell
$env:SANTIMENT_API_KEY = "your_santiment_api_key_here"
$env:LUNARCRUSH_API_KEY = "your_lunarcrush_api_key_here"
$env:OPENAI_API_KEY = "your_openai_api_key_here"
$env:MESSARI_API_KEY = "your_messari_api_key_here"
$env:STOCKGEIST_API_KEY = "your_stockgeist_api_key_here"
```

#### Linux/Mac (Bash):
```bash
export SANTIMENT_API_KEY="your_santiment_api_key_here"
export LUNARCRUSH_API_KEY="your_lunarcrush_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"
export MESSARI_API_KEY="your_messari_api_key_here"
export STOCKGEIST_API_KEY="your_stockgeist_api_key_here"
```

#### Permanent Setup (.env file):
Create a `.env` file in your trading bot directory:
```env
SANTIMENT_API_KEY=your_santiment_api_key_here
LUNARCRUSH_API_KEY=your_lunarcrush_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
MESSARI_API_KEY=your_messari_api_key_here
STOCKGEIST_API_KEY=your_stockgeist_api_key_here
```

### 🚀 Enhanced Features Available

Once configured, your trading bot will have access to:

- **🎯 Multi-source Sentiment Aggregation**: Combines sentiment from all professional sources
- **🧠 AI-Powered Analysis**: OpenAI GPT-4 provides contextual market analysis
- **📊 On-chain Metrics**: Santiment provides blockchain-based sentiment indicators
- **📱 Social Media Intelligence**: LunarCrush aggregates sentiment across platforms
- **💼 Professional Market Data**: Messari provides institutional-grade market intelligence
- **⚖️ Weighted Scoring System**: Intelligent combination of all sentiment sources
- **🎚️ Confidence Filtering**: Only act on high-confidence sentiment signals

### 🔍 Testing Your Setup

Run the professional sentiment test suite:
```bash
python test_professional_sentiment.py
```

This will verify all API keys are working and demonstrate the enhanced sentiment analysis capabilities.

### 💡 Cost Optimization Tips

1. **Start with Free Tiers**: Most providers offer free tiers for testing
2. **OpenAI Usage**: Very cost-effective for occasional market analysis
3. **Santiment**: Focus on specific metrics rather than full data downloads
4. **LunarCrush**: Use for real-time sentiment during active trading periods
5. **Messari**: Leverage for fundamental analysis and market intelligence

### 🛡️ Fallback System

The enhanced sentiment collector includes intelligent fallbacks:
- If professional APIs are unavailable, falls back to Twitter/Reddit analysis
- Graceful degradation ensures trading continues with available data
- Error handling prevents API failures from stopping trading operations

### 📈 Integration Status

✅ **EnhancedSocialSentimentCollector** - Fully integrated and operational  
✅ **Professional API Methods** - All 5 third-party APIs implemented  
✅ **Weighted Scoring System** - Multi-source sentiment aggregation  
✅ **Error Handling & Fallbacks** - Robust operation even with API failures  
✅ **Trading System Integration** - Ready for use in integrated_trading_launcher.py  

Your trading bot now has **institutional-grade sentiment analysis capabilities**!