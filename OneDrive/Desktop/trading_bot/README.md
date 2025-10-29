# 🤖 Smart Crypto Trading Bot

An advanced, AI-powered cryptocurrency trading bot that combines market data analysis, social sentiment monitoring, on-chain analysis, and sophisticated risk management to execute profitable trades automatically.

## ✨ Features

### Core Trading Features
- **Multi-Exchange Support**: Binance, Coinbase Pro, and more
- **Real-time Market Data**: WebSocket feeds for tick-level data
- **Advanced Technical Analysis**: RSI, Bollinger Bands, VWAP, and custom indicators
- **AI-Powered Sentiment Analysis**: Twitter, Reddit, and news sentiment using transformer models
- **On-Chain Analysis**: Whale tracking, liquidity monitoring, rugpull detection
- **Risk Management**: Stop-loss, take-profit, position sizing, daily loss limits
- **Paper Trading**: Test strategies without real money

### Safety & Risk Controls
- **Emergency Stop**: Instantly halt all trading and close positions
- **Rugpull Detection**: AI-powered scam token detection
- **Circuit Breakers**: Global and per-instrument trading halts
- **Position Limits**: Maximum position size and count controls
- **Daily Loss Limits**: Automatic shutdown on excessive losses

### User Interface
- **Web Dashboard**: Real-time monitoring and control interface
- **Manual Trading**: Execute trades manually through the GUI
- **Live Alerts**: Real-time notifications for important events
- **Performance Monitoring**: P&L tracking, position management
- **Activity Logs**: Complete audit trail of all actions

### Technical Infrastructure
- **Containerized Deployment**: Docker and Docker Compose support
- **Monitoring & Observability**: Prometheus metrics, Grafana dashboards
- **Scalable Architecture**: Microservices with message queues
- **Data Storage**: PostgreSQL, TimescaleDB for time-series data, Redis for caching

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (optional but recommended)
- API keys for exchanges and social media platforms

### Option 1: Simple Setup (Local Development)

1. **Clone and Setup**
```bash
git clone <your-repo>
cd trading_bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run the Application**
```bash
python launcher.py
```

3. **Access Dashboard**
Open your browser and go to: http://localhost:8000

### Option 2: Docker Setup (Recommended for Production)

1. **Clone the Repository**
```bash
git clone <your-repo>
cd trading_bot
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

3. **Start with Docker Compose**
```bash
docker-compose up -d
```

4. **Access Services**
- Trading Bot Dashboard: http://localhost:8000
- Grafana Monitoring: http://localhost:3000 (admin/admin)
- Prometheus Metrics: http://localhost:9090

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your configuration:

```bash
# Trading Configuration
ENVIRONMENT=development
MAX_POSITION_SIZE=1000.0
DAILY_LOSS_LIMIT=-500.0
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.15

# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret
BINANCE_TESTNET=true

COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET=your_coinbase_secret
COINBASE_PASSPHRASE=your_coinbase_passphrase
COINBASE_SANDBOX=true

# Social Media APIs
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/trading_bot
REDIS_URL=redis://localhost:6379

# Risk Management
SENTIMENT_THRESHOLD=0.6
RUGPULL_THRESHOLD=0.3
MIN_CONFIDENCE=0.6
MAX_DAILY_TRADES=50

# Monitoring
LOG_LEVEL=INFO
```

### Trading Pairs Configuration

Edit `config.json` to customize trading pairs:

```json
{
  "trading": {
    "enabled_exchanges": ["binance"],
    "trading_pairs": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"],
    "max_positions": 5,
    "position_size_base": 0.02
  }
}
```

## 🎯 Usage Guide

### Starting the Bot

1. **Access the Dashboard**: Go to http://localhost:8000
2. **Check Status**: Verify all systems are green
3. **Start Trading**: Click the "▶️ Start Bot" button
4. **Monitor**: Watch real-time updates and alerts

### Manual Trading

1. **Navigate to Manual Trade Section** in the dashboard
2. **Enter Trade Details**:
   - Symbol (e.g., BTCUSDT)
   - Side (Buy/Sell)
   - Size (amount to trade)
   - Price (optional, leave blank for market order)
3. **Execute Trade**: Click "Execute Trade"

### Emergency Procedures

- **Emergency Stop**: Click "🚨 EMERGENCY STOP" to immediately halt all trading and close positions
- **Regular Stop**: Click "⏹️ Stop Bot" to stop new trades but keep existing positions
- **Position Management**: Monitor and manually close positions from the dashboard

## 📊 Dashboard Features

### Status Overview
- **Bot Status**: Running/Stopped/Emergency
- **Total P&L**: Real-time profit/loss
- **Open Positions**: Number of active trades
- **Active Alerts**: Current system alerts

### Real-time Data
- **Live Position Updates**: Current positions with P&L
- **Market Data**: Price feeds and technical indicators
- **Sentiment Analysis**: Social media sentiment scores
- **Activity Log**: Real-time event logging

### Controls
- **Start/Stop Bot**: Control trading execution
- **Emergency Stop**: Immediate halt with position closure
- **Manual Trading**: Execute individual trades
- **Configuration**: Adjust settings and parameters

## 🛡️ Risk Management

### Built-in Safety Features

1. **Position Limits**
   - Maximum position size per trade
   - Maximum number of open positions
   - Daily loss limits with automatic shutdown

2. **Stop Loss & Take Profit**
   - Automatic stop-loss orders
   - Take-profit targets
   - Trailing stops (configurable)

3. **Rugpull Detection**
   - AI-powered scam token identification
   - Liquidity monitoring
   - Whale activity tracking

4. **Circuit Breakers**
   - Global trading halt on market anomalies
   - Per-instrument circuit breakers
   - Volatility-based trading pauses

### Best Practices

1. **Start Small**: Begin with small position sizes and low risk limits
2. **Paper Trade First**: Test strategies thoroughly before live trading
3. **Monitor Actively**: Keep an eye on the dashboard, especially initially
4. **Set Conservative Limits**: Use appropriate stop-losses and position limits
5. **Regular Reviews**: Monitor performance and adjust strategies

## 🔧 Development

### Project Structure
```
trading_bot/
├── src/
│   ├── api/              # WebSocket and API management
│   ├── data/             # Data collection and processing
│   ├── models/           # Trading strategies and ML models
│   ├── execution/        # Order management and execution
│   └── config.py         # Configuration management
├── docker/               # Docker configurations
├── tests/                # Test suites
├── launcher.py           # Simple application launcher
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Container orchestration
└── README.md            # This file
```

### Adding New Features

1. **New Exchange**: Add exchange support in `src/data/market_data.py`
2. **New Indicators**: Extend `calculate_indicators()` methods
3. **New Strategies**: Create strategy classes in `src/models/`
4. **New Risk Controls**: Add to `src/execution/order_manager.py`

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test
python -m pytest tests/test_trading_strategy.py
```

## 📈 Monitoring & Observability

### Prometheus Metrics
- Trading performance metrics
- System health indicators
- API call rates and errors
- Position and P&L tracking

### Grafana Dashboards
- Real-time trading performance
- Risk metrics visualization
- System resource monitoring
- Alert management

### Logging
- Structured JSON logging
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Centralized log aggregation
- Audit trails for all trades

## 🔒 Security

### Best Practices
1. **API Key Security**: Store keys in environment variables or secrets manager
2. **Network Security**: Use firewall rules and VPN access
3. **Access Control**: Implement role-based access for the dashboard
4. **Audit Logging**: Complete audit trails for all actions
5. **Regular Updates**: Keep dependencies and containers updated

### Production Deployment
1. **Use HTTPS**: Enable SSL/TLS for web access
2. **Database Security**: Use encrypted connections and strong passwords
3. **Container Security**: Run containers as non-root users
4. **Monitoring**: Set up security monitoring and alerting

## 🚨 Troubleshooting

### Common Issues

**Bot Won't Start**
- Check API keys are configured correctly
- Verify database connection
- Check logs for specific error messages

**No Market Data**
- Verify exchange API credentials
- Check network connectivity
- Ensure exchange symbols are correct

**Trades Not Executing**
- Check account balances
- Verify exchange permissions
- Review risk limit settings

**Dashboard Not Loading**
- Check if application is running (port 8000)
- Verify no firewall blocking
- Check browser console for errors

### Getting Help

1. **Check Logs**: Review application logs for error messages
2. **Verify Configuration**: Ensure all required settings are configured
3. **Test Connectivity**: Verify API connections are working
4. **Documentation**: Review this README and inline documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This trading bot is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software. Always:

- Trade only with money you can afford to lose
- Test thoroughly with paper trading before live trading
- Start with small amounts and gradually increase
- Monitor the bot actively, especially initially
- Understand the risks involved in automated trading

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.

---

**Happy Trading! 🚀📈**