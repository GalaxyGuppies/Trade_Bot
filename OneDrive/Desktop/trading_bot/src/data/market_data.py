"""
Market data collector for multiple exchanges
"""
import asyncio
import ccxt
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import websockets
import pandas as pd

logger = logging.getLogger(__name__)

class MarketDataCollector:
    def __init__(self):
        self.exchanges = {}
        self.websocket_connections = {}
        self.symbol_data = {}
        self.running = False
        
        # Initialize exchanges
        self.init_exchanges()
    
    def init_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Binance
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': '',  # Add your API keys to config
                'secret': '',
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Add more exchanges as needed
            logger.info("Exchanges initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def start(self):
        """Start market data collection"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting market data collection...")
        
        # Start WebSocket connections for real-time data
        await self.start_websocket_feeds()
        
        # Start periodic data collection
        asyncio.create_task(self.collect_historical_data())
    
    async def stop(self):
        """Stop market data collection"""
        self.running = False
        logger.info("Stopping market data collection...")
        
        # Close WebSocket connections
        for ws in self.websocket_connections.values():
            if ws and not ws.closed:
                await ws.close()
    
    async def start_websocket_feeds(self):
        """Start WebSocket feeds for real-time data"""
        # Binance WebSocket stream
        symbols = ["btcusdt", "ethusdt", "solusdt"]  # Add more as needed
        
        for symbol in symbols:
            asyncio.create_task(self.binance_websocket_feed(symbol))
    
    async def binance_websocket_feed(self, symbol: str):
        """Connect to Binance WebSocket feed for a symbol"""
        uri = f"wss://stream.binance.com:9443/ws/{symbol}@ticker"
        
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    self.websocket_connections[f"binance_{symbol}"] = websocket
                    logger.info(f"Connected to Binance WebSocket for {symbol}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        data = json.loads(message)
                        await self.process_binance_ticker(data)
                        
            except Exception as e:
                logger.error(f"Binance WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def process_binance_ticker(self, data: Dict[str, Any]):
        """Process Binance ticker data"""
        symbol = data.get('s', '').upper()
        
        if symbol:
            ticker_data = {
                'symbol': symbol,
                'price': float(data.get('c', 0)),
                'volume': float(data.get('v', 0)),
                'change_24h': float(data.get('P', 0)),
                'high_24h': float(data.get('h', 0)),
                'low_24h': float(data.get('l', 0)),
                'timestamp': datetime.now(),
                'exchange': 'binance'
            }
            
            self.symbol_data[symbol] = ticker_data
            
            # Calculate technical indicators
            await self.calculate_indicators(symbol)
    
    async def calculate_indicators(self, symbol: str):
        """Calculate technical indicators for a symbol"""
        try:
            # Get historical data for calculations
            exchange = self.exchanges.get('binance')
            if not exchange:
                return
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if len(df) < 20:
                return
            
            # Calculate indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['rsi'] = self.calculate_rsi(df['close'])
            df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
            
            # Update symbol data with indicators
            latest = df.iloc[-1]
            if symbol in self.symbol_data:
                self.symbol_data[symbol].update({
                    'sma_20': latest['sma_20'],
                    'rsi': latest['rsi'],
                    'bb_upper': latest['bb_upper'],
                    'bb_lower': latest['bb_lower'],
                })
                
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    async def collect_historical_data(self):
        """Periodically collect historical data for analysis"""
        while self.running:
            try:
                for exchange_name, exchange in self.exchanges.items():
                    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
                    
                    for symbol in symbols:
                        try:
                            # Fetch recent trades
                            trades = exchange.fetch_trades(symbol, limit=50)
                            
                            # Process trades for volume analysis
                            await self.process_trades(symbol, trades, exchange_name)
                            
                        except Exception as e:
                            logger.error(f"Error fetching data for {symbol} on {exchange_name}: {e}")
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in historical data collection: {e}")
                await asyncio.sleep(60)
    
    async def process_trades(self, symbol: str, trades: List[Dict], exchange_name: str):
        """Process trade data for volume analysis"""
        if not trades:
            return
        
        # Calculate volume metrics
        total_volume = sum(trade['amount'] for trade in trades)
        avg_price = sum(trade['price'] * trade['amount'] for trade in trades) / total_volume if total_volume > 0 else 0
        
        # Update symbol data
        symbol_key = symbol.replace('/', '')
        if symbol_key in self.symbol_data:
            self.symbol_data[symbol_key].update({
                'recent_volume': total_volume,
                'avg_trade_price': avg_price,
                'trade_count': len(trades)
            })
    
    async def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current data for a symbol"""
        return self.symbol_data.get(symbol.upper())
    
    async def get_orderbook(self, symbol: str, exchange_name: str = 'binance') -> Optional[Dict[str, Any]]:
        """Get orderbook data for a symbol"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if exchange:
                orderbook = exchange.fetch_order_book(symbol)
                return {
                    'symbol': symbol,
                    'bids': orderbook['bids'][:10],  # Top 10 bids
                    'asks': orderbook['asks'][:10],  # Top 10 asks
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
        
        return None
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get overall market summary"""
        summary = {
            'total_symbols': len(self.symbol_data),
            'active_connections': len(self.websocket_connections),
            'last_update': datetime.now().isoformat(),
            'symbols': {}
        }
        
        for symbol, data in self.symbol_data.items():
            summary['symbols'][symbol] = {
                'price': data.get('price', 0),
                'change_24h': data.get('change_24h', 0),
                'volume': data.get('volume', 0)
            }
        
        return summary