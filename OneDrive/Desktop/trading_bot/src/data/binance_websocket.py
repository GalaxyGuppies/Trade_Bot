import asyncio
import json
import logging
import websockets
from typing import Dict, Callable, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    """Real-time Binance WebSocket client for price feeds"""
    
    def __init__(self):
        self.ws = None
        self.price_cache = {}
        self.callbacks = {}
        self.subscribed_symbols = set()
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_ping_time = 0
        
        # Binance WebSocket endpoints
        self.base_url = "wss://stream.binance.com:443"
        self.data_stream_url = "wss://data-stream.binance.vision"  # Alternative endpoint
        self.stream_url = f"{self.base_url}/stream"
        self.fallback_stream_url = f"{self.data_stream_url}/stream"
        
        logger.info("🔗 Binance WebSocket client initialized")
    
    async def connect(self):
        """Connect to Binance WebSocket"""
        try:
            if self.subscribed_symbols:
                # Build combined stream URL
                streams = "/".join([f"{symbol.lower()}@ticker" for symbol in self.subscribed_symbols])
                urls_to_try = [
                    f"{self.stream_url}?streams={streams}",
                    f"{self.fallback_stream_url}?streams={streams}"
                ]
            else:
                # Start with BTC ticker for initial connection
                urls_to_try = [
                    f"{self.base_url}/ws/btcusdt@ticker",
                    f"{self.data_stream_url}/ws/btcusdt@ticker"
                ]
            
            for url in urls_to_try:
                try:
                    logger.info(f"🔌 Connecting to Binance WebSocket: {url}")
                    
                    self.ws = await websockets.connect(
                        url,
                        ping_interval=30,
                        ping_timeout=10,
                        close_timeout=10
                    )
                    
                    self.running = True
                    self.reconnect_attempts = 0
                    logger.info("✅ Connected to Binance WebSocket")
                    
                    # Start listening for messages
                    await self._listen()
                    return  # Success, exit function
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to connect to {url}: {e}")
                    continue
            
            # If all URLs failed
            raise Exception("All WebSocket endpoints failed")
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            await self._handle_reconnect()
    
    async def _listen(self):
        """Listen for WebSocket messages"""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                    
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("🔌 WebSocket connection closed")
            if self.running:
                await self._handle_reconnect()
        except Exception as e:
            logger.error(f"❌ WebSocket listen error: {e}")
            if self.running:
                await self._handle_reconnect()
    
    async def _handle_message(self, data: dict):
        """Process incoming WebSocket messages"""
        try:
            # Handle combined stream format
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                ticker_data = data['data']
            else:
                # Handle single stream format
                ticker_data = data
                stream_name = None
            
            # Process ticker data
            if 'c' in ticker_data and 's' in ticker_data:  # 'c' = current price, 's' = symbol
                symbol = ticker_data['s']
                price = float(ticker_data['c'])
                
                # Update price cache
                self.price_cache[symbol] = {
                    'price': price,
                    'timestamp': time.time(),
                    'symbol': symbol,
                    'volume': float(ticker_data.get('v', 0)),  # 24hr volume
                    'change': float(ticker_data.get('P', 0))   # 24hr price change %
                }
                
                # Call registered callbacks
                await self._notify_callbacks(symbol, price)
                
                logger.debug(f"📈 {symbol}: ${price:,.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error handling message: {e}")
    
    async def _notify_callbacks(self, symbol: str, price: float):
        """Notify registered callbacks of price updates"""
        if symbol in self.callbacks:
            for callback in self.callbacks[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol, price)
                    else:
                        callback(symbol, price)
                except Exception as e:
                    logger.error(f"❌ Callback error for {symbol}: {e}")
    
    def subscribe_symbol(self, symbol: str, callback: Optional[Callable] = None):
        """Subscribe to price updates for a symbol"""
        symbol_upper = symbol.upper()
        self.subscribed_symbols.add(symbol_upper)
        
        if callback:
            if symbol_upper not in self.callbacks:
                self.callbacks[symbol_upper] = []
            self.callbacks[symbol_upper].append(callback)
        
        logger.info(f"📊 Subscribed to {symbol_upper} price feed")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get cached price for a symbol"""
        symbol_upper = symbol.upper()
        if symbol_upper in self.price_cache:
            cached_data = self.price_cache[symbol_upper]
            # Check if price is fresh (less than 30 seconds old)
            if time.time() - cached_data['timestamp'] < 30:
                return cached_data['price']
        return None
    
    def get_market_data(self, symbol: str) -> Optional[dict]:
        """Get full market data for a symbol"""
        symbol_upper = symbol.upper()
        return self.price_cache.get(symbol_upper)
    
    async def _handle_reconnect(self):
        """Handle WebSocket reconnection"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"❌ Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            self.running = False
            return
        
        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 30)  # Exponential backoff, max 30s
        
        logger.info(f"🔄 Reconnecting in {wait_time}s (attempt {self.reconnect_attempts})")
        await asyncio.sleep(wait_time)
        
        if self.running:
            await self.connect()
    
    async def start(self, symbols: list = None):
        """Start the WebSocket client with specified symbols"""
        if symbols:
            for symbol in symbols:
                self.subscribe_symbol(symbol)
        
        # Default symbols for major cryptocurrencies
        default_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'MATICUSDT', 
                          'DOGEUSDT', 'LINKUSDT', 'UNIUSDT', 'AVAXUSDT']
        
        for symbol in default_symbols:
            self.subscribe_symbol(symbol)
        
        await self.connect()
    
    async def stop(self):
        """Stop the WebSocket client"""
        self.running = False
        if self.ws:
            await self.ws.close()
        logger.info("🛑 Binance WebSocket client stopped")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.ws is not None and not self.ws.closed


# Global WebSocket client instance
_websocket_client = None

async def get_websocket_client() -> BinanceWebSocketClient:
    """Get or create the global WebSocket client"""
    global _websocket_client
    
    if _websocket_client is None:
        _websocket_client = BinanceWebSocketClient()
        
        # Start the client in background
        asyncio.create_task(_websocket_client.start())
        
        # Wait a moment for initial connection
        await asyncio.sleep(2)
    
    return _websocket_client

async def get_realtime_price(symbol: str) -> Optional[float]:
    """Get real-time price from WebSocket feed"""
    client = await get_websocket_client()
    return client.get_price(symbol)

async def get_realtime_market_data(symbol: str) -> Optional[dict]:
    """Get real-time market data from WebSocket feed"""
    client = await get_websocket_client()
    return client.get_market_data(symbol)