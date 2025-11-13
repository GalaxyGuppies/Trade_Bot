"""
Real Exchange Integration for Live Trading
Supports both Binance (CEX) and Jupiter (Solana DEX)
"""

import asyncio
import logging
import ccxt
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import aiohttp

# Add WebSocket client import
from ..data.binance_websocket import get_websocket_client, get_realtime_price
from ..data.alternative_prices import get_unrestricted_price
from ..data.dexscreener_provider import get_dex_price, get_dex_token_info
from .non_restricted_exchange import get_non_restricted_exchange

logger = logging.getLogger(__name__)

@dataclass
class OrderResult:
    """Result of order execution"""
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    error_message: Optional[str] = None
    transaction_hash: Optional[str] = None

@dataclass
class TokenInfo:
    """Token information with address"""
    symbol: str
    address: str
    name: str
    decimals: int
    price_usd: float

class ExchangeManager:
    """Manages real exchange connections for live trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = {}
        self.connected_exchanges = []
        
        # Token address mappings for Solana
        self.solana_tokens = {
            'SOL': 'So11111111111111111111111111111111111111112',
            'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
            'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
            'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
            'SRM': 'SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt',
            'FTT': 'AGFEad2et2ZJif9jaGpdMixQqvW5i81aBdvKe7PHNfz3',
            # 🐋 WHALE-TRACKED TOKENS (Nov 2025)
            'BANGERS': '3wppuwUMAGgxnX75Aqr4W91xYWaN6RjxjCUFiPZUpump',  # Primary whale token
            'TRUMP': '6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN',    # Secondary whale token
            'BASED': 'EMAGfmV5bMzYEtgda43ZmCYwmLL7SaMi2RVqaRPjpump',    # Third whale token
            # Previous tokens (kept for reference):
            'USELESS': 'Dz9mQ9NzkBcCsuGPFJ3r1bS4wgqKMHBPiVuniW8Mbonk',  # ✅ +$0.005 profit
            'TROLL': '63LfDmNb3MQ8mw9MtZ2To9bEA2M71kZUUGq5tiJxcqj9',     # ✅ +$3.49 profit
            'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',       # dogwifhat
            'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',       # Jupiter
            'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',     # Bonk
            'PYTH': 'HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3',     # Pyth Network
        }
        
        # Binance symbol mappings
        self.binance_symbols = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT', 
            'SOL': 'SOLUSDT',
            'BNB': 'BNBUSDT',
            'ADA': 'ADAUSDT'
        }
        
        logger.info("🔗 Exchange Manager initialized")
    
    async def initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize Binance (if API keys available)
            await self._init_binance()
            
            # Initialize Jupiter for Solana DEX
            await self._init_jupiter()
            
            # Initialize WebSocket price feeds
            await self._init_websocket_feeds()
            
            # Initialize non-restricted exchanges
            await self._init_non_restricted_exchanges()
            
            logger.info(f"✅ Exchanges initialized: {', '.join(self.connected_exchanges)}")
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def _init_websocket_feeds(self):
        """Initialize real-time WebSocket price feeds"""
        try:
            self.websocket_client = await get_websocket_client()
            logger.info("📡 Real-time price feeds initialized")
        except Exception as e:
            logger.error(f"WebSocket initialization failed: {e}")
            self.websocket_client = None
    
    async def _init_non_restricted_exchanges(self):
        """Initialize non-restricted exchange connections"""
        try:
            self.non_restricted_exchange = await get_non_restricted_exchange()
            if self.non_restricted_exchange.is_available():
                logger.info("🌍 Non-restricted exchanges initialized")
                return True
            else:
                logger.warning("⚠️ No non-restricted exchanges available")
                return False
        except Exception as e:
            logger.error(f"Non-restricted exchange initialization failed: {e}")
            self.non_restricted_exchange = None
            return False
    
    async def _init_binance(self):
        """Initialize Binance exchange"""
        try:
            # Check for Binance API keys in config
            api_keys = self.config.get('api_keys', {})
            binance_config = api_keys.get('binance', {})
            
            if binance_config.get('api_key') and binance_config.get('secret'):
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': binance_config['api_key'],
                    'secret': binance_config['secret'],
                    'sandbox': binance_config.get('sandbox', True),  # Start with testnet
                    'enableRateLimit': True,
                })
                
                # Test connection
                balance = await self.exchanges['binance'].fetch_balance()
                self.connected_exchanges.append('binance')
                logger.info("✅ Binance connected successfully")
                
            else:
                logger.warning("⚠️ Binance API keys not found - using simulation mode")
                
        except Exception as e:
            logger.error(f"Binance initialization failed: {e}")
    
    async def _init_jupiter(self):
        """Initialize Jupiter for Solana DEX trading"""
        try:
            # Jupiter Ultra API endpoint (Updated November 2025)
            self.jupiter_api = "https://api.jup.ag/ultra"
            
            # Test Jupiter connection
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.jup.ag/tokens/v1") as response:
                    if response.status == 200:
                        tokens_data = await response.json()
                        self.connected_exchanges.append('jupiter')
                        logger.info("✅ Jupiter (Solana DEX) connected successfully")
                    else:
                        logger.warning("⚠️ Jupiter connection failed")
                        
        except Exception as e:
            logger.error(f"Jupiter initialization failed: {e}")
    
    async def get_token_info(self, symbol: str) -> Optional[TokenInfo]:
        """Get token information including address"""
        try:
            # Check if we have the token address in our local registry FIRST
            # This allows us to get prices even if DexScreener doesn't have the token yet
            if symbol.upper() in self.solana_tokens:
                address = self.solana_tokens[symbol.upper()]
                price = await self._get_solana_token_price(symbol)
                
                return TokenInfo(
                    symbol=symbol.upper(),
                    address=address,
                    name=f"{symbol.upper()} Token",
                    decimals=9 if symbol.upper() == 'SOL' else 6,
                    price_usd=price
                )
            
            # Check Binance symbols
            elif symbol.upper() in self.binance_symbols:
                binance_symbol = self.binance_symbols[symbol.upper()]
                price = await self._get_binance_price(binance_symbol)
                
                return TokenInfo(
                    symbol=symbol.upper(),
                    address=binance_symbol,  # Use Binance symbol as "address"
                    name=f"{symbol.upper()} Token",
                    decimals=8,
                    price_usd=price
                )
            
            else:
                # Try to discover new token
                return await self._discover_token(symbol)
                
        except Exception as e:
            logger.error(f"Error getting token info for {symbol}: {e}")
            return None
    
    async def _get_solana_token_price(self, symbol: str) -> float:
        """Get Solana token price from Jupiter"""
        try:
            # First, try DexScreener for real Solana DEX prices
            dex_price = await get_dex_price(symbol)
            if dex_price and dex_price > 0:
                logger.info(f"📊 DexScreener price for {symbol}: ${dex_price}")
                return dex_price
            
            # For SOL, try WebSocket first (from SOLUSDT)
            if symbol.upper() == 'SOL' and hasattr(self, 'websocket_client') and self.websocket_client:
                ws_price = await get_realtime_price('SOLUSDT')
                if ws_price and ws_price > 0:
                    logger.info(f"📡 Real-time SOL price from WebSocket: ${ws_price}")
                    return ws_price
            
            # Try Jupiter Price API for tokens in our registry (works even if not on DexScreener!)
            token_address = self.solana_tokens.get(symbol.upper())
            if token_address:
                try:
                    async with aiohttp.ClientSession() as session:
                        # Use Jupiter Ultra Price API (Updated November 2025)
                        url = f"https://api.jup.ag/price/v2?ids={token_address}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'data' in data and token_address in data['data']:
                                    jupiter_price = float(data['data'][token_address]['price'])
                                    logger.info(f"🪐 Jupiter price for {symbol}: ${jupiter_price}")
                                    return jupiter_price
                except Exception as jup_error:
                    logger.warning(f"Jupiter price API failed for {symbol}: {jup_error}")
            
            # Try unrestricted alternative APIs
            alt_price = await get_unrestricted_price(symbol)
            if alt_price and alt_price > 0:
                return alt_price
            
            # Fallback to CoinGecko then static prices
            coingecko_price = await self._get_coingecko_price(symbol)
            if coingecko_price > 0:
                return coingecko_price
            return self._get_fallback_price(symbol)
            
        except Exception as e:
            logger.error(f"Error getting Solana price for {symbol}: {e}")
            # Try CoinGecko before fallback
            coingecko_price = await self._get_coingecko_price(symbol)
            if coingecko_price > 0:
                return coingecko_price
            return self._get_fallback_price(symbol)
    
    async def _get_binance_price(self, symbol: str) -> float:
        """Get price from Binance"""
        try:
            # First, try unrestricted alternative APIs
            alt_price = await get_unrestricted_price(symbol.replace('USDT', '').replace('USDC', ''))
            if alt_price and alt_price > 0:
                return alt_price
            
            # Try WebSocket real-time price if available
            if hasattr(self, 'websocket_client') and self.websocket_client:
                ws_price = await get_realtime_price(symbol)
                if ws_price and ws_price > 0:
                    logger.info(f"📡 Real-time price from WebSocket for {symbol}: ${ws_price}")
                    return ws_price
            
            # Fallback to REST API (may fail with 451)
            if 'binance' in self.exchanges:
                ticker = await self.exchanges['binance'].fetch_ticker(symbol)
                return float(ticker['last'])
            else:
                # Try public Binance API
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            price = float(data['price'])
                            logger.info(f"💰 Real price from Binance API for {symbol}: ${price}")
                            return price
            
            # Try CoinGecko for major symbols
            base_symbol = symbol.replace('USDT', '').replace('USDC', '')
            coingecko_price = await self._get_coingecko_price(base_symbol)
            if coingecko_price > 0:
                return coingecko_price
                
            return self._get_fallback_price(symbol)
            
        except Exception as e:
            logger.error(f"Error getting Binance price for {symbol}: {e}")
            # Try CoinGecko before fallback
            base_symbol = symbol.replace('USDT', '').replace('USDC', '')
            coingecko_price = await self._get_coingecko_price(base_symbol)
            if coingecko_price > 0:
                return coingecko_price
            return self._get_fallback_price(symbol)
    
    def _get_fallback_price(self, symbol: str) -> float:
        """Get fallback prices when APIs are unavailable"""
        fallback_prices = {
            'BTC': 67000.0,
            'BTCUSDT': 67000.0,
            'ETH': 2400.0,
            'ETHUSDT': 2400.0,
            'SOL': 165.0,
            'SOLUSDT': 165.0,
            'USDC': 1.0,
            'USDCUSDT': 1.0,
            'USDT': 1.0,
            'ADA': 0.35,
            'ADAUSDT': 0.35,
            'MATIC': 0.45,
            'MATICUSDT': 0.45,
            'DOGE': 0.15,
            'DOGEUSDT': 0.15,
            'LINK': 12.5,
            'LINKUSDT': 12.5,
            'UNI': 8.5,
            'UNIUSDT': 8.5,
            'AVAX': 24.0,
            'AVAXUSDT': 24.0
        }
        
        price = fallback_prices.get(symbol.upper(), 0.0)
        if price > 0:
            logger.info(f"💰 Using fallback price for {symbol}: ${price}")
        return price
    
    async def _get_coingecko_price(self, symbol: str) -> float:
        """Get real-time price from CoinGecko API"""
        try:
            # Map symbols to CoinGecko IDs
            coingecko_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum', 
                'SOL': 'solana',
                'USDC': 'usd-coin',
                'USDT': 'tether',
                'ADA': 'cardano',
                'MATIC': 'matic-network',
                'DOGE': 'dogecoin',
                'LINK': 'chainlink',
                'UNI': 'uniswap',
                'AVAX': 'avalanche-2'
            }
            
            coin_id = coingecko_map.get(symbol.upper())
            if not coin_id:
                return 0.0
                
            async with aiohttp.ClientSession() as session:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = float(data[coin_id]['usd'])
                        logger.info(f"💰 Real price from CoinGecko for {symbol}: ${price}")
                        return price
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting CoinGecko price for {symbol}: {e}")
            return 0.0
    
    async def _discover_token(self, symbol: str) -> Optional[TokenInfo]:
        """Discover new tokens from various sources"""
        try:
            # Try Jupiter token list
            if 'jupiter' in self.connected_exchanges:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.jupiter_api}/tokens") as response:
                        if response.status == 200:
                            tokens = await response.json()
                            for token in tokens:
                                if token.get('symbol', '').upper() == symbol.upper():
                                    price = await self._get_token_price_by_address(token['address'])
                                    return TokenInfo(
                                        symbol=token['symbol'],
                                        address=token['address'],
                                        name=token.get('name', ''),
                                        decimals=token.get('decimals', 9),
                                        price_usd=price
                                    )
            
            logger.warning(f"Token {symbol} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error discovering token {symbol}: {e}")
            return None
    
    async def _get_token_price_by_address(self, address: str) -> float:
        """Get token price by contract address"""
        try:
            async with aiohttp.ClientSession() as session:
                # Use Jupiter Ultra Price API (Updated November 2025)
                url = f"https://api.jup.ag/price/v2?ids={address}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and address in data['data']:
                            return float(data['data'][address]['price'])
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting price for address {address}: {e}")
            return 0.0
    
    async def execute_trade(self, symbol: str, side: str, amount: float, 
                          order_type: str = "market") -> OrderResult:
        """Execute a trade on the best available exchange"""
        try:
            logger.info(f"🔄 Executing {side} {amount} {symbol}")
            
            # Get token info
            token_info = await self.get_token_info(symbol)
            if not token_info:
                return OrderResult(
                    success=False,
                    error_message=f"Token {symbol} not found"
                )
            
            # Determine best exchange
            exchange_name = await self._select_best_exchange(symbol)
            
            if exchange_name == 'non_restricted':
                return await self._execute_non_restricted_trade(symbol, side, amount, order_type)
            elif exchange_name == 'binance':
                return await self._execute_binance_trade(symbol, side, amount, order_type)
            elif exchange_name == 'jupiter':
                return await self._execute_jupiter_trade(token_info, side, amount)
            else:
                # Simulation mode
                return await self._simulate_trade(token_info, side, amount)
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def _select_best_exchange(self, symbol: str) -> str:
        """Select the best exchange for trading a symbol"""
        # Check if non-restricted exchanges are available
        if hasattr(self, 'non_restricted_exchange') and self.non_restricted_exchange and self.non_restricted_exchange.is_available():
            return 'non_restricted'
        
        # Prefer Solana DEX for SOL and SPL tokens
        elif symbol.upper() in self.solana_tokens and 'jupiter' in self.connected_exchanges:
            return 'jupiter'
        
        # Use Binance for major cryptos (may fail with geo-restrictions)
        elif symbol.upper() in self.binance_symbols and 'binance' in self.connected_exchanges:
            return 'binance'
        
        # Default to simulation
        else:
            return 'simulation'
    
    async def _execute_binance_trade(self, symbol: str, side: str, amount: float, 
                                   order_type: str) -> OrderResult:
        """Execute trade on Binance"""
        try:
            if 'binance' not in self.exchanges:
                raise Exception("Binance not connected")
            
            binance_symbol = self.binance_symbols.get(symbol.upper())
            if not binance_symbol:
                raise Exception(f"Symbol {symbol} not available on Binance")
            
            # Execute order
            order = await self.exchanges['binance'].create_order(
                symbol=binance_symbol,
                type=order_type,
                side=side.lower(),
                amount=amount
            )
            
            return OrderResult(
                success=True,
                order_id=order['id'],
                filled_price=order.get('price'),
                filled_quantity=order.get('filled'),
            )
            
        except Exception as e:
            logger.error(f"Binance trade execution failed: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def _execute_non_restricted_trade(self, symbol: str, side: str, amount: float, 
                                          order_type: str) -> OrderResult:
        """Execute trade on non-restricted exchanges"""
        try:
            if not hasattr(self, 'non_restricted_exchange') or not self.non_restricted_exchange:
                raise Exception("Non-restricted exchange not available")
            
            logger.info(f"🌍 Executing on non-restricted exchange: {side} {amount} {symbol}")
            
            # Execute trade through non-restricted exchange
            result = await self.non_restricted_exchange.execute_trade(symbol, side, amount, order_type)
            
            if result.success:
                logger.info(f"✅ Non-restricted trade executed: {result.order_id}")
            else:
                logger.error(f"❌ Non-restricted trade failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Non-restricted trade execution failed: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def _execute_jupiter_trade(self, token_info: TokenInfo, side: str, 
                                   amount: float) -> OrderResult:
        """Execute trade on Jupiter (Solana DEX)"""
        try:
            # For now, simulate Jupiter trades
            # Real implementation would require Solana wallet integration
            logger.info(f"🔄 Jupiter trade: {side} {amount} {token_info.symbol}")
            
            # Calculate filled price with some slippage
            slippage = 0.005  # 0.5% slippage
            if side.upper() == 'BUY':
                filled_price = token_info.price_usd * (1 + slippage)
            else:
                filled_price = token_info.price_usd * (1 - slippage)
            
            return OrderResult(
                success=True,
                order_id=f"jupiter_{int(datetime.now().timestamp())}",
                filled_price=filled_price,
                filled_quantity=amount,
                transaction_hash=f"0x{'a' * 64}"  # Mock transaction hash
            )
            
        except Exception as e:
            logger.error(f"Jupiter trade execution failed: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def _simulate_trade(self, token_info: TokenInfo, side: str, 
                            amount: float) -> OrderResult:
        """Simulate trade execution for testing"""
        try:
            logger.info(f"🎭 SIMULATED trade: {side} {amount} {token_info.symbol} @ ${token_info.price_usd}")
            
            # Add realistic slippage and fees
            slippage = 0.003  # 0.3% slippage
            fee = 0.001       # 0.1% fee
            
            if side.upper() == 'BUY':
                filled_price = token_info.price_usd * (1 + slippage + fee)
            else:
                filled_price = token_info.price_usd * (1 - slippage - fee)
            
            return OrderResult(
                success=True,
                order_id=f"sim_{int(datetime.now().timestamp())}",
                filled_price=filled_price,
                filled_quantity=amount,
            )
            
        except Exception as e:
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    def get_supported_tokens(self) -> List[TokenInfo]:
        """Get list of all supported tokens"""
        tokens = []
        
        # Add Solana tokens
        for symbol, address in self.solana_tokens.items():
            tokens.append(TokenInfo(
                symbol=symbol,
                address=address,
                name=f"{symbol} Token",
                decimals=9 if symbol == 'SOL' else 6,
                price_usd=0.0  # Price would be fetched separately
            ))
        
        # Add Binance tokens
        for symbol, binance_symbol in self.binance_symbols.items():
            if symbol not in [t.symbol for t in tokens]:  # Avoid duplicates
                tokens.append(TokenInfo(
                    symbol=symbol,
                    address=binance_symbol,
                    name=f"{symbol} Token",
                    decimals=8,
                    price_usd=0.0
                ))
        
        return tokens
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balances across all exchanges"""
        balances = {}
        
        try:
            # Get Binance balances
            if 'binance' in self.exchanges:
                binance_balance = await self.exchanges['binance'].fetch_balance()
                for currency, balance_info in binance_balance.items():
                    if isinstance(balance_info, dict) and balance_info.get('free', 0) > 0:
                        balances[f"{currency} (Binance)"] = balance_info['free']
            
            # Get Solana wallet balance (would need wallet integration)
            wallet_config = self.config.get('wallet', {})
            if wallet_config.get('current_balance'):
                balances['SOL (Wallet)'] = wallet_config['current_balance'] / 168.0  # Convert USD to SOL approx
            
        except Exception as e:
            logger.error(f"Error getting balances: {e}")
        
        return balances