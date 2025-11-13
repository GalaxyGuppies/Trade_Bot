"""
DEX Aggregator Integration Module
Optimal trade routing across multiple exchanges to get best prices and reduce slippage.
Integrates with major DEX aggregators and provides smart routing capabilities.
"""

import asyncio
import logging
import requests
import json
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DEXType(Enum):
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    PANCAKESWAP = "pancakeswap"
    SUSHISWAP = "sushiswap"
    KYBERSWAP = "kyberswap"
    BALANCER = "balancer"
    CURVE = "curve"
    QUICKSWAP = "quickswap"

class AggregatorType(Enum):
    ONEINCH = "1inch"
    PARASWAP = "paraswap"
    ZEROX = "0x"
    KYBER = "kyber"
    JUPITER = "jupiter"  # For Solana
    MATCHA = "matcha"

@dataclass
class DEXPrice:
    """Price quote from a specific DEX"""
    dex: DEXType
    input_amount: float
    output_amount: float
    price_impact: float
    gas_estimate: int
    liquidity: float
    pool_address: str
    timestamp: datetime

@dataclass
class RouteStep:
    """Single step in a multi-hop route"""
    dex: DEXType
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    pool_address: str
    fee_tier: Optional[float] = None

@dataclass
class OptimalRoute:
    """Optimal trading route across DEXes"""
    aggregator: AggregatorType
    input_token: str
    output_token: str
    input_amount: float
    output_amount: float
    price_impact: float
    gas_estimate: int
    estimated_gas_cost: float
    route_steps: List[RouteStep]
    slippage_tolerance: float
    execution_time_estimate: float
    confidence_score: float
    savings_vs_worst: float
    timestamp: datetime

@dataclass
class ExecutionResult:
    """Result of trade execution"""
    route_id: str
    transaction_hash: str
    actual_input_amount: float
    actual_output_amount: float
    actual_gas_used: int
    actual_gas_cost: float
    execution_time: float
    slippage: float
    success: bool
    error_message: Optional[str]
    timestamp: datetime

class DEXAggregator:
    """
    Advanced DEX aggregator for optimal trade routing
    Finds best prices across multiple DEXes and aggregators
    """
    
    def __init__(self, database_path: str = "trading_bot.db"):
        self.database_path = database_path
        self.price_cache = {}
        self.route_cache = {}
        self.execution_history = []
        
        # API keys and endpoints
        self.api_keys = {
            '1inch': None,  # Get from config
            'paraswap': None,
            '0x': None,
            'kyber': None
        }
        
        self.api_endpoints = {
            '1inch': 'https://api.1inch.io/v5.0',
            'paraswap': 'https://apiv5.paraswap.io',
            '0x': 'https://api.0x.org',
            'kyber': 'https://aggregator-api.kyberswap.com',
            'jupiter': 'https://quote-api.jup.ag/v6'  # Solana
        }
        
        # Default settings
        self.default_slippage = 0.01  # 1%
        self.max_hops = 3
        self.cache_duration = 30  # seconds
        self.min_liquidity = 10000  # $10k minimum liquidity
        
        # Gas price tracking
        self.current_gas_price = 20  # gwei
        self.gas_price_cache_time = None
        
        self._init_database()
        logger.info("💱 DEX Aggregator initialized")
    
    def _init_database(self):
        """Initialize DEX aggregator database tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # DEX prices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dex_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dex TEXT NOT NULL,
                    token_in TEXT NOT NULL,
                    token_out TEXT NOT NULL,
                    input_amount REAL,
                    output_amount REAL,
                    price_impact REAL,
                    gas_estimate INTEGER,
                    liquidity REAL,
                    pool_address TEXT,
                    timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Optimal routes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimal_routes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    route_id TEXT UNIQUE,
                    aggregator TEXT,
                    input_token TEXT,
                    output_token TEXT,
                    input_amount REAL,
                    output_amount REAL,
                    price_impact REAL,
                    gas_estimate INTEGER,
                    estimated_gas_cost REAL,
                    slippage_tolerance REAL,
                    execution_time_estimate REAL,
                    confidence_score REAL,
                    savings_vs_worst REAL,
                    route_data TEXT,
                    timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Execution results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    route_id TEXT,
                    transaction_hash TEXT,
                    actual_input_amount REAL,
                    actual_output_amount REAL,
                    actual_gas_used INTEGER,
                    actual_gas_cost REAL,
                    execution_time REAL,
                    slippage REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("DEX aggregator database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing DEX database: {e}")
    
    async def get_optimal_route(self, token_in: str, token_out: str, 
                              amount_in: float, slippage: float = None,
                              chain: str = "ethereum") -> OptimalRoute:
        """
        Find optimal trading route across all DEXes and aggregators
        """
        try:
            slippage = slippage or self.default_slippage
            cache_key = f"{token_in}_{token_out}_{amount_in}_{slippage}_{chain}"
            
            # Check cache
            if cache_key in self.route_cache:
                cached_route, cache_time = self.route_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.cache_duration:
                    logger.info("📋 Using cached optimal route")
                    return cached_route
            
            logger.info(f"🔍 Finding optimal route: {amount_in} {token_in} → {token_out}")
            
            # Get quotes from multiple aggregators in parallel
            aggregator_quotes = await asyncio.gather(
                self._get_1inch_quote(token_in, token_out, amount_in, slippage, chain),
                self._get_paraswap_quote(token_in, token_out, amount_in, slippage, chain),
                self._get_0x_quote(token_in, token_out, amount_in, slippage, chain),
                self._get_kyber_quote(token_in, token_out, amount_in, slippage, chain),
                return_exceptions=True
            )
            
            # Filter successful quotes
            valid_quotes = []
            for i, quote in enumerate(aggregator_quotes):
                if isinstance(quote, OptimalRoute):
                    valid_quotes.append(quote)
                elif isinstance(quote, Exception):
                    logger.warning(f"Aggregator {i} failed: {quote}")
            
            if not valid_quotes:
                logger.error("No valid quotes received from aggregators")
                return self._create_fallback_route(token_in, token_out, amount_in, slippage)
            
            # Select best route based on multiple criteria
            best_route = self._select_best_route(valid_quotes)
            
            # Cache the result
            self.route_cache[cache_key] = (best_route, datetime.now())
            
            # Store in database
            await self._store_optimal_route(best_route)
            
            logger.info(f"✅ Optimal route found: {best_route.output_amount:.6f} {token_out} "
                       f"via {best_route.aggregator.value} (savings: {best_route.savings_vs_worst:.2%})")
            
            return best_route
            
        except Exception as e:
            logger.error(f"Error finding optimal route: {e}")
            return self._create_fallback_route(token_in, token_out, amount_in, slippage)
    
    async def _get_1inch_quote(self, token_in: str, token_out: str, 
                             amount_in: float, slippage: float, chain: str) -> OptimalRoute:
        """Get quote from 1inch aggregator"""
        try:
            # Map chain to 1inch chain ID
            chain_ids = {
                'ethereum': '1',
                'bsc': '56',
                'polygon': '137',
                'arbitrum': '42161',
                'optimism': '10'
            }
            
            chain_id = chain_ids.get(chain, '1')
            base_url = f"{self.api_endpoints['1inch']}/{chain_id}"
            
            # Convert amount to wei (assuming 18 decimals)
            amount_wei = int(amount_in * 10**18)
            
            # Get quote
            quote_url = f"{base_url}/quote"
            params = {
                'fromTokenAddress': token_in,
                'toTokenAddress': token_out,
                'amount': str(amount_wei)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(quote_url, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"1inch API error: {response.status}")
                    
                    data = await response.json()
                    
                    output_amount = float(data['toTokenAmount']) / 10**18
                    gas_estimate = int(data.get('estimatedGas', 150000))
                    
                    # Create route steps (simplified)
                    route_steps = [RouteStep(
                        dex=DEXType.UNISWAP_V2,  # Simplified
                        token_in=token_in,
                        token_out=token_out,
                        amount_in=amount_in,
                        amount_out=output_amount,
                        pool_address="0x" + "0" * 40  # Placeholder
                    )]
                    
                    return OptimalRoute(
                        aggregator=AggregatorType.ONEINCH,
                        input_token=token_in,
                        output_token=token_out,
                        input_amount=amount_in,
                        output_amount=output_amount,
                        price_impact=0.01,  # Would calculate from data
                        gas_estimate=gas_estimate,
                        estimated_gas_cost=gas_estimate * self.current_gas_price * 1e-9,
                        route_steps=route_steps,
                        slippage_tolerance=slippage,
                        execution_time_estimate=15.0,
                        confidence_score=0.9,
                        savings_vs_worst=0.0,  # Will be calculated later
                        timestamp=datetime.now()
                    )
            
        except Exception as e:
            logger.error(f"1inch quote error: {e}")
            raise e
    
    async def _get_paraswap_quote(self, token_in: str, token_out: str,
                                amount_in: float, slippage: float, chain: str) -> OptimalRoute:
        """Get quote from ParaSwap aggregator"""
        try:
            # ParaSwap network mapping
            networks = {
                'ethereum': '1',
                'bsc': '56',
                'polygon': '137',
                'arbitrum': '42161',
                'optimism': '10'
            }
            
            network = networks.get(chain, '1')
            base_url = f"{self.api_endpoints['paraswap']}/prices"
            
            amount_wei = int(amount_in * 10**18)
            
            params = {
                'srcToken': token_in,
                'destToken': token_out,
                'amount': str(amount_wei),
                'network': network,
                'side': 'SELL'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"ParaSwap API error: {response.status}")
                    
                    data = await response.json()
                    price_route = data['priceRoute']
                    
                    output_amount = float(price_route['destAmount']) / 10**18
                    gas_estimate = int(price_route.get('gasCost', 200000))
                    
                    # Parse route steps
                    route_steps = []
                    for route in price_route.get('bestRoute', []):
                        for swap in route.get('swaps', []):
                            route_steps.append(RouteStep(
                                dex=DEXType.UNISWAP_V2,  # Simplified
                                token_in=swap['srcToken'],
                                token_out=swap['destToken'],
                                amount_in=float(swap['srcAmount']) / 10**18,
                                amount_out=float(swap['destAmount']) / 10**18,
                                pool_address=swap.get('swapExchanges', [{}])[0].get('exchange', '')
                            ))
                    
                    return OptimalRoute(
                        aggregator=AggregatorType.PARASWAP,
                        input_token=token_in,
                        output_token=token_out,
                        input_amount=amount_in,
                        output_amount=output_amount,
                        price_impact=0.01,
                        gas_estimate=gas_estimate,
                        estimated_gas_cost=gas_estimate * self.current_gas_price * 1e-9,
                        route_steps=route_steps,
                        slippage_tolerance=slippage,
                        execution_time_estimate=20.0,
                        confidence_score=0.85,
                        savings_vs_worst=0.0,
                        timestamp=datetime.now()
                    )
            
        except Exception as e:
            logger.error(f"ParaSwap quote error: {e}")
            raise e
    
    async def _get_0x_quote(self, token_in: str, token_out: str,
                          amount_in: float, slippage: float, chain: str) -> OptimalRoute:
        """Get quote from 0x aggregator"""
        try:
            if chain != 'ethereum':
                raise Exception("0x only supports Ethereum mainnet in this demo")
            
            base_url = f"{self.api_endpoints['0x']}/swap/v1/quote"
            amount_wei = int(amount_in * 10**18)
            
            params = {
                'sellToken': token_in,
                'buyToken': token_out,
                'sellAmount': str(amount_wei),
                'slippagePercentage': str(slippage)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"0x API error: {response.status}")
                    
                    data = await response.json()
                    
                    output_amount = float(data['buyAmount']) / 10**18
                    gas_estimate = int(data.get('estimatedGas', 180000))
                    
                    # Create simplified route
                    route_steps = [RouteStep(
                        dex=DEXType.UNISWAP_V2,
                        token_in=token_in,
                        token_out=token_out,
                        amount_in=amount_in,
                        amount_out=output_amount,
                        pool_address="0x" + "0" * 40
                    )]
                    
                    return OptimalRoute(
                        aggregator=AggregatorType.ZEROX,
                        input_token=token_in,
                        output_token=token_out,
                        input_amount=amount_in,
                        output_amount=output_amount,
                        price_impact=float(data.get('estimatedPriceImpact', 0.01)),
                        gas_estimate=gas_estimate,
                        estimated_gas_cost=gas_estimate * self.current_gas_price * 1e-9,
                        route_steps=route_steps,
                        slippage_tolerance=slippage,
                        execution_time_estimate=12.0,
                        confidence_score=0.88,
                        savings_vs_worst=0.0,
                        timestamp=datetime.now()
                    )
            
        except Exception as e:
            logger.error(f"0x quote error: {e}")
            raise e
    
    async def _get_kyber_quote(self, token_in: str, token_out: str,
                             amount_in: float, slippage: float, chain: str) -> OptimalRoute:
        """Get quote from KyberSwap aggregator"""
        try:
            # KyberSwap chain mapping
            chain_mapping = {
                'ethereum': 'ethereum',
                'bsc': 'bsc',
                'polygon': 'polygon',
                'arbitrum': 'arbitrum',
                'optimism': 'optimism'
            }
            
            kyber_chain = chain_mapping.get(chain, 'ethereum')
            base_url = f"{self.api_endpoints['kyber']}/{kyber_chain}/route/encode"
            
            amount_wei = int(amount_in * 10**18)
            
            payload = {
                'tokenIn': token_in,
                'tokenOut': token_out,
                'amountIn': str(amount_wei),
                'to': '0x' + '0' * 40,  # Placeholder recipient
                'slippageTolerance': int(slippage * 10000)  # In basis points
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(base_url, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"KyberSwap API error: {response.status}")
                    
                    data = await response.json()
                    
                    output_amount = float(data['outputAmount']) / 10**18
                    gas_estimate = int(data.get('gasUsd', 3) / (self.current_gas_price * 1e-9))
                    
                    route_steps = [RouteStep(
                        dex=DEXType.KYBERSWAP,
                        token_in=token_in,
                        token_out=token_out,
                        amount_in=amount_in,
                        amount_out=output_amount,
                        pool_address="0x" + "0" * 40
                    )]
                    
                    return OptimalRoute(
                        aggregator=AggregatorType.KYBER,
                        input_token=token_in,
                        output_token=token_out,
                        input_amount=amount_in,
                        output_amount=output_amount,
                        price_impact=0.015,  # Estimated
                        gas_estimate=gas_estimate,
                        estimated_gas_cost=float(data.get('gasUsd', 3)),
                        route_steps=route_steps,
                        slippage_tolerance=slippage,
                        execution_time_estimate=18.0,
                        confidence_score=0.82,
                        savings_vs_worst=0.0,
                        timestamp=datetime.now()
                    )
            
        except Exception as e:
            logger.error(f"KyberSwap quote error: {e}")
            raise e
    
    def _select_best_route(self, routes: List[OptimalRoute]) -> OptimalRoute:
        """Select the best route from multiple options"""
        if not routes:
            raise Exception("No routes available for selection")
        
        if len(routes) == 1:
            routes[0].savings_vs_worst = 0.0
            return routes[0]
        
        # Sort by output amount (descending)
        routes.sort(key=lambda r: r.output_amount, reverse=True)
        
        best_route = routes[0]
        worst_output = min(route.output_amount for route in routes)
        
        # Calculate savings vs worst route
        for route in routes:
            if worst_output > 0:
                route.savings_vs_worst = (route.output_amount - worst_output) / worst_output
        
        # Advanced selection criteria
        scored_routes = []
        for route in routes:
            # Score based on multiple factors
            output_score = route.output_amount / best_route.output_amount  # 0-1
            gas_score = 1.0 - (route.estimated_gas_cost / max(r.estimated_gas_cost for r in routes))  # 0-1
            confidence_score = route.confidence_score  # 0-1
            time_score = 1.0 - (route.execution_time_estimate / max(r.execution_time_estimate for r in routes))  # 0-1
            
            # Weighted composite score
            composite_score = (
                output_score * 0.5 +      # 50% weight on output amount
                gas_score * 0.2 +         # 20% weight on gas efficiency
                confidence_score * 0.2 +  # 20% weight on confidence
                time_score * 0.1          # 10% weight on speed
            )
            
            scored_routes.append((route, composite_score))
        
        # Select route with highest composite score
        scored_routes.sort(key=lambda x: x[1], reverse=True)
        selected_route = scored_routes[0][0]
        
        logger.info(f"🎯 Selected {selected_route.aggregator.value}: "
                   f"Output: {selected_route.output_amount:.6f}, "
                   f"Gas: ${selected_route.estimated_gas_cost:.2f}, "
                   f"Score: {scored_routes[0][1]:.3f}")
        
        return selected_route
    
    async def execute_route(self, route: OptimalRoute, 
                          wallet_address: str = None) -> ExecutionResult:
        """Execute the optimal route (simulation for demo)"""
        try:
            route_id = f"route_{int(time.time())}"
            logger.info(f"🚀 Executing route {route_id} via {route.aggregator.value}")
            
            # Simulate execution delay
            await asyncio.sleep(2)
            
            # Simulate transaction execution
            success = True  # In real implementation, would execute actual transaction
            tx_hash = f"0x{'a' * 64}"  # Simulated transaction hash
            
            # Simulate some slippage
            actual_slippage = min(route.slippage_tolerance * 0.5, 0.005)  # Half of tolerance or 0.5%
            actual_output = route.output_amount * (1 - actual_slippage)
            
            # Simulate gas usage
            gas_variance = 1.0 + (hash(route_id) % 200 - 100) / 1000  # ±10% variance
            actual_gas_used = int(route.gas_estimate * gas_variance)
            actual_gas_cost = actual_gas_used * self.current_gas_price * 1e-9
            
            execution_result = ExecutionResult(
                route_id=route_id,
                transaction_hash=tx_hash,
                actual_input_amount=route.input_amount,
                actual_output_amount=actual_output,
                actual_gas_used=actual_gas_used,
                actual_gas_cost=actual_gas_cost,
                execution_time=15.5,  # Simulated execution time
                slippage=actual_slippage,
                success=success,
                error_message=None if success else "Execution failed",
                timestamp=datetime.now()
            )
            
            # Store execution result
            await self._store_execution_result(execution_result)
            self.execution_history.append(execution_result)
            
            if success:
                logger.info(f"✅ Route executed successfully: {actual_output:.6f} tokens received "
                           f"(slippage: {actual_slippage:.3%}, gas: ${actual_gas_cost:.2f})")
            else:
                logger.error(f"❌ Route execution failed: {execution_result.error_message}")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing route: {e}")
            return ExecutionResult(
                route_id=f"failed_{int(time.time())}",
                transaction_hash="",
                actual_input_amount=route.input_amount,
                actual_output_amount=0.0,
                actual_gas_used=0,
                actual_gas_cost=0.0,
                execution_time=0.0,
                slippage=0.0,
                success=False,
                error_message=str(e),
                timestamp=datetime.now()
            )
    
    async def _update_gas_price(self):
        """Update current gas price from network"""
        try:
            if (self.gas_price_cache_time and 
                (datetime.now() - self.gas_price_cache_time).total_seconds() < 60):
                return  # Use cached price
            
            # In real implementation, would fetch from gas station API
            # For demo, simulate gas price fluctuation
            import random
            self.current_gas_price = random.uniform(15, 30)  # 15-30 gwei
            self.gas_price_cache_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating gas price: {e}")
    
    def _create_fallback_route(self, token_in: str, token_out: str, 
                             amount_in: float, slippage: float) -> OptimalRoute:
        """Create fallback route when all aggregators fail"""
        return OptimalRoute(
            aggregator=AggregatorType.ONEINCH,  # Default fallback
            input_token=token_in,
            output_token=token_out,
            input_amount=amount_in,
            output_amount=amount_in * 0.95,  # Assume 5% slippage
            price_impact=0.05,
            gas_estimate=200000,
            estimated_gas_cost=200000 * self.current_gas_price * 1e-9,
            route_steps=[RouteStep(
                dex=DEXType.UNISWAP_V2,
                token_in=token_in,
                token_out=token_out,
                amount_in=amount_in,
                amount_out=amount_in * 0.95,
                pool_address="0x" + "0" * 40
            )],
            slippage_tolerance=slippage,
            execution_time_estimate=30.0,
            confidence_score=0.5,
            savings_vs_worst=0.0,
            timestamp=datetime.now()
        )
    
    async def _store_optimal_route(self, route: OptimalRoute):
        """Store optimal route in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            route_id = f"route_{int(time.time())}_{hash(route.input_token + route.output_token) % 10000}"
            
            cursor.execute('''
                INSERT INTO optimal_routes (
                    route_id, aggregator, input_token, output_token, input_amount,
                    output_amount, price_impact, gas_estimate, estimated_gas_cost,
                    slippage_tolerance, execution_time_estimate, confidence_score,
                    savings_vs_worst, route_data, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                route_id, route.aggregator.value, route.input_token, route.output_token,
                route.input_amount, route.output_amount, route.price_impact,
                route.gas_estimate, route.estimated_gas_cost, route.slippage_tolerance,
                route.execution_time_estimate, route.confidence_score,
                route.savings_vs_worst, json.dumps([asdict(step) for step in route.route_steps]),
                route.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing optimal route: {e}")
    
    async def _store_execution_result(self, result: ExecutionResult):
        """Store execution result in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO execution_results (
                    route_id, transaction_hash, actual_input_amount, actual_output_amount,
                    actual_gas_used, actual_gas_cost, execution_time, slippage,
                    success, error_message, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.route_id, result.transaction_hash, result.actual_input_amount,
                result.actual_output_amount, result.actual_gas_used, result.actual_gas_cost,
                result.execution_time, result.slippage, result.success,
                result.error_message, result.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing execution result: {e}")
    
    def get_aggregator_performance(self) -> Dict:
        """Get performance statistics for different aggregators"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get aggregator performance over last 24 hours
            cursor.execute('''
                SELECT 
                    r.aggregator,
                    COUNT(*) as route_count,
                    AVG(r.output_amount) as avg_output,
                    AVG(r.estimated_gas_cost) as avg_gas_cost,
                    AVG(r.savings_vs_worst) as avg_savings,
                    AVG(e.slippage) as avg_slippage,
                    AVG(CASE WHEN e.success THEN 1.0 ELSE 0.0 END) as success_rate
                FROM optimal_routes r
                LEFT JOIN execution_results e ON r.route_id = e.route_id
                WHERE r.timestamp > datetime('now', '-24 hours')
                GROUP BY r.aggregator
            ''')
            
            performance_data = {}
            for row in cursor.fetchall():
                performance_data[row[0]] = {
                    'route_count': row[1],
                    'avg_output': row[2] or 0,
                    'avg_gas_cost': row[3] or 0,
                    'avg_savings': row[4] or 0,
                    'avg_slippage': row[5] or 0,
                    'success_rate': row[6] or 0
                }
            
            conn.close()
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting aggregator performance: {e}")
            return {}
    
    def get_savings_summary(self) -> Dict:
        """Get summary of savings achieved through optimal routing"""
        try:
            if not self.execution_history:
                return {"total_savings": 0, "avg_savings": 0, "execution_count": 0}
            
            recent_executions = [ex for ex in self.execution_history 
                               if (datetime.now() - ex.timestamp).total_seconds() < 86400]  # Last 24h
            
            if not recent_executions:
                return {"total_savings": 0, "avg_savings": 0, "execution_count": 0}
            
            # Calculate savings based on gas cost reduction and better prices
            total_gas_saved = sum(max(0, 250000 - ex.actual_gas_used) * self.current_gas_price * 1e-9 
                                for ex in recent_executions)
            
            # Estimate price improvement savings (would need baseline comparison)
            estimated_price_savings = sum(ex.actual_output_amount * 0.002 for ex in recent_executions)  # 0.2% improvement
            
            total_savings = total_gas_saved + estimated_price_savings
            avg_savings = total_savings / len(recent_executions)
            
            return {
                "total_savings": total_savings,
                "avg_savings": avg_savings,
                "execution_count": len(recent_executions),
                "gas_savings": total_gas_saved,
                "price_savings": estimated_price_savings
            }
            
        except Exception as e:
            logger.error(f"Error calculating savings summary: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Test the DEX aggregator"""
    aggregator = DEXAggregator()
    
    # Test token addresses (example)
    WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    USDC = "0xA0b86a33E6411c3Ce98e6d9b4B3C61a6F2B0C1D2"
    
    print("💱 Testing DEX Aggregator")
    print("=" * 50)
    
    # Test optimal route finding
    print("\n🔍 Finding optimal route...")
    amount_in = 1.0  # 1 WETH
    
    try:
        optimal_route = await aggregator.get_optimal_route(
            token_in=WETH,
            token_out=USDC, 
            amount_in=amount_in,
            slippage=0.01  # 1%
        )
        
        print(f"Best route found via {optimal_route.aggregator.value}")
        print(f"Input: {optimal_route.input_amount} WETH")
        print(f"Output: {optimal_route.output_amount:.2f} USDC")
        print(f"Price impact: {optimal_route.price_impact:.2%}")
        print(f"Estimated gas cost: ${optimal_route.estimated_gas_cost:.2f}")
        print(f"Savings vs worst: {optimal_route.savings_vs_worst:.2%}")
        
        # Test execution
        print(f"\n🚀 Executing route...")
        execution_result = await aggregator.execute_route(optimal_route)
        
        if execution_result.success:
            print(f"✅ Execution successful!")
            print(f"Actual output: {execution_result.actual_output_amount:.2f} USDC")
            print(f"Actual slippage: {execution_result.slippage:.3%}")
            print(f"Actual gas cost: ${execution_result.actual_gas_cost:.2f}")
        else:
            print(f"❌ Execution failed: {execution_result.error_message}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Performance summary
    print(f"\n📊 Aggregator Performance:")
    performance = aggregator.get_aggregator_performance()
    for aggregator_name, stats in performance.items():
        print(f"{aggregator_name}: {stats['route_count']} routes, "
              f"{stats['success_rate']:.1%} success rate, "
              f"{stats['avg_savings']:.2%} avg savings")
    
    # Savings summary
    savings = aggregator.get_savings_summary()
    print(f"\n💰 Savings Summary:")
    print(f"Total savings: ${savings.get('total_savings', 0):.2f}")
    print(f"Average per trade: ${savings.get('avg_savings', 0):.2f}")
    print(f"Executions: {savings.get('execution_count', 0)}")

if __name__ == "__main__":
    asyncio.run(main())