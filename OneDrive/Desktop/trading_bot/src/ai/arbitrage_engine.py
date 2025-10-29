"""
Real-Time Arbitrage Detection Engine
Multi-dimensional arbitrage opportunities across exchanges, chains, and protocols
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import heapq
from collections import defaultdict

logger = logging.getLogger(__name__)

class ArbitrageType(Enum):
    SIMPLE = "simple"  # Same asset, different exchanges
    TRIANGULAR = "triangular"  # 3+ assets on same exchange
    CROSS_CHAIN = "cross_chain"  # Same asset, different blockchains
    DEX_CEX = "dex_cex"  # DEX vs CEX arbitrage
    FUNDING_RATE = "funding_rate"  # Futures vs spot arbitrage
    FLASH_LOAN = "flash_loan"  # Flash loan arbitrage
    YIELD_FARMING = "yield_farming"  # Cross-protocol yield arbitrage

@dataclass
class ArbitrageOpportunity:
    type: ArbitrageType
    symbol: str
    profit_usd: float
    profit_percentage: float
    required_capital: float
    execution_time_seconds: float
    confidence: float  # 0-1
    expiry_time: datetime
    complexity: int  # 1-5 (1=simple, 5=very complex)
    
    # Execution details
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    
    # Additional details for complex arbitrage
    execution_steps: List[Dict] = field(default_factory=list)
    gas_costs: float = 0.0
    slippage_estimate: float = 0.0
    bridge_fees: float = 0.0
    
    # Risk factors
    liquidity_risk: float = 0.0  # 0-1
    execution_risk: float = 0.0  # 0-1
    bridge_risk: float = 0.0  # 0-1

@dataclass
class ExchangeData:
    exchange: str
    symbol: str
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    last_updated: datetime
    fees: Dict[str, float]  # maker, taker fees
    
class PriceData:
    """Real-time price data aggregator"""
    
    def __init__(self):
        self.exchanges = [
            'binance', 'coinbase', 'kraken', 'ftx', 'bybit',
            'kucoin', 'huobi', 'okx', 'gate', 'bitfinex'
        ]
        
        self.dex_protocols = [
            'uniswap_v3', 'sushiswap', 'pancakeswap', 'curve',
            'balancer', '1inch', 'quickswap', 'spookyswap'
        ]
        
        self.price_cache = {}
        self.last_update = {}
        
    async def get_all_prices(self, symbol: str) -> Dict[str, ExchangeData]:
        """Get prices from all exchanges for symbol"""
        price_data = {}
        
        # Get CEX prices
        for exchange in self.exchanges:
            try:
                data = await self._get_cex_price(exchange, symbol)
                if data:
                    price_data[exchange] = data
            except Exception as e:
                logger.debug(f"Failed to get price from {exchange}: {e}")
        
        # Get DEX prices
        for protocol in self.dex_protocols:
            try:
                data = await self._get_dex_price(protocol, symbol)
                if data:
                    price_data[f"dex_{protocol}"] = data
            except Exception as e:
                logger.debug(f"Failed to get price from {protocol}: {e}")
        
        return price_data
    
    async def _get_cex_price(self, exchange: str, symbol: str) -> Optional[ExchangeData]:
        """Get price from centralized exchange"""
        # Placeholder - in production would use CCXT or direct APIs
        
        # Mock price data with some variation
        base_price = 45000  # Base BTC price
        price_variation = np.random.uniform(-0.001, 0.001)  # ±0.1% variation
        
        bid_price = base_price * (1 + price_variation - 0.0005)
        ask_price = base_price * (1 + price_variation + 0.0005)
        
        # Different exchanges have different fees
        exchange_fees = {
            'binance': {'maker': 0.001, 'taker': 0.001},
            'coinbase': {'maker': 0.005, 'taker': 0.005},
            'kraken': {'maker': 0.0016, 'taker': 0.0026},
            'ftx': {'maker': 0.0002, 'taker': 0.0007},
            'bybit': {'maker': 0.001, 'taker': 0.001}
        }
        
        return ExchangeData(
            exchange=exchange,
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_volume=np.random.uniform(10, 100),
            ask_volume=np.random.uniform(10, 100),
            last_updated=datetime.now(),
            fees=exchange_fees.get(exchange, {'maker': 0.001, 'taker': 0.001})
        )
    
    async def _get_dex_price(self, protocol: str, symbol: str) -> Optional[ExchangeData]:
        """Get price from DEX protocol"""
        # Mock DEX price data
        base_price = 45000
        price_variation = np.random.uniform(-0.002, 0.002)  # ±0.2% variation (higher for DEX)
        
        # DEX prices often have wider spreads
        bid_price = base_price * (1 + price_variation - 0.001)
        ask_price = base_price * (1 + price_variation + 0.001)
        
        # DEX fees (gas + swap fees)
        dex_fees = {
            'uniswap_v3': {'maker': 0.003, 'taker': 0.003},  # 0.3% swap fee
            'sushiswap': {'maker': 0.003, 'taker': 0.003},
            'pancakeswap': {'maker': 0.0025, 'taker': 0.0025},
            'curve': {'maker': 0.0004, 'taker': 0.0004}
        }
        
        return ExchangeData(
            exchange=f"dex_{protocol}",
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_volume=np.random.uniform(5, 50),  # Lower liquidity
            ask_volume=np.random.uniform(5, 50),
            last_updated=datetime.now(),
            fees=dex_fees.get(protocol, {'maker': 0.003, 'taker': 0.003})
        )

class SimpleArbitrageDetector:
    """Detects simple arbitrage opportunities across exchanges"""
    
    def __init__(self, min_profit_percentage: float = 0.5):
        self.min_profit_percentage = min_profit_percentage
        
    async def detect_opportunities(self, price_data: Dict[str, ExchangeData]) -> List[ArbitrageOpportunity]:
        """Detect simple arbitrage opportunities"""
        opportunities = []
        
        exchanges = list(price_data.keys())
        
        # Compare all exchange pairs
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                exchange1 = exchanges[i]
                exchange2 = exchanges[j]
                
                data1 = price_data[exchange1]
                data2 = price_data[exchange2]
                
                # Check both directions
                opp1 = self._check_arbitrage_direction(data1, data2, exchange1, exchange2)
                if opp1:
                    opportunities.append(opp1)
                
                opp2 = self._check_arbitrage_direction(data2, data1, exchange2, exchange1)
                if opp2:
                    opportunities.append(opp2)
        
        return opportunities
    
    def _check_arbitrage_direction(self, buy_data: ExchangeData, sell_data: ExchangeData, 
                                 buy_exchange: str, sell_exchange: str) -> Optional[ArbitrageOpportunity]:
        """Check arbitrage in one direction"""
        
        # Buy at ask price, sell at bid price
        buy_price = buy_data.ask_price
        sell_price = sell_data.bid_price
        
        # Calculate fees
        buy_fee_rate = buy_data.fees['taker']
        sell_fee_rate = sell_data.fees['taker']
        
        # Net prices after fees
        net_buy_price = buy_price * (1 + buy_fee_rate)
        net_sell_price = sell_price * (1 - sell_fee_rate)
        
        if net_sell_price <= net_buy_price:
            return None
        
        # Calculate profit
        profit_per_unit = net_sell_price - net_buy_price
        profit_percentage = (profit_per_unit / net_buy_price) * 100
        
        if profit_percentage < self.min_profit_percentage:
            return None
        
        # Estimate available capital and volume
        available_volume = min(buy_data.ask_volume, sell_data.bid_volume)
        required_capital = net_buy_price * available_volume
        profit_usd = profit_per_unit * available_volume
        
        # Calculate execution time (simple arbitrage is fast)
        execution_time = 30.0  # 30 seconds
        
        # Calculate confidence based on spread and volume
        confidence = self._calculate_confidence(buy_data, sell_data, profit_percentage)
        
        return ArbitrageOpportunity(
            type=ArbitrageType.SIMPLE,
            symbol=buy_data.symbol,
            profit_usd=profit_usd,
            profit_percentage=profit_percentage,
            required_capital=required_capital,
            execution_time_seconds=execution_time,
            confidence=confidence,
            expiry_time=datetime.now() + timedelta(minutes=5),
            complexity=1,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=net_buy_price,
            sell_price=net_sell_price,
            slippage_estimate=0.001,  # 0.1% slippage estimate
            execution_steps=[
                {'step': 1, 'action': 'buy', 'exchange': buy_exchange, 'amount': available_volume},
                {'step': 2, 'action': 'sell', 'exchange': sell_exchange, 'amount': available_volume}
            ]
        )
    
    def _calculate_confidence(self, buy_data: ExchangeData, sell_data: ExchangeData, profit_percentage: float) -> float:
        """Calculate confidence in arbitrage opportunity"""
        confidence_factors = []
        
        # Profit margin confidence (higher profit = higher confidence)
        profit_confidence = min(profit_percentage / 2.0, 1.0)  # Cap at 100%
        confidence_factors.append(profit_confidence)
        
        # Volume confidence (higher volume = higher confidence)
        min_volume = min(buy_data.ask_volume, sell_data.bid_volume)
        volume_confidence = min(min_volume / 10.0, 1.0)  # Normalize to volume of 10
        confidence_factors.append(volume_confidence)
        
        # Data freshness (recent data = higher confidence)
        now = datetime.now()
        buy_age = (now - buy_data.last_updated).total_seconds()
        sell_age = (now - sell_data.last_updated).total_seconds()
        max_age = max(buy_age, sell_age)
        
        freshness_confidence = max(0, 1 - (max_age / 300))  # 5 minute decay
        confidence_factors.append(freshness_confidence)
        
        return np.mean(confidence_factors)

class TriangularArbitrageDetector:
    """Detects triangular arbitrage opportunities on single exchange"""
    
    def __init__(self):
        self.common_triangles = [
            ('BTC', 'ETH', 'USDT'),
            ('BTC', 'SOL', 'USDT'),
            ('ETH', 'SOL', 'USDT'),
            ('BTC', 'ADA', 'USDT'),
            ('ETH', 'LINK', 'USDT')
        ]
    
    async def detect_opportunities(self, exchange: str) -> List[ArbitrageOpportunity]:
        """Detect triangular arbitrage on specific exchange"""
        opportunities = []
        
        for triangle in self.common_triangles:
            base, intermediate, quote = triangle
            
            # Get prices for all three pairs
            try:
                pair1_data = await self._get_pair_data(exchange, f"{base}/{intermediate}")
                pair2_data = await self._get_pair_data(exchange, f"{intermediate}/{quote}")
                pair3_data = await self._get_pair_data(exchange, f"{base}/{quote}")
                
                if all([pair1_data, pair2_data, pair3_data]):
                    # Check both directions
                    opp1 = self._check_triangular_direction(
                        pair1_data, pair2_data, pair3_data, triangle, 'forward'
                    )
                    if opp1:
                        opportunities.append(opp1)
                    
                    opp2 = self._check_triangular_direction(
                        pair1_data, pair2_data, pair3_data, triangle, 'reverse'
                    )
                    if opp2:
                        opportunities.append(opp2)
                        
            except Exception as e:
                logger.debug(f"Error checking triangle {triangle}: {e}")
        
        return opportunities
    
    async def _get_pair_data(self, exchange: str, pair: str) -> Optional[ExchangeData]:
        """Get pair data for triangular arbitrage"""
        # Mock implementation - in production would get real data
        base_price = np.random.uniform(0.1, 100)
        spread = base_price * 0.001
        
        return ExchangeData(
            exchange=exchange,
            symbol=pair,
            bid_price=base_price - spread/2,
            ask_price=base_price + spread/2,
            bid_volume=np.random.uniform(10, 100),
            ask_volume=np.random.uniform(10, 100),
            last_updated=datetime.now(),
            fees={'maker': 0.001, 'taker': 0.001}
        )
    
    def _check_triangular_direction(self, pair1_data: ExchangeData, pair2_data: ExchangeData,
                                  pair3_data: ExchangeData, triangle: Tuple, direction: str) -> Optional[ArbitrageOpportunity]:
        """Check triangular arbitrage in specific direction"""
        
        if direction == 'forward':
            # Buy base with quote, sell base for intermediate, sell intermediate for quote
            step1_price = pair3_data.ask_price  # Buy base/quote
            step2_price = pair1_data.bid_price  # Sell base/intermediate
            step3_price = pair2_data.bid_price  # Sell intermediate/quote
        else:
            # Reverse direction
            step1_price = pair2_data.ask_price  # Buy intermediate/quote
            step2_price = pair1_data.ask_price  # Buy base/intermediate
            step3_price = pair3_data.bid_price  # Sell base/quote
        
        # Calculate theoretical profit (simplified)
        if direction == 'forward':
            final_amount = (1 / step1_price) * step2_price * step3_price
        else:
            final_amount = (1 / step1_price) * (1 / step2_price) * step3_price
        
        # Account for fees (3 trades)
        fee_rate = 0.001  # 0.1% per trade
        final_amount_after_fees = final_amount * (1 - fee_rate) ** 3
        
        if final_amount_after_fees <= 1.0:
            return None
        
        profit_percentage = (final_amount_after_fees - 1.0) * 100
        
        if profit_percentage < 0.1:  # Minimum 0.1% profit
            return None
        
        # Calculate required capital and volume
        min_volume = min(pair1_data.ask_volume, pair2_data.bid_volume, pair3_data.bid_volume)
        required_capital = min_volume * step1_price
        profit_usd = (final_amount_after_fees - 1.0) * required_capital
        
        return ArbitrageOpportunity(
            type=ArbitrageType.TRIANGULAR,
            symbol=f"{triangle[0]}/{triangle[1]}/{triangle[2]}",
            profit_usd=profit_usd,
            profit_percentage=profit_percentage,
            required_capital=required_capital,
            execution_time_seconds=60.0,  # 1 minute for 3 trades
            confidence=0.7,  # Lower confidence due to complexity
            expiry_time=datetime.now() + timedelta(minutes=2),
            complexity=3,
            buy_exchange=pair1_data.exchange,
            sell_exchange=pair1_data.exchange,
            buy_price=step1_price,
            sell_price=step3_price,
            execution_steps=[
                {'step': 1, 'action': 'trade', 'pair': triangle, 'direction': direction}
            ]
        )

class CrossChainArbitrageDetector:
    """Detects arbitrage opportunities across different blockchains"""
    
    def __init__(self):
        self.supported_chains = {
            'ethereum': ['binance', 'coinbase', 'uniswap_v3'],
            'bsc': ['pancakeswap', 'binance'],
            'polygon': ['quickswap', 'sushiswap'],
            'avalanche': ['traderjoe', 'pangolin'],
            'solana': ['raydium', 'orca']
        }
        
        self.bridge_costs = {
            ('ethereum', 'bsc'): 0.001,  # 0.1% bridge fee
            ('ethereum', 'polygon'): 0.0005,
            ('ethereum', 'avalanche'): 0.002,
            ('ethereum', 'solana'): 0.003
        }
        
        self.bridge_times = {
            ('ethereum', 'bsc'): 300,  # 5 minutes
            ('ethereum', 'polygon'): 180,  # 3 minutes
            ('ethereum', 'avalanche'): 600,  # 10 minutes
            ('ethereum', 'solana'): 900   # 15 minutes
        }
    
    async def detect_opportunities(self, symbol: str) -> List[ArbitrageOpportunity]:
        """Detect cross-chain arbitrage opportunities"""
        opportunities = []
        
        chain_pairs = list(self.supported_chains.keys())
        
        for i in range(len(chain_pairs)):
            for j in range(i + 1, len(chain_pairs)):
                chain1 = chain_pairs[i]
                chain2 = chain_pairs[j]
                
                try:
                    # Get best prices on each chain
                    price1 = await self._get_best_chain_price(chain1, symbol)
                    price2 = await self._get_best_chain_price(chain2, symbol)
                    
                    if price1 and price2:
                        # Check both directions
                        opp1 = self._check_cross_chain_direction(price1, price2, chain1, chain2, symbol)
                        if opp1:
                            opportunities.append(opp1)
                        
                        opp2 = self._check_cross_chain_direction(price2, price1, chain2, chain1, symbol)
                        if opp2:
                            opportunities.append(opp2)
                            
                except Exception as e:
                    logger.debug(f"Error checking cross-chain {chain1}-{chain2}: {e}")
        
        return opportunities
    
    async def _get_best_chain_price(self, chain: str, symbol: str) -> Optional[Dict]:
        """Get best price available on specific chain"""
        exchanges = self.supported_chains.get(chain, [])
        
        best_bid = 0
        best_ask = float('inf')
        best_volume = 0
        
        for exchange in exchanges:
            try:
                # Mock price data
                base_price = 45000 * np.random.uniform(0.998, 1.002)
                spread = base_price * 0.001
                
                bid_price = base_price - spread/2
                ask_price = base_price + spread/2
                volume = np.random.uniform(10, 50)
                
                if bid_price > best_bid:
                    best_bid = bid_price
                
                if ask_price < best_ask:
                    best_ask = ask_price
                    
                best_volume = max(best_volume, volume)
                
            except Exception:
                continue
        
        if best_ask == float('inf'):
            return None
        
        return {
            'chain': chain,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'volume': best_volume,
            'symbol': symbol
        }
    
    def _check_cross_chain_direction(self, buy_chain_data: Dict, sell_chain_data: Dict,
                                   buy_chain: str, sell_chain: str, symbol: str) -> Optional[ArbitrageOpportunity]:
        """Check cross-chain arbitrage in one direction"""
        
        buy_price = buy_chain_data['best_ask']
        sell_price = sell_chain_data['best_bid']
        
        # Get bridge costs and time
        bridge_key = tuple(sorted([buy_chain, sell_chain]))
        bridge_fee_rate = self.bridge_costs.get(bridge_key, 0.005)  # Default 0.5%
        bridge_time = self.bridge_times.get(bridge_key, 600)  # Default 10 minutes
        
        # Calculate net prices after fees and bridge costs
        net_buy_price = buy_price * (1 + 0.001)  # 0.1% trading fee
        net_sell_price = sell_price * (1 - 0.001 - bridge_fee_rate)  # Trading fee + bridge fee
        
        if net_sell_price <= net_buy_price:
            return None
        
        profit_per_unit = net_sell_price - net_buy_price
        profit_percentage = (profit_per_unit / net_buy_price) * 100
        
        if profit_percentage < 1.0:  # Minimum 1% for cross-chain
            return None
        
        # Calculate volume and capital
        available_volume = min(buy_chain_data['volume'], sell_chain_data['volume'])
        required_capital = net_buy_price * available_volume
        profit_usd = profit_per_unit * available_volume
        
        # Calculate execution time (includes bridge time)
        execution_time = bridge_time + 120  # Bridge time + execution buffer
        
        return ArbitrageOpportunity(
            type=ArbitrageType.CROSS_CHAIN,
            symbol=symbol,
            profit_usd=profit_usd,
            profit_percentage=profit_percentage,
            required_capital=required_capital,
            execution_time_seconds=execution_time,
            confidence=0.6,  # Lower confidence due to bridge risk
            expiry_time=datetime.now() + timedelta(minutes=30),
            complexity=4,
            buy_exchange=buy_chain,
            sell_exchange=sell_chain,
            buy_price=net_buy_price,
            sell_price=net_sell_price,
            bridge_fees=buy_price * available_volume * bridge_fee_rate,
            bridge_risk=0.3,  # Bridge risk factor
            execution_steps=[
                {'step': 1, 'action': 'buy', 'chain': buy_chain, 'amount': available_volume},
                {'step': 2, 'action': 'bridge', 'from': buy_chain, 'to': sell_chain, 'time': bridge_time},
                {'step': 3, 'action': 'sell', 'chain': sell_chain, 'amount': available_volume}
            ]
        )

class ArbitrageEngine:
    """Main arbitrage detection and execution engine"""
    
    def __init__(self):
        self.price_data = PriceData()
        self.simple_detector = SimpleArbitrageDetector()
        self.triangular_detector = TriangularArbitrageDetector()
        self.cross_chain_detector = CrossChainArbitrageDetector()
        
        self.opportunity_queue = []  # Priority queue
        self.active_opportunities = {}
        
    async def scan_all_opportunities(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Scan for all types of arbitrage opportunities"""
        all_opportunities = []
        
        for symbol in symbols:
            try:
                # Get price data
                price_data = await self.price_data.get_all_prices(symbol)
                
                # 1. Simple arbitrage
                simple_opps = await self.simple_detector.detect_opportunities(price_data)
                all_opportunities.extend(simple_opps)
                
                # 2. Cross-chain arbitrage
                cross_chain_opps = await self.cross_chain_detector.detect_opportunities(symbol)
                all_opportunities.extend(cross_chain_opps)
                
                # 3. Triangular arbitrage (check major exchanges)
                for exchange in ['binance', 'coinbase', 'kraken']:
                    triangular_opps = await self.triangular_detector.detect_opportunities(exchange)
                    all_opportunities.extend(triangular_opps)
                
            except Exception as e:
                logger.error(f"Error scanning opportunities for {symbol}: {e}")
        
        # Sort by profit potential
        all_opportunities.sort(key=lambda x: x.profit_usd * x.confidence, reverse=True)
        
        return all_opportunities
    
    def rank_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Rank opportunities by profit-risk ratio"""
        
        def calculate_score(opp: ArbitrageOpportunity) -> float:
            # Base score from profit and confidence
            base_score = opp.profit_usd * opp.confidence
            
            # Penalty for complexity
            complexity_penalty = 1 - (opp.complexity - 1) * 0.1
            
            # Penalty for execution time
            time_penalty = 1 - (opp.execution_time_seconds / 3600)  # 1 hour max
            
            # Risk adjustments
            risk_adjustment = 1 - (opp.liquidity_risk + opp.execution_risk + opp.bridge_risk) / 3
            
            return base_score * complexity_penalty * time_penalty * risk_adjustment
        
        # Calculate scores and sort
        for opp in opportunities:
            opp.metadata = opp.metadata or {}
            opp.metadata['score'] = calculate_score(opp)
        
        return sorted(opportunities, key=lambda x: x.metadata.get('score', 0), reverse=True)
    
    async def monitor_real_time(self, symbols: List[str], callback=None):
        """Real-time monitoring of arbitrage opportunities"""
        logger.info(f"Starting real-time arbitrage monitoring for {symbols}")
        
        while True:
            try:
                # Scan for opportunities
                opportunities = await self.scan_all_opportunities(symbols)
                
                # Rank opportunities
                ranked_opportunities = self.rank_opportunities(opportunities)
                
                # Filter for high-quality opportunities
                quality_opportunities = [
                    opp for opp in ranked_opportunities
                    if opp.confidence > 0.7 and opp.profit_percentage > 0.5
                ]
                
                # Update opportunity queue
                self._update_opportunity_queue(quality_opportunities)
                
                # Callback with new opportunities
                if callback and quality_opportunities:
                    await callback(quality_opportunities)
                
                # Log summary
                if quality_opportunities:
                    best_opp = quality_opportunities[0]
                    logger.info(f"Best opportunity: {best_opp.type.value} - "
                              f"{best_opp.profit_percentage:.2f}% profit - "
                              f"${best_opp.profit_usd:.2f} - "
                              f"{best_opp.confidence:.1%} confidence")
                
                # Wait before next scan
                await asyncio.sleep(5)  # 5-second intervals
                
            except Exception as e:
                logger.error(f"Error in real-time monitoring: {e}")
                await asyncio.sleep(10)  # Longer wait on error
    
    def _update_opportunity_queue(self, opportunities: List[ArbitrageOpportunity]):
        """Update priority queue with new opportunities"""
        current_time = datetime.now()
        
        # Remove expired opportunities
        self.opportunity_queue = [
            opp for opp in self.opportunity_queue
            if opp.expiry_time > current_time
        ]
        
        # Add new opportunities
        for opp in opportunities:
            # Avoid duplicates
            opp_key = f"{opp.type.value}_{opp.symbol}_{opp.buy_exchange}_{opp.sell_exchange}"
            
            if opp_key not in self.active_opportunities:
                heapq.heappush(self.opportunity_queue, (-opp.metadata.get('score', 0), opp))
                self.active_opportunities[opp_key] = opp
    
    def get_best_opportunities(self, count: int = 5) -> List[ArbitrageOpportunity]:
        """Get best current opportunities"""
        opportunities = []
        temp_queue = []
        
        # Extract best opportunities while preserving queue
        for _ in range(min(count, len(self.opportunity_queue))):
            if self.opportunity_queue:
                score, opp = heapq.heappop(self.opportunity_queue)
                opportunities.append(opp)
                temp_queue.append((score, opp))
        
        # Restore queue
        for item in temp_queue:
            heapq.heappush(self.opportunity_queue, item)
        
        return opportunities

# Example usage and testing
async def test_arbitrage_engine():
    """Test arbitrage detection engine"""
    engine = ArbitrageEngine()
    
    # Test symbols
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    print("Scanning for arbitrage opportunities...")
    opportunities = await engine.scan_all_opportunities(test_symbols)
    
    print(f"\nFound {len(opportunities)} arbitrage opportunities:")
    
    for i, opp in enumerate(opportunities[:5]):  # Show top 5
        print(f"\n{i+1}. {opp.type.value.upper()} Arbitrage:")
        print(f"   Symbol: {opp.symbol}")
        print(f"   Profit: ${opp.profit_usd:.2f} ({opp.profit_percentage:.2f}%)")
        print(f"   Required Capital: ${opp.required_capital:.2f}")
        print(f"   Confidence: {opp.confidence:.1%}")
        print(f"   Execution Time: {opp.execution_time_seconds:.0f}s")
        print(f"   Complexity: {opp.complexity}/5")
        print(f"   Buy: {opp.buy_exchange} @ ${opp.buy_price:.2f}")
        print(f"   Sell: {opp.sell_exchange} @ ${opp.sell_price:.2f}")
    
    return opportunities

async def monitor_arbitrage_callback(opportunities: List[ArbitrageOpportunity]):
    """Callback for real-time arbitrage monitoring"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New arbitrage opportunities detected:")
    
    for opp in opportunities[:3]:  # Show top 3
        print(f"  {opp.type.value}: {opp.symbol} - "
              f"{opp.profit_percentage:.2f}% profit - "
              f"${opp.profit_usd:.2f}")

if __name__ == "__main__":
    # Test the arbitrage engine
    asyncio.run(test_arbitrage_engine())
    
    # Uncomment to test real-time monitoring
    # engine = ArbitrageEngine()
    # asyncio.run(engine.monitor_real_time(['BTC/USDT', 'ETH/USDT'], monitor_arbitrage_callback))