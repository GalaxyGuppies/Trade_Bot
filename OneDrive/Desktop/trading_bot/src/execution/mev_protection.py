"""
Advanced MEV Protection and Smart Order Execution
Protects against sandwich attacks, front-running, and optimizes order execution
"""

import asyncio
import random
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class ExecutionStrategy(Enum):
    STEALTH = "stealth"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    DYNAMIC = "dynamic"
    AGGRESSIVE = "aggressive"

@dataclass
class Order:
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: Optional[float] = None
    order_type: OrderType = OrderType.MARKET
    time_in_force: str = "GTC"
    reduce_only: bool = False
    stop_price: Optional[float] = None
    metadata: Dict = None

@dataclass
class MEVRiskAssessment:
    sandwich_risk: float  # 0-1
    front_running_risk: float  # 0-1
    overall_risk: str  # 'low', 'medium', 'high', 'extreme'
    recommended_strategy: ExecutionStrategy
    max_slippage: float
    recommended_chunks: int
    reasoning: str

@dataclass
class ExecutionResult:
    success: bool
    filled_amount: float
    average_price: float
    total_fees: float
    slippage: float
    execution_time: float
    mev_protection_used: bool
    strategy_used: ExecutionStrategy
    chunks_executed: int
    metadata: Dict

class MEVDetector:
    """Detects MEV vulnerabilities and sandwich attack risks"""
    
    def __init__(self):
        self.mempool_monitor = MempoolMonitor()
        self.sandwich_detector = SandwichDetector()
        self.frontrun_detector = FrontRunDetector()
        
    async def assess_risk(self, order: Order, market_data: Dict) -> MEVRiskAssessment:
        """Assess MEV risk for a given order"""
        try:
            # Get current market conditions
            orderbook = market_data.get('orderbook', {})
            recent_trades = market_data.get('recent_trades', [])
            mempool_data = await self.mempool_monitor.get_mempool_state(order.symbol)
            
            # Calculate different risk factors
            sandwich_risk = await self.sandwich_detector.calculate_risk(order, orderbook, mempool_data)
            frontrun_risk = await self.frontrun_detector.calculate_risk(order, mempool_data)
            
            # Determine overall risk level
            overall_risk = self._calculate_overall_risk(sandwich_risk, frontrun_risk)
            
            # Recommend execution strategy
            strategy = self._recommend_strategy(sandwich_risk, frontrun_risk, order)
            
            # Calculate optimal parameters
            max_slippage = self._calculate_max_slippage(overall_risk, order.amount)
            recommended_chunks = self._calculate_optimal_chunks(order, overall_risk)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(sandwich_risk, frontrun_risk, strategy)
            
            return MEVRiskAssessment(
                sandwich_risk=sandwich_risk,
                front_running_risk=frontrun_risk,
                overall_risk=overall_risk,
                recommended_strategy=strategy,
                max_slippage=max_slippage,
                recommended_chunks=recommended_chunks,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in MEV risk assessment: {e}")
            return MEVRiskAssessment(
                sandwich_risk=0.5,
                front_running_risk=0.5,
                overall_risk='medium',
                recommended_strategy=ExecutionStrategy.STEALTH,
                max_slippage=0.005,
                recommended_chunks=3,
                reasoning="Error in risk assessment, using conservative defaults"
            )
    
    def _calculate_overall_risk(self, sandwich_risk: float, frontrun_risk: float) -> str:
        """Calculate overall MEV risk level"""
        max_risk = max(sandwich_risk, frontrun_risk)
        
        if max_risk < 0.25:
            return 'low'
        elif max_risk < 0.5:
            return 'medium'
        elif max_risk < 0.75:
            return 'high'
        else:
            return 'extreme'
    
    def _recommend_strategy(self, sandwich_risk: float, frontrun_risk: float, order: Order) -> ExecutionStrategy:
        """Recommend optimal execution strategy based on risks"""
        if sandwich_risk > 0.7 or frontrun_risk > 0.7:
            return ExecutionStrategy.STEALTH
        elif order.amount > self._get_market_impact_threshold(order.symbol):
            return ExecutionStrategy.ICEBERG
        elif sandwich_risk > 0.4:
            return ExecutionStrategy.TWAP
        else:
            return ExecutionStrategy.DYNAMIC
    
    def _calculate_max_slippage(self, risk_level: str, amount: float) -> float:
        """Calculate maximum acceptable slippage"""
        base_slippage = {
            'low': 0.001,      # 0.1%
            'medium': 0.003,   # 0.3%
            'high': 0.005,     # 0.5%
            'extreme': 0.01    # 1.0%
        }
        
        # Adjust for order size
        size_multiplier = min(1 + (amount / 100000), 2.0)  # Max 2x adjustment
        
        return base_slippage.get(risk_level, 0.005) * size_multiplier
    
    def _calculate_optimal_chunks(self, order: Order, risk_level: str) -> int:
        """Calculate optimal number of chunks for order splitting"""
        base_chunks = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'extreme': 5
        }
        
        # Adjust for order size
        if order.amount > 50000:  # Large order
            return base_chunks.get(risk_level, 2) + 2
        elif order.amount > 10000:  # Medium order
            return base_chunks.get(risk_level, 2) + 1
        else:
            return base_chunks.get(risk_level, 2)
    
    def _generate_reasoning(self, sandwich_risk: float, frontrun_risk: float, strategy: ExecutionStrategy) -> str:
        """Generate human-readable reasoning for the recommendation"""
        reasons = []
        
        if sandwich_risk > 0.5:
            reasons.append(f"High sandwich attack risk ({sandwich_risk:.1%})")
        
        if frontrun_risk > 0.5:
            reasons.append(f"High front-running risk ({frontrun_risk:.1%})")
        
        if strategy == ExecutionStrategy.STEALTH:
            reasons.append("Using stealth execution to avoid detection")
        elif strategy == ExecutionStrategy.ICEBERG:
            reasons.append("Using iceberg orders to hide order size")
        
        return "; ".join(reasons) if reasons else "Low MEV risk detected"
    
    def _get_market_impact_threshold(self, symbol: str) -> float:
        """Get market impact threshold for symbol"""
        # Placeholder - in production would be dynamic based on liquidity
        thresholds = {
            'BTC/USDT': 100000,
            'ETH/USDT': 50000,
            'SOL/USDT': 25000
        }
        return thresholds.get(symbol, 10000)

class MempoolMonitor:
    """Monitors mempool for pending transactions that might affect orders"""
    
    async def get_mempool_state(self, symbol: str) -> Dict:
        """Get current mempool state for symbol"""
        # Placeholder - in production would connect to mempool APIs
        return {
            'pending_large_orders': [],
            'arbitrage_activity': 'low',
            'bot_activity': 'medium',
            'gas_price_trend': 'stable'
        }

class SandwichDetector:
    """Detects sandwich attack risks"""
    
    async def calculate_risk(self, order: Order, orderbook: Dict, mempool_data: Dict) -> float:
        """Calculate sandwich attack risk (0-1)"""
        try:
            risk_factors = []
            
            # 1. Order size vs liquidity
            liquidity_risk = self._assess_liquidity_risk(order, orderbook)
            risk_factors.append(liquidity_risk * 0.4)
            
            # 2. Mempool activity
            mempool_risk = self._assess_mempool_risk(mempool_data)
            risk_factors.append(mempool_risk * 0.3)
            
            # 3. Market conditions
            market_risk = self._assess_market_conditions_risk(orderbook)
            risk_factors.append(market_risk * 0.3)
            
            return min(sum(risk_factors), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating sandwich risk: {e}")
            return 0.5
    
    def _assess_liquidity_risk(self, order: Order, orderbook: Dict) -> float:
        """Assess risk based on order size vs available liquidity"""
        try:
            if order.side == 'buy':
                levels = orderbook.get('asks', [])
            else:
                levels = orderbook.get('bids', [])
            
            if not levels:
                return 1.0
            
            # Calculate depth needed for order
            cumulative_volume = 0.0
            for price, volume in levels:
                cumulative_volume += float(volume)
                if cumulative_volume >= order.amount:
                    break
            
            # Risk increases as order size approaches available liquidity
            if cumulative_volume == 0:
                return 1.0
                
            liquidity_ratio = order.amount / cumulative_volume
            return min(liquidity_ratio * 2, 1.0)  # Risk increases non-linearly
            
        except Exception:
            return 0.5
    
    def _assess_mempool_risk(self, mempool_data: Dict) -> float:
        """Assess risk based on mempool activity"""
        bot_activity = mempool_data.get('bot_activity', 'medium')
        arbitrage_activity = mempool_data.get('arbitrage_activity', 'low')
        
        activity_scores = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'extreme': 1.0
        }
        
        bot_risk = activity_scores.get(bot_activity, 0.5)
        arb_risk = activity_scores.get(arbitrage_activity, 0.2)
        
        return (bot_risk + arb_risk) / 2
    
    def _assess_market_conditions_risk(self, orderbook: Dict) -> float:
        """Assess risk based on current market conditions"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 1.0
            
            # Calculate spread
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # Higher spread = higher risk
            spread_risk = min(spread * 1000, 1.0)  # Normalize
            
            # Calculate orderbook imbalance
            bid_volume = sum(float(bid[1]) for bid in bids[:5])
            ask_volume = sum(float(ask[1]) for ask in asks[:5])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance = abs(bid_volume - ask_volume) / total_volume
            else:
                imbalance = 1.0
            
            return (spread_risk + imbalance) / 2
            
        except Exception:
            return 0.5

class FrontRunDetector:
    """Detects front-running risks"""
    
    async def calculate_risk(self, order: Order, mempool_data: Dict) -> float:
        """Calculate front-running risk (0-1)"""
        try:
            # Simplified front-running risk calculation
            pending_orders = mempool_data.get('pending_large_orders', [])
            gas_trend = mempool_data.get('gas_price_trend', 'stable')
            
            # Higher risk with more pending large orders
            pending_risk = min(len(pending_orders) / 10, 1.0)
            
            # Higher risk with rising gas prices (indicates competition)
            gas_risk = {
                'falling': 0.2,
                'stable': 0.4,
                'rising': 0.7,
                'spiking': 1.0
            }.get(gas_trend, 0.4)
            
            return (pending_risk + gas_risk) / 2
            
        except Exception as e:
            logger.error(f"Error calculating front-running risk: {e}")
            return 0.5

class StealthExecutor:
    """Executes orders with stealth techniques to avoid detection"""
    
    async def execute_protected(self, order: Order, **kwargs) -> ExecutionResult:
        """Execute order with stealth protection"""
        try:
            start_time = time.time()
            
            # Break order into random-sized chunks
            chunks = self._create_stealth_chunks(order)
            
            # Execute with random delays
            results = []
            for chunk in chunks:
                # Random delay between 1-10 seconds
                delay = random.uniform(1, 10)
                await asyncio.sleep(delay)
                
                # Execute chunk
                chunk_result = await self._execute_chunk(chunk)
                results.append(chunk_result)
            
            # Aggregate results
            execution_result = self._aggregate_results(results, start_time)
            execution_result.mev_protection_used = True
            execution_result.strategy_used = ExecutionStrategy.STEALTH
            execution_result.chunks_executed = len(chunks)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error in stealth execution: {e}")
            return ExecutionResult(
                success=False,
                filled_amount=0.0,
                average_price=0.0,
                total_fees=0.0,
                slippage=0.0,
                execution_time=time.time() - start_time,
                mev_protection_used=True,
                strategy_used=ExecutionStrategy.STEALTH,
                chunks_executed=0,
                metadata={'error': str(e)}
            )
    
    def _create_stealth_chunks(self, order: Order) -> List[Order]:
        """Create random-sized chunks to avoid pattern detection"""
        chunks = []
        remaining_amount = order.amount
        min_chunk_size = order.amount * 0.05  # Minimum 5% of order
        max_chunk_size = order.amount * 0.3   # Maximum 30% of order
        
        while remaining_amount > min_chunk_size:
            if remaining_amount <= max_chunk_size:
                chunk_size = remaining_amount
            else:
                # Random chunk size between min and max
                chunk_size = random.uniform(min_chunk_size, min(max_chunk_size, remaining_amount))
            
            chunk = Order(
                symbol=order.symbol,
                side=order.side,
                amount=chunk_size,
                price=order.price,
                order_type=order.order_type,
                metadata={'parent_order': order, 'chunk_index': len(chunks)}
            )
            
            chunks.append(chunk)
            remaining_amount -= chunk_size
        
        return chunks
    
    async def _execute_chunk(self, chunk: Order) -> Dict:
        """Execute a single chunk (placeholder)"""
        # Simulate execution
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        return {
            'filled_amount': chunk.amount,
            'average_price': chunk.price or 100.0,  # Placeholder
            'fees': chunk.amount * 0.001,  # 0.1% fee
            'slippage': random.uniform(0.0001, 0.002)  # Random slippage
        }
    
    def _aggregate_results(self, results: List[Dict], start_time: float) -> ExecutionResult:
        """Aggregate chunk execution results"""
        total_filled = sum(r['filled_amount'] for r in results)
        total_fees = sum(r['fees'] for r in results)
        
        if total_filled > 0:
            weighted_price = sum(r['average_price'] * r['filled_amount'] for r in results) / total_filled
            average_slippage = sum(r['slippage'] * r['filled_amount'] for r in results) / total_filled
        else:
            weighted_price = 0.0
            average_slippage = 0.0
        
        return ExecutionResult(
            success=total_filled > 0,
            filled_amount=total_filled,
            average_price=weighted_price,
            total_fees=total_fees,
            slippage=average_slippage,
            execution_time=time.time() - start_time,
            mev_protection_used=True,
            strategy_used=ExecutionStrategy.STEALTH,
            chunks_executed=len(results),
            metadata={'chunk_results': results}
        )

class IcebergExecutor:
    """Executes large orders using iceberg strategy"""
    
    async def execute_protected(self, order: Order, **kwargs) -> ExecutionResult:
        """Execute order with iceberg strategy"""
        try:
            start_time = time.time()
            
            # Calculate iceberg parameters
            visible_size = self._calculate_visible_size(order)
            
            # Execute in iceberg fashion
            results = []
            remaining_amount = order.amount
            
            while remaining_amount > 0:
                chunk_size = min(visible_size, remaining_amount)
                
                chunk = Order(
                    symbol=order.symbol,
                    side=order.side,
                    amount=chunk_size,
                    price=order.price,
                    order_type=OrderType.LIMIT,
                    metadata={'iceberg_chunk': True}
                )
                
                chunk_result = await self._execute_chunk(chunk)
                results.append(chunk_result)
                
                remaining_amount -= chunk_result['filled_amount']
                
                # Small delay between chunks
                await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Aggregate results
            execution_result = self._aggregate_results(results, start_time)
            execution_result.strategy_used = ExecutionStrategy.ICEBERG
            execution_result.chunks_executed = len(results)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error in iceberg execution: {e}")
            return ExecutionResult(
                success=False,
                filled_amount=0.0,
                average_price=0.0,
                total_fees=0.0,
                slippage=0.0,
                execution_time=time.time() - start_time,
                mev_protection_used=True,
                strategy_used=ExecutionStrategy.ICEBERG,
                chunks_executed=0,
                metadata={'error': str(e)}
            )
    
    def _calculate_visible_size(self, order: Order) -> float:
        """Calculate optimal visible size for iceberg order"""
        # Typically 5-20% of total order size
        return order.amount * random.uniform(0.05, 0.2)
    
    async def _execute_chunk(self, chunk: Order) -> Dict:
        """Execute iceberg chunk"""
        # Placeholder execution
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        return {
            'filled_amount': chunk.amount,
            'average_price': chunk.price or 100.0,
            'fees': chunk.amount * 0.001,
            'slippage': random.uniform(0.0001, 0.001)
        }
    
    def _aggregate_results(self, results: List[Dict], start_time: float) -> ExecutionResult:
        """Aggregate iceberg execution results"""
        # Same as StealthExecutor aggregation
        total_filled = sum(r['filled_amount'] for r in results)
        total_fees = sum(r['fees'] for r in results)
        
        if total_filled > 0:
            weighted_price = sum(r['average_price'] * r['filled_amount'] for r in results) / total_filled
            average_slippage = sum(r['slippage'] * r['filled_amount'] for r in results) / total_filled
        else:
            weighted_price = 0.0
            average_slippage = 0.0
        
        return ExecutionResult(
            success=total_filled > 0,
            filled_amount=total_filled,
            average_price=weighted_price,
            total_fees=total_fees,
            slippage=average_slippage,
            execution_time=time.time() - start_time,
            mev_protection_used=True,
            strategy_used=ExecutionStrategy.ICEBERG,
            chunks_executed=len(results),
            metadata={'chunk_results': results}
        )

class SmartOrderExecution:
    """Main smart order execution engine with MEV protection"""
    
    def __init__(self):
        self.mev_detector = MEVDetector()
        self.execution_strategies = {
            ExecutionStrategy.STEALTH: StealthExecutor(),
            ExecutionStrategy.ICEBERG: IcebergExecutor(),
            # Add more strategies as needed
        }
    
    async def execute_order(self, order: Order, market_data: Dict) -> ExecutionResult:
        """Execute order with optimal strategy and MEV protection"""
        try:
            # Assess MEV risk
            mev_risk = await self.mev_detector.assess_risk(order, market_data)
            
            logger.info(f"MEV Risk Assessment: {mev_risk.overall_risk} ({mev_risk.reasoning})")
            
            # Select execution strategy
            strategy = mev_risk.recommended_strategy
            
            # Get executor
            executor = self.execution_strategies.get(strategy)
            if not executor:
                # Fallback to stealth execution
                executor = self.execution_strategies[ExecutionStrategy.STEALTH]
                strategy = ExecutionStrategy.STEALTH
            
            # Execute with protection
            result = await executor.execute_protected(
                order=order,
                mev_protection=True,
                slippage_tolerance=mev_risk.max_slippage,
                recommended_chunks=mev_risk.recommended_chunks
            )
            
            # Add MEV risk assessment to metadata
            result.metadata['mev_risk_assessment'] = mev_risk
            
            return result
            
        except Exception as e:
            logger.error(f"Error in smart order execution: {e}")
            return ExecutionResult(
                success=False,
                filled_amount=0.0,
                average_price=0.0,
                total_fees=0.0,
                slippage=0.0,
                execution_time=0.0,
                mev_protection_used=False,
                strategy_used=ExecutionStrategy.STEALTH,
                chunks_executed=0,
                metadata={'error': str(e)}
            )

# Example usage
async def test_smart_execution():
    """Test smart order execution"""
    executor = SmartOrderExecution()
    
    # Create test order
    test_order = Order(
        symbol='BTC/USDT',
        side='buy',
        amount=50000,  # Large order
        price=45000,
        order_type=OrderType.MARKET
    )
    
    # Mock market data
    market_data = {
        'orderbook': {
            'bids': [['44990', '1000'], ['44980', '2000']],
            'asks': [['45010', '1500'], ['45020', '2500']]
        },
        'recent_trades': []
    }
    
    # Execute order
    result = await executor.execute_order(test_order, market_data)
    
    print(f"Execution Result:")
    print(f"  Success: {result.success}")
    print(f"  Filled Amount: {result.filled_amount}")
    print(f"  Average Price: {result.average_price}")
    print(f"  Total Fees: {result.total_fees}")
    print(f"  Slippage: {result.slippage:.4f}")
    print(f"  Strategy Used: {result.strategy_used}")
    print(f"  MEV Protection: {result.mev_protection_used}")
    print(f"  Chunks Executed: {result.chunks_executed}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_smart_execution())