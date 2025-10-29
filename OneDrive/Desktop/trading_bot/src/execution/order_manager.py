"""
Order manager for executing trades with risk controls
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import ccxt

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class Position:
    def __init__(self, symbol: str, side: str, size: float, entry_price: float):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.current_price = entry_price
        self.pnl = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.timestamp = datetime.now()
        self.status = "open"

class Order:
    def __init__(self, symbol: str, side: str, order_type: str, size: float, price: Optional[float] = None):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.type = order_type
        self.size = size
        self.price = price
        self.filled_size = 0.0
        self.status = OrderStatus.PENDING
        self.timestamp = datetime.now()
        self.exchange_id = None
        self.error_message = None

class OrderManager:
    def __init__(self):
        self.exchanges = {}
        self.positions = {}
        self.orders = {}
        self.running = False
        
        # Risk limits
        self.max_position_size = 1000.0  # USD
        self.max_daily_loss = -500.0  # USD
        self.max_positions = 10
        self.daily_pnl = 0.0
        
        # Order tracking
        self.pending_orders = {}
        self.stop_loss_orders = {}
        self.take_profit_orders = {}
        
        # Initialize exchanges
        self.init_exchanges()
    
    def init_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Binance (you'll need to add your API credentials)
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': '',  # Add your API key
                'secret': '',  # Add your secret
                'sandbox': True,  # Use testnet for testing
                'enableRateLimit': True,
            })
            
            logger.info("Order manager exchanges initialized")
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def start(self):
        """Start the order manager"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting order manager...")
        
        # Start monitoring loops
        asyncio.create_task(self.monitor_orders())
        asyncio.create_task(self.monitor_positions())
        asyncio.create_task(self.update_position_pnl())
    
    async def stop(self):
        """Stop the order manager"""
        self.running = False
        logger.info("Stopping order manager...")
    
    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on a signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            size = signal['size']
            confidence = signal['confidence']
            
            logger.info(f"Executing trade: {action} {size} {symbol} (confidence: {confidence:.2f})")
            
            # Risk checks
            risk_check = await self.check_risk_limits(symbol, action, size)
            if not risk_check['allowed']:
                logger.warning(f"Trade rejected by risk check: {risk_check['reason']}")
                return {'success': False, 'error': risk_check['reason']}
            
            # Calculate position size in base currency
            current_price = await self.get_current_price(symbol)
            if not current_price:
                return {'success': False, 'error': 'Unable to get current price'}
            
            position_value = size * current_price
            
            # Create order
            order = Order(
                symbol=symbol,
                side=action,
                order_type=OrderType.MARKET.value,
                size=size,
                price=current_price
            )
            
            # Execute order
            result = await self.submit_order(order)
            
            if result['success']:
                # Create position if order is filled
                if action == 'buy':
                    position = Position(symbol, 'long', size, current_price)
                else:
                    position = Position(symbol, 'short', size, current_price)
                
                # Set stop loss and take profit
                if 'stop_loss' in signal and signal['stop_loss']:
                    position.stop_loss = signal['stop_loss']
                    await self.set_stop_loss(position)
                
                if 'take_profit' in signal and signal['take_profit']:
                    position.take_profit = signal['take_profit']
                    await self.set_take_profit(position)
                
                self.positions[position.id] = position
                
                logger.info(f"Trade executed successfully: {position.id}")
                return {'success': True, 'position_id': position.id, 'order_id': order.id}
            else:
                logger.error(f"Trade execution failed: {result['error']}")
                return {'success': False, 'error': result['error']}
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'success': False, 'error': str(e)}
    
    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """Submit an order to the exchange"""
        try:
            exchange = self.exchanges.get('binance')
            if not exchange:
                return {'success': False, 'error': 'Exchange not available'}
            
            # For testing, we'll simulate order execution
            # In production, you'd actually submit to the exchange
            logger.info(f"Simulating order submission: {order.side} {order.size} {order.symbol}")
            
            # Simulate successful execution
            order.status = OrderStatus.FILLED
            order.filled_size = order.size
            order.exchange_id = f"sim_{order.id[:8]}"
            
            self.orders[order.id] = order
            
            return {'success': True, 'order_id': order.id, 'exchange_id': order.exchange_id}
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            return {'success': False, 'error': str(e)}
    
    async def check_risk_limits(self, symbol: str, action: str, size: float) -> Dict[str, Any]:
        """Check if trade passes risk limits"""
        # Check maximum positions
        if len(self.positions) >= self.max_positions:
            return {'allowed': False, 'reason': 'Maximum positions limit reached'}
        
        # Check daily P&L limit
        if self.daily_pnl <= self.max_daily_loss:
            return {'allowed': False, 'reason': 'Daily loss limit reached'}
        
        # Check position size
        current_price = await self.get_current_price(symbol)
        if current_price:
            position_value = size * current_price
            if position_value > self.max_position_size:
                return {'allowed': False, 'reason': 'Position size too large'}
        
        # Check if already have position in this symbol
        existing_position = self.get_position_by_symbol(symbol)
        if existing_position:
            return {'allowed': False, 'reason': 'Already have position in this symbol'}
        
        return {'allowed': True, 'reason': 'All checks passed'}
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            exchange = self.exchanges.get('binance')
            if exchange:
                ticker = exchange.fetch_ticker(symbol)
                return ticker['last']
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        
        # Fallback to mock prices for testing
        mock_prices = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 3000.0,
            'SOLUSDT': 100.0
        }
        
        return mock_prices.get(symbol)
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get existing position for a symbol"""
        for position in self.positions.values():
            if position.symbol == symbol and position.status == 'open':
                return position
        return None
    
    async def set_stop_loss(self, position: Position):
        """Set stop loss order for a position"""
        try:
            if position.side == 'long':
                stop_side = 'sell'
            else:
                stop_side = 'buy'
            
            stop_order = Order(
                symbol=position.symbol,
                side=stop_side,
                order_type=OrderType.STOP_LOSS.value,
                size=position.size,
                price=position.stop_loss
            )
            
            # In production, submit stop loss order to exchange
            self.stop_loss_orders[position.id] = stop_order
            logger.info(f"Stop loss set for position {position.id} at {position.stop_loss}")
            
        except Exception as e:
            logger.error(f"Error setting stop loss: {e}")
    
    async def set_take_profit(self, position: Position):
        """Set take profit order for a position"""
        try:
            if position.side == 'long':
                tp_side = 'sell'
            else:
                tp_side = 'buy'
            
            tp_order = Order(
                symbol=position.symbol,
                side=tp_side,
                order_type=OrderType.TAKE_PROFIT.value,
                size=position.size,
                price=position.take_profit
            )
            
            # In production, submit take profit order to exchange
            self.take_profit_orders[position.id] = tp_order
            logger.info(f"Take profit set for position {position.id} at {position.take_profit}")
            
        except Exception as e:
            logger.error(f"Error setting take profit: {e}")
    
    async def close_position(self, position_id: str, reason: str = "manual") -> Dict[str, Any]:
        """Close a position"""
        try:
            position = self.positions.get(position_id)
            if not position:
                return {'success': False, 'error': 'Position not found'}
            
            if position.status != 'open':
                return {'success': False, 'error': 'Position already closed'}
            
            # Create closing order
            if position.side == 'long':
                close_side = 'sell'
            else:
                close_side = 'buy'
            
            close_order = Order(
                symbol=position.symbol,
                side=close_side,
                order_type=OrderType.MARKET.value,
                size=position.size
            )
            
            # Execute closing order
            result = await self.submit_order(close_order)
            
            if result['success']:
                position.status = 'closed'
                
                # Calculate final P&L
                current_price = await self.get_current_price(position.symbol)
                if current_price:
                    if position.side == 'long':
                        position.pnl = (current_price - position.entry_price) * position.size
                    else:
                        position.pnl = (position.entry_price - current_price) * position.size
                    
                    self.daily_pnl += position.pnl
                
                # Remove stop loss and take profit orders
                self.stop_loss_orders.pop(position_id, None)
                self.take_profit_orders.pop(position_id, None)
                
                logger.info(f"Position {position_id} closed: {reason}, P&L: {position.pnl:.2f}")
                return {'success': True, 'pnl': position.pnl}
            else:
                return {'success': False, 'error': result['error']}
                
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def close_all_positions(self) -> Dict[str, Any]:
        """Close all open positions (emergency stop)"""
        logger.critical("Closing all positions (emergency stop)")
        
        results = []
        for position_id, position in self.positions.items():
            if position.status == 'open':
                result = await self.close_position(position_id, "emergency_stop")
                results.append({'position_id': position_id, 'result': result})
        
        return {'closed_positions': results}
    
    async def monitor_orders(self):
        """Monitor pending orders"""
        while self.running:
            try:
                # Check status of pending orders
                for order_id, order in self.orders.items():
                    if order.status == OrderStatus.PENDING:
                        # In production, check order status with exchange
                        pass
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(1)
    
    async def monitor_positions(self):
        """Monitor positions for stop loss and take profit"""
        while self.running:
            try:
                for position_id, position in self.positions.items():
                    if position.status != 'open':
                        continue
                    
                    current_price = await self.get_current_price(position.symbol)
                    if not current_price:
                        continue
                    
                    position.current_price = current_price
                    
                    # Check stop loss
                    if position.stop_loss:
                        if position.side == 'long' and current_price <= position.stop_loss:
                            await self.close_position(position_id, "stop_loss")
                        elif position.side == 'short' and current_price >= position.stop_loss:
                            await self.close_position(position_id, "stop_loss")
                    
                    # Check take profit
                    if position.take_profit:
                        if position.side == 'long' and current_price >= position.take_profit:
                            await self.close_position(position_id, "take_profit")
                        elif position.side == 'short' and current_price <= position.take_profit:
                            await self.close_position(position_id, "take_profit")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(5)
    
    async def update_position_pnl(self):
        """Update P&L for all positions"""
        while self.running:
            try:
                for position in self.positions.values():
                    if position.status == 'open':
                        current_price = await self.get_current_price(position.symbol)
                        if current_price:
                            if position.side == 'long':
                                position.pnl = (current_price - position.entry_price) * position.size
                            else:
                                position.pnl = (position.entry_price - current_price) * position.size
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error updating position P&L: {e}")
                await asyncio.sleep(10)
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions"""
        return [
            {
                'id': pos.id,
                'symbol': pos.symbol,
                'side': pos.side,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'pnl': pos.pnl,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'timestamp': pos.timestamp.isoformat(),
                'status': pos.status
            }
            for pos in self.positions.values()
        ]
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders"""
        return [
            {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'type': order.type,
                'size': order.size,
                'price': order.price,
                'filled_size': order.filled_size,
                'status': order.status.value,
                'timestamp': order.timestamp.isoformat()
            }
            for order in self.orders.values()
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get order manager status"""
        return {
            'running': self.running,
            'positions_count': len([p for p in self.positions.values() if p.status == 'open']),
            'orders_count': len(self.orders),
            'daily_pnl': self.daily_pnl,
            'total_position_value': sum(p.size * p.current_price for p in self.positions.values() if p.status == 'open'),
            'last_update': datetime.now().isoformat()
        }