"""
WebSocket manager for real-time dashboard updates
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific client"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_update(self, data: Dict[str, Any]):
        """Send a structured update to all clients"""
        message = json.dumps(data)
        await self.broadcast(message)
    
    async def send_alert(self, level: str, message: str, component: str = "system"):
        """Send an alert to all clients"""
        alert_data = {
            "type": "alert",
            "data": {
                "level": level,
                "message": message,
                "component": component,
                "timestamp": datetime.now().isoformat()
            }
        }
        await self.send_update(alert_data)
    
    async def send_trade_update(self, trade_data: Dict[str, Any]):
        """Send a trade execution update"""
        update_data = {
            "type": "trade_update",
            "data": trade_data
        }
        await self.send_update(update_data)
    
    async def send_market_data(self, symbol: str, price: float, volume: float):
        """Send market data update"""
        market_data = {
            "type": "market_data",
            "data": {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": datetime.now().isoformat()
            }
        }
        await self.send_update(market_data)