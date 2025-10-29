"""
Main FastAPI application for the trading bot with web-based GUI
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from src.data.market_data import MarketDataCollector
from src.data.social_sentiment import SocialSentimentCollector
from src.models.strategy_engine import StrategyEngine
from src.execution.order_manager import OrderManager
from src.api.websocket_manager import WebSocketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "market_data": None,
    "social_sentiment": None,
    "strategy_engine": None,
    "order_manager": None,
    "websocket_manager": None,
    "bot_status": "stopped",
    "positions": {},
    "pnl": 0.0,
    "alerts": []
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting trading bot application...")
    
    # Initialize components
    app_state["websocket_manager"] = WebSocketManager()
    app_state["market_data"] = MarketDataCollector()
    app_state["social_sentiment"] = SocialSentimentCollector()
    app_state["strategy_engine"] = StrategyEngine()
    app_state["order_manager"] = OrderManager()
    
    # Start background tasks
    yield
    
    # Cleanup
    logger.info("Shutting down trading bot application...")
    if app_state["market_data"]:
        await app_state["market_data"].stop()

app = FastAPI(
    title="Smart Trading Bot",
    description="Advanced crypto trading bot with social sentiment and on-chain analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for API
class BotConfig(BaseModel):
    max_position_size: float = 1000.0
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    sentiment_threshold: float = 0.6
    rugpull_threshold: float = 0.3
    enabled_exchanges: List[str] = ["binance"]
    trading_pairs: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

class TradeSignal(BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float
    reasoning: str

class Position(BaseModel):
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    pnl: float
    timestamp: datetime

class Alert(BaseModel):
    level: str  # "info", "warning", "critical"
    message: str
    timestamp: datetime
    component: str

# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard HTML"""
    return HTMLResponse(content=get_dashboard_html())

@app.get("/api/status")
async def get_bot_status():
    """Get current bot status and metrics"""
    return {
        "status": app_state["bot_status"],
        "uptime": "0:00:00",  # TODO: Calculate actual uptime
        "positions_count": len(app_state["positions"]),
        "total_pnl": app_state["pnl"],
        "alerts_count": len(app_state["alerts"]),
        "last_update": datetime.now().isoformat()
    }

@app.post("/api/bot/start")
async def start_bot():
    """Start the trading bot"""
    if app_state["bot_status"] == "running":
        raise HTTPException(status_code=400, detail="Bot is already running")
    
    logger.info("Starting trading bot...")
    app_state["bot_status"] = "running"
    
    # Start market data collection
    if app_state["market_data"]:
        await app_state["market_data"].start()
    
    # Start social sentiment collection
    if app_state["social_sentiment"]:
        await app_state["social_sentiment"].start()
    
    return {"message": "Bot started successfully", "status": "running"}

@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot"""
    if app_state["bot_status"] == "stopped":
        raise HTTPException(status_code=400, detail="Bot is already stopped")
    
    logger.info("Stopping trading bot...")
    app_state["bot_status"] = "stopped"
    
    # Stop all components
    if app_state["market_data"]:
        await app_state["market_data"].stop()
    
    if app_state["social_sentiment"]:
        await app_state["social_sentiment"].stop()
    
    return {"message": "Bot stopped successfully", "status": "stopped"}

@app.post("/api/bot/emergency_stop")
async def emergency_stop():
    """Emergency stop - close all positions and halt trading"""
    logger.critical("EMERGENCY STOP TRIGGERED")
    
    # Close all positions
    if app_state["order_manager"]:
        await app_state["order_manager"].close_all_positions()
    
    app_state["bot_status"] = "emergency_stopped"
    app_state["alerts"].append(Alert(
        level="critical",
        message="Emergency stop triggered - all positions closed",
        timestamp=datetime.now(),
        component="system"
    ))
    
    return {"message": "Emergency stop executed", "status": "emergency_stopped"}

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    return list(app_state["positions"].values())

@app.get("/api/alerts")
async def get_alerts():
    """Get recent alerts"""
    return app_state["alerts"][-50:]  # Return last 50 alerts

@app.post("/api/trade")
async def manual_trade(signal: TradeSignal):
    """Execute a manual trade"""
    if app_state["order_manager"]:
        result = await app_state["order_manager"].execute_trade(signal)
        return {"message": "Trade executed", "result": result}
    else:
        raise HTTPException(status_code=500, detail="Order manager not initialized")

@app.get("/api/market_data/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for a symbol"""
    if app_state["market_data"]:
        data = await app_state["market_data"].get_symbol_data(symbol)
        return data
    else:
        raise HTTPException(status_code=500, detail="Market data not available")

@app.get("/api/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """Get sentiment data for a symbol"""
    if app_state["social_sentiment"]:
        sentiment = await app_state["social_sentiment"].get_symbol_sentiment(symbol)
        return sentiment
    else:
        raise HTTPException(status_code=500, detail="Sentiment data not available")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await app_state["websocket_manager"].connect(websocket)
    try:
        while True:
            # Keep connection alive and send updates
            await asyncio.sleep(1)
            await app_state["websocket_manager"].send_update({
                "type": "status_update",
                "data": {
                    "bot_status": app_state["bot_status"],
                    "positions": list(app_state["positions"].values()),
                    "pnl": app_state["pnl"],
                    "timestamp": datetime.now().isoformat()
                }
            })
    except WebSocketDisconnect:
        app_state["websocket_manager"].disconnect(websocket)

def get_dashboard_html():
    """Generate the dashboard HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Trading Bot Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status-bar {
            display: flex;
            justify-content: space-around;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        
        .status-item {
            text-align: center;
        }
        
        .status-value {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .status-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-start {
            background: #28a745;
            color: white;
        }
        
        .btn-stop {
            background: #dc3545;
            color: white;
        }
        
        .btn-emergency {
            background: #ff4757;
            color: white;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        .card h3 {
            margin-bottom: 15px;
            color: #ffd700;
        }
        
        .position-item, .alert-item {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        .position-profit {
            color: #28a745;
        }
        
        .position-loss {
            color: #dc3545;
        }
        
        .alert-critical {
            border-left: 4px solid #dc3545;
        }
        
        .alert-warning {
            border-left: 4px solid #ffc107;
        }
        
        .alert-info {
            border-left: 4px solid #17a2b8;
        }
        
        .manual-trade {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background: rgba(255,255,255,0.1);
            color: white;
        }
        
        .form-group input::placeholder {
            color: rgba(255,255,255,0.6);
        }
        
        .status-running {
            color: #28a745;
        }
        
        .status-stopped {
            color: #dc3545;
        }
        
        .status-emergency {
            color: #ff4757;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Smart Trading Bot Dashboard</h1>
            <p>Advanced Crypto Trading with AI-Powered Sentiment Analysis</p>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-value" id="bot-status">STOPPED</div>
                <div class="status-label">Bot Status</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="total-pnl">$0.00</div>
                <div class="status-label">Total P&L</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="positions-count">0</div>
                <div class="status-label">Open Positions</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="alerts-count">0</div>
                <div class="status-label">Active Alerts</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn btn-start" onclick="startBot()">▶️ Start Bot</button>
            <button class="btn btn-stop" onclick="stopBot()">⏹️ Stop Bot</button>
            <button class="btn btn-emergency" onclick="emergencyStop()">🚨 EMERGENCY STOP</button>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Current Positions</h3>
                <div id="positions-list">
                    <p>No open positions</p>
                </div>
            </div>
            
            <div class="card">
                <h3>🚨 Recent Alerts</h3>
                <div id="alerts-list">
                    <p>No alerts</p>
                </div>
            </div>
        </div>
        
        <div class="manual-trade">
            <h3>💱 Manual Trade Execution</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div class="form-group">
                    <label>Symbol</label>
                    <input type="text" id="trade-symbol" placeholder="BTCUSDT" value="BTCUSDT">
                </div>
                <div class="form-group">
                    <label>Side</label>
                    <select id="trade-side">
                        <option value="buy">Buy</option>
                        <option value="sell">Sell</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Size</label>
                    <input type="number" id="trade-size" placeholder="100" step="0.01">
                </div>
                <div class="form-group">
                    <label>Price (optional)</label>
                    <input type="number" id="trade-price" placeholder="Market" step="0.01">
                </div>
            </div>
            <button class="btn btn-start" onclick="executeTrade()" style="margin-top: 15px;">Execute Trade</button>
        </div>
    </div>

    <script>
        let ws = null;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status_update') {
                    updateDashboard(data.data);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket connection closed. Attempting to reconnect...');
                setTimeout(connectWebSocket, 5000);
            };
        }
        
        function updateDashboard(data) {
            // Update status
            const statusElement = document.getElementById('bot-status');
            statusElement.textContent = data.bot_status.toUpperCase();
            statusElement.className = 'status-value status-' + data.bot_status;
            
            // Update P&L
            document.getElementById('total-pnl').textContent = `$${data.pnl.toFixed(2)}`;
            
            // Update positions count
            document.getElementById('positions-count').textContent = data.positions.length;
            
            // Update positions list
            const positionsList = document.getElementById('positions-list');
            if (data.positions.length === 0) {
                positionsList.innerHTML = '<p>No open positions</p>';
            } else {
                positionsList.innerHTML = data.positions.map(pos => `
                    <div class="position-item">
                        <strong>${pos.symbol}</strong> - ${pos.side.toUpperCase()}<br>
                        Size: ${pos.size} | Entry: $${pos.entry_price}<br>
                        <span class="${pos.pnl >= 0 ? 'position-profit' : 'position-loss'}">
                            P&L: $${pos.pnl.toFixed(2)}
                        </span>
                    </div>
                `).join('');
            }
        }
        
        async function startBot() {
            try {
                const response = await fetch('/api/bot/start', { method: 'POST' });
                const result = await response.json();
                if (response.ok) {
                    alert('Bot started successfully!');
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }
        
        async function stopBot() {
            try {
                const response = await fetch('/api/bot/stop', { method: 'POST' });
                const result = await response.json();
                if (response.ok) {
                    alert('Bot stopped successfully!');
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }
        
        async function emergencyStop() {
            if (confirm('Are you sure you want to trigger EMERGENCY STOP? This will close all positions immediately!')) {
                try {
                    const response = await fetch('/api/bot/emergency_stop', { method: 'POST' });
                    const result = await response.json();
                    if (response.ok) {
                        alert('Emergency stop executed! All positions have been closed.');
                    } else {
                        alert(`Error: ${result.detail}`);
                    }
                } catch (error) {
                    alert(`Error: ${error.message}`);
                }
            }
        }
        
        async function executeTrade() {
            const symbol = document.getElementById('trade-symbol').value;
            const side = document.getElementById('trade-side').value;
            const size = parseFloat(document.getElementById('trade-size').value);
            const price = document.getElementById('trade-price').value;
            
            if (!symbol || !size) {
                alert('Please fill in symbol and size');
                return;
            }
            
            const trade = {
                symbol: symbol,
                side: side,
                size: size,
                confidence: 1.0,
                reasoning: 'Manual trade from dashboard'
            };
            
            if (price) {
                trade.price = parseFloat(price);
            }
            
            try {
                const response = await fetch('/api/trade', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(trade)
                });
                
                const result = await response.json();
                if (response.ok) {
                    alert('Trade executed successfully!');
                } else {
                    alert(`Error: ${result.detail}`);
                }
            } catch (error) {
                alert(`Error: ${error.message}`);
            }
        }
        
        // Initialize dashboard
        window.onload = function() {
            connectWebSocket();
            
            // Load initial data
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateDashboard({
                        bot_status: data.status,
                        pnl: data.total_pnl,
                        positions: []
                    });
                });
        };
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )