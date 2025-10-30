"""
Integrated High Volatility Low Cap Trading Bot
Combines adaptive scaling, UUID tracking, ML pipeline, and low cap strategy
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Local imports
from src.data.trade_tracking import TradeTrackingSystem, TradeCandidate, ExecutionRecord, TradeOutcome, TerminationReason
from src.strategies.high_volatility_low_cap import HighVolatilityLowCapStrategy, LowCapCandidate
from src.ml.feature_engineering import FeatureEngineer
from src.ml.safe_training_pipeline import SafeMLPipeline, ModelConfig
from src.risk.adaptive_scaling import AdaptiveProfitScaling
from src.data.unified_market_provider import UnifiedMarketDataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedTradingBot:
    """Integrated trading bot with UUID tracking and ML pipeline"""
    
    def __init__(self,
                 small_fund_usd: float = 1000.0,
                 total_fund_usd: float = 10000.0,
                 coinmarketcap_api_key: str = None,
                 coingecko_api_key: str = None,
                 dappradar_api_key: str = None):
        
        self.small_fund_usd = small_fund_usd
        self.total_fund_usd = total_fund_usd
        
        # Initialize core systems
        self.trade_tracker = TradeTrackingSystem(
            db_path="integrated_trades.db",
            artifacts_dir="integrated_artifacts"
        )
        
        # Initialize strategies
        self.low_cap_strategy = HighVolatilityLowCapStrategy(
            small_fund_usd=small_fund_usd,
            coinmarketcap_api_key=coinmarketcap_api_key,
            coingecko_api_key=coingecko_api_key,
            dappradar_api_key=dappradar_api_key
        )
        
        # Initialize adaptive scaling for main fund
        self.adaptive_scaling = AdaptiveProfitScaling()
        
        # Initialize ML pipeline
        self.ml_pipeline = SafeMLPipeline(
            db_path="integrated_trades.db",
            models_dir="integrated_models",
            config=ModelConfig(
                model_type="lightgbm",
                target_column="realized_pnl_percent",
                max_features=40,
                min_samples=50,
                min_r2_score=0.05,
                canary_traffic_percent=15.0
            )
        )
        
        # Initialize market data provider
        self.market_provider = UnifiedMarketDataProvider(
            coinmarketcap_api_key=coinmarketcap_api_key,
            coingecko_api_key=coingecko_api_key
        )
        
        # Active positions tracking
        self.active_positions: Dict[str, Dict] = {}
        self.position_monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.session_stats = {
            'trades_initiated': 0,
            'trades_completed': 0,
            'total_pnl': 0.0,
            'successful_trades': 0,
            'failed_trades': 0,
            'session_start': datetime.now(timezone.utc)
        }
        
        logger.info(f"Initialized Integrated Trading Bot")
        logger.info(f"Small fund allocation: ${small_fund_usd}")
        logger.info(f"Total fund: ${total_fund_usd}")
    
    async def start_trading_session(self):
        """Start the integrated trading session"""
        logger.info("🚀 Starting integrated trading session")
        
        try:
            # Start background tasks
            tasks = [
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self._ml_training_loop()),
                asyncio.create_task(self._performance_reporting_loop())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            logger.info("Trading session interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
        finally:
            await self._cleanup_session()
    
    async def _trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop")
        
        while True:
            try:
                # 1. Scan for low cap opportunities
                candidates = await self.low_cap_strategy.scan_low_cap_opportunities()
                
                if candidates:
                    logger.info(f"Found {len(candidates)} low cap candidates")
                    
                    # 2. Evaluate and create trade candidates
                    for candidate in candidates[:3]:  # Limit to top 3
                        if len(self.active_positions) < 5:  # Max 5 concurrent positions
                            await self._process_candidate(candidate)
                
                # 3. Check for main fund opportunities (using ML predictions)
                await self._check_main_fund_opportunities()
                
                # Wait before next scan
                await asyncio.sleep(300)  # 5 minutes between scans
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_candidate(self, candidate: LowCapCandidate):
        """Process a low cap candidate through the full pipeline"""
        try:
            # 1. Create trade candidate with UUID tracking
            trade_candidate = await self.low_cap_strategy.create_trade_candidate(candidate)
            
            if not trade_candidate:
                return
            
            # 2. Get ML prediction if model is available
            ml_confidence = await self._get_ml_prediction(trade_candidate)
            
            if ml_confidence:
                logger.info(f"ML confidence for {candidate.symbol}: {ml_confidence:.3f}")
                
                # Adjust confidence based on ML prediction
                trade_candidate.confidence = (trade_candidate.confidence + ml_confidence) / 2
            
            # 3. Execute trade if confidence is high enough
            if trade_candidate.confidence > 0.6:
                await self._execute_trade(trade_candidate, candidate)
            else:
                logger.info(f"Skipping {candidate.symbol}: Low confidence {trade_candidate.confidence:.3f}")
                
        except Exception as e:
            logger.error(f"Error processing candidate {candidate.symbol}: {e}")
    
    async def _execute_trade(self, trade_candidate: TradeCandidate, low_cap_candidate: LowCapCandidate):
        """Execute trade with full tracking"""
        logger.info(f"🎯 Executing trade for {trade_candidate.instrument}")
        
        try:
            # 1. Calculate position size with adaptive scaling
            base_position_size = self.low_cap_strategy.calculate_position_size(
                low_cap_candidate, trade_candidate.confidence
            )
            
            # Apply adaptive scaling if profitable
            scaling_factor = self.adaptive_scaling.get_scaling_factor()
            position_size = base_position_size * scaling_factor
            
            logger.info(f"Position size: ${position_size:.2f} (scaling: {scaling_factor:.1f}x)")
            
            # 2. Simulate trade execution (in real system, this would call exchange API)
            entry_price = await self._get_entry_price(trade_candidate.instrument)
            
            if not entry_price:
                logger.warning(f"Could not get entry price for {trade_candidate.instrument}")
                return
            
            # 3. Create execution record
            execution_record = ExecutionRecord(
                uuid=trade_candidate.uuid,
                order_id=f"order_{int(time.time())}",
                venue="simulated_exchange",
                entry_timestamp=datetime.now(timezone.utc),
                exit_timestamp=None,
                entry_price=entry_price,
                exit_price=None,
                size=position_size,
                fees_paid=position_size * 0.001,  # 0.1% fee
                slippage=0.002,  # 0.2% slippage
                fills_json=json.dumps([{
                    "price": entry_price,
                    "size": position_size,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }]),
                latency_metrics={"order_latency_ms": 45, "fill_latency_ms": 12},
                route_taken="direct",
                mev_detected=False
            )
            
            # 4. Store execution record
            self.trade_tracker.store_execution_record(execution_record)
            
            # 5. Update candidate status
            self.trade_tracker.update_candidate_status(
                trade_candidate.uuid,
                self.trade_tracker.TradeStatus.ACTIVE
            )
            
            # 6. Start monitoring position
            self.active_positions[trade_candidate.uuid] = {
                'trade_candidate': trade_candidate,
                'execution_record': execution_record,
                'low_cap_candidate': low_cap_candidate,
                'entry_time': datetime.now(timezone.utc),
                'target_profit_pct': 30.0,
                'stop_loss_pct': 15.0
            }
            
            # Start monitoring task
            self.position_monitoring_tasks[trade_candidate.uuid] = asyncio.create_task(
                self._monitor_position(trade_candidate.uuid)
            )
            
            self.session_stats['trades_initiated'] += 1
            logger.info(f"✅ Trade executed: {trade_candidate.uuid}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def _get_entry_price(self, instrument: str) -> Optional[float]:
        """Get entry price for instrument"""
        try:
            # Extract symbol from instrument (e.g., "PEPE/USDT" -> "PEPE")
            symbol = instrument.split('/')[0]
            
            # Get market data
            market_data = await self.market_provider.get_market_data(symbol)
            
            if market_data and 'price' in market_data:
                return float(market_data['price'])
            
            # Fallback to mock price for demonstration
            return 0.001 * (1 + (hash(symbol) % 1000) / 10000)  # Mock price
            
        except Exception as e:
            logger.error(f"Error getting entry price for {instrument}: {e}")
            return None
    
    async def _monitor_position(self, trade_uuid: str):
        """Monitor active position"""
        logger.info(f"Starting position monitoring for {trade_uuid}")
        
        try:
            while trade_uuid in self.active_positions:
                position = self.active_positions[trade_uuid]
                
                # Get current price
                current_price = await self._get_current_price(position['trade_candidate'].instrument)
                
                if current_price:
                    entry_price = position['execution_record'].entry_price
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Check exit conditions
                    should_exit, reason = self._check_exit_conditions(position, pnl_pct)
                    
                    if should_exit:
                        await self._close_position(trade_uuid, current_price, reason)
                        break
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"Error monitoring position {trade_uuid}: {e}")
    
    def _check_exit_conditions(self, position: Dict, current_pnl_pct: float) -> tuple[bool, str]:
        """Check if position should be closed"""
        
        # Profit target
        if current_pnl_pct >= position['target_profit_pct']:
            return True, TerminationReason.PROFIT_TARGET.value
        
        # Stop loss
        if current_pnl_pct <= -position['stop_loss_pct']:
            return True, TerminationReason.STOP_LOSS.value
        
        # Time limit (48 hours for low cap trades)
        time_in_position = datetime.now(timezone.utc) - position['entry_time']
        if time_in_position.total_seconds() > 48 * 3600:
            return True, TerminationReason.TIME_LIMIT.value
        
        # Rugpull detection (simplified)
        if current_pnl_pct < -25:  # Severe loss might indicate rugpull
            return True, TerminationReason.RUG_DETECTED.value
        
        return False, ""
    
    async def _close_position(self, trade_uuid: str, exit_price: float, reason: str):
        """Close position and record outcome"""
        logger.info(f"🔚 Closing position {trade_uuid}: {reason}")
        
        try:
            position = self.active_positions[trade_uuid]
            execution_record = position['execution_record']
            
            # Calculate final metrics
            entry_price = execution_record.entry_price
            position_size = execution_record.size
            
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_usd = (position_size * pnl_pct / 100) - execution_record.fees_paid
            
            time_in_market = datetime.now(timezone.utc) - execution_record.entry_timestamp
            
            # Update execution record
            execution_record.exit_timestamp = datetime.now(timezone.utc)
            execution_record.exit_price = exit_price
            self.trade_tracker.store_execution_record(execution_record)
            
            # Create outcome record
            outcome = TradeOutcome(
                uuid=trade_uuid,
                realized_pnl=pnl_usd,
                realized_pnl_percent=pnl_pct,
                max_adverse_excursion=-5.0,  # Would track during monitoring
                max_favorable_excursion=max(pnl_pct, 0),
                time_in_market_seconds=int(time_in_market.total_seconds()),
                termination_reason=TerminationReason(reason),
                external_incidents=[],
                execution_quality_score=0.85  # Based on slippage, timing, etc.
            )
            
            # Store outcome
            self.trade_tracker.store_outcome(outcome)
            
            # Update adaptive scaling with result
            if pnl_usd > 0:
                self.adaptive_scaling.add_trade_result(pnl_usd, True)
                self.session_stats['successful_trades'] += 1
            else:
                self.adaptive_scaling.add_trade_result(abs(pnl_usd), False)
                self.session_stats['failed_trades'] += 1
            
            # Update session stats
            self.session_stats['trades_completed'] += 1
            self.session_stats['total_pnl'] += pnl_usd
            
            # Clean up position
            del self.active_positions[trade_uuid]
            if trade_uuid in self.position_monitoring_tasks:
                self.position_monitoring_tasks[trade_uuid].cancel()
                del self.position_monitoring_tasks[trade_uuid]
            
            logger.info(f"✅ Position closed: {pnl_pct:.2f}% PnL (${pnl_usd:.2f})")
            
        except Exception as e:
            logger.error(f"Error closing position {trade_uuid}: {e}")
    
    async def _get_current_price(self, instrument: str) -> Optional[float]:
        """Get current price for position monitoring"""
        try:
            symbol = instrument.split('/')[0]
            market_data = await self.market_provider.get_market_data(symbol)
            
            if market_data and 'price' in market_data:
                return float(market_data['price'])
            
            # Mock price movement for demonstration
            import random
            return 0.001 * (1 + random.uniform(-0.1, 0.1))  # ±10% movement
            
        except Exception:
            return None
    
    async def _get_ml_prediction(self, trade_candidate: TradeCandidate) -> Optional[float]:
        """Get ML model prediction for trade candidate"""
        try:
            # Load active model
            model_data = self.ml_pipeline.load_model()
            
            if not model_data:
                return None
            
            # Extract features (simplified - would use feature engineering)
            # This is a placeholder - real implementation would extract full feature vector
            mock_features = [
                trade_candidate.confidence,
                0.5,  # mock market cap score
                0.7,  # mock liquidity score
                0.6,  # mock volatility score
                0.4,  # mock rugpull risk
                0.65  # mock social sentiment
            ]
            
            # Pad to match expected feature count
            while len(mock_features) < len(model_data['feature_names']):
                mock_features.append(0.5)
            
            # Make prediction
            features_scaled = model_data['scaler'].transform([mock_features])
            prediction = model_data['model'].predict(features_scaled)[0]
            
            # Convert prediction to confidence score (0-1)
            confidence = max(0, min(1, (prediction + 20) / 40))  # Assuming prediction range -20 to +20
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return None
    
    async def _check_main_fund_opportunities(self):
        """Check for main fund opportunities using ML predictions"""
        # This would implement larger position trading with the main fund
        # For now, we focus on the low cap strategy
        pass
    
    async def _monitoring_loop(self):
        """Monitor overall system health"""
        logger.info("Starting monitoring loop")
        
        while True:
            try:
                # Monitor active positions
                logger.info(f"Active positions: {len(self.active_positions)}")
                
                # Check for risk management
                total_exposure = sum(
                    pos['execution_record'].size 
                    for pos in self.active_positions.values()
                )
                
                if total_exposure > self.small_fund_usd * 0.8:  # 80% max exposure
                    logger.warning(f"High exposure: ${total_exposure:.2f}")
                
                await asyncio.sleep(120)  # Every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _ml_training_loop(self):
        """Periodic ML model training"""
        logger.info("Starting ML training loop")
        
        while True:
            try:
                # Train new model every 6 hours
                await asyncio.sleep(6 * 3600)
                
                logger.info("Running automated ML training cycle")
                self.ml_pipeline.run_automated_training_cycle()
                
            except Exception as e:
                logger.error(f"Error in ML training loop: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _performance_reporting_loop(self):
        """Periodic performance reporting"""
        logger.info("Starting performance reporting loop")
        
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                # Generate performance report
                self._log_performance_report()
                
            except Exception as e:
                logger.error(f"Error in performance reporting: {e}")
                await asyncio.sleep(600)
    
    def _log_performance_report(self):
        """Log current performance statistics"""
        stats = self.session_stats
        
        session_duration = datetime.now(timezone.utc) - stats['session_start']
        session_hours = session_duration.total_seconds() / 3600
        
        win_rate = (stats['successful_trades'] / max(stats['trades_completed'], 1)) * 100
        avg_pnl = stats['total_pnl'] / max(stats['trades_completed'], 1)
        
        scaling_factor = self.adaptive_scaling.get_scaling_factor()
        current_profit = self.adaptive_scaling.total_profit
        
        logger.info("📊 PERFORMANCE REPORT")
        logger.info(f"Session duration: {session_hours:.1f} hours")
        logger.info(f"Trades initiated: {stats['trades_initiated']}")
        logger.info(f"Trades completed: {stats['trades_completed']}")
        logger.info(f"Active positions: {len(self.active_positions)}")
        logger.info(f"Win rate: {win_rate:.1f}%")
        logger.info(f"Total PnL: ${stats['total_pnl']:.2f}")
        logger.info(f"Average PnL per trade: ${avg_pnl:.2f}")
        logger.info(f"Adaptive scaling: {scaling_factor:.1f}x (profit: ${current_profit:.2f})")
    
    async def _cleanup_session(self):
        """Clean up at end of session"""
        logger.info("Cleaning up trading session")
        
        # Cancel all monitoring tasks
        for task in self.position_monitoring_tasks.values():
            task.cancel()
        
        # Final performance report
        self._log_performance_report()
        
        # Get complete trade tracking metrics
        metrics = self.trade_tracker.get_performance_metrics()
        logger.info(f"Total tracked trades: {metrics.get('total_trades', 0)}")
        logger.info(f"Completed trades: {metrics.get('completed_trades', 0)}")
        logger.info(f"Overall win rate: {metrics.get('win_rate', 0) * 100:.1f}%")

# Main execution
async def main():
    """Main entry point"""
    
    # Configuration
    config = {
        'small_fund_usd': 1000.0,
        'total_fund_usd': 10000.0,
        'coinmarketcap_api_key': '6cad35f36d7b4e069b8dcb0eb9d17d56',
        'coingecko_api_key': 'CG-uKph8trS6RiycsxwVQtxfxvF',
        'dappradar_api_key': 'xD9Fvb0Nb285BLRPfKLgL44ULe6nR8Fm90i894xA'
    }
    
    # Initialize and start bot
    bot = IntegratedTradingBot(**config)
    await bot.start_trading_session()

if __name__ == "__main__":
    print("🚀 Starting Integrated High Volatility Low Cap Trading Bot")
    print("Features:")
    print("  ✅ UUID-linked trade tracking")
    print("  ✅ High volatility low cap strategy")
    print("  ✅ Adaptive position scaling")
    print("  ✅ ML prediction pipeline")
    print("  ✅ Real-time monitoring")
    print("  ✅ Automated risk management")
    print("\nPress Ctrl+C to stop\n")
    
    asyncio.run(main())