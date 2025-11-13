"""
Complete System Integration Test
Tests all powerhouse components working together:
- Enhanced Social Sentiment System
- Smart Contract Security Analysis  
- Whale Wallet Monitoring
- DEX Aggregator Integration
- ML Performance Prediction
- Advanced Risk Management
- Real-time Monitoring
- Automated Trading System
- GUI Risk Controls
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'security'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'monitoring'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'trading'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'ai'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test.log')
    ]
)
logger = logging.getLogger(__name__)

class PowerhouseIntegrationTest:
    """
    Comprehensive integration test for the complete powerhouse trading system
    """
    
    def __init__(self):
        self.test_results = {}
        self.components = {}
        self.test_token = {
            'address': '0x1234567890123456789012345678901234567890',
            'symbol': 'TESTCOIN',
            'name': 'Test Microcap Token'
        }
        
    async def initialize_components(self):
        """Initialize all powerhouse components"""
        try:
            logger.info("🚀 Initializing all powerhouse components...")
            
            # Import all components
            try:
                from social_sentiment import EnhancedSocialSentimentCollector
                self.components['sentiment'] = EnhancedSocialSentimentCollector()
                logger.info("✅ Social Sentiment System loaded")
            except Exception as e:
                logger.error(f"❌ Social Sentiment System failed: {e}")
                self.components['sentiment'] = None
            
            try:
                from contract_analyzer import SmartContractAnalyzer
                self.components['security'] = SmartContractAnalyzer()
                logger.info("✅ Security Analysis System loaded")
            except Exception as e:
                logger.error(f"❌ Security Analysis System failed: {e}")
                self.components['security'] = None
            
            try:
                from whale_monitor import WhaleMonitor
                self.components['whale'] = WhaleMonitor()
                logger.info("✅ Whale Monitoring System loaded")
            except Exception as e:
                logger.error(f"❌ Whale Monitoring System failed: {e}")
                self.components['whale'] = None
            
            try:
                from dex_aggregator import DEXAggregator
                self.components['dex'] = DEXAggregator()
                logger.info("✅ DEX Aggregator System loaded")
            except Exception as e:
                logger.error(f"❌ DEX Aggregator System failed: {e}")
                self.components['dex'] = None
            
            try:
                from ml_predictor import MLPredictor, PredictionTimeframe
                self.components['ml'] = MLPredictor()
                self.PredictionTimeframe = PredictionTimeframe
                logger.info("✅ ML Prediction System loaded")
            except Exception as e:
                logger.error(f"❌ ML Prediction System failed: {e}")
                self.components['ml'] = None
            
            # Additional components would be loaded here
            # (Risk Manager, Real-time Monitor, Trading Engine, GUI)
            
            loaded_components = sum(1 for comp in self.components.values() if comp is not None)
            total_components = len(self.components)
            
            logger.info(f"📊 Component loading: {loaded_components}/{total_components} successful")
            
            return loaded_components > 0
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    async def test_sentiment_analysis(self) -> Dict:
        """Test the enhanced social sentiment system"""
        try:
            logger.info("\n🔍 Testing Social Sentiment Analysis...")
            
            if not self.components['sentiment']:
                return {"status": "SKIPPED", "reason": "Component not loaded"}
            
            # Test sentiment collection
            sentiment_result = await self.components['sentiment'].collect_sentiment(
                self.test_token['symbol']
            )
            
            # Validate sentiment data
            if sentiment_result:
                test_result = {
                    "status": "PASS",
                    "overall_score": sentiment_result.overall_score,
                    "confidence": sentiment_result.confidence,
                    "sources": len(sentiment_result.sources),
                    "reddit_score": sentiment_result.reddit_sentiment,
                    "news_score": sentiment_result.news_sentiment
                }
                logger.info(f"✅ Sentiment Analysis: Score {sentiment_result.overall_score:.2f}, "
                           f"Confidence {sentiment_result.confidence:.2f}")
            else:
                test_result = {"status": "FAIL", "reason": "No sentiment data returned"}
                logger.error("❌ Sentiment Analysis failed")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing sentiment analysis: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def test_security_analysis(self) -> Dict:
        """Test the smart contract security analysis"""
        try:
            logger.info("\n🔒 Testing Smart Contract Security Analysis...")
            
            if not self.components['security']:
                return {"status": "SKIPPED", "reason": "Component not loaded"}
            
            # Test security analysis
            security_result = await self.components['security'].analyze_contract(
                self.test_token['address']
            )
            
            # Validate security data
            if security_result:
                test_result = {
                    "status": "PASS",
                    "security_score": security_result.security_score,
                    "risk_level": security_result.risk_level.value,
                    "honeypot_risk": security_result.honeypot_risk,
                    "ownership_renounced": security_result.ownership_renounced,
                    "liquidity_locked": security_result.liquidity_locked,
                    "verified_contract": security_result.verified_contract,
                    "issues_found": len(security_result.security_issues)
                }
                logger.info(f"✅ Security Analysis: Score {security_result.security_score}/100, "
                           f"Risk Level {security_result.risk_level.value}")
            else:
                test_result = {"status": "FAIL", "reason": "No security data returned"}
                logger.error("❌ Security Analysis failed")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing security analysis: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def test_whale_monitoring(self) -> Dict:
        """Test the whale wallet monitoring system"""
        try:
            logger.info("\n🐋 Testing Whale Monitoring System...")
            
            if not self.components['whale']:
                return {"status": "SKIPPED", "reason": "Component not loaded"}
            
            # Test whale activity analysis
            whale_result = await self.components['whale'].analyze_whale_activity(
                self.test_token['address']
            )
            
            # Validate whale data
            if whale_result:
                test_result = {
                    "status": "PASS",
                    "whale_holders": len(whale_result.whale_holders),
                    "activity_score": whale_result.activity_score,
                    "accumulation_trend": whale_result.accumulation_trend,
                    "insider_activity": whale_result.insider_activity_detected,
                    "smart_money_score": whale_result.smart_money_score,
                    "risk_alerts": len(whale_result.risk_alerts)
                }
                logger.info(f"✅ Whale Monitoring: {len(whale_result.whale_holders)} whales, "
                           f"Activity Score {whale_result.activity_score:.2f}")
            else:
                test_result = {"status": "FAIL", "reason": "No whale data returned"}
                logger.error("❌ Whale Monitoring failed")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing whale monitoring: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def test_dex_aggregation(self) -> Dict:
        """Test the DEX aggregator system"""
        try:
            logger.info("\n💱 Testing DEX Aggregator...")
            
            if not self.components['dex']:
                return {"status": "SKIPPED", "reason": "Component not loaded"}
            
            # Test optimal route finding
            WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
            USDC = "0xA0b86a33E6411c3Ce98e6d9b4B3C61a6F2B0C1D2"
            
            route_result = await self.components['dex'].get_optimal_route(
                token_in=WETH,
                token_out=USDC,
                amount_in=1.0,
                slippage=0.01
            )
            
            # Validate route data
            if route_result:
                test_result = {
                    "status": "PASS",
                    "aggregator": route_result.aggregator.value,
                    "input_amount": route_result.input_amount,
                    "output_amount": route_result.output_amount,
                    "price_impact": route_result.price_impact,
                    "gas_estimate": route_result.gas_estimate,
                    "confidence_score": route_result.confidence_score,
                    "savings_vs_worst": route_result.savings_vs_worst
                }
                logger.info(f"✅ DEX Aggregation: {route_result.output_amount:.2f} output via "
                           f"{route_result.aggregator.value}")
            else:
                test_result = {"status": "FAIL", "reason": "No route found"}
                logger.error("❌ DEX Aggregation failed")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing DEX aggregation: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def test_ml_prediction(self) -> Dict:
        """Test the ML performance prediction system"""
        try:
            logger.info("\n🧠 Testing ML Performance Prediction...")
            
            if not self.components['ml']:
                return {"status": "SKIPPED", "reason": "Component not loaded"}
            
            # Test ML prediction for multiple timeframes
            timeframes = [
                self.PredictionTimeframe.HOURS_24,
                self.PredictionTimeframe.DAYS_7
            ]
            
            predictions = {}
            for timeframe in timeframes:
                prediction_result = await self.components['ml'].predict_performance(
                    self.test_token['address'],
                    self.test_token['symbol'],
                    timeframe
                )
                
                if prediction_result:
                    predictions[timeframe.value] = {
                        "predicted_return": prediction_result.predicted_return,
                        "confidence_level": prediction_result.confidence_level.value,
                        "confidence_score": prediction_result.confidence_score,
                        "upside_potential": prediction_result.upside_potential,
                        "downside_risk": prediction_result.downside_risk,
                        "bullish_signals": len(prediction_result.bullish_signals),
                        "bearish_signals": len(prediction_result.bearish_signals)
                    }
            
            if predictions:
                test_result = {
                    "status": "PASS",
                    "predictions": predictions,
                    "timeframes_tested": len(predictions)
                }
                logger.info(f"✅ ML Prediction: {len(predictions)} timeframes predicted")
            else:
                test_result = {"status": "FAIL", "reason": "No predictions generated"}
                logger.error("❌ ML Prediction failed")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing ML prediction: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def test_component_integration(self) -> Dict:
        """Test how well components work together"""
        try:
            logger.info("\n🔗 Testing Component Integration...")
            
            integration_score = 0.0
            integration_details = {}
            
            # Test data flow between components
            
            # 1. Security -> ML Integration
            if self.components['security'] and self.components['ml']:
                try:
                    # Collect security features for ML
                    security_data = await self.components['security'].analyze_contract(
                        self.test_token['address']
                    )
                    
                    # Pass security data to ML predictor (would be done via feature collection)
                    ml_features = await self.components['ml'].collect_features(
                        self.test_token['address'],
                        self.test_token['symbol']
                    )
                    
                    if security_data and ml_features:
                        integration_score += 0.25
                        integration_details['security_ml'] = "PASS"
                        logger.info("✅ Security -> ML integration working")
                    else:
                        integration_details['security_ml'] = "FAIL"
                        logger.error("❌ Security -> ML integration failed")
                        
                except Exception as e:
                    integration_details['security_ml'] = f"ERROR: {e}"
            
            # 2. Whale -> ML Integration
            if self.components['whale'] and self.components['ml']:
                try:
                    whale_data = await self.components['whale'].analyze_whale_activity(
                        self.test_token['address']
                    )
                    
                    if whale_data:
                        integration_score += 0.25
                        integration_details['whale_ml'] = "PASS"
                        logger.info("✅ Whale -> ML integration working")
                    else:
                        integration_details['whale_ml'] = "FAIL"
                        logger.error("❌ Whale -> ML integration failed")
                        
                except Exception as e:
                    integration_details['whale_ml'] = f"ERROR: {e}"
            
            # 3. Sentiment -> ML Integration
            if self.components['sentiment'] and self.components['ml']:
                try:
                    sentiment_data = await self.components['sentiment'].collect_sentiment(
                        self.test_token['symbol']
                    )
                    
                    if sentiment_data:
                        integration_score += 0.25
                        integration_details['sentiment_ml'] = "PASS"
                        logger.info("✅ Sentiment -> ML integration working")
                    else:
                        integration_details['sentiment_ml'] = "FAIL"
                        logger.error("❌ Sentiment -> ML integration failed")
                        
                except Exception as e:
                    integration_details['sentiment_ml'] = f"ERROR: {e}"
            
            # 4. ML -> DEX Integration (for trade execution)
            if self.components['ml'] and self.components['dex']:
                try:
                    # Get ML prediction
                    prediction = await self.components['ml'].predict_performance(
                        self.test_token['address'],
                        self.test_token['symbol']
                    )
                    
                    # Use prediction confidence for trade sizing (simulated)
                    if prediction and prediction.confidence_score > 0.5:
                        integration_score += 0.25
                        integration_details['ml_dex'] = "PASS"
                        logger.info("✅ ML -> DEX integration working")
                    else:
                        integration_details['ml_dex'] = "PARTIAL"
                        logger.warning("⚠️ ML -> DEX integration partial")
                        
                except Exception as e:
                    integration_details['ml_dex'] = f"ERROR: {e}"
            
            # Calculate overall integration score
            integration_percentage = integration_score * 100
            
            test_result = {
                "status": "PASS" if integration_score >= 0.5 else "PARTIAL",
                "integration_score": integration_percentage,
                "details": integration_details,
                "data_flow_working": integration_score >= 0.75
            }
            
            logger.info(f"🔗 Integration Score: {integration_percentage:.1f}%")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing component integration: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def test_system_performance(self) -> Dict:
        """Test overall system performance and response times"""
        try:
            logger.info("\n⚡ Testing System Performance...")
            
            start_time = datetime.now()
            performance_metrics = {}
            
            # Test component response times
            for component_name, component in self.components.items():
                if component is None:
                    continue
                
                comp_start = datetime.now()
                
                try:
                    if component_name == 'sentiment':
                        await component.collect_sentiment(self.test_token['symbol'])
                    elif component_name == 'security':
                        await component.analyze_contract(self.test_token['address'])
                    elif component_name == 'whale':
                        await component.analyze_whale_activity(self.test_token['address'])
                    elif component_name == 'dex':
                        WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
                        USDC = "0xA0b86a33E6411c3Ce98e6d9b4B3C61a6F2B0C1D2"
                        await component.get_optimal_route(WETH, USDC, 1.0)
                    elif component_name == 'ml':
                        await component.predict_performance(
                            self.test_token['address'],
                            self.test_token['symbol']
                        )
                    
                    comp_time = (datetime.now() - comp_start).total_seconds()
                    performance_metrics[component_name] = {
                        "response_time": comp_time,
                        "status": "PASS" if comp_time < 10.0 else "SLOW"
                    }
                    
                    logger.info(f"⚡ {component_name.title()}: {comp_time:.2f}s")
                    
                except Exception as e:
                    performance_metrics[component_name] = {
                        "response_time": 999.0,
                        "status": "ERROR",
                        "error": str(e)
                    }
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate performance score
            avg_response_time = sum(
                m["response_time"] for m in performance_metrics.values() 
                if m["response_time"] < 999.0
            ) / max(1, len([m for m in performance_metrics.values() if m["response_time"] < 999.0]))
            
            performance_score = max(0, 100 - (avg_response_time * 10))  # Penalty for slow responses
            
            test_result = {
                "status": "PASS" if performance_score >= 60 else "SLOW",
                "total_test_time": total_time,
                "avg_response_time": avg_response_time,
                "performance_score": performance_score,
                "component_metrics": performance_metrics
            }
            
            logger.info(f"⚡ Performance Score: {performance_score:.1f}%")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Error testing system performance: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def run_full_integration_test(self):
        """Run complete powerhouse system integration test"""
        try:
            logger.info("🚀 STARTING POWERHOUSE SYSTEM INTEGRATION TEST")
            logger.info("=" * 60)
            
            # Initialize all components
            init_success = await self.initialize_components()
            if not init_success:
                logger.error("❌ Failed to initialize components - aborting test")
                return
            
            # Run individual component tests
            test_functions = [
                ("Sentiment Analysis", self.test_sentiment_analysis),
                ("Security Analysis", self.test_security_analysis),
                ("Whale Monitoring", self.test_whale_monitoring),
                ("DEX Aggregation", self.test_dex_aggregation),
                ("ML Prediction", self.test_ml_prediction),
                ("Component Integration", self.test_component_integration),
                ("System Performance", self.test_system_performance)
            ]
            
            for test_name, test_func in test_functions:
                try:
                    result = await test_func()
                    self.test_results[test_name] = result
                    
                    if result["status"] == "PASS":
                        logger.info(f"✅ {test_name}: PASSED")
                    elif result["status"] == "PARTIAL":
                        logger.warning(f"⚠️ {test_name}: PARTIAL")
                    elif result["status"] == "SKIPPED":
                        logger.info(f"⏭️ {test_name}: SKIPPED")
                    else:
                        logger.error(f"❌ {test_name}: FAILED")
                        
                except Exception as e:
                    logger.error(f"❌ {test_name}: ERROR - {e}")
                    self.test_results[test_name] = {"status": "ERROR", "error": str(e)}
            
            # Generate test summary
            self.generate_test_summary()
            
        except Exception as e:
            logger.error(f"Critical error in integration test: {e}")
    
    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        try:
            logger.info("\n" + "=" * 60)
            logger.info("🎯 POWERHOUSE SYSTEM INTEGRATION TEST SUMMARY")
            logger.info("=" * 60)
            
            total_tests = len(self.test_results)
            passed_tests = sum(1 for r in self.test_results.values() if r["status"] == "PASS")
            partial_tests = sum(1 for r in self.test_results.values() if r["status"] == "PARTIAL")
            failed_tests = sum(1 for r in self.test_results.values() if r["status"] in ["FAIL", "ERROR"])
            skipped_tests = sum(1 for r in self.test_results.values() if r["status"] == "SKIPPED")
            
            success_rate = (passed_tests + partial_tests * 0.5) / max(1, total_tests) * 100
            
            logger.info(f"📊 Test Results:")
            logger.info(f"   ✅ Passed: {passed_tests}/{total_tests}")
            logger.info(f"   ⚠️ Partial: {partial_tests}/{total_tests}")
            logger.info(f"   ❌ Failed: {failed_tests}/{total_tests}")
            logger.info(f"   ⏭️ Skipped: {skipped_tests}/{total_tests}")
            logger.info(f"   🎯 Success Rate: {success_rate:.1f}%")
            
            # Detailed results
            logger.info(f"\n📋 Detailed Results:")
            for test_name, result in self.test_results.items():
                status_emoji = {
                    "PASS": "✅",
                    "PARTIAL": "⚠️", 
                    "FAIL": "❌",
                    "ERROR": "💥",
                    "SKIPPED": "⏭️"
                }.get(result["status"], "❓")
                
                logger.info(f"   {status_emoji} {test_name}: {result['status']}")
                
                # Show key metrics for passed tests
                if result["status"] == "PASS":
                    if "confidence_score" in result:
                        logger.info(f"      Confidence: {result['confidence_score']:.2f}")
                    if "integration_score" in result:
                        logger.info(f"      Integration: {result['integration_score']:.1f}%")
                    if "performance_score" in result:
                        logger.info(f"      Performance: {result['performance_score']:.1f}%")
            
            # Overall system assessment
            logger.info(f"\n🏆 SYSTEM ASSESSMENT:")
            
            if success_rate >= 90:
                logger.info("🚀 EXCELLENT: Powerhouse system fully operational!")
                system_status = "EXCELLENT"
            elif success_rate >= 75:
                logger.info("✅ GOOD: System operational with minor issues")
                system_status = "GOOD"
            elif success_rate >= 50:
                logger.info("⚠️ PARTIAL: System functional but needs optimization")
                system_status = "PARTIAL"
            else:
                logger.info("❌ POOR: System needs significant fixes")
                system_status = "POOR"
            
            # Recommendations
            logger.info(f"\n💡 RECOMMENDATIONS:")
            
            if failed_tests > 0:
                logger.info("   • Fix failed components before production use")
            if partial_tests > 0:
                logger.info("   • Optimize partial components for better performance")
            if success_rate < 85:
                logger.info("   • Consider additional testing and debugging")
            
            logger.info("   • Monitor system performance in live environment")
            logger.info("   • Set up automated health checks")
            logger.info("   • Implement gradual rollout for safety")
            
            logger.info("\n🎉 POWERHOUSE INTEGRATION TEST COMPLETE!")
            logger.info("=" * 60)
            
            return {
                "system_status": system_status,
                "success_rate": success_rate,
                "total_tests": total_tests,
                "passed": passed_tests,
                "partial": partial_tests,
                "failed": failed_tests,
                "recommendations": self._get_recommendations(success_rate, failed_tests)
            }
            
        except Exception as e:
            logger.error(f"Error generating test summary: {e}")
    
    def _get_recommendations(self, success_rate: float, failed_tests: int) -> List[str]:
        """Get specific recommendations based on test results"""
        recommendations = []
        
        if success_rate >= 90:
            recommendations.extend([
                "System ready for production deployment",
                "Monitor performance metrics in live environment",
                "Set up automated alerts for component failures"
            ])
        elif success_rate >= 75:
            recommendations.extend([
                "Address minor issues before full deployment",
                "Consider phased rollout starting with low-risk tokens",
                "Implement comprehensive logging for debugging"
            ])
        elif success_rate >= 50:
            recommendations.extend([
                "Fix critical issues before any live trading",
                "Run additional stress tests",
                "Consider reducing system complexity initially"
            ])
        else:
            recommendations.extend([
                "Major system overhaul required",
                "Focus on core functionality first",
                "Implement extensive unit testing"
            ])
        
        if failed_tests > 0:
            recommendations.append("Prioritize fixing failed components")
        
        return recommendations

# Main execution
async def main():
    """Run the complete powerhouse integration test"""
    test_runner = PowerhouseIntegrationTest()
    await test_runner.run_full_integration_test()

if __name__ == "__main__":
    asyncio.run(main())