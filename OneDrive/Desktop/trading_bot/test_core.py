#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.trade_tracking import TradeTrackingSystem

def test_core_system():
    print("🧪 Testing Core Trade Tracking System")
    print("=" * 40)
    
    # Initialize system
    tracker = TradeTrackingSystem(db_path="demo_trades.db")
    print("✅ Trade Tracking System initialized")
    
    # Create candidate
    candidate = tracker.create_candidate(
        instrument="BTC/USDT",
        model_version="demo_v1.0", 
        confidence=0.75,
        params_ptr="demo_params",
        created_by="demo_strategy"
    )
    print(f"✅ Created candidate: {candidate.uuid[:8]}...")
    
    # Get metrics
    metrics = tracker.get_performance_metrics()
    print(f"✅ System metrics: {metrics['total_trades']} total trades")
    
    # Test complete record
    record = tracker.get_complete_trade_record(candidate.uuid)
    print(f"✅ Complete record retrieved with keys: {list(record.keys())}")
    
    print("\n🎉 Core system test successful!")
    return True

if __name__ == "__main__":
    test_core_system()