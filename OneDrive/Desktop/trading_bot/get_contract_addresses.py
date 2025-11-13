#!/usr/bin/env python3
"""
Script to retrieve contract addresses for active trades
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

def get_active_trade_contracts(db_path: str = "trading_bot.db") -> List[Dict[str, Any]]:
    """
    Retrieve contract addresses for all active trades
    
    Returns:
        List of active trades with their contract addresses
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get active trades with contract addresses
            cursor.execute("""
                SELECT t.symbol, t.entry_price, t.quantity, t.timestamp, 
                       t.pnl, t.confidence, t.market_cap, t.rugpull_risk,
                       mc.contract_address, mc.daily_volume, mc.volatility_score
                FROM trades t
                LEFT JOIN microcap_candidates mc ON t.symbol = mc.symbol
                WHERE t.status = 'active'
                ORDER BY t.timestamp DESC
            """)
            
            results = cursor.fetchall()
            
            if not results:
                print("ℹ️  No active trades found in database")
                return []
            
            active_trades = []
            for row in results:
                trade = {
                    'symbol': row[0],
                    'entry_price': row[1],
                    'quantity': row[2], 
                    'timestamp': row[3],
                    'pnl': row[4],
                    'confidence': row[5],
                    'market_cap': row[6],
                    'rugpull_risk': row[7],
                    'contract_address': row[8] or 'Unknown',
                    'daily_volume': row[9],
                    'volatility_score': row[10]
                }
                active_trades.append(trade)
            
            return active_trades
            
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return []

def get_all_candidate_contracts(db_path: str = "trading_bot.db") -> List[Dict[str, Any]]:
    """
    Get all token candidates with their contract addresses
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, contract_address, market_cap, daily_volume, 
                       volatility_score, rugpull_risk, confidence, timestamp
                FROM microcap_candidates
                ORDER BY timestamp DESC
                LIMIT 20
            """)
            
            results = cursor.fetchall()
            
            candidates = []
            for row in results:
                candidate = {
                    'symbol': row[0],
                    'contract_address': row[1],
                    'market_cap': row[2],
                    'daily_volume': row[3],
                    'volatility_score': row[4],
                    'rugpull_risk': row[5],
                    'confidence': row[6],
                    'timestamp': row[7]
                }
                candidates.append(candidate)
            
            return candidates
            
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return []

def print_active_trades():
    """Display active trades with contract addresses"""
    print("\n🔍 ACTIVE TRADES - CONTRACT ADDRESSES")
    print("=" * 80)
    
    active_trades = get_active_trade_contracts()
    
    if not active_trades:
        print("📝 No active trades found")
        return
    
    for i, trade in enumerate(active_trades, 1):
        print(f"\n🎯 Trade #{i}: {trade['symbol']}")
        print(f"   📧 Contract Address: {trade['contract_address']}")
        print(f"   💰 Entry Price: ${trade['entry_price']:.4f}")
        print(f"   📊 Quantity: {trade['quantity']:.4f}")
        print(f"   📈 Market Cap: ${trade['market_cap']:,.0f}")
        print(f"   🎯 Confidence: {trade['confidence']:.1%}")
        print(f"   ⚠️  Rugpull Risk: {trade['rugpull_risk']:.1%}")
        print(f"   🕒 Entry Time: {trade['timestamp']}")
        
        # Add blockchain explorer links
        if trade['contract_address'] and trade['contract_address'] != 'Unknown':
            if trade['contract_address'].startswith('0x'):
                # Ethereum-based
                print(f"   🔗 Etherscan: https://etherscan.io/token/{trade['contract_address']}")
                print(f"   🔗 Uniswap: https://app.uniswap.org/#/tokens/ethereum/{trade['contract_address']}")
            else:
                # Assume Solana
                print(f"   🔗 Solscan: https://solscan.io/token/{trade['contract_address']}")
                print(f"   🔗 Jupiter: https://jup.ag/swap/SOL-{trade['contract_address']}")

def print_recent_candidates():
    """Display recent candidates with contract addresses"""
    print("\n\n🔍 RECENT TOKEN CANDIDATES - CONTRACT ADDRESSES")
    print("=" * 80)
    
    candidates = get_all_candidate_contracts()
    
    if not candidates:
        print("📝 No candidates found")
        return
    
    for i, candidate in enumerate(candidates, 1):
        print(f"\n📊 Candidate #{i}: {candidate['symbol']}")
        print(f"   📧 Contract Address: {candidate['contract_address']}")
        print(f"   💰 Market Cap: ${candidate['market_cap']:,.0f}")
        print(f"   📊 Daily Volume: ${candidate['daily_volume']:,.0f}")
        print(f"   📈 Volatility: {candidate['volatility_score']:.2f}")
        print(f"   🎯 Confidence: {candidate['confidence']:.1%}")
        print(f"   ⚠️  Rugpull Risk: {candidate['rugpull_risk']:.1%}")
        print(f"   🕒 Discovered: {candidate['timestamp']}")

def export_to_json():
    """Export active trades and candidates to JSON file"""
    try:
        data = {
            'active_trades': get_active_trade_contracts(),
            'recent_candidates': get_all_candidate_contracts(),
            'export_timestamp': datetime.now().isoformat(),
            'total_active_trades': len(get_active_trade_contracts())
        }
        
        filename = f"contract_addresses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n💾 Data exported to: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None

if __name__ == "__main__":
    print("🤖 Trading Bot - Contract Address Lookup Tool")
    print("=" * 50)
    
    # Show active trades
    print_active_trades()
    
    # Show recent candidates
    print_recent_candidates()
    
    # Export to JSON
    export_to_json()
    
    print("\n✅ Contract address lookup complete!")