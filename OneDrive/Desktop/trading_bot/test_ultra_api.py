"""
Test Jupiter Ultra API with the updated real_trader.py
"""
import sys
import os

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

from real_trader import RealTrader

print("🧪 Testing Jupiter Ultra API")
print("="*60)

# Your private key (will be used for address only, not signing in this test)
private_key = "2qNP2hU1b8KsjcY2w1qxH6b2Y8DHGhrNE1tUmfKK5bUvBSnrBGqBPNraFFRoBE9vV7xT2ExGjwQXaLngs8rSJAC7"

try:
    # Initialize trader
    print("\n1️⃣ Initializing RealTrader...")
    trader = RealTrader(private_key)
    
    # Test get_jupiter_quote (now using Ultra API)
    print("\n2️⃣ Testing get_jupiter_quote with Ultra API...")
    print("   Requesting SOL → USDC swap quote...")
    
    order = trader.get_jupiter_quote(
        input_mint='So11111111111111111111111111111111111111112',  # SOL
        output_mint='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
        amount=1000000  # 0.001 SOL
    )
    
    if order:
        print("\n✅ SUCCESS! Jupiter Ultra API is working!")
        print(f"   In Amount: {order.get('inAmount')}")
        print(f"   Out Amount: {order.get('outAmount')}")
        print(f"   Slippage: {order.get('slippageBps')} bps")
        print(f"   Price Impact: {order.get('priceImpactPct')}%")
        print(f"   Transaction ready: {'transaction' in order}")
        
        print("\n🎉 Real trading is ready to use!")
        print("   Start the GUI and enable real trading!")
    else:
        print("\n❌ Failed to get order from Jupiter Ultra API")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
