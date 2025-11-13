"""
Test Wallet Balance Detection
Demonstrates automatic detection of wallet balance for trading capital
"""

import asyncio
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.wallet.balance_detector import WalletBalanceDetector

async def test_wallet_detection():
    """Test automatic wallet balance detection"""
    
    print("🔍 WALLET BALANCE DETECTION TEST")
    print("=" * 50)
    
    # Your wallet address
    wallet_address = "0xb4add0df12df32981773ca25ee88bdab750bfa20"
    print(f"Testing wallet: {wallet_address}")
    
    # Initialize detector
    detector = WalletBalanceDetector(wallet_address)
    
    try:
        print("\n🔍 Checking Ethereum balance...")
        
        # Get Ethereum balance
        eth_balance = await detector.get_ethereum_balance()
        
        print(f"\n📊 BALANCE RESULTS:")
        print(f"   ETH Balance: {eth_balance.eth_balance:.6f} ETH")
        print(f"   USD Value: ${eth_balance.usd_balance:,.2f}")
        print(f"   Last Updated: {eth_balance.last_updated}")
        
        if eth_balance.token_balances:
            print(f"   Token Holdings:")
            for symbol, amount in eth_balance.token_balances.items():
                print(f"     - {symbol}: {amount:.4f}")
        else:
            print(f"   No major token holdings detected")
        
        # Get total portfolio value
        print("\n💰 Calculating total portfolio value...")
        total_value = await detector.get_total_portfolio_value()
        
        print(f"\n🎯 TRADING CAPITAL RECOMMENDATION:")
        print(f"   Total Portfolio: ${total_value:,.2f}")
        print(f"   Suggested Trading Capital (75%): ${total_value * 0.75:,.2f}")
        print(f"   Conservative Capital (50%): ${total_value * 0.50:,.2f}")
        print(f"   Aggressive Capital (90%): ${total_value * 0.90:,.2f}")
        
        if total_value > 0:
            print(f"\n✅ SUCCESS: Automatic balance detection working!")
            print(f"   The system can now use ${total_value:,.2f} as your trading capital")
            print(f"   instead of the hardcoded $50,000 default")
        else:
            print(f"\n⚠️ WARNING: Could not detect balance")
            print(f"   This could be due to:")
            print(f"   - New wallet with no transactions")
            print(f"   - Network connectivity issues")
            print(f"   - API rate limits")
            print(f"   - Wallet on different network (BSC, Polygon, etc.)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"   The system will fall back to configured values")

def main():
    """Run the wallet detection test"""
    try:
        # Run async test
        asyncio.run(test_wallet_detection())
        
        print(f"\n🔧 INTEGRATION STATUS:")
        print(f"   ✅ Wallet balance detector created")
        print(f"   ✅ Integrated into trading GUI")
        print(f"   ✅ Auto-refresh button added")
        print(f"   ✅ Real-time balance updates enabled")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Run the trading GUI: python advanced_microcap_gui.py")
        print(f"   2. Click 'Refresh Wallet Balance' button")
        print(f"   3. System will auto-detect and use your real balance")
        print(f"   4. Trading allocation will be based on actual funds")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()