"""
Test SOL Balance Detection for Your Wallet
Address: 6zpXi3eJSDVxBJUBaK9gs72hGZ8ViYjBFBrqT3Hpxk8x
"""

import asyncio
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.wallet.multichain_detector import MultiChainWalletDetector

async def test_sol_balance():
    """Test SOL balance detection for your specific wallet"""
    
    print("🔍 SOL BALANCE DETECTION TEST")
    print("=" * 50)
    
    # Your wallet addresses
    wallet_addresses = {
        'ethereum': "0xb4add0df12df32981773ca25ee88bdab750bfa20",
        'solana': "6zpXi3eJSDVxBJUBaK9gs72hGZ8ViYjBFBrqT3Hpxk8x"
    }
    
    print(f"🔧 Testing Multi-Chain Detection:")
    print(f"   Ethereum: {wallet_addresses['ethereum']}")
    print(f"   Solana:   {wallet_addresses['solana']}")
    
    try:
        # Initialize multi-chain detector
        detector = MultiChainWalletDetector(wallet_addresses)
        
        print("\n🔍 Checking Solana balance...")
        sol_balance = await detector.get_solana_balance(wallet_addresses['solana'])
        
        print(f"\n📊 SOLANA RESULTS:")
        print(f"   SOL Balance: {sol_balance.native_balance:.6f} SOL")
        print(f"   USD Value: ${sol_balance.usd_balance:,.2f}")
        print(f"   Last Updated: {sol_balance.last_updated}")
        
        print(f"\n🔍 Checking Ethereum balance...")
        eth_balance = await detector.get_ethereum_balance(wallet_addresses['ethereum'])
        
        print(f"\n📊 ETHEREUM RESULTS:")
        print(f"   ETH Balance: {eth_balance.native_balance:.6f} ETH")
        print(f"   USD Value: ${eth_balance.usd_balance:,.2f}")
        
        print(f"\n💰 Getting total portfolio value...")
        total_value = await detector.get_total_portfolio_value()
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Total Portfolio: ${total_value:,.2f}")
        
        if total_value > 0:
            print(f"\n✅ SUCCESS: Detected wallet balance!")
            print(f"   🚀 Ready for 12-hour microcap trading")
            print(f"   💰 Trading Capital Available: ${total_value:,.2f}")
            
            # Trading allocation suggestions
            print(f"\n📊 TRADING ALLOCATION SUGGESTIONS:")
            print(f"   Conservative (25%): ${total_value * 0.25:,.2f}")
            print(f"   Moderate (50%):     ${total_value * 0.50:,.2f}")
            print(f"   Aggressive (75%):   ${total_value * 0.75:,.2f}")
            
        else:
            print(f"\n⚠️ No balance detected on either network")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sol_balance())