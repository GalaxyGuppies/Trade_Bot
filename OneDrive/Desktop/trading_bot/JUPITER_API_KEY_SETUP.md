# Jupiter API Key Setup Guide

## Why You Need a Jupiter API Key

As of November 2025, the Jupiter Ultra Swap API requires authentication via an API key. Without it, you'll get **401 Unauthorized** errors when trying to execute trades.

## Error You're Seeing

```
❌ Order failed: 401
   Response: {"code":401,"message":"Unauthorized"}
❌ Failed to get quote for BANGERS
```

## How to Get Your Free Jupiter API Key

### Step 1: Visit Jupiter Portal
Go to: **https://portal.jup.ag**

### Step 2: Generate API Key
1. Sign in with your Solana wallet
2. Navigate to the API Keys section
3. Click "Generate New API Key"
4. Copy your API key

### Step 3: Add API Key to Config

1. Open `config.json` in your trading_bot folder
2. Find the `"solana"` section
3. Update the `"jupiter_api_key"` field:

```json
"solana": {
  "rpc_url": "https://api.mainnet-beta.solana.com",
  "websocket_url": "wss://solana-mainnet.g.alchemy.com/v2/0t0vMPzLZHMfvrbbJsdC3Hy5oqPnEqav",
  "wallet_private_key": "YOUR_PRIVATE_KEY_HERE",
  "wallet_address": "YOUR_WALLET_ADDRESS_HERE",
  "jupiter_api_key": "YOUR_JUPITER_API_KEY_HERE"
}
```

4. Save the file

### Step 4: Restart Your Bot

```powershell
python scripts\trading_bot_gui.py
```

## Important Notes

✅ **Free API Key**: No payment required - completely free!

✅ **Dynamic Rate Limits**: Rate limits scale with your swap volume automatically

✅ **Required for Trading**: Without this key, the bot cannot execute any trades

⚠️ **Keep It Secret**: Don't share your API key or commit it to GitHub

## What the Bot Does Now

The bot has been updated to:
- ✅ Load the Jupiter API key from `config.json`
- ✅ Include it in all Jupiter Ultra API requests
- ✅ Display confirmation when API key is loaded

## Testing Your Setup

Once you've added your API key, you should see:
```
✅ Jupiter API key configured
```

And trades should execute successfully instead of getting 401 errors.

## Troubleshooting

### Still getting 401 errors?
- Double-check you copied the entire API key
- Make sure there are no extra spaces or quotes
- Verify the key is inside the quotes in the config.json file

### Can't access portal.jup.ag?
- Check your internet connection
- Try using a different browser
- Make sure you have a Solana wallet installed (Phantom, Solflare, etc.)

## More Information

For detailed documentation on Jupiter Ultra API:
- **Portal**: https://portal.jup.ag
- **Docs**: https://station.jup.ag/docs/ultra
- **Rate Limits**: Dynamic scaling with your volume - no fixed limits!

---

**Status**: Your bot is ready to trade once you add the API key! 🚀
