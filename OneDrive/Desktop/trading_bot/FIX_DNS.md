# DNS Fix Guide - No VPN Needed

## Quick Fix: Change DNS Servers (Free)

### Method 1: PowerShell (Fastest)
Run PowerShell **as Administrator** and paste:

```powershell
# Change to Google DNS
netsh interface ip set dns "Wi-Fi" static 8.8.8.8
netsh interface ip add dns "Wi-Fi" 8.8.4.4 index=2

# Flush DNS cache
ipconfig /flushdns

# Test
nslookup quote-api.jup.ag 8.8.8.8
```

### Method 2: Windows Settings GUI
1. Open **Settings** → **Network & Internet**
2. Click your connection (Wi-Fi or Ethernet)
3. Click **Edit** next to "DNS server assignment"
4. Select **Manual**
5. Turn on **IPv4**
6. Set:
   - **Preferred DNS**: `8.8.8.8` (Google)
   - **Alternate DNS**: `1.1.1.1` (Cloudflare)
7. Click **Save**
8. Restart computer

---

## Alternative DNS Providers

**Google DNS** (Best for general use)
- Primary: `8.8.8.8`
- Secondary: `8.8.4.4`

**Cloudflare DNS** (Fastest)
- Primary: `1.1.1.1`
- Secondary: `1.0.0.1`

**Quad9 DNS** (Security-focused)
- Primary: `9.9.9.9`
- Secondary: `149.112.112.112`

---

## If DNS Change Doesn't Work

Then yes, use a VPN:

### Free VPN Options:
1. **Proton VPN** - Unlimited free, no logs
2. **Windscribe** - 10GB/month free

### Paid VPN (Better for trading):
1. **NordVPN** - Fast, crypto-friendly
2. **ExpressVPN** - Premium reliability
3. **Mullvad** - Anonymous, $5/month

---

## Test After Fix

Run this in PowerShell:
```powershell
cd c:\Users\tfair\OneDrive\Desktop\trading_bot
.\scripts\Activate.ps1
python test_dns.py
```

If it shows ✅, restart the GUI and enable real trading!
