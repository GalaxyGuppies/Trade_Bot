"""
Test Jupiter API connectivity with various methods
"""
import httpx
import json

print("🧪 Testing Jupiter API Connectivity\n" + "="*60)

# Method 1: Try with httpx directly
print("\n1️⃣ Testing with httpx (direct)...")
try:
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        response = client.get(
            'https://quote-api.jup.ag/v6/quote',
            params={
                'inputMint': 'So11111111111111111111111111111111111111112',  # SOL
                'outputMint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
                'amount': '1000000',  # 0.001 SOL
                'slippageBps': '50'
            }
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS! Got quote: {data.get('outAmount', 'N/A')} USDC")
            print(f"   Routes found: {len(data.get('routePlan', []))}")
        else:
            print(f"❌ Failed: Status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

# Method 2: Try with Cloudflare DNS-over-HTTPS
print("\n2️⃣ Testing with Cloudflare DoH...")
try:
    doh_url = "https://cloudflare-dns.com/dns-query"
    with httpx.Client() as client:
        # Query DNS over HTTPS
        response = client.get(
            doh_url,
            params={
                'name': 'quote-api.jup.ag',
                'type': 'A'
            },
            headers={'accept': 'application/dns-json'}
        )
        if response.status_code == 200:
            data = response.json()
            if 'Answer' in data and len(data['Answer']) > 0:
                ip = data['Answer'][0]['data']
                print(f"✅ Resolved quote-api.jup.ag to: {ip}")
                
                # Now try to connect using the IP
                print(f"   Trying to connect to {ip}...")
                response2 = client.get(
                    f'https://{ip}/v6/quote',
                    params={
                        'inputMint': 'So11111111111111111111111111111111111111112',
                        'outputMint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
                        'amount': '1000000',
                        'slippageBps': '50'
                    },
                    headers={'Host': 'quote-api.jup.ag'},
                    verify=False  # Skip SSL verification for IP-based request
                )
                print(f"   Response: {response2.status_code}")
            else:
                print("❌ No DNS answer received")
        else:
            print(f"❌ DoH failed: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("📝 If Method 1 works, real trading will work!")
print("   If both fail, the network is blocking crypto APIs completely.")
