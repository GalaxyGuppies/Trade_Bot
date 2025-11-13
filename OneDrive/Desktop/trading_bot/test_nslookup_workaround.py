"""
Workaround: Resolve DNS using Windows nslookup command
"""
import subprocess
import re
import httpx

def resolve_using_nslookup(hostname, dns_server='1.1.1.1'):
    """Use Windows nslookup command to resolve hostname"""
    try:
        result = subprocess.run(
            ['nslookup', hostname, dns_server],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout
        print(f"📋 nslookup output:\n{output}")
        
        # Parse the output - look for "Name:" followed by "Address:"
        lines = output.split('\n')
        found_name = False
        for line in lines:
            if 'Name:' in line and hostname in line:
                found_name = True
            elif found_name and 'Address:' in line:
                # Extract IP from this line
                match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
                if match:
                    ip = match.group(1)
                    # Make sure it's not the DNS server IP
                    if ip != dns_server:
                        return ip
        
        return None
    except Exception as e:
        print(f"❌ nslookup failed: {e}")
        return None

print("🧪 Testing nslookup-based DNS resolution")
print("=" * 60)

hostname = 'quote-api.jup.ag'
print(f"\n🔍 Resolving {hostname} using nslookup...")

ip = resolve_using_nslookup(hostname)

if ip:
    print(f"✅ Resolved to: {ip}")
    print(f"\n📡 Testing connection to {ip}...")
    
    try:
        # Try to connect using the IP
        with httpx.Client(timeout=15.0, verify=False) as client:
            response = client.get(
                f'https://{ip}/v6/quote',
                params={
                    'inputMint': 'So11111111111111111111111111111111111111112',
                    'outputMint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
                    'amount': '1000000',
                    'slippageBps': '50'
                },
                headers={'Host': hostname}
            )
            
            if response.status_code == 200:
                print(f"✅ SUCCESS! Jupiter API is reachable via IP!")
                data = response.json()
                print(f"   Got quote: {data.get('outAmount', 'N/A')} output tokens")
                print(f"\n🎉 Real trading WILL WORK with this workaround!")
            else:
                print(f"❌ Connection failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
else:
    print("❌ Could not resolve hostname")

print("\n" + "=" * 60)
