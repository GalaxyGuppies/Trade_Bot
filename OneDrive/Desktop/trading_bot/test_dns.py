"""
Quick DNS test for Jupiter API
"""
import socket
import urllib.request

print("🔍 Testing DNS resolution for Jupiter API...")
print("=" * 60)

# Test 1: Basic DNS lookup
print("\n1️⃣ DNS Lookup Test:")
try:
    ip = socket.gethostbyname('quote-api.jup.ag')
    print(f"✅ quote-api.jup.ag resolves to: {ip}")
except socket.gaierror as e:
    print(f"❌ DNS lookup failed: {e}")
    print("   This is why real trading fails!")

# Test 2: Try with Google DNS
print("\n2️⃣ Testing with alternative DNS (8.8.8.8):")
print("   To fix, run PowerShell as Administrator:")
print("   netsh interface ip set dns 'Wi-Fi' static 8.8.8.8")

# Test 3: Basic connectivity
print("\n3️⃣ Testing general internet connectivity:")
try:
    response = urllib.request.urlopen('https://www.google.com', timeout=5)
    print(f"✅ Internet works: Google returned {response.status}")
except Exception as e:
    print(f"❌ Internet issue: {e}")

print("\n" + "=" * 60)
print("📝 Summary:")
print("   Paper trading = Works offline (no network needed)")
print("   Real trading = Requires Jupiter API (network required)")
print("\n   If DNS fails above, real trading cannot work!")
