"""
Test Script: Verify 12-Hour Hold Time Updates
Checks all trading files to confirm hold times are set to 12 hours
"""

import re
import os

def check_hold_times():
    """Check all trading files for hold time configurations"""
    
    files_to_check = [
        'advanced_microcap_gui.py',
        'integrated_trading_launcher.py', 
        'automated_microcap_trader.py'
    ]
    
    print("🔍 CHECKING 12-HOUR HOLD TIME UPDATES")
    print("=" * 50)
    
    all_updated = True
    
    for filename in files_to_check:
        if os.path.exists(filename):
            print(f"\n📋 Checking {filename}...")
            
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Look for time limit patterns
            twelve_hour_patterns = [
                r'12\s*\*\s*3600',  # 12 * 3600
                r'12\s*hours?',      # 12 hour or 12 hours
            ]
            
            old_patterns = [
                r'48\s*\*\s*3600',  # 48 * 3600
                r'24\s*\*\s*3600',  # 24 * 3600
                r'48\s*hours?',      # 48 hour or 48 hours
                r'24\s*hours?',      # 24 hour or 24 hours (except in comments)
            ]
            
            # Check for 12-hour patterns
            found_12_hour = False
            for pattern in twelve_hour_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    found_12_hour = True
                    print(f"   ✅ Found 12-hour pattern: {matches}")
            
            # Check for old patterns
            found_old = False
            for pattern in old_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    found_old = True
                    print(f"   ⚠️ Found old pattern: {matches}")
            
            if found_12_hour and not found_old:
                print(f"   ✅ {filename} - UPDATED TO 12 HOURS")
            elif found_12_hour and found_old:
                print(f"   ⚠️ {filename} - PARTIALLY UPDATED (mixed patterns)")
                all_updated = False
            elif not found_12_hour and found_old:
                print(f"   ❌ {filename} - STILL USING OLD TIMES")
                all_updated = False
            else:
                print(f"   ❓ {filename} - NO TIME PATTERNS FOUND")
                all_updated = False
                
        else:
            print(f"   ❌ {filename} - FILE NOT FOUND")
            all_updated = False
    
    print("\n" + "=" * 50)
    if all_updated:
        print("🎯 SUCCESS: All files updated to 12-hour hold times!")
        print("\n🚀 READY TO TEST:")
        print("   - Start with small positions ($50-100)")
        print("   - Monitor 12-hour exit performance")
        print("   - Can always buy back in if needed")
    else:
        print("⚠️ WARNING: Some files may still have old hold times")
        print("   Please review the patterns above")
    
    return all_updated

if __name__ == "__main__":
    check_hold_times()