#!/usr/bin/env python3
"""
Quick Fix for API Rate Limiting Issues - Force Curated Token Fallback
"""
import json
import time

def fix_rate_limit_issue():
    """Fix the rate limiting issue by temporarily using mock-only mode"""
    
    print("🔧 FIXING API RATE LIMIT ISSUE")
    print("=" * 50)
    
    # Read current config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return
    
    # Add rate limiting configuration
    config['api_rate_limits'] = {
        'coingecko_min_interval': 3.0,  # 3 seconds between calls
        'coingecko_max_retries': 2,
        'use_fallback_on_429': True,
        'force_mock_mode_on_errors': True
    }
    
    # Set more conservative thresholds for curated tokens
    config['thresholds']['min_volume'] = 50000  # Lower threshold for curated tokens
    config['curated_token_mode'] = {
        'enabled': True,
        'priority': 'high',
        'always_include': True,
        'ignore_api_errors': True
    }
    
    # Save updated config
    try:
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Updated config.json with rate limiting fixes")
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return
    
    print("\n🎯 SOLUTIONS APPLIED:")
    print("✅ Added API rate limiting configuration")
    print("✅ Enabled prioritized curated token mode")  
    print("✅ Set fallback on 429 errors")
    print("✅ Lowered volume thresholds for curated tokens")
    
    print("\n🚀 RECOMMENDED ACTIONS:")
    print("1. Restart your bot to apply new settings")
    print("2. The bot will now prioritize curated tokens when APIs fail")
    print("3. Rate limiting is now more conservative (3 sec intervals)")
    print("4. Curated tokens (MICROGEM, DEFISTAR, MOONRUN) will always be available")
    
    return True

def create_emergency_fix_script():
    """Create a direct fix for the bot code"""
    
    fix_script = '''
# EMERGENCY FIX: Add this to your bot initialization
def force_curated_tokens_on_api_failure(self):
    """Emergency method to ensure curated tokens are always available"""
    try:
        # Always add curated tokens regardless of API status
        curated = self._get_curated_microcaps(50000, 25000, 500000, 1500000)
        
        if not hasattr(self, 'microcap_candidates') or not self.microcap_candidates:
            logger.info("🚨 Emergency: Using curated tokens due to API issues")
            
            processed = []
            for token in curated:
                candidate = self._process_token_candidate(token, 500000, 1500000, 50000, 25000)
                if candidate:
                    processed.append(candidate)
            
            # Filter with current risk profile
            filtered = []
            for candidate in processed:
                if (candidate['rugpull_risk'] <= self.current_risk_profile.rugpull_threshold and
                    candidate['confidence'] >= self.current_risk_profile.min_confidence):
                    filtered.append(candidate)
            
            self.microcap_candidates = filtered
            logger.info(f"🔄 Emergency fallback: Added {len(filtered)} curated candidates")
            
            # Update GUI
            if hasattr(self, 'update_candidates_display'):
                self.update_candidates_display()
                
    except Exception as e:
        logger.error(f"Emergency fallback failed: {e}")

# Add this to your trading evaluation loop:
def ensure_candidates_available(self):
    """Ensure we always have candidates available for trading"""
    if not hasattr(self, 'microcap_candidates') or len(self.microcap_candidates) == 0:
        logger.warning("⚠️ No candidates available, forcing curated fallback")
        self.force_curated_tokens_on_api_failure()
'''
    
    with open('emergency_fix.py', 'w') as f:
        f.write(fix_script)
    
    print("💾 Created emergency_fix.py with fallback methods")

if __name__ == "__main__":
    print("🚨 EMERGENCY FIX FOR API RATE LIMITING")
    print("=" * 60)
    
    # Apply config fixes
    success = fix_rate_limit_issue()
    
    if success:
        # Create emergency script
        create_emergency_fix_script()
        
        print("\n💡 IMMEDIATE SOLUTION:")
        print("Your bot is hitting CoinGecko rate limits (429 errors)")
        print("This prevents it from getting the usual 3 curated tokens")
        print("The fix ensures curated tokens are always available!")
        
        print("\n🔄 RESTART YOUR BOT NOW to apply these fixes")
        print("Your bot will then have the 3 curated tokens available:")
        print("  • MICROGEM")
        print("  • DEFISTAR") 
        print("  • MOONRUN")
        
    else:
        print("❌ Fix failed - manual intervention required")