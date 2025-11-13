# 🎯 Essential Moralis Endpoints for Token Verification

## **The 4 Most Critical Endpoints You Need**

### **1. 📋 TOKEN METADATA - Basic Legitimacy Check**
```
GET /erc20/{address}/metadata?chain=0x1
```

**Why Critical:** 
- Verifies token exists and has proper structure
- Reveals suspicious naming patterns
- Checks for reasonable decimal places

**Red Flags to Watch:**
- Missing name/symbol
- Unusual decimals (not 6-18)
- Names containing "test", "fake", "scam"

**Example Response:**
```json
{
  "address": "0x...",
  "name": "Uniswap",
  "symbol": "UNI", 
  "decimals": 18,
  "logo": "...",
  "thumbnail": "..."
}
```

---

### **2. 👥 TOP HOLDERS - Concentration Risk Analysis**
```
GET /erc20/{address}/owners?chain=0x1&limit=20
```

**Why Critical:**
- Detects if token is controlled by few wallets (rugpull risk)
- Reveals fake holder distribution
- Identifies potential pump & dump schemes

**Red Flags to Watch:**
- Top 5 holders own >60% of supply
- Very few total holders (<100)
- New wallets holding large amounts

**Key Metrics:**
```json
{
  "result": [
    {
      "owner_address": "0x...",
      "balance": "1000000000000000000000",
      "balance_formatted": "1000",
      "percentage_relative_to_total_supply": 5.5
    }
  ]
}
```

---

### **3. 💹 TOKEN TRANSFERS - Real Activity Verification**
```
GET /erc20/{address}/transfers?chain=0x1&limit=100
```

**Why Critical:**
- Distinguishes real trading from fake volume
- Shows if token has genuine user adoption
- Reveals wash trading patterns

**Red Flags to Watch:**
- Very few unique addresses trading
- No recent activity (last 24h)
- Repetitive transfer patterns between same wallets

**Analysis Points:**
- Unique addresses in transfers
- Recent activity frequency
- Transaction value distribution

---

### **4. 🐋 WALLET TOKEN BALANCES - Whale Legitimacy**
```
GET /{wallet_address}/erc20?chain=0x1&limit=50
```

**Why Critical:**
- Verifies if large holders are legitimate whales
- Detects wallets created just for this token
- Reveals diversification patterns

**Red Flags to Watch:**
- Wallets only holding this one token
- No token diversity (<3 different tokens)
- Wallets with identical holding patterns

---

## **🚨 Critical Red Flag Combinations**

### **IMMEDIATE AVOID Signals:**
1. **Top 5 holders own >80%** + **<50 total holders**
2. **Only 1-2 tokens per whale wallet** + **No recent transfers**
3. **Token name contains "test/fake"** + **High concentration**
4. **<20 unique trading addresses** + **New token**

### **⚠️ High Caution Signals:**
1. **Top 5 holders own 60-80%** + **Low trading volume**
2. **<100 total holders** + **Limited recent activity**
3. **Whales with poor diversification** + **Suspicious transfer patterns**

---

## **📊 Scoring System**

### **Token Safety Score (0-100):**
- **Basic Info Valid:** +30 points
- **Holder Distribution:** 
  - Low concentration (top 5 <40%): +30 points
  - Medium concentration (40-60%): +15 points
  - High concentration (>60%): +0 points
- **Real Trading Activity:** +25 points (based on unique addresses & recent activity)
- **Legitimate Whales:** +15 points (diversified portfolios)

### **Final Verdicts:**
- **80-100:** ✅ SAFE - High confidence
- **60-79:** ⚠️ CAUTION - Moderate risk
- **40-59:** 🚨 HIGH RISK - Trade carefully  
- **0-39:** ❌ AVOID - Too risky

---

## **🎯 Focused Implementation Strategy**

### **Phase 1: Core Verification (Week 1)**
- Implement the 4 essential endpoints above
- Create red flag detection system
- Build scoring algorithm

### **Phase 2: Advanced Analysis (Week 2)**
- Add whale wallet deep-dive analysis
- Implement transfer pattern recognition
- Create automated alerting system

### **Phase 3: Integration (Week 3)**
- Integrate with your trading bot
- Add real-time monitoring
- Create dashboard for quick decisions

---

## **💡 Pro Tips for Your Trading Bot**

1. **Always check ALL 4 endpoints** before any trade
2. **Set hard limits:** Never trade if concentration >80%
3. **Require minimum holders:** At least 100 unique holders
4. **Verify recent activity:** Must have transfers in last 24h
5. **Double-check whales:** Ensure they hold other legitimate tokens

This focused approach will give you **90% of the safety benefits** with just **4 API calls** per token analysis!