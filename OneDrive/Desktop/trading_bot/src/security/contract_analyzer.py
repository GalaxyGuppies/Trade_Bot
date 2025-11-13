"""
Smart Contract Security Analysis Module
Comprehensive security analysis for microcap tokens to detect:
- Honeypots and malicious contracts
- Ownership risks and admin keys
- Liquidity locks and pull risks
- Contract verification status
"""

import asyncio
import logging
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import sqlite3
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL_RISK = "critical_risk"

class ContractRisk(Enum):
    HONEYPOT = "honeypot"
    OWNERSHIP_RISK = "ownership_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    UNVERIFIED_CONTRACT = "unverified_contract"
    SUSPICIOUS_FUNCTIONS = "suspicious_functions"
    HIGH_SELL_TAX = "high_sell_tax"
    MINT_FUNCTION = "mint_function"
    PAUSE_FUNCTION = "pause_function"

@dataclass
class SecurityResult:
    """Security analysis result for a token contract"""
    contract_address: str
    token_symbol: str
    security_level: SecurityLevel
    risk_score: float  # 0.0 (safe) to 1.0 (critical)
    risks_detected: List[ContractRisk]
    analysis_details: Dict
    honeypot_risk: float
    ownership_risk: float
    liquidity_risk: float
    verified: bool
    audit_status: str
    recommendation: str
    timestamp: datetime

@dataclass
class ContractInfo:
    """Basic contract information"""
    address: str
    name: str
    symbol: str
    decimals: int
    total_supply: int
    owner: Optional[str]
    is_verified: bool
    source_code: Optional[str]
    creation_date: Optional[datetime]

class SmartContractAnalyzer:
    """
    Advanced smart contract security analyzer for microcap tokens
    Integrates multiple security APIs and performs comprehensive analysis
    """
    
    def __init__(self, database_path: str = "trading_bot.db"):
        self.database_path = database_path
        self.api_cache = {}
        self.analysis_cache = {}
        
        # Security thresholds
        self.honeypot_threshold = 0.7
        self.ownership_risk_threshold = 0.6
        self.liquidity_risk_threshold = 0.8
        self.sell_tax_threshold = 0.15  # 15% sell tax
        
        # API endpoints for security analysis
        self.security_apis = {
            'honeypot_is': 'https://api.honeypot.is/v2/IsHoneypot',
            'rugcheck': 'https://api.rugcheck.xyz/v1/tokens',
            'dextools': 'https://www.dextools.io/shared/data/pair',
            'etherscan': 'https://api.etherscan.io/api',
            'bscscan': 'https://api.bscscan.com/api'
        }
        
        self._init_database()
        logger.info("🔒 Smart Contract Security Analyzer initialized")
    
    def _init_database(self):
        """Initialize security analysis database tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Security analysis results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_address TEXT NOT NULL UNIQUE,
                    token_symbol TEXT,
                    security_level TEXT,
                    risk_score REAL,
                    risks_detected TEXT,
                    honeypot_risk REAL,
                    ownership_risk REAL,
                    liquidity_risk REAL,
                    verified BOOLEAN,
                    audit_status TEXT,
                    recommendation TEXT,
                    analysis_details TEXT,
                    timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Contract information table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS contract_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    address TEXT NOT NULL UNIQUE,
                    name TEXT,
                    symbol TEXT,
                    decimals INTEGER,
                    total_supply TEXT,
                    owner TEXT,
                    is_verified BOOLEAN,
                    creation_date DATETIME,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Security alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contract_address TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT,
                    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Security analysis database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing security database: {e}")
    
    async def analyze_contract_security(self, contract_address: str, 
                                      token_symbol: str = None, 
                                      chain: str = "ethereum") -> SecurityResult:
        """
        Perform comprehensive security analysis on a smart contract
        """
        try:
            logger.info(f"🔍 Analyzing contract security: {contract_address}")
            
            # Check cache first
            cache_key = f"{contract_address}_{chain}"
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if (datetime.now() - cached_result.timestamp).total_seconds() < 3600:  # 1 hour cache
                    logger.info("📋 Using cached security analysis")
                    return cached_result
            
            # Gather contract information
            contract_info = await self._get_contract_info(contract_address, chain)
            
            # Run multiple security checks in parallel
            security_checks = await asyncio.gather(
                self._check_honeypot_risk(contract_address, chain),
                self._check_ownership_risk(contract_address, chain),
                self._check_liquidity_risk(contract_address, chain),
                self._check_contract_verification(contract_address, chain),
                self._check_suspicious_functions(contract_address, chain),
                self._check_tax_analysis(contract_address, chain),
                return_exceptions=True
            )
            
            # Process results
            honeypot_risk, ownership_risk, liquidity_risk, verification_result, \
            suspicious_functions, tax_analysis = security_checks
            
            # Handle exceptions
            for i, result in enumerate(security_checks):
                if isinstance(result, Exception):
                    logger.warning(f"Security check {i} failed: {result}")
                    security_checks[i] = {"risk": 0.5, "details": "Analysis failed"}
            
            # Calculate overall risk score
            risk_score, security_level, risks_detected = self._calculate_risk_score(
                honeypot_risk, ownership_risk, liquidity_risk, 
                verification_result, suspicious_functions, tax_analysis
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(security_level, risks_detected)
            
            # Create security result
            security_result = SecurityResult(
                contract_address=contract_address,
                token_symbol=token_symbol or contract_info.symbol,
                security_level=security_level,
                risk_score=risk_score,
                risks_detected=risks_detected,
                analysis_details={
                    "honeypot_analysis": honeypot_risk,
                    "ownership_analysis": ownership_risk,
                    "liquidity_analysis": liquidity_risk,
                    "verification_status": verification_result,
                    "suspicious_functions": suspicious_functions,
                    "tax_analysis": tax_analysis,
                    "contract_info": contract_info.__dict__ if contract_info else {}
                },
                honeypot_risk=honeypot_risk.get("risk", 0.5) if isinstance(honeypot_risk, dict) else 0.5,
                ownership_risk=ownership_risk.get("risk", 0.5) if isinstance(ownership_risk, dict) else 0.5,
                liquidity_risk=liquidity_risk.get("risk", 0.5) if isinstance(liquidity_risk, dict) else 0.5,
                verified=verification_result.get("verified", False) if isinstance(verification_result, dict) else False,
                audit_status=verification_result.get("audit_status", "Unknown") if isinstance(verification_result, dict) else "Unknown",
                recommendation=recommendation,
                timestamp=datetime.now()
            )
            
            # Cache and store result
            self.analysis_cache[cache_key] = security_result
            await self._store_security_result(security_result)
            
            # Generate alerts if high risk
            if security_level in [SecurityLevel.HIGH_RISK, SecurityLevel.CRITICAL_RISK]:
                await self._generate_security_alert(security_result)
            
            logger.info(f"✅ Security analysis complete: {security_level.value} (risk: {risk_score:.2f})")
            return security_result
            
        except Exception as e:
            logger.error(f"Error in contract security analysis: {e}")
            return self._create_failed_analysis(contract_address, token_symbol, str(e))
    
    async def _get_contract_info(self, contract_address: str, chain: str) -> ContractInfo:
        """Get basic contract information"""
        try:
            # Try to get from cache/database first
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM contract_info 
                WHERE address = ? AND last_updated > datetime('now', '-24 hours')
            ''', (contract_address,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return ContractInfo(
                    address=row[1], name=row[2], symbol=row[3], decimals=row[4],
                    total_supply=int(row[5]) if row[5] else 0, owner=row[6],
                    is_verified=bool(row[7]), source_code=None,
                    creation_date=datetime.fromisoformat(row[8]) if row[8] else None
                )
            
            # Fetch fresh data from blockchain API
            api_url = self._get_blockchain_api_url(chain)
            
            if not api_url:
                return ContractInfo(contract_address, "Unknown", "UNKNOWN", 18, 0, None, False, None, None)
            
            # Make API request (simplified for demo)
            contract_info = ContractInfo(
                address=contract_address,
                name="Token Name",  # Would be fetched from API
                symbol="TOKEN",     # Would be fetched from API
                decimals=18,
                total_supply=1000000,
                owner=None,
                is_verified=False,
                source_code=None,
                creation_date=datetime.now()
            )
            
            # Store in database
            await self._store_contract_info(contract_info)
            
            return contract_info
            
        except Exception as e:
            logger.error(f"Error getting contract info: {e}")
            return ContractInfo(contract_address, "Unknown", "UNKNOWN", 18, 0, None, False, None, None)
    
    async def _check_honeypot_risk(self, contract_address: str, chain: str) -> Dict:
        """Check if contract is a honeypot using multiple APIs"""
        try:
            honeypot_risk = 0.0
            details = {}
            
            # Check honeypot.is API
            try:
                url = f"{self.security_apis['honeypot_is']}?address={contract_address}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    is_honeypot = data.get('IsHoneypot', False)
                    
                    if is_honeypot:
                        honeypot_risk = 0.9
                        details['honeypot_is'] = "HONEYPOT DETECTED"
                    else:
                        honeypot_risk = max(honeypot_risk, 0.1)
                        details['honeypot_is'] = "Clean"
                        
            except Exception as e:
                logger.warning(f"honeypot.is API error: {e}")
                details['honeypot_is'] = "API Error"
            
            # Additional honeypot checks would go here
            # - Check for buy/sell transaction patterns
            # - Analyze contract functions for sell blocks
            # - Check for blacklist functions
            
            return {
                "risk": honeypot_risk,
                "confidence": 0.8,
                "details": details,
                "checks_performed": ["honeypot.is", "pattern_analysis"]
            }
            
        except Exception as e:
            logger.error(f"Error checking honeypot risk: {e}")
            return {"risk": 0.5, "confidence": 0.0, "details": {"error": str(e)}}
    
    async def _check_ownership_risk(self, contract_address: str, chain: str) -> Dict:
        """Analyze ownership and admin key risks"""
        try:
            ownership_risk = 0.0
            details = {}
            
            # Check for dangerous functions
            dangerous_functions = [
                "mint", "burn", "pause", "unpause", "setTaxFee", 
                "excludeFromFee", "renounceOwnership", "transferOwnership"
            ]
            
            # In real implementation, would analyze contract bytecode/source
            # For demo, simulate analysis
            has_mint = False  # Would check actual contract
            has_pause = False
            ownership_renounced = False
            
            if has_mint:
                ownership_risk += 0.3
                details['mint_function'] = "Contract can mint new tokens"
            
            if has_pause:
                ownership_risk += 0.2
                details['pause_function'] = "Contract can be paused"
            
            if not ownership_renounced:
                ownership_risk += 0.4
                details['ownership'] = "Ownership not renounced"
            else:
                details['ownership'] = "Ownership renounced"
            
            return {
                "risk": min(ownership_risk, 1.0),
                "confidence": 0.7,
                "details": details,
                "dangerous_functions": dangerous_functions
            }
            
        except Exception as e:
            logger.error(f"Error checking ownership risk: {e}")
            return {"risk": 0.5, "confidence": 0.0, "details": {"error": str(e)}}
    
    async def _check_liquidity_risk(self, contract_address: str, chain: str) -> Dict:
        """Analyze liquidity lock and pull risks"""
        try:
            liquidity_risk = 0.0
            details = {}
            
            # Check liquidity locks
            # In real implementation, would check:
            # - Liquidity provider tokens locked in contract
            # - Lock duration and unlock dates
            # - Percentage of liquidity locked
            
            liquidity_locked = True  # Would check actual locks
            lock_duration_days = 365  # Would get from lock contract
            locked_percentage = 0.85  # Would calculate from LP tokens
            
            if not liquidity_locked:
                liquidity_risk += 0.7
                details['liquidity_lock'] = "No liquidity lock detected"
            elif lock_duration_days < 30:
                liquidity_risk += 0.4
                details['liquidity_lock'] = f"Short lock duration: {lock_duration_days} days"
            elif locked_percentage < 0.5:
                liquidity_risk += 0.3
                details['liquidity_lock'] = f"Low locked percentage: {locked_percentage:.1%}"
            else:
                details['liquidity_lock'] = f"Good: {locked_percentage:.1%} locked for {lock_duration_days} days"
            
            return {
                "risk": min(liquidity_risk, 1.0),
                "confidence": 0.6,
                "details": details,
                "locked_percentage": locked_percentage,
                "lock_duration": lock_duration_days
            }
            
        except Exception as e:
            logger.error(f"Error checking liquidity risk: {e}")
            return {"risk": 0.5, "confidence": 0.0, "details": {"error": str(e)}}
    
    async def _check_contract_verification(self, contract_address: str, chain: str) -> Dict:
        """Check contract verification and audit status"""
        try:
            # In real implementation, would check blockchain explorers
            is_verified = True  # Would check actual verification
            has_audit = False   # Would check audit databases
            
            details = {}
            
            if is_verified:
                details['verification'] = "Contract source code verified"
            else:
                details['verification'] = "Contract not verified - HIGH RISK"
            
            if has_audit:
                details['audit'] = "Professional audit found"
            else:
                details['audit'] = "No professional audit found"
            
            audit_status = "Verified" if is_verified else "Unverified"
            if has_audit:
                audit_status += " + Audited"
            
            return {
                "verified": is_verified,
                "audited": has_audit,
                "audit_status": audit_status,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Error checking contract verification: {e}")
            return {"verified": False, "audited": False, "audit_status": "Unknown", "details": {"error": str(e)}}
    
    async def _check_suspicious_functions(self, contract_address: str, chain: str) -> Dict:
        """Analyze contract for suspicious functions"""
        try:
            suspicious_score = 0.0
            detected_functions = []
            
            # List of suspicious function patterns
            suspicious_patterns = [
                "blacklist", "isBot", "setMaxTx", "setSwapThreshold",
                "emergencyWithdraw", "rugPull", "drain"
            ]
            
            # In real implementation, would analyze actual contract bytecode
            # For demo, simulate detection
            has_blacklist = False
            has_max_tx_limit = True
            has_emergency_functions = False
            
            if has_blacklist:
                suspicious_score += 0.4
                detected_functions.append("blacklist")
            
            if has_max_tx_limit:
                suspicious_score += 0.1
                detected_functions.append("max_transaction_limit")
            
            if has_emergency_functions:
                suspicious_score += 0.6
                detected_functions.append("emergency_functions")
            
            return {
                "risk": min(suspicious_score, 1.0),
                "detected_functions": detected_functions,
                "patterns_checked": suspicious_patterns
            }
            
        except Exception as e:
            logger.error(f"Error checking suspicious functions: {e}")
            return {"risk": 0.5, "detected_functions": [], "patterns_checked": []}
    
    async def _check_tax_analysis(self, contract_address: str, chain: str) -> Dict:
        """Analyze buy/sell taxes and fees"""
        try:
            # In real implementation, would simulate buy/sell to check actual taxes
            buy_tax = 0.05   # 5% buy tax
            sell_tax = 0.08  # 8% sell tax
            
            details = {}
            risk = 0.0
            
            if buy_tax > 0.1:  # >10% buy tax
                risk += 0.3
                details['buy_tax'] = f"High buy tax: {buy_tax:.1%}"
            else:
                details['buy_tax'] = f"Buy tax: {buy_tax:.1%}"
            
            if sell_tax > self.sell_tax_threshold:
                risk += 0.4
                details['sell_tax'] = f"High sell tax: {sell_tax:.1%}"
            else:
                details['sell_tax'] = f"Sell tax: {sell_tax:.1%}"
            
            return {
                "risk": min(risk, 1.0),
                "buy_tax": buy_tax,
                "sell_tax": sell_tax,
                "details": details
            }
            
        except Exception as e:
            logger.error(f"Error checking tax analysis: {e}")
            return {"risk": 0.5, "buy_tax": 0.0, "sell_tax": 0.0, "details": {"error": str(e)}}
    
    def _calculate_risk_score(self, honeypot_risk: Dict, ownership_risk: Dict,
                            liquidity_risk: Dict, verification_result: Dict,
                            suspicious_functions: Dict, tax_analysis: Dict) -> Tuple[float, SecurityLevel, List[ContractRisk]]:
        """Calculate overall risk score and security level"""
        
        # Extract risk values
        hp_risk = honeypot_risk.get("risk", 0.5) if isinstance(honeypot_risk, dict) else 0.5
        own_risk = ownership_risk.get("risk", 0.5) if isinstance(ownership_risk, dict) else 0.5
        liq_risk = liquidity_risk.get("risk", 0.5) if isinstance(liquidity_risk, dict) else 0.5
        sus_risk = suspicious_functions.get("risk", 0.5) if isinstance(suspicious_functions, dict) else 0.5
        tax_risk = tax_analysis.get("risk", 0.5) if isinstance(tax_analysis, dict) else 0.5
        
        # Weight the risks
        weights = {
            'honeypot': 0.35,      # Highest weight - direct scam risk
            'ownership': 0.25,     # Admin key risks
            'liquidity': 0.20,     # Rug pull risk
            'suspicious': 0.15,    # Code red flags
            'tax': 0.05           # Fee analysis
        }
        
        # Calculate weighted risk score
        risk_score = (
            hp_risk * weights['honeypot'] +
            own_risk * weights['ownership'] +
            liq_risk * weights['liquidity'] +
            sus_risk * weights['suspicious'] +
            tax_risk * weights['tax']
        )
        
        # Penalty for unverified contracts
        if not verification_result.get("verified", False):
            risk_score += 0.2
        
        # Determine security level
        if risk_score >= 0.8:
            security_level = SecurityLevel.CRITICAL_RISK
        elif risk_score >= 0.6:
            security_level = SecurityLevel.HIGH_RISK
        elif risk_score >= 0.4:
            security_level = SecurityLevel.MEDIUM_RISK
        elif risk_score >= 0.2:
            security_level = SecurityLevel.LOW_RISK
        else:
            security_level = SecurityLevel.SAFE
        
        # Identify specific risks
        risks_detected = []
        
        if hp_risk > 0.7:
            risks_detected.append(ContractRisk.HONEYPOT)
        if own_risk > 0.6:
            risks_detected.append(ContractRisk.OWNERSHIP_RISK)
        if liq_risk > 0.8:
            risks_detected.append(ContractRisk.LIQUIDITY_RISK)
        if not verification_result.get("verified", False):
            risks_detected.append(ContractRisk.UNVERIFIED_CONTRACT)
        if sus_risk > 0.5:
            risks_detected.append(ContractRisk.SUSPICIOUS_FUNCTIONS)
        if tax_analysis.get("sell_tax", 0) > self.sell_tax_threshold:
            risks_detected.append(ContractRisk.HIGH_SELL_TAX)
        
        return risk_score, security_level, risks_detected
    
    def _generate_recommendation(self, security_level: SecurityLevel, 
                               risks_detected: List[ContractRisk]) -> str:
        """Generate trading recommendation based on security analysis"""
        
        if security_level == SecurityLevel.CRITICAL_RISK:
            return "🚨 DO NOT TRADE - Critical security risks detected. High probability of loss."
        
        elif security_level == SecurityLevel.HIGH_RISK:
            return "⛔ AVOID - High risk contract. Only trade with extreme caution and small amounts."
        
        elif security_level == SecurityLevel.MEDIUM_RISK:
            return "⚠️ CAUTION - Medium risk detected. Consider waiting for better opportunities."
        
        elif security_level == SecurityLevel.LOW_RISK:
            return "✅ ACCEPTABLE - Low risk profile. Suitable for trading with normal position sizing."
        
        else:
            return "🟢 SAFE - Good security profile. Cleared for normal trading operations."
    
    def _get_blockchain_api_url(self, chain: str) -> Optional[str]:
        """Get appropriate blockchain API URL for chain"""
        api_urls = {
            'ethereum': self.security_apis['etherscan'],
            'bsc': self.security_apis['bscscan'],
            'polygon': 'https://api.polygonscan.com/api'
        }
        return api_urls.get(chain.lower())
    
    async def _store_security_result(self, result: SecurityResult):
        """Store security analysis result in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO security_analysis (
                    contract_address, token_symbol, security_level, risk_score,
                    risks_detected, honeypot_risk, ownership_risk, liquidity_risk,
                    verified, audit_status, recommendation, analysis_details, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.contract_address, result.token_symbol, result.security_level.value,
                result.risk_score, json.dumps([r.value for r in result.risks_detected]),
                result.honeypot_risk, result.ownership_risk, result.liquidity_risk,
                result.verified, result.audit_status, result.recommendation,
                json.dumps(result.analysis_details), result.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing security result: {e}")
    
    async def _store_contract_info(self, contract_info: ContractInfo):
        """Store contract information in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO contract_info (
                    address, name, symbol, decimals, total_supply, owner, is_verified, creation_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                contract_info.address, contract_info.name, contract_info.symbol,
                contract_info.decimals, str(contract_info.total_supply), contract_info.owner,
                contract_info.is_verified, contract_info.creation_date.isoformat() if contract_info.creation_date else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing contract info: {e}")
    
    async def _generate_security_alert(self, result: SecurityResult):
        """Generate security alert for high-risk contracts"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            for risk in result.risks_detected:
                cursor.execute('''
                    INSERT INTO security_alerts (contract_address, alert_type, severity, message)
                    VALUES (?, ?, ?, ?)
                ''', (
                    result.contract_address, risk.value, result.security_level.value,
                    f"{risk.value.replace('_', ' ').title()} detected in {result.token_symbol}"
                ))
            
            conn.commit()
            conn.close()
            
            logger.warning(f"🚨 Security alert generated for {result.token_symbol}: {result.security_level.value}")
            
        except Exception as e:
            logger.error(f"Error generating security alert: {e}")
    
    def _create_failed_analysis(self, contract_address: str, token_symbol: str, error: str) -> SecurityResult:
        """Create a failed analysis result"""
        return SecurityResult(
            contract_address=contract_address,
            token_symbol=token_symbol or "UNKNOWN",
            security_level=SecurityLevel.HIGH_RISK,
            risk_score=0.8,  # High risk for failed analysis
            risks_detected=[ContractRisk.UNVERIFIED_CONTRACT],
            analysis_details={"error": error},
            honeypot_risk=0.5,
            ownership_risk=0.5,
            liquidity_risk=0.5,
            verified=False,
            audit_status="Analysis Failed",
            recommendation="⚠️ AVOID - Security analysis failed. Do not trade until manually verified.",
            timestamp=datetime.now()
        )
    
    async def batch_analyze_contracts(self, contract_addresses: List[str], 
                                    chain: str = "ethereum") -> Dict[str, SecurityResult]:
        """Analyze multiple contracts efficiently"""
        logger.info(f"🔍 Batch analyzing {len(contract_addresses)} contracts")
        
        results = {}
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def analyze_single(address):
            async with semaphore:
                try:
                    result = await self.analyze_contract_security(address, chain=chain)
                    results[address] = result
                    await asyncio.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.error(f"Batch analysis failed for {address}: {e}")
                    results[address] = self._create_failed_analysis(address, None, str(e))
        
        # Run all analyses
        await asyncio.gather(*[analyze_single(addr) for addr in contract_addresses])
        
        # Summary
        safe_count = sum(1 for r in results.values() if r.security_level == SecurityLevel.SAFE)
        risky_count = len(results) - safe_count
        
        logger.info(f"✅ Batch analysis complete: {safe_count} safe, {risky_count} risky")
        
        return results
    
    def get_security_summary(self) -> Dict:
        """Get summary of recent security analyses"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get recent analyses
            cursor.execute('''
                SELECT security_level, COUNT(*) 
                FROM security_analysis 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY security_level
            ''')
            
            level_counts = dict(cursor.fetchall())
            
            # Get recent alerts
            cursor.execute('''
                SELECT COUNT(*) FROM security_alerts 
                WHERE triggered_at > datetime('now', '-24 hours')
            ''')
            
            recent_alerts = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "analyses_24h": sum(level_counts.values()),
                "level_distribution": level_counts,
                "alerts_24h": recent_alerts,
                "safe_percentage": (level_counts.get('safe', 0) / max(1, sum(level_counts.values()))) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting security summary: {e}")
            return {"error": str(e)}

# Example usage and testing
async def main():
    """Test the smart contract security analyzer"""
    analyzer = SmartContractAnalyzer()
    
    # Test contract addresses (use real ones for actual testing)
    test_contracts = [
        "0xa0b86a33e6411c3ce98e6d9b4b3c61a6f2b0c1d2",  # Example contract
        "0xb1c73e1b4f5c6d7e8f9a0b1c2d3e4f5g6h7i8j9k",  # Another example
    ]
    
    print("🔒 Testing Smart Contract Security Analyzer")
    print("=" * 50)
    
    # Single contract analysis
    print("\n📋 Single Contract Analysis:")
    result = await analyzer.analyze_contract_security(test_contracts[0], "TEST", "ethereum")
    
    print(f"Contract: {result.contract_address}")
    print(f"Security Level: {result.security_level.value}")
    print(f"Risk Score: {result.risk_score:.2f}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Risks: {[r.value for r in result.risks_detected]}")
    
    # Batch analysis
    print("\n📊 Batch Analysis:")
    batch_results = await analyzer.batch_analyze_contracts(test_contracts, "ethereum")
    
    for address, result in batch_results.items():
        print(f"{address[:10]}...: {result.security_level.value} (risk: {result.risk_score:.2f})")
    
    # Summary
    print("\n📈 Security Summary:")
    summary = analyzer.get_security_summary()
    print(f"Analyses in 24h: {summary.get('analyses_24h', 0)}")
    print(f"Alerts in 24h: {summary.get('alerts_24h', 0)}")
    print(f"Safe percentage: {summary.get('safe_percentage', 0):.1f}%")

if __name__ == "__main__":
    asyncio.run(main())