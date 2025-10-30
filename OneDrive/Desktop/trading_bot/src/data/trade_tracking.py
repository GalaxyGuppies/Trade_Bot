"""
End-to-End Trade Tracking System with UUID Linking
Implements complete research → decision → execution → outcome pipeline
"""

import uuid
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TerminationReason(Enum):
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_LIMIT = "time_limit"
    MANUAL_EXIT = "manual_exit"
    EMERGENCY_EXIT = "emergency_exit"
    LIQUIDITY_DRIED = "liquidity_dried"
    RUG_DETECTED = "rug_detected"

class TradeStatus(Enum):
    CANDIDATE = "candidate"
    QUEUED = "queued"
    EXECUTING = "executing"
    ACTIVE = "active"
    CLOSED = "closed"
    FAILED = "failed"

@dataclass
class TradeCandidate:
    """Core trade candidate with UUID linking"""
    uuid: str
    timestamp: datetime
    instrument: str
    contract_address: Optional[str]
    model_version: str
    confidence: float
    params_ptr: str  # JSON pointer to parameter snapshot
    status: TradeStatus
    created_by: str  # Strategy name that created this candidate
    
    @classmethod
    def generate(cls, instrument: str, model_version: str, confidence: float, 
                 params_ptr: str, created_by: str, contract_address: str = None):
        return cls(
            uuid=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            instrument=instrument,
            contract_address=contract_address,
            model_version=model_version,
            confidence=confidence,
            params_ptr=params_ptr,
            status=TradeStatus.CANDIDATE,
            created_by=created_by
        )

@dataclass
class ResearchDocument:
    """Immutable research snapshot keyed by UUID"""
    uuid: str
    timestamp: datetime
    instrument: str
    contract_address: Optional[str]
    creator_wallets: List[str]
    initial_market_cap: float
    total_supply: float
    liquidity_pools: Dict[str, float]  # pool_address -> liquidity_usd
    liquidity_provider_addresses: List[str]
    holder_count: int
    on_chain_age_days: int
    audit_status: str
    verified_source_flag: bool
    social_snippets_ptr: str  # Pointer to raw social data
    news_snippets_ptr: str    # Pointer to raw news data
    social_sentiment_scores: Dict[str, float]
    news_sentiment_scores: Dict[str, float]
    whale_snapshot_ptr: str   # Pointer to whale analysis
    rugpull_heuristic_scores: Dict[str, float]
    trade_rationale: str
    model_feature_vector: List[float]
    parameter_snapshot: Dict[str, Any]

@dataclass
class ExecutionRecord:
    """Trade execution details with full audit trail"""
    uuid: str
    order_id: str
    venue: str
    entry_timestamp: Optional[datetime]
    exit_timestamp: Optional[datetime]
    entry_price: Optional[float]
    exit_price: Optional[float]
    size: float
    fees_paid: float
    slippage: float
    fills_json: str  # JSON string of all fills
    latency_metrics: Dict[str, float]
    route_taken: str
    mev_detected: bool

@dataclass
class TradeOutcome:
    """Final trade results and metrics"""
    uuid: str
    realized_pnl: float
    realized_pnl_percent: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    time_in_market_seconds: int
    termination_reason: TerminationReason
    external_incidents: List[str]  # Exchange outages, MEV events, etc.
    execution_quality_score: float

@dataclass
class Artifact:
    """Reference to stored raw data artifacts"""
    pointer_id: str
    storage_path: str
    artifact_type: str
    created_at: datetime
    size_bytes: int
    checksum: str

class TradeTrackingSystem:
    """Complete end-to-end trade tracking with UUID linking"""
    
    def __init__(self, db_path: str = "trade_tracking.db", artifacts_dir: str = "artifacts"):
        self.db_path = db_path
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize database with complete schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Candidates table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            uuid TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            instrument TEXT NOT NULL,
            contract_address TEXT,
            model_version TEXT NOT NULL,
            confidence REAL NOT NULL,
            params_ptr TEXT NOT NULL,
            status TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Research documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_docs (
            uuid TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            instrument TEXT NOT NULL,
            contract_address TEXT,
            creator_wallets TEXT,  -- JSON array
            initial_market_cap REAL,
            total_supply REAL,
            liquidity_pools TEXT,  -- JSON object
            liquidity_provider_addresses TEXT,  -- JSON array
            holder_count INTEGER,
            on_chain_age_days INTEGER,
            audit_status TEXT,
            verified_source_flag BOOLEAN,
            social_snippets_ptr TEXT,
            news_snippets_ptr TEXT,
            social_sentiment_scores TEXT,  -- JSON object
            news_sentiment_scores TEXT,    -- JSON object
            whale_snapshot_ptr TEXT,
            rugpull_heuristic_scores TEXT, -- JSON object
            trade_rationale TEXT,
            model_feature_vector TEXT,     -- JSON array
            parameter_snapshot TEXT,       -- JSON object
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Executions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS executions (
            uuid TEXT NOT NULL,
            order_id TEXT NOT NULL,
            venue TEXT NOT NULL,
            entry_timestamp TEXT,
            exit_timestamp TEXT,
            entry_price REAL,
            exit_price REAL,
            size REAL NOT NULL,
            fees_paid REAL DEFAULT 0,
            slippage REAL DEFAULT 0,
            fills_json TEXT,
            latency_metrics TEXT,  -- JSON object
            route_taken TEXT,
            mev_detected BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (uuid, order_id),
            FOREIGN KEY (uuid) REFERENCES candidates(uuid)
        )
        """)
        
        # Outcomes table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            uuid TEXT PRIMARY KEY,
            realized_pnl REAL NOT NULL,
            realized_pnl_percent REAL NOT NULL,
            max_adverse_excursion REAL,
            max_favorable_excursion REAL,
            time_in_market_seconds INTEGER,
            termination_reason TEXT NOT NULL,
            external_incidents TEXT,  -- JSON array
            execution_quality_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (uuid) REFERENCES candidates(uuid)
        )
        """)
        
        # Artifacts table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
            pointer_id TEXT PRIMARY KEY,
            storage_path TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            size_bytes INTEGER,
            checksum TEXT
        )
        """)
        
        # Experiments table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            run_id TEXT PRIMARY KEY,
            start_timestamp TEXT NOT NULL,
            end_timestamp TEXT,
            params_snapshot_ptr TEXT,
            aggregated_metrics TEXT,  -- JSON object
            sample_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indices for fast lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_timestamp ON candidates(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_instrument ON candidates(instrument)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_research_contract ON research_docs(contract_address)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_uuid ON executions(uuid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_uuid ON outcomes(uuid)")
        
        conn.commit()
        conn.close()
        logger.info("Database initialized with complete schema")
    
    def create_candidate(self, instrument: str, model_version: str, confidence: float,
                        params_ptr: str, created_by: str, contract_address: str = None) -> TradeCandidate:
        """Create new trade candidate with UUID"""
        candidate = TradeCandidate.generate(
            instrument=instrument,
            model_version=model_version,
            confidence=confidence,
            params_ptr=params_ptr,
            created_by=created_by,
            contract_address=contract_address
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO candidates (uuid, timestamp, instrument, contract_address,
                              model_version, confidence, params_ptr, status, created_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate.uuid,
            candidate.timestamp.isoformat(),
            candidate.instrument,
            candidate.contract_address,
            candidate.model_version,
            candidate.confidence,
            candidate.params_ptr,
            candidate.status.value,
            candidate.created_by
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created candidate {candidate.uuid} for {instrument}")
        return candidate
    
    def store_research_document(self, research_doc: ResearchDocument):
        """Store immutable research snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO research_docs (
            uuid, timestamp, instrument, contract_address, creator_wallets,
            initial_market_cap, total_supply, liquidity_pools, liquidity_provider_addresses,
            holder_count, on_chain_age_days, audit_status, verified_source_flag,
            social_snippets_ptr, news_snippets_ptr, social_sentiment_scores,
            news_sentiment_scores, whale_snapshot_ptr, rugpull_heuristic_scores,
            trade_rationale, model_feature_vector, parameter_snapshot
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            research_doc.uuid,
            research_doc.timestamp.isoformat(),
            research_doc.instrument,
            research_doc.contract_address,
            json.dumps(research_doc.creator_wallets),
            research_doc.initial_market_cap,
            research_doc.total_supply,
            json.dumps(research_doc.liquidity_pools),
            json.dumps(research_doc.liquidity_provider_addresses),
            research_doc.holder_count,
            research_doc.on_chain_age_days,
            research_doc.audit_status,
            research_doc.verified_source_flag,
            research_doc.social_snippets_ptr,
            research_doc.news_snippets_ptr,
            json.dumps(research_doc.social_sentiment_scores),
            json.dumps(research_doc.news_sentiment_scores),
            research_doc.whale_snapshot_ptr,
            json.dumps(research_doc.rugpull_heuristic_scores),
            research_doc.trade_rationale,
            json.dumps(research_doc.model_feature_vector),
            json.dumps(research_doc.parameter_snapshot)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored research document for {research_doc.uuid}")
    
    def store_execution_record(self, execution: ExecutionRecord):
        """Store execution details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO executions (
            uuid, order_id, venue, entry_timestamp, exit_timestamp,
            entry_price, exit_price, size, fees_paid, slippage,
            fills_json, latency_metrics, route_taken, mev_detected
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.uuid,
            execution.order_id,
            execution.venue,
            execution.entry_timestamp.isoformat() if execution.entry_timestamp else None,
            execution.exit_timestamp.isoformat() if execution.exit_timestamp else None,
            execution.entry_price,
            execution.exit_price,
            execution.size,
            execution.fees_paid,
            execution.slippage,
            execution.fills_json,
            json.dumps(execution.latency_metrics),
            execution.route_taken,
            execution.mev_detected
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored execution record for {execution.uuid}")
    
    def store_outcome(self, outcome: TradeOutcome):
        """Store final trade outcome"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO outcomes (
            uuid, realized_pnl, realized_pnl_percent, max_adverse_excursion,
            max_favorable_excursion, time_in_market_seconds, termination_reason,
            external_incidents, execution_quality_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            outcome.uuid,
            outcome.realized_pnl,
            outcome.realized_pnl_percent,
            outcome.max_adverse_excursion,
            outcome.max_favorable_excursion,
            outcome.time_in_market_seconds,
            outcome.termination_reason.value,
            json.dumps(outcome.external_incidents),
            outcome.execution_quality_score
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored outcome for {outcome.uuid}: {outcome.realized_pnl_percent:.2f}%")
    
    def store_artifact(self, data: bytes, artifact_type: str, 
                      file_extension: str = "") -> str:
        """Store raw artifact and return pointer"""
        pointer_id = str(uuid.uuid4())
        filename = f"{pointer_id}{file_extension}"
        storage_path = self.artifacts_dir / filename
        
        # Write data to file
        with open(storage_path, 'wb') as f:
            f.write(data)
        
        # Create artifact record
        import hashlib
        checksum = hashlib.sha256(data).hexdigest()
        
        artifact = Artifact(
            pointer_id=pointer_id,
            storage_path=str(storage_path),
            artifact_type=artifact_type,
            created_at=datetime.now(timezone.utc),
            size_bytes=len(data),
            checksum=checksum
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO artifacts (pointer_id, storage_path, artifact_type,
                             created_at, size_bytes, checksum)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            artifact.pointer_id,
            artifact.storage_path,
            artifact.artifact_type,
            artifact.created_at.isoformat(),
            artifact.size_bytes,
            artifact.checksum
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored artifact {pointer_id} ({len(data)} bytes)")
        return pointer_id
    
    def get_complete_trade_record(self, trade_uuid: str) -> Dict[str, Any]:
        """Get complete end-to-end trade record by UUID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get candidate
        cursor.execute("SELECT * FROM candidates WHERE uuid = ?", (trade_uuid,))
        candidate_row = cursor.fetchone()
        
        if not candidate_row:
            return {"error": "Trade UUID not found"}
        
        # Get research document
        cursor.execute("SELECT * FROM research_docs WHERE uuid = ?", (trade_uuid,))
        research_row = cursor.fetchone()
        
        # Get executions
        cursor.execute("SELECT * FROM executions WHERE uuid = ?", (trade_uuid,))
        execution_rows = cursor.fetchall()
        
        # Get outcome
        cursor.execute("SELECT * FROM outcomes WHERE uuid = ?", (trade_uuid,))
        outcome_row = cursor.fetchone()
        
        conn.close()
        
        return {
            "uuid": trade_uuid,
            "candidate": dict(zip([desc[0] for desc in cursor.description], candidate_row)) if candidate_row else None,
            "research": dict(zip([desc[0] for desc in cursor.description], research_row)) if research_row else None,
            "executions": [dict(zip([desc[0] for desc in cursor.description], row)) for row in execution_rows],
            "outcome": dict(zip([desc[0] for desc in cursor.description], outcome_row)) if outcome_row else None
        }
    
    def update_candidate_status(self, trade_uuid: str, new_status: TradeStatus):
        """Update candidate status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        UPDATE candidates SET status = ? WHERE uuid = ?
        """, (new_status.value, trade_uuid))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated {trade_uuid} status to {new_status.value}")

    def get_performance_metrics(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Get aggregated performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query with optional date filtering
        where_clause = ""
        params = []
        if start_date:
            where_clause += " AND c.timestamp >= ?"
            params.append(start_date)
        if end_date:
            where_clause += " AND c.timestamp <= ?"
            params.append(end_date)
        
        # Get comprehensive metrics
        cursor.execute(f"""
        SELECT 
            COUNT(*) as total_trades,
            COUNT(o.uuid) as completed_trades,
            AVG(o.realized_pnl_percent) as avg_return_pct,
            SUM(o.realized_pnl) as total_pnl,
            COUNT(CASE WHEN o.realized_pnl > 0 THEN 1 END) * 1.0 / COUNT(o.uuid) as win_rate,
            AVG(o.time_in_market_seconds) as avg_time_in_market,
            AVG(c.confidence) as avg_confidence,
            MAX(o.realized_pnl_percent) as max_win_pct,
            MIN(o.realized_pnl_percent) as max_loss_pct
        FROM candidates c
        LEFT JOIN outcomes o ON c.uuid = o.uuid
        WHERE 1=1 {where_clause}
        """, params)
        
        metrics = dict(zip([desc[0] for desc in cursor.description], cursor.fetchone()))
        
        # Get termination reason breakdown
        cursor.execute(f"""
        SELECT termination_reason, COUNT(*) as count
        FROM candidates c
        JOIN outcomes o ON c.uuid = o.uuid
        WHERE 1=1 {where_clause}
        GROUP BY termination_reason
        """, params)
        
        termination_breakdown = dict(cursor.fetchall())
        metrics['termination_breakdown'] = termination_breakdown
        
        conn.close()
        return metrics

# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    tracker = TradeTrackingSystem()
    
    # Example: Create a trade candidate
    candidate = tracker.create_candidate(
        instrument="PEPE/USDT",
        model_version="v1.2.3",
        confidence=0.75,
        params_ptr="params_snapshot_001",
        created_by="high_volatility_strategy",
        contract_address="0x6982508145454Ce325dDbE47a25d4ec3d2311933"
    )
    
    print(f"Created candidate: {candidate.uuid}")
    
    # Example: Store research document
    research = ResearchDocument(
        uuid=candidate.uuid,
        timestamp=datetime.now(timezone.utc),
        instrument="PEPE/USDT",
        contract_address="0x6982508145454Ce325dDbE47a25d4ec3d2311933",
        creator_wallets=["0x123...", "0x456..."],
        initial_market_cap=1000000.0,
        total_supply=420690000000000,
        liquidity_pools={"uniswap_v3": 500000.0, "sushiswap": 250000.0},
        liquidity_provider_addresses=["0x789...", "0xabc..."],
        holder_count=15420,
        on_chain_age_days=45,
        audit_status="unaudited",
        verified_source_flag=False,
        social_snippets_ptr="social_001",
        news_snippets_ptr="news_001",
        social_sentiment_scores={"twitter": 0.65, "reddit": 0.72, "telegram": 0.58},
        news_sentiment_scores={"coindesk": 0.45, "cointelegraph": 0.55},
        whale_snapshot_ptr="whale_001",
        rugpull_heuristic_scores={"liquidity_score": 0.8, "dev_behavior": 0.6, "token_distribution": 0.7},
        trade_rationale="High social sentiment + recent listing + good liquidity",
        model_feature_vector=[0.75, 0.65, 0.8, 0.45, 0.6],
        parameter_snapshot={"stop_loss": 0.15, "take_profit": 0.3, "position_size": 0.02}
    )
    
    tracker.store_research_document(research)
    print("Stored research document")
    
    # Get complete record
    complete_record = tracker.get_complete_trade_record(candidate.uuid)
    print(f"Complete record keys: {complete_record.keys()}")