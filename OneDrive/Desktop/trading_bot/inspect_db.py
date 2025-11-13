#!/usr/bin/env python3
"""
Database inspector to find contract addresses
"""
import sqlite3
import json
from datetime import datetime

def inspect_database(db_path: str = "trading_bot.db"):
    """Inspect what's in the database"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print(f"📊 Database: {db_path}")
            print(f"📋 Tables found: {len(tables)}")
            
            for table in tables:
                table_name = table[0]
                print(f"\n📊 Table: {table_name}")
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                print(f"   Columns: {[col[1] for col in columns]}")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"   Rows: {count}")
                
                # Show sample data
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    rows = cursor.fetchall()
                    for i, row in enumerate(rows, 1):
                        print(f"   Row {i}: {row}")
                        
    except Exception as e:
        print(f"❌ Error: {e}")

def get_microcap_trades():
    """Get trades from the specific microcap system"""
    databases_to_check = [
        "trading_bot.db",
        "trade_tracking.db", 
        "demo_trades.db"
    ]
    
    for db_file in databases_to_check:
        print(f"\n🔍 Checking: {db_file}")
        try:
            inspect_database(db_file)
        except Exception as e:
            print(f"❌ Cannot access {db_file}: {e}")

if __name__ == "__main__":
    print("🔍 DATABASE INSPECTOR - Contract Address Hunt")
    print("=" * 60)
    get_microcap_trades()