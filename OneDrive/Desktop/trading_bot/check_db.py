#!/usr/bin/env python3
import sqlite3
import os

db_path = 'trade_tracking.db'

if os.path.exists(db_path):
    print(f"Database file exists: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables found: {tables}")
    
    # Check if positions table exists
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"  Rows: {count}")
        
        if count > 0 and count < 20:  # Show some sample data for small tables
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            rows = cursor.fetchall()
            for i, row in enumerate(rows):
                print(f"  {i+1}. {row}")
    
    conn.close()
else:
    print(f"Database file does not exist: {db_path}")