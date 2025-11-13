#!/usr/bin/env python3
"""
Test transaction encoding to debug Jupiter /execute submission
"""

import base58
import base64
from solders.transaction import VersionedTransaction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.instruction import Instruction
from solders.pubkey import Pubkey
from solders.hash import Hash

# Create a test transaction
kp = Keypair()
program_id = Pubkey.default()
instructions = [Instruction(program_id, bytes([]), [])]
recent_blockhash = Hash.new_unique()
message = MessageV0.try_compile(kp.pubkey(), instructions, [], recent_blockhash)

# Sign it
signed_tx = VersionedTransaction(message, [kp])

# Test serialization
tx_bytes = bytes(signed_tx)
print(f"Transaction bytes length: {len(tx_bytes)}")
print(f"First 20 bytes (hex): {tx_bytes[:20].hex()}")

# Try base58 encoding
tx_base58 = base58.b58encode(tx_bytes).decode('utf-8')
print(f"\nBase58 encoded length: {len(tx_base58)}")
print(f"Base58 (first 50 chars): {tx_base58[:50]}")

# Also check what Jupiter sends us (base64)
tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')
print(f"\nBase64 encoded length: {len(tx_base64)}")
print(f"Base64 (first 50 chars): {tx_base64[:50]}")

# Test round-trip
decoded_base58 = base58.b58decode(tx_base58)
print(f"\nRound-trip base58 matches: {decoded_base58 == tx_bytes}")

decoded_base64 = base64.b64decode(tx_base64)
print(f"Round-trip base64 matches: {decoded_base64 == tx_bytes}")
