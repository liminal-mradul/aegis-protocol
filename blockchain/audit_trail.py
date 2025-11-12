import hashlib
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class AuditBlock:
    block_hash: str
    previous_hash: str
    timestamp: float
    transactions: List[Dict]
    nonce: int
    merkle_root: str

class AuditTrail:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = 4
        
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        genesis_block = AuditBlock(
            block_hash="0" * 64,
            previous_hash="0" * 64,
            timestamp=time.time(),
            transactions=[],
            nonce=0,
            merkle_root="0" * 64
        )
        self.chain.append(genesis_block)
    
    def record_operation(self, operation_type: str, node_id: str, 
                        details: Dict, signature: str):
        transaction = {
            'type': operation_type,
            'node_id': node_id,
            'timestamp': time.time(),
            'details': details,
            'signature': signature,
            'transaction_id': self._generate_transaction_id(operation_type, node_id, details)
        }
        
        self.pending_transactions.append(transaction)
        
        if len(self.pending_transactions) >= 10:
            asyncio.create_task(self._mine_block())
    
    def _generate_transaction_id(self, operation_type: str, node_id: str, details: Dict) -> str:
        data = f"{operation_type}{node_id}{json.dumps(details, sort_keys=True)}{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    async def _mine_block(self):
        if not self.pending_transactions:
            return
        
        previous_block = self.chain[-1]
        merkle_root = self._compute_merkle_root(self.pending_transactions)
        
        nonce = 0
        while True:
            block_data = {
                'previous_hash': previous_block.block_hash,
                'timestamp': time.time(),
                'transactions': self.pending_transactions.copy(),
                'merkle_root': merkle_root,
                'nonce': nonce
            }
            
            block_hash = hashlib.sha256(json.dumps(block_data, sort_keys=True).encode()).hexdigest()
            
            if block_hash[:self.difficulty] == "0" * self.difficulty:
                new_block = AuditBlock(
                    block_hash=block_hash,
                    previous_hash=previous_block.block_hash,
                    timestamp=time.time(),
                    transactions=self.pending_transactions.copy(),
                    nonce=nonce,
                    merkle_root=merkle_root
                )
                
                self.chain.append(new_block)
                self.pending_transactions = []
                break
            
            nonce += 1
            await asyncio.sleep(0)
    
    def _compute_merkle_root(self, transactions: List[Dict]) -> str:
        if not transactions:
            return "0" * 64
        
        transaction_hashes = [tx['transaction_id'] for tx in transactions]
        
        while len(transaction_hashes) > 1:
            new_hashes = []
            for i in range(0, len(transaction_hashes), 2):
                if i + 1 < len(transaction_hashes):
                    combined = transaction_hashes[i] + transaction_hashes[i + 1]
                else:
                    combined = transaction_hashes[i] + transaction_hashes[i]
                
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_hashes.append(new_hash)
            
            transaction_hashes = new_hashes
        
        return transaction_hashes[0]
    
    def verify_integrity(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            if current_block.previous_hash != previous_block.block_hash:
                return False
            
            block_data = {
                'previous_hash': current_block.previous_hash,
                'timestamp': current_block.timestamp,
                'transactions': current_block.transactions,
                'merkle_root': current_block.merkle_root,
                'nonce': current_block.nonce
            }
            
            computed_hash = hashlib.sha256(
                json.dumps(block_data, sort_keys=True).encode()
            ).hexdigest()
            
            if computed_hash != current_block.block_hash:
                return False
        
        return True
