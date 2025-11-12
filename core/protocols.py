import hashlib
import time
import json
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from threading import Lock
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from cachetools import TTLCache, LRUCache
import blake3
import orjson

class NodeStatus(Enum):
    HEALTHY = "healthy"
    SUSPECTED = "suspected" 
    BYZANTINE = "byzantine"

@dataclass
class NodeState:
    id: str
    public_key: str
    stake: int
    reputation: float
    status: NodeStatus
    last_heartbeat: float

class ByzantineFaultTolerance:
    def __init__(self, total_nodes: int, fault_threshold: float = 0.33):
        self.total_nodes = total_nodes
        self.fault_threshold = fault_threshold
        
        # OPTIMIZED: Better data structures
        self.node_states: Dict[str, NodeState] = {}
        self.vote_history = defaultdict(dict)
        self.consensus_cache = TTLCache(maxsize=1000, ttl=300)
        
        # OPTIMIZED: Fine-grained locks
        self.node_lock = Lock()
        self.vote_lock = Lock()
        
        # OPTIMIZED: Precomputed thresholds
        self._precomputed_thresholds = self._precompute_thresholds()

    def _precompute_thresholds(self) -> Dict[int, float]:
        """Precompute consensus thresholds for performance"""
        thresholds = {}
        for n in range(3, 101):
            thresholds[n] = (1 - self.fault_threshold) * n
        return thresholds

    def register_node(self, node_id: str, public_key: str, initial_stake: int = 1000):
        with self.node_lock:
            self.node_states[node_id] = NodeState(
                id=node_id,
                public_key=public_key,
                stake=initial_stake,
                reputation=0.5,
                status=NodeStatus.HEALTHY,
                last_heartbeat=time.time()
            )

    def process_vote_batch(self, vote_round: int, votes: List[Tuple[str, Dict]]) -> bool:
        """OPTIMIZED: Process multiple votes in batch"""
        with self.vote_lock:
            round_votes = self.vote_history[vote_round]
            
            for node_id, vote_data in votes:
                if node_id in self.node_states:
                    round_votes[node_id] = {
                        'vote': vote_data,
                        'timestamp': time.time(),
                        'stake_weight': self.node_states[node_id].stake
                    }
            
            return self._check_consensus_optimized(vote_round)

    def process_vote(self, vote_round: int, node_id: str, vote_data: Dict) -> bool:
        """OPTIMIZED: Single vote processing"""
        with self.vote_lock:
            if node_id not in self.node_states:
                return False
            
            if vote_round not in self.vote_history:
                self.vote_history[vote_round] = {}
            
            self.vote_history[vote_round][node_id] = {
                'vote': vote_data,
                'timestamp': time.time(),
                'stake_weight': self.node_states[node_id].stake
            }
            
            return self._check_consensus_optimized(vote_round)

    def _check_consensus_optimized(self, vote_round: int) -> bool:
        """OPTIMIZED: Fast consensus checking"""
        if vote_round not in self.vote_history:
            return False
        
        votes = self.vote_history[vote_round]
        if not votes:
            return False
        
        # OPTIMIZED: Use precomputed threshold
        required_stake = self._precomputed_thresholds.get(
            len(self.node_states), 
            (1 - self.fault_threshold) * sum(s.stake for s in self.node_states.values())
        )
        
        voted_stake = sum(vote['stake_weight'] for vote in votes.values())
        
        if voted_stake < required_stake:
            return False
        
        # OPTIMIZED: Hash-based vote counting
        vote_hashes = defaultdict(float)
        for vote in votes.values():
            vote_hash = blake3.blake3(orjson.dumps(vote['vote'])).hexdigest()
            vote_hashes[vote_hash] += vote['stake_weight']
        
        max_vote_stake = max(vote_hashes.values())
        return max_vote_stake / voted_stake > 0.66

    def validate_byzantine_proof(self, node_id: str, proof: Dict) -> bool:
        required_fields = {'signature', 'timestamp', 'message_hash', 'sequence'}
        if not all(field in proof for field in required_fields):
            return False
        
        if node_id not in self.node_states:
            return False
        
        return self._verify_signature(proof['signature'], proof['message_hash'], node_id)

    def _verify_signature(self, signature: str, message_hash: str, node_id: str) -> bool:
        # FIXED: Use full hash comparison
        expected_signature = hashlib.sha256(f"{message_hash}{node_id}".encode()).hexdigest()
        return signature == expected_signature

class SecureAggregation:
    def __init__(self, node_count: int):
        self.node_count = node_count
        self.aggregation_sessions = LRUCache(maxsize=500)  # OPTIMIZED: LRU cache
        self.session_lock = Lock()
    
    def initialize_secure_session(self, session_id: str, participants: List[str], 
                                threshold: int = None) -> Dict:
        if threshold is None:
            threshold = len(participants) // 2 + 1
        
        with self.session_lock:
            self.aggregation_sessions[session_id] = {
                'participants': set(participants),
                'threshold': threshold,
                'contributions': {},
                'masks': {},
                'completed': False,
                'created_at': time.time()
            }
        
        return {
            'session_id': session_id,
            'threshold': threshold,
            'participant_count': len(participants)
        }
    
    def submit_encrypted_gradient(self, session_id: str, node_id: str, 
                                encrypted_data: bytes, security_proof: Dict) -> bool:
        with self.session_lock:
            if session_id not in self.aggregation_sessions:
                return False
            
            session = self.aggregation_sessions[session_id]
            if node_id not in session['participants']:
                return False
            
            if not self._validate_security_proof(security_proof, encrypted_data, node_id):
                return False
            
            session['contributions'][node_id] = {
                'data': encrypted_data,
                'timestamp': time.time(),
                'proof': security_proof
            }
            
            return True
    
    def _validate_security_proof(self, proof: Dict, data: bytes, node_id: str) -> bool:
        required = {'nonce', 'commitment', 'timestamp'}
        if not all(field in proof for field in required):
            return False
        
        # OPTIMIZED: Use faster BLAKE3
        expected_commitment = blake3.blake3(data + proof['nonce'].encode()).hexdigest()
        return proof['commitment'] == expected_commitment
    
    def compute_secure_aggregate(self, session_id: str) -> Optional[bytes]:
        with self.session_lock:
            if session_id not in self.aggregation_sessions:
                return None
            
            session = self.aggregation_sessions[session_id]
            if len(session['contributions']) < session['threshold']:
                return None
            
            aggregated = self._simulate_secure_aggregation(session['contributions'])
            session['completed'] = True
            return aggregated
    
    def _simulate_secure_aggregation(self, contributions: Dict) -> bytes:
        # OPTIMIZED: Use faster hash
        combined_data = b""
        for contrib in contributions.values():
            combined_data += contrib['data']
        
        return blake3.blake3(combined_data).digest()

class ProductionDPEngine:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget = epsilon
        self.budget_lock = Lock()
        self.audit_log = LRUCache(maxsize=10000)  # OPTIMIZED: LRU cache
        
        # OPTIMIZED: Precompute noise parameters
        self.gaussian_sigma = self._precompute_gaussian_sigma()
        self.laplace_scale = self._precompute_laplace_scale()
    
    def _precompute_gaussian_sigma(self) -> float:
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def _precompute_laplace_scale(self) -> float:
        return 1.0 / self.epsilon

    def gaussian_noise_vectorized(self, values: np.ndarray, sensitivity: float) -> np.ndarray:
        """OPTIMIZED: Vectorized noise addition"""
        noise = np.random.normal(0, sensitivity * self.gaussian_sigma, values.shape)
        self.privacy_budget -= self.epsilon * 0.1
        return values + noise

    def gaussian_noise(self, value: float, sensitivity: float) -> float:
        with self.budget_lock:
            if self.privacy_budget <= 0:
                raise ValueError("Privacy budget exhausted")
            
            # OPTIMIZED: Use precomputed sigma
            noise = np.random.normal(0, sensitivity * self.gaussian_sigma)
            self.privacy_budget -= self.epsilon * 0.1
            self._log_operation('gaussian', sensitivity, self.epsilon * 0.1)
            
            return value + noise
    
    def laplace_noise(self, value: float, sensitivity: float) -> float:
        with self.budget_lock:
            if self.privacy_budget <= 0:
                raise ValueError("Privacy budget exhausted")
            
            # OPTIMIZED: Use precomputed scale
            noise = np.random.laplace(0, sensitivity * self.laplace_scale)
            self.privacy_budget -= self.epsilon * 0.2
            self._log_operation('laplace', sensitivity, self.epsilon * 0.2)
            
            return value + noise
    
    def _log_operation(self, mechanism: str, sensitivity: float, epsilon_used: float):
        # OPTIMIZED: Use LRU cache instead of unlimited list
        log_key = f"{mechanism}_{time.time()}"
        self.audit_log[log_key] = {
            'timestamp': time.time(),
            'mechanism': mechanism,
            'sensitivity': sensitivity,
            'epsilon_used': epsilon_used,
            'remaining_budget': self.privacy_budget
        }
    
    def advanced_composition(self, operations: int, target_delta: float) -> float:
        return self.epsilon * np.sqrt(2 * operations * np.log(1 / target_delta))
    
    def get_dp_metrics(self) -> Dict:
        with self.budget_lock:
            total_ops = len(self.audit_log)
            return {
                'remaining_epsilon': self.privacy_budget,
                'total_operations': total_ops,
                'estimated_composition': self.advanced_composition(total_ops, 1e-5),
                'audit_trail_size': total_ops
            }
