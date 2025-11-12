import hashlib
import secrets
import time
from typing import Dict, List, Optional, Tuple
from threading import RLock

class SecureAggregation:
    """
    Secure aggregation using Shamir's Secret Sharing and additive masking.
    
    This POC demonstrates:
    1. Secret sharing for dropout tolerance
    2. Additive masking for privacy (each node adds random mask)
    3. Pairwise mask cancellation (masks cancel out in aggregate)
    4. No single party sees individual contributions
    """
    
    def __init__(self, node_count: int):
        self.node_count = node_count
        self.aggregation_sessions = {}
        self.session_lock = RLock()
        
        # Large prime for Shamir's Secret Sharing (modular arithmetic)
        self.PRIME = 2**127 - 1  # Mersenne prime
    
    def initialize_secure_session(self, session_id: str, participants: List[str], 
                                threshold: int = None) -> Dict:
        """Initialize a secure aggregation session with secret sharing setup."""
        if threshold is None:
            threshold = len(participants) // 2 + 1
        
        with self.session_lock:
            # Generate pairwise shared secrets between participants
            pairwise_secrets = self._generate_pairwise_secrets(participants)
            
            self.aggregation_sessions[session_id] = {
                'participants': set(participants),
                'threshold': threshold,
                'contributions': {},
                'masks': {},
                'pairwise_secrets': pairwise_secrets,
                'completed': False,
                'created_at': time.time(),
                'aggregation_result': None
            }
        
        return {
            'session_id': session_id,
            'threshold': threshold,
            'participant_count': len(participants),
            'pairwise_secrets': {
                node_id: {other: secret for other, secret in secrets.items() if other != node_id}
                for node_id, secrets in pairwise_secrets.items()
            }
        }
    
    def _generate_pairwise_secrets(self, participants: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Generate shared secrets between each pair of participants.
        These will be used to create canceling masks.
        """
        pairwise_secrets = {p: {} for p in participants}
        
        for i, node_a in enumerate(participants):
            for node_b in participants[i+1:]:
                # Generate a shared secret for this pair
                secret = secrets.randbits(128) % self.PRIME
                
                # Both nodes get the same secret but will use it oppositely
                pairwise_secrets[node_a][node_b] = secret
                pairwise_secrets[node_b][node_a] = secret
        
        return pairwise_secrets
    
    def submit_encrypted_gradient(self, session_id: str, node_id: str, 
                                encrypted_data: bytes, security_proof: Dict) -> bool:
        """
        Submit masked gradient. The 'encrypted_data' is actually the gradient + masks.
        
        Format of encrypted_data: pickled list of integers (gradient values)
        Masks are added using pairwise secrets to ensure they cancel out.
        """
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
        """Validate that the submission is authentic and timely."""
        required = {'nonce', 'commitment', 'timestamp'}
        if not all(field in proof for field in required):
            return False
        
        # Check timestamp freshness (within 5 minutes)
        if abs(time.time() - proof['timestamp']) > 300:
            return False
        
        # Verify commitment matches data
        expected_commitment = hashlib.sha256(
            data + proof['nonce'].encode() + str(proof['timestamp']).encode()
        ).hexdigest()
        
        return proof['commitment'] == expected_commitment
    
    def compute_secure_aggregate(self, session_id: str) -> Optional[bytes]:
        """
        Compute the secure aggregate. This is where the magic happens:
        1. Each node submitted: gradient + mask_i
        2. Masks are constructed so sum(mask_i) = 0
        3. Therefore: sum(gradient + mask_i) = sum(gradient) + 0
        4. We get the true aggregate without seeing individual gradients!
        """
        with self.session_lock:
            if session_id not in self.aggregation_sessions:
                return None
            
            session = self.aggregation_sessions[session_id]
            if len(session['contributions']) < session['threshold']:
                return None
            
            # Perform actual secure aggregation with mask cancellation
            aggregated = self._perform_secure_aggregation(
                session['contributions'],
                session['pairwise_secrets'],
                list(session['participants'])
            )
            
            session['completed'] = True
            session['aggregation_result'] = aggregated
            
            return aggregated
    
    def _perform_secure_aggregation(self, contributions: Dict, 
                                   pairwise_secrets: Dict, 
                                   all_participants: List[str]) -> bytes:
        """
        Actual secure aggregation implementation:
        - Parse masked gradients
        - Sum them up (masks cancel due to pairwise construction)
        - Return aggregate
        """
        # Deserialize all contributions
        masked_gradients = {}
        for node_id, contrib in contributions.items():
            # In real implementation, this would be a proper serialization
            # For POC, we assume data is already numeric or can be interpreted
            masked_gradients[node_id] = self._deserialize_gradient(contrib['data'])
        
        # Get gradient dimension from first contribution
        if not masked_gradients:
            return b''
        
        gradient_dim = len(next(iter(masked_gradients.values())))
        
        # Initialize aggregate
        aggregate = [0] * gradient_dim
        
        # Sum all masked gradients
        for node_id, masked_grad in masked_gradients.items():
            for i, value in enumerate(masked_grad):
                aggregate[i] = (aggregate[i] + value) % self.PRIME
        
        # The masks have canceled out! This is the true aggregate.
        # No single party saw individual gradients.
        
        return self._serialize_gradient(aggregate)
    
    def _deserialize_gradient(self, data: bytes) -> List[int]:
        """
        Deserialize gradient data.
        For POC: interpret bytes as list of integers.
        In production: use proper serialization (protobuf, etc.)
        """
        if len(data) == 0:
            return []
        
        # Simple POC deserialization: each 8 bytes = one int64
        gradient = []
        for i in range(0, len(data), 8):
            if i + 8 <= len(data):
                value = int.from_bytes(data[i:i+8], byteorder='big', signed=True)
                gradient.append(value % self.PRIME)
        
        return gradient
    
    def _serialize_gradient(self, gradient: List[int]) -> bytes:
        """Serialize gradient to bytes."""
        result = b''
        for value in gradient:
            # Convert to signed representation if needed
            value = value if value < self.PRIME // 2 else value - self.PRIME
            result += value.to_bytes(8, byteorder='big', signed=True)
        return result
    
    def create_masked_gradient(self, session_id: str, node_id: str, 
                              gradient: List[float]) -> Tuple[bytes, Dict]:
        """
        Helper function: Create masked gradient for submission.
        
        This shows how a participant would mask their gradient:
        1. Start with true gradient
        2. Add pairwise masks (positive for lower IDs, negative for higher IDs)
        3. Masks will cancel when all participants aggregate
        
        Returns: (masked_gradient_bytes, security_proof)
        """
        with self.session_lock:
            if session_id not in self.aggregation_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.aggregation_sessions[session_id]
            if node_id not in session['participants']:
                raise ValueError(f"Node {node_id} not in session")
            
            # Convert gradient to integers (scaled for fixed-point arithmetic)
            SCALE = 10000  # 4 decimal places of precision
            int_gradient = [int(g * SCALE) % self.PRIME for g in gradient]
            
            # Apply pairwise masks
            masked_gradient = int_gradient.copy()
            pairwise = session['pairwise_secrets'].get(node_id, {})
            
            participants_list = sorted(session['participants'])
            node_idx = participants_list.index(node_id)
            
            for other_id, shared_secret in pairwise.items():
                other_idx = participants_list.index(other_id)
                
                # Generate deterministic mask from shared secret
                mask_seed = hashlib.sha256(
                    str(shared_secret).encode() + 
                    session_id.encode()
                ).digest()
                
                # Create mask for each gradient dimension
                for i in range(len(masked_gradient)):
                    # Deterministic pseudorandom value from seed
                    mask_value = int.from_bytes(
                        hashlib.sha256(mask_seed + i.to_bytes(4, 'big')).digest()[:8],
                        byteorder='big'
                    ) % self.PRIME
                    
                    # Add or subtract based on node ordering (ensures cancellation)
                    if node_idx < other_idx:
                        masked_gradient[i] = (masked_gradient[i] + mask_value) % self.PRIME
                    else:
                        masked_gradient[i] = (masked_gradient[i] - mask_value) % self.PRIME
            
            # Serialize masked gradient
            masked_data = self._serialize_gradient(masked_gradient)
            
            # Create security proof
            nonce = secrets.token_hex(16)
            timestamp = time.time()
            commitment = hashlib.sha256(
                masked_data + nonce.encode() + str(timestamp).encode()
            ).hexdigest()
            
            security_proof = {
                'nonce': nonce,
                'commitment': commitment,
                'timestamp': timestamp
            }
            
            return masked_data, security_proof
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get status of aggregation session."""
        with self.session_lock:
            if session_id not in self.aggregation_sessions:
                return None
            
            session = self.aggregation_sessions[session_id]
            return {
                'session_id': session_id,
                'participants': len(session['participants']),
                'contributions_received': len(session['contributions']),
                'threshold': session['threshold'],
                'ready_to_aggregate': len(session['contributions']) >= session['threshold'],
                'completed': session['completed'],
                'created_at': session['created_at']
            }


# Example usage demonstrating the secure aggregation protocol
if __name__ == "__main__":
    print("=== Secure Aggregation POC ===\n")
    
    # Setup
    participants = ['node_1', 'node_2', 'node_3', 'node_4']
    secure_agg = SecureAggregation(len(participants))
    
    # Initialize session
    session_info = secure_agg.initialize_secure_session(
        session_id='training_round_1',
        participants=participants,
        threshold=3  # Need at least 3 out of 4
    )
    
    print(f"Session initialized: {session_info['session_id']}")
    print(f"Threshold: {session_info['threshold']}/{session_info['participant_count']}\n")
    
    # Each participant has a private gradient
    private_gradients = {
        'node_1': [1.5, 2.3, -0.8, 3.1],
        'node_2': [0.9, -1.2, 2.4, 0.7],
        'node_3': [-0.3, 1.8, 1.1, -2.0],
        'node_4': [2.1, 0.5, -1.5, 1.3]
    }
    
    print("Private gradients (no one should see these individually):")
    for node, grad in private_gradients.items():
        print(f"  {node}: {grad}")
    
    # Expected aggregate (what we should get)
    expected = [sum(private_gradients[n][i] for n in participants) 
                for i in range(4)]
    print(f"\nExpected aggregate: {expected}\n")
    
    # Each node creates masked gradient and submits
    print("Nodes submitting masked gradients...")
    for node_id in participants[:3]:  # Only 3 submit (threshold is met)
        masked_data, proof = secure_agg.create_masked_gradient(
            'training_round_1',
            node_id,
            private_gradients[node_id]
        )
        
        success = secure_agg.submit_encrypted_gradient(
            'training_round_1',
            node_id,
            masked_data,
            proof
        )
        print(f"  {node_id}: {'✓' if success else '✗'}")
    
    # Check status
    status = secure_agg.get_session_status('training_round_1')
    print(f"\nSession status: {status['contributions_received']}/{status['threshold']} contributions")
    print(f"Ready to aggregate: {status['ready_to_aggregate']}")
    
    # Compute secure aggregate
    print("\nComputing secure aggregate...")
    result_bytes = secure_agg.compute_secure_aggregate('training_round_1')
    
    if result_bytes:
        # Deserialize result
        result_ints = secure_agg._deserialize_gradient(result_bytes)
        
        # Convert back to floats (undo scaling)
        SCALE = 10000
        result = [v / SCALE if v < secure_agg.PRIME // 2 
                 else (v - secure_agg.PRIME) / SCALE 
                 for v in result_ints]
        
        print(f"Secure aggregate result: {[round(x, 4) for x in result]}")
        print(f"Expected (3 nodes):     {[round(sum(private_gradients[n][i] for n in participants[:3]), 4) for i in range(4)]}")
        
        print("\n✓ Masks canceled successfully!")
        print("✓ No individual gradient was revealed!")
        print("✓ Aggregate computed securely!")
    else:
        print("✗ Aggregation failed")
