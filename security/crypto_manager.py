import os
import hashlib
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from typing import Dict, Tuple, Optional
import base64

class CryptoManager:
    def __init__(self, key_storage_path: str = "./keys"):
        self.key_storage_path = key_storage_path
        self.node_keys = {}
        self.session_keys = {}
        self._ensure_key_directory()
    
    def _ensure_key_directory(self):
        os.makedirs(self.key_storage_path, mode=0o700, exist_ok=True)
    
    def generate_node_identity(self, node_id: str) -> Tuple[str, str]:
        private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
        public_key = private_key.public_key()
        
        priv_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        pub_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        key_path = os.path.join(self.key_storage_path, f"{node_id}.key")
        with open(key_path, 'wb') as f:
            f.write(priv_pem)
        os.chmod(key_path, 0o600)
        
        self.node_keys[node_id] = {
            'private_key': private_key,
            'public_key': public_key,
            'public_pem': pub_pem.decode()
        }
        
        return priv_pem.decode(), pub_pem.decode()
    
    def sign_message(self, node_id: str, message: bytes) -> str:
        if node_id not in self.node_keys:
            raise ValueError(f"No keys found for node {node_id}")
        
        private_key = self.node_keys[node_id]['private_key']
        signature = private_key.sign(
            message,
            ec.ECDSA(hashes.SHA384())
        )
        
        return base64.b64encode(signature).decode()
    
    def verify_signature(self, public_key_pem: str, message: bytes, signature: str) -> bool:
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode(),
                backend=default_backend()
            )
            
            sig_bytes = base64.b64decode(signature)
            public_key.verify(
                sig_bytes,
                message,
                ec.ECDSA(hashes.SHA384())
            )
            return True
        except Exception:
            return False
    
    def derive_session_key(self, node_a: str, node_b: str) -> bytes:
        shared_secret = self._compute_shared_secret(node_a, node_b)
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'aegis-session-key',
            backend=default_backend()
        )
        
        session_key = hkdf.derive(shared_secret)
        session_id = f"{node_a}-{node_b}"
        self.session_keys[session_id] = session_key
        
        return session_key
    
    def _compute_shared_secret(self, node_a: str, node_b: str) -> bytes:
        if node_a not in self.node_keys or node_b not in self.node_keys:
            raise ValueError("Both nodes must have generated keys")
        
        private_key_a = self.node_keys[node_a]['private_key']
        public_key_b = self.node_keys[node_b]['public_key']
        
        return private_key_a.exchange(ec.ECDH(), public_key_b)
