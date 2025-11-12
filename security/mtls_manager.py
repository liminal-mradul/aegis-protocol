import ssl
from typing import Dict, Tuple
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from datetime import datetime, timedelta

class mTLSManager:
    def __init__(self, crypto_manager):
        self.crypto_manager = crypto_manager
        self.certificates = {}
    
    def generate_certificate(self, node_id: str, hostname: str) -> Tuple[str, str]:
        private_key = self.crypto_manager.node_keys[node_id]['private_key']
        
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aegis Network"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())
        
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        self.certificates[node_id] = cert_pem.decode()
        return cert_pem.decode(), key_pem.decode()
    
    def create_ssl_context(self, node_id: str, ca_cert: str) -> ssl.SSLContext:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.verify_mode = ssl.CERT_REQUIRED
        
        cert_pem, key_pem = self.generate_certificate(node_id, f"node-{node_id}.aegis")
        
        context.load_cert_chain(
            certfile=self._pem_to_file(cert_pem),
            keyfile=self._pem_to_file(key_pem)
        )
        
        context.load_verify_locations(cafile=self._pem_to_file(ca_cert))
        context.check_hostname = False
        
        return context
    
    def _pem_to_file(self, pem_data: str) -> str:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pem') as f:
            f.write(pem_data)
            return f.name
