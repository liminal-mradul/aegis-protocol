import json
import hashlib
from typing import Dict, List, Optional
from datetime import datetime
from threading import RLock

class ModelVersion:
    def __init__(self, model_id: str, version: int, checksum: str, metadata: Dict):
        self.model_id = model_id
        self.version = version
        self.checksum = checksum
        self.metadata = metadata
        self.created_at = datetime.now()
        self.shard_locations = []

class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.model_versions = {}
        self.registry_lock = RLock()
    
    def register_model(self, model_id: str, initial_version: ModelVersion) -> bool:
        with self.registry_lock:
            if model_id in self.models:
                return False
            
            self.models[model_id] = {
                'current_version': initial_version.version,
                'versions': [initial_version],
                'created_at': datetime.now()
            }
            self.model_versions[f"{model_id}_{initial_version.version}"] = initial_version
            return True
    
    def create_new_version(self, model_id: str, parent_version: int, 
                          new_checksum: str, metadata: Dict) -> Optional[ModelVersion]:
        with self.registry_lock:
            if model_id not in self.models:
                return None
            
            current_version = self.models[model_id]['current_version']
            new_version = current_version + 1
            
            version = ModelVersion(model_id, new_version, new_checksum, metadata)
            self.models[model_id]['versions'].append(version)
            self.models[model_id]['current_version'] = new_version
            self.model_versions[f"{model_id}_{new_version}"] = version
            
            return version
    
    def get_model_shards(self, model_id: str, version: int) -> List[str]:
        key = f"{model_id}_{version}"
        if key in self.model_versions:
            return self.model_versions[key].shard_locations
        return []
    
    def validate_model_integrity(self, model_id: str, version: int, 
                               shard_data: Dict[str, bytes]) -> bool:
        key = f"{model_id}_{version}"
        if key not in self.model_versions:
            return False
        
        expected_checksum = self.model_versions[key].checksum
        return self._compute_shards_checksum(shard_data) == expected_checksum
    
    def _compute_shards_checksum(self, shard_data: Dict[str, bytes]) -> str:
        combined = b"".join(shard_data.values())
        return hashlib.sha256(combined).hexdigest()
