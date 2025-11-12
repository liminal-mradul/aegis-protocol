from typing import Dict, List, Optional, Any
from threading import Lock
import time
import json
import zlib
import pickle
from collections import OrderedDict
from dataclasses import dataclass

@dataclass
class OptimizedRecord:
    key: str
    value: bytes  # Compressed storage
    timestamp: float
    size: int
    access_count: int

class DistributedDB:
    def __init__(self, max_memory_mb: int = 100, compression_level: int = 6):
        # OPTIMIZED: Memory management
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_level = compression_level
        
        # OPTIMIZED: LRU ordering with memory tracking
        self.data = OrderedDict()
        self.indices = {}
        
        # OPTIMIZED: Memory management
        self.current_memory = 0
        self.hit_count = 0
        self.miss_count = 0
        
        # OPTIMIZED: Fine-grained locking
        self.global_lock = Lock()
        self.locks = {}
        
    def set(self, key: str, value: Dict, node_id: str = None) -> bool:
        # OPTIMIZED: Compression for large values
        serialized = pickle.dumps(value)
        
        if len(serialized) > 100:  # Only compress larger values
            compressed_value = zlib.compress(serialized, self.compression_level)
            stored_value = compressed_value
            stored_size = len(compressed_value)
        else:
            stored_value = serialized
            stored_size = len(serialized)
        
        with self.global_lock:
            # OPTIMIZED: Memory management
            if self.current_memory + stored_size > self.max_memory_bytes:
                self._evict_until_fit(stored_size)
            
            if key not in self.locks:
                self.locks[key] = Lock()
            
            with self.locks[key]:
                record = OptimizedRecord(
                    key=key,
                    value=stored_value,
                    timestamp=time.time(),
                    size=stored_size,
                    access_count=0
                )
                
                self.data[key] = {
                    'value': record,
                    'timestamp': time.time(),
                    'node_id': node_id
                }
                
                # Move to end for LRU
                self.data.move_to_end(key)
                self.current_memory += stored_size
                
                return True
    
    def get(self, key: str) -> Optional[Dict]:
        with self.global_lock:
            if key not in self.data:
                self.miss_count += 1
                return None
            
            # OPTIMIZED: LRU update
            self.data.move_to_end(key)
            record = self.data[key]['value']
            record.access_count += 1
            record.timestamp = time.time()
            
            with self.locks.get(key, Lock()):
                try:
                    # Decompress if needed
                    if record.value[:2] == b'x\x9c':  # zlib header
                        decompressed = zlib.decompress(record.value)
                        value = pickle.loads(decompressed)
                    else:
                        value = pickle.loads(record.value)
                    
                    self.hit_count += 1
                    return value
                except:
                    return None
    
    def _evict_until_fit(self, required_size: int):
        """OPTIMIZED: LRU eviction"""
        while self.data and (self.current_memory + required_size > self.max_memory_bytes):
            key, data = self.data.popitem(last=False)  # Remove from beginning (LRU)
            record = data['value']
            self.current_memory -= record.size
            
            # Remove lock
            if key in self.locks:
                del self.locks[key]
    
    def delete(self, key: str) -> bool:
        with self.global_lock:
            if key not in self.data:
                return False
            
            with self.locks[key]:
                record = self.data[key]['value']
                self.current_memory -= record.size
                del self.data[key]
                del self.locks[key]
                return True
    
    def query(self, pattern: str) -> List[Dict]:
        """OPTIMIZED: Pattern querying"""
        results = []
        with self.global_lock:
            for key, data in self.data.items():
                if pattern in key:
                    value = self.get(key)
                    if value is not None:
                        results.append({
                            'key': key,
                            'value': value,
                            'timestamp': data['timestamp'],
                            'node_id': data['node_id']
                        })
        return results
    
    def get_stats(self) -> Dict:
        """OPTIMIZED: Statistics with hit rate"""
        with self.global_lock:
            hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
            
            return {
                'total_keys': len(self.data),
                'total_locks': len(self.locks),
                'memory_usage_mb': self.current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'total_hits': self.hit_count,
                'total_misses': self.miss_count
            }
