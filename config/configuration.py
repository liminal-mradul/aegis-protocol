import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    host: str
    port: int
    username: str
    password: str
    database: str
    pool_size: int = 10
    timeout: int = 30

@dataclass
class SecurityConfig:
    enable_mtls: bool = True
    enable_audit: bool = True
    require_2fa: bool = False
    session_timeout: int = 3600
    max_login_attempts: int = 5

@dataclass
class MonitoringConfig:
    enable_prometheus: bool = True
    enable_grafana: bool = True
    metrics_port: int = 9090
    scrape_interval: int = 30

class ConfigurationManager:
    def __init__(self, config_path: str = "./config"):
        self.config_path = config_path
        self.environment = Environment(os.getenv('AEGIS_ENV', 'development'))
        self.configs = {}
        self._load_configurations()
    
    def _load_configurations(self):
        base_config = self._load_yaml_file('base.yaml')
        env_config = self._load_yaml_file(f'{self.environment.value}.yaml')
        
        self.configs = self._deep_merge(base_config, env_config)
        
        secrets_config = self._load_secrets()
        self.configs = self._deep_merge(self.configs, secrets_config)
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.config_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _load_secrets(self) -> Dict[str, Any]:
        secrets = {}
        prefix = 'AEGIS_'
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                secrets[config_key] = value
        
        return secrets
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        result = base.copy()
        
        for key, value in update.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_database_config(self) -> DatabaseConfig:
        db_config = self.configs.get('database', {})
        return DatabaseConfig(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            username=db_config.get('username', 'aegis'),
            password=db_config.get('password', ''),
            database=db_config.get('database', 'aegis'),
            pool_size=db_config.get('pool_size', 10),
            timeout=db_config.get('timeout', 30)
        )
    
    def get_security_config(self) -> SecurityConfig:
        security_config = self.configs.get('security', {})
        return SecurityConfig(
            enable_mtls=security_config.get('enable_mtls', True),
            enable_audit=security_config.get('enable_audit', True),
            require_2fa=security_config.get('require_2fa', False),
            session_timeout=security_config.get('session_timeout', 3600),
            max_login_attempts=security_config.get('max_login_attempts', 5)
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        monitoring_config = self.configs.get('monitoring', {})
        return MonitoringConfig(
            enable_prometheus=monitoring_config.get('enable_prometheus', True),
            enable_grafana=monitoring_config.get('enable_grafana', True),
            metrics_port=monitoring_config.get('metrics_port', 9090),
            scrape_interval=monitoring_config.get('scrape_interval', 30)
        )
    
    def hot_reload(self):
        self._load_configurations()
