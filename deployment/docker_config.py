from typing import Dict, List

class DockerConfig:
    def __init__(self):
        self.base_image = "python:3.9-slim"
        self.working_dir = "/app"
        
    def generate_dockerfile(self, requirements: List[str] = None) -> str:
        if requirements is None:
            requirements = [
                "grpcio==1.60.0",
                "numpy==1.24.3", 
                "cryptography==41.0.8",
                "fastapi==0.104.1"
            ]
        
        requirements_str = "\n".join(requirements)
        
        dockerfile = f"""
FROM {self.base_image}

WORKDIR {self.working_dir}

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 aegis
USER aegis

EXPOSE 8000 8001

CMD ["python", "main.py"]
"""
        return dockerfile
    
    def generate_docker_compose(self, node_count: int = 3) -> Dict:
        services = {}
        
        for i in range(node_count):
            services[f"node_{i}"] = {
                'build': '.',
                'ports': [
                    f"{8000 + i}:8000",
                    f"{8001 + i}:8001"
                ],
                'environment': [
                    f'NODE_ID=node_{i}',
                    f'TOTAL_NODES={node_count}',
                    'AEGIS_ENV=production'
                ],
                'volumes': [
                    './config:/app/config',
                    './logs:/app/logs'
                ],
                'healthcheck': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3
                }
            }
        
        return {
            'version': '3.8',
            'services': services
        }
