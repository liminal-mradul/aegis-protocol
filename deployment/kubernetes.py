import yaml
from typing import Dict, List

class KubernetesDeployment:
    def __init__(self, cluster_name: str, namespace: str = "aegis"):
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.deployment_templates = {}
        
    def generate_node_deployment(self, node_id: str, config: Dict) -> str:
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'aegis-node-{node_id}',
                'namespace': self.namespace,
                'labels': {
                    'app': 'aegis-node',
                    'node-id': node_id
                }
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': 'aegis-node',
                        'node-id': node_id
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'aegis-node',
                            'node-id': node_id
                        },
                        'annotations': {
                            'prometheus.io/scrape': 'true',
                            'prometheus.io/port': '8000'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'aegis-node',
                            'image': 'aegis/node:production',
                            'imagePullPolicy': 'Always',
                            'ports': [
                                {'containerPort': 8000, 'name': 'http'},
                                {'containerPort': 8001, 'name': 'grpc'}
                            ],
                            'env': [
                                {'name': 'NODE_ID', 'value': node_id},
                                {'name': 'TOTAL_NODES', 'value': str(config.get('total_nodes', 3))},
                                {'name': 'K8S_POD_IP', 'valueFrom': {'fieldRef': {'fieldPath': 'status.podIP'}}},
                                {'name': 'LOG_LEVEL', 'value': 'INFO'}
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '1Gi',
                                    'cpu': '500m',
                                    'ephemeral-storage': '10Gi'
                                },
                                'limits': {
                                    'memory': '2Gi',
                                    'cpu': '1000m',
                                    'ephemeral-storage': '20Gi'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': 8000},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/ready', 'port': 8000},
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            },
                            'securityContext': {
                                'runAsNonRoot': True,
                                'runAsUser': 1000,
                                'allowPrivilegeEscalation': False
                            }
                        }],
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000
                        }
                    }
                }
            }
        }
        
        return yaml.dump(deployment, default_flow_style=False)
    
    def generate_service_mesh_config(self) -> Dict:
        return {
            'apiVersion': 'networking.istio.io/v1beta1',
            'kind': 'PeerAuthentication',
            'metadata': {
                'name': 'aegis-mtls',
                'namespace': self.namespace
            },
            'spec': {
                'mtls': {'mode': 'STRICT'}
            }
        }
    
    def generate_monitoring_config(self) -> Dict:
        return {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': 'aegis-nodes',
                'namespace': self.namespace
            },
            'spec': {
                'selector': {
                    'matchLabels': {'app': 'aegis-node'}
                },
                'endpoints': [{
                    'port': 'http',
                    'path': '/metrics',
                    'interval': '30s'
                }]
            }
        }
