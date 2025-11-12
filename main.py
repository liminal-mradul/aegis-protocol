#!/usr/bin/env python3
"""
Aegis Constellation - Main Entry Point for Distributed Nodes
Run this on each machine to join the network
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.node import ProductionNode
from security.crypto_manager import CryptoManager

# Try to import discovery, create stub if not available
try:
    from discovery import NodeDiscovery
except ImportError:
    print("WARNING: discovery.py not found. Using stub implementation.")
    class NodeDiscovery:
        def __init__(self, *args, **kwargs):
            self.known_nodes = {}
        async def start(self):
            await asyncio.sleep(float('inf'))
        async def stop(self):
            pass
        def get_known_nodes(self):
            return []
        def add_seed_nodes(self, seeds):
            pass
        def enable_multicast_discovery(self):
            pass
        def set_registry_server(self, url):
            pass

# Import FastAPI app separately to handle errors better
try:
    from api.rest_api import app
except Exception as e:
    print(f"WARNING: Failed to import API: {e}")
    print("Creating stub FastAPI app...")
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/")
    async def root():
        return {"message": "Aegis Node (Limited Mode)", "status": "operational"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}

from config.configuration import ConfigurationManager
import uvicorn


class AegisNode:
    """Main Aegis Node - Runs all components"""
    
    def __init__(self, node_id: str, config_path: str = "./config"):
        self.node_id = node_id
        self.config_manager = ConfigurationManager(config_path)
        
        # Load configuration
        config = self.config_manager.configs
        self.base_port = config.get('base_port', 8000)
        self.total_nodes = config.get('node_count', 3)
        
        # Calculate ports for this node
        node_index = int(node_id.split('_')[-1]) if '_' in node_id else 0
        self.http_port = self.base_port + node_index
        self.grpc_port = self.http_port + 1000
        
        # Initialize components
        self.crypto_manager = CryptoManager()
        self.production_node = ProductionNode(
            node_id=node_id,
            total_nodes=self.total_nodes,
            port=self.grpc_port
        )
        
        # Generate node identity
        priv_key, pub_key = self.crypto_manager.generate_node_identity(node_id)
        
        # Initialize discovery
        self.discovery = NodeDiscovery(
            node_id=node_id,
            host="0.0.0.0",
            port=self.http_port,
            grpc_port=self.grpc_port,
            public_key=pub_key,
            capabilities=["training", "inference", "consensus"]
        )
        
        self.logger = logging.getLogger(f"AegisNode-{node_id}")
        self.running = False
        
        # API server
        self.api_server = None
    
    async def start(self):
        """Start all node components"""
        self.logger.info(f"Starting Aegis Node: {self.node_id}")
        self.logger.info(f"  HTTP Port: {self.http_port}")
        self.logger.info(f"  gRPC Port: {self.grpc_port}")
        
        self.running = True
        
        # Configure discovery based on environment
        await self._configure_discovery()
        
        # Start core node
        await self.production_node.initialize_node()
        self.logger.info("✓ Core node initialized")
        
        # Start discovery
        discovery_task = asyncio.create_task(self.discovery.start())
        self.logger.info("✓ Discovery service started")
        
        # Start API server
        api_task = asyncio.create_task(self._start_api_server())
        self.logger.info(f"✓ API server starting on port {self.http_port}")
        
        # Wait for discovery to find nodes
        await asyncio.sleep(5)
        nodes = self.discovery.get_known_nodes()
        self.logger.info(f"✓ Discovered {len(nodes)} peer nodes")
        
        # Connect to discovered nodes
        await self._connect_to_peers()
        
        self.logger.info(f"🚀 Node {self.node_id} is fully operational!")
        
        # Keep running
        try:
            await asyncio.gather(discovery_task, api_task)
        except asyncio.CancelledError:
            self.logger.info("Shutting down...")
    
    async def _configure_discovery(self):
        """Configure node discovery based on environment"""
        # Try to get seed nodes from environment
        seed_nodes_env = os.getenv('AEGIS_SEED_NODES', '')
        if seed_nodes_env:
            seeds = []
            for seed in seed_nodes_env.split(','):
                try:
                    host, http_port, grpc_port = seed.strip().split(':')
                    seeds.append((host, int(http_port), int(grpc_port)))
                except:
                    self.logger.warning(f"Invalid seed node format: {seed}")
            
            if seeds:
                self.discovery.add_seed_nodes(seeds)
                self.logger.info(f"Added {len(seeds)} seed nodes from environment")
        
        # Enable multicast for local network discovery
        self.discovery.enable_multicast_discovery()
        
        # Try to use registry if specified
        registry_url = os.getenv('AEGIS_REGISTRY', None)
        if registry_url:
            self.discovery.set_registry_server(registry_url)
            self.logger.info(f"Using registry: {registry_url}")
    
    async def _start_api_server(self):
        """Start the REST API server"""
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.http_port,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def _connect_to_peers(self):
        """Connect to discovered peer nodes"""
        nodes = self.discovery.get_known_nodes()
        
        for node in nodes:
            if node.node_id != self.node_id:
                try:
                    # Register with RPC service
                    self.production_node.rpc_service.connected_nodes[node.node_id] = {
                        'address': node.host,
                        'port': node.grpc_port,
                        'public_key': node.public_key,
                        'capabilities': node.capabilities,
                        'avg_latency': 0.0,
                        'success_rate': 0.5
                    }
                    
                    self.logger.info(f"Registered peer: {node.node_id} at {node.host}:{node.grpc_port}")
                
                except Exception as e:
                    self.logger.warning(f"Failed to connect to {node.node_id}: {e}")
    
    async def stop(self):
        """Gracefully stop the node"""
        self.logger.info("Stopping node...")
        self.running = False
        
        await self.discovery.stop()
        await self.production_node.shutdown()
        
        self.logger.info("Node stopped")


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('aegis_node.log')
        ]
    )


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Aegis Constellation Node')
    parser.add_argument('--node-id', required=True, help='Unique node identifier')
    parser.add_argument('--config', default='./config', help='Config directory path')
    parser.add_argument('--seeds', nargs='*', help='Seed nodes: host:http_port:grpc_port')
    parser.add_argument('--registry', help='Registry server URL')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger("main")
    
    # Set environment variables if provided
    if args.seeds:
        os.environ['AEGIS_SEED_NODES'] = ','.join(args.seeds)
    if args.registry:
        os.environ['AEGIS_REGISTRY'] = args.registry
    
    # Create and start node
    node = AegisNode(args.node_id, args.config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(node.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await node.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await node.stop()


if __name__ == "__main__":
    asyncio.run(main())
