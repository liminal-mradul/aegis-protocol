from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import time
import uvicorn

app = FastAPI(
    title="Aegis Constellation API",
    description="Enterprise decentralized computing platform",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== PYDANTIC MODELS (Fixed for v2) =====

class CommitRequest(BaseModel):
    """Request model for code commits"""
    code: str
    message: str
    domain: str
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "def hello(): return 'world'",
                "message": "Initial commit",
                "domain": "ai",
                "tags": ["python", "function"]
            }
        }


class TrainingRequest(BaseModel):
    """Request model for federated training - FIXED: renamed model_config to training_config"""
    training_config: Dict = Field(
        description="Training configuration including model architecture and hyperparameters"
    )
    participants: List[str] = Field(
        description="List of participating node IDs"
    )
    max_rounds: int = Field(
        default=10,
        description="Maximum number of training rounds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "training_config": {
                    "model_type": "neural_network",
                    "layers": [128, 64, 32],
                    "learning_rate": 0.001
                },
                "participants": ["node_1", "node_2", "node_3"],
                "max_rounds": 10
            }
        }


class NodeStatusResponse(BaseModel):
    """Response model for node status"""
    node_id: str
    status: str
    reputation: float
    balance: float
    commits_count: int
    uptime_seconds: float = 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "node_id": "node_1",
                "status": "healthy",
                "reputation": 0.85,
                "balance": 1500.0,
                "commits_count": 42,
                "uptime_seconds": 3600.0
            }
        }


class CommitResponse(BaseModel):
    """Response model for commit submission"""
    commit_id: str
    reward: float
    processing_time_ms: float
    security_level: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "commit_id": "abc123def456",
                "reward": 15.5,
                "processing_time_ms": 45.2,
                "security_level": "high"
            }
        }


class TrainingResponse(BaseModel):
    """Response model for training initiation"""
    session_id: str
    status: str
    estimated_duration_minutes: float = 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "training_session_xyz789",
                "status": "started",
                "estimated_duration_minutes": 30.0
            }
        }


# ===== GLOBAL STATE (will be injected by main.py) =====
# This will be set when the node initializes
_node_instance = None
_start_time = time.time()


def set_node_instance(node):
    """Set the global node instance for API access"""
    global _node_instance
    _node_instance = node


# ===== API ENDPOINTS =====

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Aegis Constellation v5.0 API",
        "status": "operational",
        "uptime_seconds": time.time() - _start_time,
        "node_id": _node_instance.node_id if _node_instance else "unknown",
        "version": "5.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "nodes": "/nodes",
            "commits": "/nodes/{node_id}/commits",
            "training": "/training",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "5.0.0",
        "uptime_seconds": time.time() - _start_time
    }
    
    # Add node-specific health if available
    if _node_instance:
        try:
            node_health = await _node_instance.check_health()
            health_data.update({
                "node_id": _node_instance.node_id,
                "node_health": node_health
            })
        except Exception as e:
            health_data["node_error"] = str(e)
    
    return health_data


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    if _node_instance and _node_instance.is_running:
        return {"ready": True}
    return {"ready": False}


@app.get("/nodes", response_model=List[NodeStatusResponse])
async def get_nodes():
    """Get list of all nodes in the network"""
    if not _node_instance:
        return []
    
    try:
        # Get nodes from discovery service
        if hasattr(_node_instance, 'discovery'):
            discovered_nodes = _node_instance.discovery.get_known_nodes()
            
            return [
                NodeStatusResponse(
                    node_id=node.node_id,
                    status="healthy" if time.time() - node.last_seen < 120 else "stale",
                    reputation=node.reputation,
                    balance=0.0,  # TODO: integrate with cryptocurrency module
                    commits_count=0,  # TODO: integrate with commit tracking
                    uptime_seconds=time.time() - node.last_seen
                )
                for node in discovered_nodes
            ]
        
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get nodes: {str(e)}")


@app.get("/nodes/{node_id}", response_model=NodeStatusResponse)
async def get_node(node_id: str):
    """Get specific node information"""
    if not _node_instance:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    # If requesting info about this node
    if node_id == _node_instance.node_id:
        health = await _node_instance.check_health()
        return NodeStatusResponse(
            node_id=node_id,
            status="healthy",
            reputation=health.get('reputation', 0.5),
            balance=health.get('balance', 0.0),
            commits_count=health.get('commits', 0),
            uptime_seconds=time.time() - _start_time
        )
    
    # Check discovered nodes
    if hasattr(_node_instance, 'discovery'):
        node_info = _node_instance.discovery.get_node_by_id(node_id)
        if node_info:
            return NodeStatusResponse(
                node_id=node_info.node_id,
                status="healthy" if time.time() - node_info.last_seen < 120 else "stale",
                reputation=node_info.reputation,
                balance=0.0,
                commits_count=0,
                uptime_seconds=time.time() - node_info.last_seen
            )
    
    raise HTTPException(status_code=404, detail=f"Node {node_id} not found")


@app.post("/nodes/{node_id}/commits", response_model=CommitResponse)
async def submit_commit(node_id: str, commit: CommitRequest):
    """Submit a code commit to the network"""
    if not _node_instance:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    try:
        # Prepare commit data
        commit_data = {
            'code': commit.code,
            'message': commit.message,
            'domain': commit.domain,
            'tags': commit.tags,
            'complexity': len(commit.code) / 100.0  # Simple complexity metric
        }
        
        # Submit to node
        result = await _node_instance.submit_secure_commit(commit_data)
        
        return CommitResponse(
            commit_id=result['commit_id'],
            reward=result['reward'],
            processing_time_ms=result['processing_time_ms'],
            security_level=result['security_level']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training", response_model=TrainingResponse)
async def start_training(training: TrainingRequest):
    """Start a federated training session"""
    if not _node_instance:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    try:
        # Generate session ID
        session_id = f"training_{int(time.time())}_{_node_instance.node_id}"
        
        # Initiate training through node
        result = _node_instance.initiate_secure_training(
            model_id=session_id,
            participants=training.participants
        )
        
        # Estimate duration based on rounds and participants
        estimated_minutes = training.max_rounds * len(training.participants) * 2.0 / 60.0
        
        return TrainingResponse(
            session_id=result['session_id'],
            status="started",
            estimated_duration_minutes=estimated_minutes
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/{session_id}")
async def get_training_status(session_id: str):
    """Get status of a training session"""
    if not _node_instance:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    # Check if session exists in model registry
    if hasattr(_node_instance, 'model_registry') and session_id in _node_instance.model_registry:
        session_info = _node_instance.model_registry[session_id]
        return {
            "session_id": session_id,
            "status": session_info.get('status', 'unknown'),
            "participants": session_info.get('participants', []),
            "created_at": session_info.get('created_at', 0)
        }
    
    raise HTTPException(status_code=404, detail=f"Training session {session_id} not found")


@app.get("/metrics")
async def get_metrics():
    """Get node metrics for monitoring"""
    if not _node_instance:
        return {"error": "Node not initialized"}
    
    metrics = {
        "timestamp": time.time(),
        "uptime_seconds": time.time() - _start_time,
        "node_id": _node_instance.node_id
    }
    
    try:
        # Get node health metrics
        health = await _node_instance.check_health()
        metrics.update(health)
        
        # Get network metrics if available
        if hasattr(_node_instance, 'rpc_service'):
            network_metrics = _node_instance.rpc_service.get_network_metrics()
            metrics['network'] = network_metrics
        
        # Get discovery metrics if available
        if hasattr(_node_instance, 'discovery'):
            metrics['discovered_nodes'] = len(_node_instance.discovery.get_known_nodes())
    
    except Exception as e:
        metrics['error'] = str(e)
    
    return metrics


@app.get("/sla")
async def get_sla_compliance():
    """Get SLA compliance metrics"""
    # TODO: Implement actual SLA tracking
    return {
        "sla_target": 99.9,
        "current_uptime": 99.95,
        "status": "compliant",
        "last_incident": None,
        "mtbf_hours": 720.0,  # Mean time between failures
        "mttr_minutes": 5.0    # Mean time to recovery
    }


@app.get("/consensus/status")
async def get_consensus_status():
    """Get Byzantine consensus status"""
    if not _node_instance or not hasattr(_node_instance, 'bft_engine'):
        raise HTTPException(status_code=503, detail="Consensus not available")
    
    return {
        "enabled": True,
        "node_count": len(_node_instance.bft_engine.node_states),
        "fault_tolerance": f"{int(_node_instance.bft_engine.fault_threshold * 100)}%",
        "status": "operational"
    }


# ===== UTILITY FUNCTIONS =====

def run_api_server(host: str = "0.0.0.0", port: int = 8000, node_instance=None):
    """Run the API server with optional node instance"""
    if node_instance:
        set_node_instance(node_instance)
    
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_api_server()
