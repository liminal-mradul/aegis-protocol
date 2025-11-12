import time
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class ScalingAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"

@dataclass
class ScalingMetrics:
    cpu_usage: float
    memory_usage: float
    network_throughput: float
    request_latency: float
    error_rate: float
    queue_depth: int

class AutoScaler:
    def __init__(self, min_nodes: int = 3, max_nodes: int = 50, 
                 target_cpu: float = 70.0, target_latency: float = 100.0):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.target_cpu = target_cpu
        self.target_latency = target_latency
        self.current_nodes = min_nodes
        self.metrics_history = []
        self.scaling_cooldown = 300
        self.last_scaling_time = 0
        
    async def analyze_and_scale(self, current_metrics: ScalingMetrics) -> ScalingAction:
        self.metrics_history.append(current_metrics)
        
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        if time.time() - self.last_scaling_time < self.scaling_cooldown:
            return ScalingAction.MAINTAIN
        
        action = self._determine_scaling_action(current_metrics)
        
        if action != ScalingAction.MAINTAIN:
            await self._execute_scaling(action)
            self.last_scaling_time = time.time()
        
        return action
    
    def _determine_scaling_action(self, metrics: ScalingMetrics) -> ScalingAction:
        cpu_violation = metrics.cpu_usage > self.target_cpu
        latency_violation = metrics.request_latency > self.target_latency
        high_error_rate = metrics.error_rate > 5.0
        
        scale_up_conditions = cpu_violation or latency_violation or high_error_rate
        
        low_utilization = (metrics.cpu_usage < self.target_cpu * 0.5 and 
                          metrics.memory_usage < 50.0 and
                          metrics.request_latency < self.target_latency * 0.7)
        
        if scale_up_conditions and self.current_nodes < self.max_nodes:
            return ScalingAction.SCALE_UP
        elif low_utilization and self.current_nodes > self.min_nodes:
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.MAINTAIN
    
    async def _execute_scaling(self, action: ScalingAction):
        if action == ScalingAction.SCALE_UP:
            new_count = min(self.current_nodes * 2, self.max_nodes)
            await self._scale_to(new_count)
        elif action == ScalingAction.SCALE_DOWN:
            new_count = max(self.current_nodes // 2, self.min_nodes)
            await self._scale_to(new_count)
    
    async def _scale_to(self, new_count: int):
        scaling_delta = new_count - self.current_nodes
        
        if scaling_delta > 0:
            await self._add_nodes(scaling_delta)
        elif scaling_delta < 0:
            await self._remove_nodes(abs(scaling_delta))
        
        self.current_nodes = new_count
    
    async def _add_nodes(self, count: int):
        for i in range(count):
            new_node_id = f"auto-scale-{int(time.time())}-{i}"
            await self._deploy_new_node(new_node_id)
    
    async def _remove_nodes(self, count: int):
        pass
    
    async def _deploy_new_node(self, node_id: str):
        pass
