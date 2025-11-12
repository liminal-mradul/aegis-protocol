import math
import random
import numpy as np
from typing import Dict, List, Any

class DifferentialPrivacyEngine:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget_used = 0.0
        self.query_history = []
        
    def add_noise_laplace(self, true_value: float, sensitivity: float) -> float:
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        noisy_value = true_value + noise
        self.privacy_budget_used += self.epsilon
        
        self.query_history.append({
            'type': 'laplace', 'true_value': true_value, 'noisy_value': noisy_value,
            'sensitivity': sensitivity, 'epsilon_used': self.epsilon
        })
        
        return noisy_value
    
    def add_noise_gaussian(self, true_value: float, sensitivity: float) -> float:
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma)
        noisy_value = true_value + noise
        self.privacy_budget_used += self.epsilon
        
        self.query_history.append({
            'type': 'gaussian', 'true_value': true_value, 'noisy_value': noisy_value,
            'sensitivity': sensitivity, 'epsilon_used': self.epsilon, 'delta_used': self.delta
        })
        
        return noisy_value
    
    def exponential_mechanism(self, candidates: List[Any], quality_scores: Dict[Any, float], sensitivity: float) -> Any:
        probabilities = {}
        total = 0.0
        
        for candidate in candidates:
            score = quality_scores.get(candidate, 0.0)
            probability = math.exp(self.epsilon * score / (2 * sensitivity))
            probabilities[candidate] = probability
            total += probability
        
        for candidate in probabilities:
            probabilities[candidate] /= total
        
        selected = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]
        self.privacy_budget_used += self.epsilon
        
        self.query_history.append({
            'type': 'exponential', 'candidates': len(candidates), 'selected': selected, 'epsilon_used': self.epsilon
        })
        
        return selected
    
    def check_privacy_budget(self, max_budget: float) -> bool:
        return self.privacy_budget_used <= max_budget
    
    def get_privacy_metrics(self) -> Dict:
        return {
            'epsilon_total': self.epsilon, 'delta': self.delta,
            'privacy_budget_used': self.privacy_budget_used,
            'queries_processed': len(self.query_history),
            'remaining_budget': max(0, self.epsilon - self.privacy_budget_used)
        }
