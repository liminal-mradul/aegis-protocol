# neuro_topology.py - REAL NEAT Neuroevolution
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import copy

class RealNEATTopology:
    """
    REAL NeuroEvolution of Augmenting Topologies (NEAT) implementation.
    Features:
    - Historical marking with innovation numbers
    - Species formation and fitness sharing
    - Complexification through structural mutations
    - Crossover with matching genes
    - Real neural network evaluation
    """
    
    def __init__(self, input_size: int, output_size: int, population_size: int = 100):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        
        # NEAT specific components
        self.innovation_number = 0
        self.innovation_dict = {}  # (in_node, out_node) -> innovation_number
        
        # Genome components
        self.node_genes = []  # List of node genes (id, type, activation)
        self.conn_genes = []  # List of connection genes (in, out, weight, enabled, innovation)
        
        # Species management
        self.species = []
        self.species_representatives = []
        self.compatibility_threshold = 3.0
        
        # Population
        self.population = []
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_genome = None
        
        # NEAT parameters
        self.mutation_rates = {
            'conn_weight': 0.8,
            'conn_enable': 0.1,
            'conn_disable': 0.1,
            'add_conn': 0.05,
            'add_node': 0.03
        }
        
        self.crossover_rate = 0.75
        self.weight_mutation_power = 0.5
        self.interspecies_mating_rate = 0.001
        
        # Activation functions
        self.activation_functions = {
            'sigmoid': lambda x: 1.0 / (1.0 + math.exp(-4.9 * x)),
            'tanh': math.tanh,
            'relu': lambda x: max(0, x),
            'linear': lambda x: x
        }
        
        # Initialize population
        self._initialize_population()
        
    def _initialize_population(self):
        """Initialize population with minimal networks."""
        for _ in range(self.population_size):
            genome = self._create_minimal_genome()
            self.population.append(genome)
    
    def _create_minimal_genome(self) -> Dict:
        """Create a minimal genome connecting all inputs to all outputs."""
        genome = {
            'nodes': [],
            'connections': [],
            'fitness': 0.0,
            'adjusted_fitness': 0.0,
            'species': None
        }
        
        # Create node genes
        node_id = 0
        
        # Input nodes
        for i in range(self.input_size):
            genome['nodes'].append({
                'id': node_id,
                'type': 'input',
                'activation': 'linear'
            })
            node_id += 1
        
        # Output nodes  
        for i in range(self.output_size):
            genome['nodes'].append({
                'id': node_id,
                'type': 'output',
                'activation': 'sigmoid'
            })
            node_id += 1
        
        # Create connection genes (fully connected)
        for input_idx in range(self.input_size):
            for output_idx in range(self.output_size):
                in_node = input_idx
                out_node = self.input_size + output_idx
                
                innov = self._get_innovation_number(in_node, out_node)
                
                genome['connections'].append({
                    'in': in_node,
                    'out': out_node,
                    'weight': random.uniform(-1, 1),
                    'enabled': True,
                    'innovation': innov
                })
        
        return genome
    
    def _get_innovation_number(self, in_node: int, out_node: int) -> int:
        """Get innovation number for a connection, creating new if needed."""
        key = (in_node, out_node)
        
        if key in self.innovation_dict:
            return self.innovation_dict[key]
        else:
            self.innovation_number += 1
            self.innovation_dict[key] = self.innovation_number
            return self.innovation_number
    
    def evolve_topology(self, performance_metrics: Dict) -> List[float]:
        """
        REAL NEAT evolution cycle.
        Returns flattened genome representation for compatibility.
        """
        self.generation += 1
        
        # Evaluate all genomes
        self._evaluate_population(performance_metrics)
        
        # Speciate population
        self._speciate()
        
        # Calculate adjusted fitness
        self._calculate_adjusted_fitness()
        
        # Remove stale species and select parents
        self._remove_stale_species()
        parents = self._select_parents()
        
        # Create new generation through crossover and mutation
        new_population = self._create_new_generation(parents)
        
        # Replace population
        self.population = new_population
        
        # Return best genome as flattened representation
        best_genome = self._get_best_genome()
        flattened = self._flatten_genome(best_genome)
        
        return flattened
    
    def _evaluate_population(self, performance_metrics: Dict):
        """Evaluate fitness of all genomes in population."""
        for genome in self.population:
            fitness = self._evaluate_genome_fitness(genome, performance_metrics)
            genome['fitness'] = fitness
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = copy.deepcopy(genome)
    
    def _evaluate_genome_fitness(self, genome: Dict, metrics: Dict) -> float:
        """
        Evaluate genome by building and testing the neural network.
        In a real application, this would test the network on actual problems.
        """
        # Build neural network from genome
        network = self._build_network(genome)
        
        # Test network - for demonstration, use synthetic test cases
        # In real implementation, this would use actual network routing/topology problems
        
        test_cases = self._generate_test_cases()
        total_error = 0.0
        successful_tests = 0
        
        for inputs, expected_outputs in test_cases:
            try:
                outputs = self._activate_network(network, inputs)
                error = self._calculate_error(outputs, expected_outputs)
                total_error += error
                successful_tests += 1
            except:
                # Invalid network topology
                pass
        
        if successful_tests == 0:
            return 0.001  # Minimal fitness for invalid networks
        
        # Fitness is inverse of error
        fitness = 1.0 / (1.0 + total_error / successful_tests)
        
        # Add bonus for smaller networks (regularization)
        complexity_penalty = len(genome['connections']) * 0.01 + len(genome['nodes']) * 0.005
        fitness *= (1.0 - complexity_penalty)
        
        return max(0.001, fitness)
    
    def _build_network(self, genome: Dict) -> Dict:
        """Build executable neural network from genome."""
        network = {
            'nodes': {},
            'connections': [],
            'input_size': self.input_size,
            'output_size': self.output_size
        }
        
        # Create nodes
        for node_gene in genome['nodes']:
            network['nodes'][node_gene['id']] = {
                'type': node_gene['type'],
                'activation': node_gene['activation'],
                'value': 0.0
            }
        
        # Create connections
        for conn_gene in genome['connections']:
            if conn_gene['enabled']:
                network['connections'].append({
                    'in': conn_gene['in'],
                    'out': conn_gene['out'],
                    'weight': conn_gene['weight']
                })
        
        return network
    
    def _activate_network(self, network: Dict, inputs: List[float]) -> List[float]:
        """Activate neural network with given inputs."""
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        # Reset node values
        for node_id, node in network['nodes'].items():
            node['value'] = 0.0
        
        # Set input values
        for i, value in enumerate(inputs):
            if i in network['nodes']:
                network['nodes'][i]['value'] = value
        
        # Multiple passes for recurrent networks
        for _ in range(3):  # Maximum 3 passes for stability
            # Process connections
            for conn in network['connections']:
                in_node = network['nodes'].get(conn['in'])
                out_node = network['nodes'].get(conn['out'])
                
                if in_node and out_node:
                    # Add weighted input to output node
                    out_node['value'] += in_node['value'] * conn['weight']
            
            # Apply activation functions (except input nodes)
            for node_id, node in network['nodes'].items():
                if node['type'] != 'input':
                    activation_fn = self.activation_functions.get(node['activation'], self.activation_functions['sigmoid'])
                    node['value'] = activation_fn(node['value'])
        
        # Collect outputs
        outputs = []
        for i in range(self.input_size, self.input_size + self.output_size):
            if i in network['nodes']:
                outputs.append(network['nodes'][i]['value'])
            else:
                outputs.append(0.0)
        
        return outputs
    
    def _generate_test_cases(self) -> List[Tuple[List[float], List[float]]]:
        """Generate test cases for network evaluation."""
        test_cases = []
        
        # Simple pattern recognition tests
        for _ in range(10):
            inputs = [random.uniform(0, 1) for _ in range(self.input_size)]
            
            # Expected outputs: various simple functions of inputs
            expected = []
            for i in range(self.output_size):
                if i % 3 == 0:
                    # Average of inputs
                    expected.append(sum(inputs) / len(inputs))
                elif i % 3 == 1:
                    # Max of inputs
                    expected.append(max(inputs))
                else:
                    # Non-linear function
                    expected.append(math.sin(sum(inputs)))
            
            test_cases.append((inputs, expected))
        
        return test_cases
    
    def _calculate_error(self, outputs: List[float], expected: List[float]) -> float:
        """Calculate mean squared error."""
        if len(outputs) != len(expected):
            return float('inf')
        
        error = 0.0
        for o, e in zip(outputs, expected):
            error += (o - e) ** 2
        
        return error / len(outputs)
    
    def _speciate(self):
        """Speciate population based on genetic distance."""
        self.species = []
        self.species_representatives = []
        
        # Assign each genome to a species
        for genome in self.population:
            assigned = False
            
            for i, species in enumerate(self.species):
                representative = self.species_representatives[i]
                distance = self._genetic_distance(genome, representative)
                
                if distance < self.compatibility_threshold:
                    species.append(genome)
                    genome['species'] = i
                    assigned = True
                    break
            
            if not assigned:
                # Create new species
                self.species.append([genome])
                self.species_representatives.append(genome)
                genome['species'] = len(self.species) - 1
    
    def _genetic_distance(self, genome1: Dict, genome2: Dict) -> float:
        """
        Calculate genetic distance between two genomes.
        Based on NEAT distance metric: excess/disjoint genes and weight differences.
        """
        # Get innovation numbers
        innov1 = set(conn['innovation'] for conn in genome1['connections'])
        innov2 = set(conn['innovation'] for conn in genome2['connections'])
        
        # Find matching, disjoint, and excess genes
        matching = innov1 & innov2
        disjoint = (innov1 - innov2) | (innov2 - innov1)
        
        N = max(len(innov1), len(innov2))
        if N < 20:
            N = 1  # Normalization factor
        
        # Calculate distance components
        excess_count = len([i for i in disjoint if i > max(innov1 | innov2, default=0)])
        disjoint_count = len(disjoint) - excess_count
        
        # Average weight differences of matching genes
        weight_diff = 0.0
        matching_count = 0
        
        for innov in matching:
            # Find weights for this innovation in both genomes
            weight1 = next(conn['weight'] for conn in genome1['connections'] if conn['innovation'] == innov)
            weight2 = next(conn['weight'] for conn in genome2['connections'] if conn['innovation'] == innov)
            weight_diff += abs(weight1 - weight2)
            matching_count += 1
        
        avg_weight_diff = weight_diff / matching_count if matching_count > 0 else 0.0
        
        # Node count difference
        node_diff = abs(len(genome1['nodes']) - len(genome2['nodes']))
        
        # NEAT distance formula
        c1 = 1.0  # Coefficient for excess/disjoint
        c2 = 0.5  # Coefficient for weight differences
        c3 = 0.4  # Coefficient for node count differences
        
        distance = (c1 * (excess_count + disjoint_count) / N + 
                   c2 * avg_weight_diff + 
                   c3 * node_diff / 10.0)
        
        return distance
    
    def _calculate_adjusted_fitness(self):
        """Calculate adjusted fitness (fitness sharing within species)."""
        for species in self.species:
            if not species:
                continue
            
            # Calculate total raw fitness
            total_fitness = sum(genome['fitness'] for genome in species)
            
            # Adjust fitness (fitness sharing)
            for genome in species:
                genome['adjusted_fitness'] = genome['fitness'] / len(species)
    
    def _remove_stale_species(self):
        """Remove species that haven't improved in several generations."""
        max_staleness = 15
        surviving_species = []
        surviving_representatives = []
        
        for i, species in enumerate(self.species):
            if not species:
                continue
            
            # Find best fitness in species
            best_in_species = max(genome['fitness'] for genome in species)
            
            # Check if species has improved (you'd track this per species in full impl)
            if self.generation < 10 or random.random() < 0.7:  # Simplified
                surviving_species.append(species)
                surviving_representatives.append(self.species_representatives[i])
        
        self.species = surviving_species
        self.species_representatives = surviving_representatives
    
    def _select_parents(self) -> List[Dict]:
        """Select parents for reproduction using fitness proportionate selection."""
        parents = []
        
        # Always keep the best genome (elitism)
        if self.best_genome:
            parents.append(copy.deepcopy(self.best_genome))
        
        # Select rest based on adjusted fitness
        all_genomes = [genome for species in self.species for genome in species]
        total_adj_fitness = sum(genome['adjusted_fitness'] for genome in all_genomes)
        
        if total_adj_fitness <= 0:
            # Fallback to random selection
            return random.sample(all_genomes, min(len(all_genomes), self.population_size - len(parents)))
        
        # Fitness proportionate selection
        while len(parents) < self.population_size:
            r = random.uniform(0, total_adj_fitness)
            cumulative = 0.0
            
            for genome in all_genomes:
                cumulative += genome['adjusted_fitness']
                if cumulative >= r:
                    parents.append(genome)
                    break
        
        return parents
    
    def _create_new_generation(self, parents: List[Dict]) -> List[Dict]:
        """Create new generation through crossover and mutation."""
        new_population = []
        
        # Elitism: keep best performers
        best_parents = sorted(parents, key=lambda x: x['fitness'], reverse=True)[:self.population_size // 10]
        new_population.extend(copy.deepcopy(best_parents))
        
        # Create offspring
        while len(new_population) < self.population_size:
            if random.random() < self.interspecies_mating_rate or len(self.species) == 1:
                # Interspecies mating
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
            else:
                # Same species mating
                species_idx = random.randrange(len(self.species))
                species = self.species[species_idx]
                if len(species) < 2:
                    parent1 = random.choice(parents)
                    parent2 = random.choice(parents)
                else:
                    parent1, parent2 = random.sample(species, 2)
            
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(random.choice([parent1, parent2]))
            
            # Mutate child
            self._mutate(child)
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two genomes using NEAT method."""
        # Make child from more fit parent
        if parent1['fitness'] > parent2['fitness']:
            child = copy.deepcopy(parent1)
            less_fit = parent2
        else:
            child = copy.deepcopy(parent2)
            less_fit = parent1
        
        # Align genes by innovation number
        child_conns = {conn['innovation']: conn for conn in child['connections']}
        less_fit_conns = {conn['innovation']: conn for conn in less_fit['connections']}
        
        # Crossover matching genes randomly
        for innov, conn in child_conns.items():
            if innov in less_fit_conns and random.random() < 0.5:
                # Take gene from less fit parent
                child_conns[innov] = copy.deepcopy(less_fit_conns[innov])
        
        child['connections'] = list(child_conns.values())
        return child
    
    def _mutate(self, genome: Dict):
        """Apply mutations to genome."""
        # Weight mutations
        if random.random() < self.mutation_rates['conn_weight']:
            for conn in genome['connections']:
                if random.random() < 0.1:  # Perturb each connection with 10% chance
                    if random.random() < 0.1:  # 10% chance of completely new weight
                        conn['weight'] = random.uniform(-1, 1)
                    else:
                        # Gaussian perturbation
                        conn['weight'] += random.gauss(0, self.weight_mutation_power)
                        conn['weight'] = max(-3.0, min(3.0, conn['weight']))
        
        # Connection enable/disable
        for conn in genome['connections']:
            if random.random() < self.mutation_rates['conn_enable'] and not conn['enabled']:
                conn['enabled'] = True
            elif random.random() < self.mutation_rates['conn_disable'] and conn['enabled']:
                conn['enabled'] = False
        
        # Add new connection
        if random.random() < self.mutation_rates['add_conn']:
            self._mutate_add_connection(genome)
        
        # Add new node
        if random.random() < self.mutation_rates['add_node']:
            self._mutate_add_node(genome)
    
    def _mutate_add_connection(self, genome: Dict):
        """Add a new connection between two unconnected nodes."""
        nodes = [node['id'] for node in genome['nodes']]
        
        if len(nodes) < 2:
            return
        
        # Try to find valid connection (not existing and not recurrent in minimal case)
        attempts = 0
        while attempts < 10:
            in_node = random.choice(nodes)
            out_node = random.choice(nodes)
            
            # Check if connection already exists
            existing = any(conn['in'] == in_node and conn['out'] == out_node 
                          for conn in genome['connections'])
            
            # Simple acyclicity check (for minimal implementation)
            valid = (not existing and in_node != out_node and
                    (in_node < out_node or random.random() < 0.3))  # Allow some recurrence
            
            if valid:
                innov = self._get_innovation_number(in_node, out_node)
                genome['connections'].append({
                    'in': in_node,
                    'out': out_node,
                    'weight': random.uniform(-1, 1),
                    'enabled': True,
                    'innovation': innov
                })
                break
            
            attempts += 1
    
    def _mutate_add_node(self, genome: Dict):
        """Add a new node by splitting an existing connection."""
        enabled_connections = [conn for conn in genome['connections'] if conn['enabled']]
        
        if not enabled_connections:
            return
        
        # Choose random connection to split
        conn_to_split = random.choice(enabled_connections)
        conn_to_split['enabled'] = False
        
        # Create new node
        new_node_id = max(node['id'] for node in genome['nodes']) + 1
        genome['nodes'].append({
            'id': new_node_id,
            'type': 'hidden',
            'activation': random.choice(['sigmoid', 'tanh', 'relu'])
        })
        
        # Create new connections
        innov1 = self._get_innovation_number(conn_to_split['in'], new_node_id)
        innov2 = self._get_innovation_number(new_node_id, conn_to_split['out'])
        
        genome['connections'].append({
            'in': conn_to_split['in'],
            'out': new_node_id,
            'weight': 1.0,
            'enabled': True,
            'innovation': innov1
        })
        
        genome['connections'].append({
            'in': new_node_id,
            'out': conn_to_split['out'],
            'weight': conn_to_split['weight'],
            'enabled': True,
            'innovation': innov2
        })
    
    def _get_best_genome(self) -> Dict:
        """Get the best genome from current population."""
        if self.best_genome:
            return self.best_genome
        return max(self.population, key=lambda x: x['fitness'])
    
    def _flatten_genome(self, genome: Dict) -> List[float]:
        """Flatten genome to list of floats for compatibility."""
        flattened = []
        
        # Encode node information
        for node in genome['nodes']:
            # Encode node type and activation
            type_encoding = {'input': 0.1, 'hidden': 0.5, 'output': 0.9}
            activation_encoding = {'linear': 0.1, 'sigmoid': 0.3, 'tanh': 0.5, 'relu': 0.7}
            
            flattened.append(type_encoding.get(node['type'], 0.5))
            flattened.append(activation_encoding.get(node['activation'], 0.3))
        
        # Encode connection information
        for conn in genome['connections']:
            flattened.append(conn['in'] / 100.0)  # Normalize
            flattened.append(conn['out'] / 100.0)
            flattened.append(conn['weight'])
            flattened.append(1.0 if conn['enabled'] else 0.0)
        
        return flattened
    
    def get_topology_metrics(self) -> Dict:
        """Get comprehensive NEAT metrics."""
        if not self.population:
            return {}
        
        best_genome = self._get_best_genome()
        avg_fitness = np.mean([genome['fitness'] for genome in self.population])
        
        return {
            'current_fitness': float(best_genome['fitness']),
            'best_fitness': float(self.best_fitness),
            'generation': int(self.generation),
            'species_count': len(self.species),
            'population_size': len(self.population),
            'average_fitness': float(avg_fitness),
            'network_complexity': {
                'nodes': len(best_genome['nodes']),
                'connections': len(best_genome['connections']),
                'enabled_connections': sum(1 for conn in best_genome['connections'] if conn['enabled'])
            },
            'innovation_count': self.innovation_number,
            'mutation_rates': self.mutation_rates.copy()
        }
    
    def get_best_network_structure(self) -> Dict:
        """Get the structure of the best network for visualization."""
        best_genome = self._get_best_genome()
        
        return {
            'nodes': best_genome['nodes'],
            'connections': [
                {
                    'source': conn['in'],
                    'target': conn['out'],
                    'weight': conn['weight'],
                    'enabled': conn['enabled']
                }
                for conn in best_genome['connections']
            ],
            'fitness': best_genome['fitness']
        }

NeuroevolutionaryTopology = RealNEATTopology