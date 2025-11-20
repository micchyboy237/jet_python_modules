# Mathematical Foundations Lab - Interactive Exploration
# Context Engineering Course: From Foundations to Frontier Systems
# Module 00: Mathematical Foundations - Interactive Laboratory

"""
Mathematical Foundations Lab: Interactive Exploration
====================================================

This laboratory notebook provides hands-on exploration of the four mathematical
pillars of Context Engineering:

1. Context Formalization: C = A(c₁, c₂, ..., c₆)
2. Optimization Theory: F* = arg max E[Reward(C)]
3. Information Theory: I(Context; Query) maximization
4. Bayesian Inference: P(Strategy|Evidence) updating

Learning Approach:
- Start with intuitive examples
- Build mathematical understanding through visualization
- Implement working algorithms
- Apply to real context engineering problems

Prerequisites: Basic Python, NumPy, Matplotlib
Estimated Time: 2-3 hours for complete exploration
"""

# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import seaborn as sns
from scipy import optimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

from jet.file.utils import save_file
from jet.logger import CustomLogger
import os
import shutil

# ============================================================================
# OUTPUT & LOGGING SETUP
# ============================================================================

BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

main_logger = CustomLogger(
    name="math_foundations_lab",
    filename=os.path.join(BASE_OUTPUT_DIR, "main.log"),
    console_level="INFO",
    level="DEBUG",
    overwrite=True
)
main_logger.info("=" * 80)
main_logger.info("MATHEMATICAL FOUNDATIONS LAB STARTED")
main_logger.info("=" * 80)


def create_example_dir(example_name: str) -> str:
    example_dir = os.path.join(BASE_OUTPUT_DIR, example_name)
    os.makedirs(example_dir, exist_ok=True)
    return example_dir


def get_example_logger(example_name: str, example_dir: str) -> CustomLogger:
    log_file = os.path.join(example_dir, "run.log")
    log = CustomLogger(
        name=f"ex_{example_name.lower().replace(' ', '_').replace(':', '')}",
        filename=log_file,
        console_level="INFO",
        level="DEBUG",
        fmt="%(asctime)s | %(message)s",
        overwrite=True
    )
    log.info("")
    log.info("=" * 80)
    log.info(f"EXAMPLE: {example_name}")
    log.info("=" * 80)
    return log

# ==============================================================================
# SECTION 1: CONTEXT FORMALIZATION - From Intuition to Mathematics
# ==============================================================================

@dataclass
class ContextComponent:
    """Mathematical representation of a context component"""
    component_type: str
    content: str
    relevance_score: float
    token_count: int
    quality_metrics: Dict[str, float]
    
    def __post_init__(self):
        """Validate component after initialization"""
        assert 0 <= self.relevance_score <= 1, "Relevance score must be between 0 and 1"
        assert self.token_count >= 0, "Token count must be non-negative"

class ContextAssemblyFunction:
    """
    Mathematical implementation of C = A(c₁, c₂, c₃, c₄, c₅, c₆)
    
    Where:
    - c₁ = Instructions (system prompts, role definitions)
    - c₂ = Knowledge (external information, facts, data)
    - c₃ = Tools (available functions, APIs, capabilities)
    - c₄ = Memory (conversation history, learned patterns)
    - c₅ = State (current situation, user context, environment)
    - c₆ = Query (immediate user request, specific question)
    """
    
    def __init__(self, max_tokens: int = 1000):
        self.max_tokens = max_tokens
        self.assembly_history = []
        
    def assemble_context(self, components: List[ContextComponent], 
                        strategy: str = 'weighted') -> Dict:
        """
        Core mathematical assembly function: C = A(c₁, c₂, c₃, c₄, c₅, c₆)
        
        Args:
            components: List of context components (c₁, c₂, ..., c₆)
            strategy: Assembly strategy ('linear', 'weighted', 'hierarchical')
            
        Returns:
            Assembled context with metadata
        """
        
        if strategy == 'linear':
            return self._linear_assembly(components)
        elif strategy == 'weighted':
            return self._weighted_assembly(components)
        elif strategy == 'hierarchical':
            return self._hierarchical_assembly(components)
        else:
            raise ValueError(f"Unknown assembly strategy: {strategy}")
    
    def _linear_assembly(self, components: List[ContextComponent]) -> Dict:
        """Simple linear concatenation assembly"""
        
        total_tokens = 0
        assembled_content = []
        included_components = []
        
        for component in components:
            if total_tokens + component.token_count <= self.max_tokens:
                assembled_content.append(f"=== {component.component_type.upper()} ===")
                assembled_content.append(component.content)
                assembled_content.append("")  # Spacing
                total_tokens += component.token_count
                included_components.append(component)
            else:
                break
        
        return {
            'assembled_context': '\n'.join(assembled_content),
            'total_tokens': total_tokens,
            'included_components': len(included_components),
            'assembly_strategy': 'linear',
            'utilization_rate': total_tokens / self.max_tokens
        }
    
    def _weighted_assembly(self, components: List[ContextComponent]) -> Dict:
        """Weighted assembly based on relevance scores"""
        
        # Sort by relevance score (descending)
        sorted_components = sorted(components, key=lambda c: c.relevance_score, reverse=True)
        
        total_tokens = 0
        assembled_content = []
        included_components = []
        total_relevance = 0
        
        for component in sorted_components:
            if total_tokens + component.token_count <= self.max_tokens:
                assembled_content.append(f"=== {component.component_type.upper()} ===")
                assembled_content.append(component.content)
                assembled_content.append("")
                total_tokens += component.token_count
                included_components.append(component)
                total_relevance += component.relevance_score
        
        avg_relevance = total_relevance / len(included_components) if included_components else 0
        
        return {
            'assembled_context': '\n'.join(assembled_content),
            'total_tokens': total_tokens,
            'included_components': len(included_components),
            'assembly_strategy': 'weighted',
            'utilization_rate': total_tokens / self.max_tokens,
            'average_relevance': avg_relevance
        }
    
    def _hierarchical_assembly(self, components: List[ContextComponent]) -> Dict:
        """Hierarchical assembly with structured organization"""
        
        # Group components by type
        component_groups = {}
        for component in components:
            comp_type = component.component_type
            if comp_type not in component_groups:
                component_groups[comp_type] = []
            component_groups[comp_type].append(component)
        
        # Define hierarchy order
        hierarchy_order = ['instructions', 'query', 'knowledge', 'tools', 'memory', 'state']
        
        total_tokens = 0
        assembled_content = []
        included_components = []
        
        for comp_type in hierarchy_order:
            if comp_type in component_groups:
                # Sort components within type by relevance
                type_components = sorted(component_groups[comp_type], 
                                       key=lambda c: c.relevance_score, reverse=True)
                
                for component in type_components:
                    if total_tokens + component.token_count <= self.max_tokens:
                        if not any(c.component_type == comp_type for c in included_components):
                            assembled_content.append(f"\n=== {comp_type.upper()} LAYER ===")
                        
                        assembled_content.append(component.content)
                        assembled_content.append("")
                        total_tokens += component.token_count
                        included_components.append(component)
        
        return {
            'assembled_context': '\n'.join(assembled_content),
            'total_tokens': total_tokens,
            'included_components': len(included_components),
            'assembly_strategy': 'hierarchical',
            'utilization_rate': total_tokens / self.max_tokens,
            'hierarchy_coverage': len(set(c.component_type for c in included_components))
        }

# ==============================================================================
# SECTION 2: OPTIMIZATION THEORY - Finding the Best Assembly Function
# ==============================================================================

def create_optimization_landscape():
    """
    Visualize context optimization as a mathematical landscape
    
    Shows how different assembly parameters affect context quality
    """
    
    log.info("\n2.1 Context Optimization Landscape")
    log.info("-" * 38)
    
    # Define parameter space
    relevance_weights = np.linspace(0, 1, 50)
    completeness_weights = np.linspace(0, 1, 50)
    
    # Create meshgrid for 3D surface
    R, C = np.meshgrid(relevance_weights, completeness_weights)
    
    # Objective function: Quality = f(relevance_weight, completeness_weight)
    def context_quality_function(r_weight, c_weight):
        """
        Simulated context quality function
        Quality depends on balance between relevance and completeness
        """
        # Optimal balance is around 0.6 relevance, 0.4 completeness
        relevance_term = r_weight * (1 - abs(r_weight - 0.6))
        completeness_term = c_weight * (1 - abs(c_weight - 0.4))
        
        # Add interaction term (synergy between relevance and completeness)
        interaction_term = 0.3 * r_weight * c_weight
        
        # Add noise to make it realistic
        noise = 0.05 * np.sin(10 * r_weight) * np.cos(10 * c_weight)
        
        return relevance_term + completeness_term + interaction_term + noise
    
    # Calculate quality for all parameter combinations
    Quality = context_quality_function(R, C)
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surface = ax1.plot_surface(R, C, Quality, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Relevance Weight')
    ax1.set_ylabel('Completeness Weight')
    ax1.set_zlabel('Context Quality')
    ax1.set_title('Context Optimization Landscape')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(R, C, Quality, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('Relevance Weight')
    ax2.set_ylabel('Completeness Weight')
    ax2.set_title('Quality Contours')
    
    # Find and mark the optimum
    max_idx = np.unravel_index(np.argmax(Quality), Quality.shape)
    optimal_r = R[max_idx]
    optimal_c = C[max_idx]
    optimal_quality = Quality[max_idx]
    
    ax1.scatter([optimal_r], [optimal_c], [optimal_quality], 
               color='red', s=100, label=f'Optimum ({optimal_r:.2f}, {optimal_c:.2f})')
    ax2.scatter([optimal_r], [optimal_c], color='red', s=100, 
               label=f'Optimum ({optimal_r:.2f}, {optimal_c:.2f})')
    
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    log.info("Optimal parameters found:")
    log.info(f"  Relevance weight: {optimal_r:.3f}")
    log.info(f"  Completeness weight: {optimal_c:.3f}")
    log.info(f"  Maximum quality: {optimal_quality:.3f}")
    
    return R, C, Quality, (optimal_r, optimal_c, optimal_quality)

# 2.2 Mathematical Optimization Implementation

class ContextOptimizer:
    """
    Mathematical optimization for context assembly
    
    Implements: F* = arg max E[Reward(C)]
    """
    
    def __init__(self):
        self.optimization_history = []
        
    def objective_function(self, params: np.ndarray, components: List[ContextComponent],
                          user_feedback_data: List[Dict] = None) -> float:
        """
        Mathematical objective function to maximize
        
        Args:
            params: [relevance_weight, completeness_weight, efficiency_weight]
            components: Available context components
            user_feedback_data: Historical user feedback for learning
            
        Returns:
            Quality score to maximize (negative for minimization algorithms)
        """
        
        relevance_weight, completeness_weight, efficiency_weight = params
        
        # Ensure weights sum to 1 (constraint)
        total_weight = relevance_weight + completeness_weight + efficiency_weight
        if total_weight == 0:
            return -1000  # Invalid solution
        
        # Normalize weights
        relevance_weight /= total_weight
        completeness_weight /= total_weight
        efficiency_weight /= total_weight
        
        # Calculate quality components
        avg_relevance = np.mean([c.relevance_score for c in components])
        completeness = len(components) / 6  # Assuming 6 ideal component types
        total_tokens = sum(c.token_count for c in components)
        efficiency = (avg_relevance * len(components)) / (total_tokens + 1)
        
        # Composite quality score
        quality = (relevance_weight * avg_relevance + 
                  completeness_weight * completeness + 
                  efficiency_weight * efficiency)
        
        # Add user feedback influence if available
        if user_feedback_data:
            feedback_score = np.mean([fb.get('satisfaction', 0.5) for fb in user_feedback_data])
            quality = 0.8 * quality + 0.2 * feedback_score
        
        return quality
    
    def optimize_assembly_strategy(self, components: List[ContextComponent],
                                  method: str = 'scipy') -> Dict:
        """
        Find optimal assembly strategy using mathematical optimization
        
        Args:
            components: Available context components
            method: Optimization method ('scipy', 'grid_search', 'gradient_descent')
            
        Returns:
            Optimization results with optimal parameters
        """
        
        if method == 'scipy':
            return self._scipy_optimization(components)
        elif method == 'grid_search':
            return self._grid_search_optimization(components)
        elif method == 'gradient_descent':
            return self._gradient_descent_optimization(components)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _scipy_optimization(self, components: List[ContextComponent]) -> Dict:
        """Use SciPy's optimization algorithms"""
        
        # Define constraints: weights must be non-negative and sum to reasonable value
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0]},  # relevance_weight >= 0
            {'type': 'ineq', 'fun': lambda x: x[1]},  # completeness_weight >= 0
            {'type': 'ineq', 'fun': lambda x: x[2]},  # efficiency_weight >= 0
            {'type': 'ineq', 'fun': lambda x: 3 - sum(x)},  # sum <= 3
            {'type': 'ineq', 'fun': lambda x: sum(x) - 0.1},  # sum >= 0.1
        ]
        
        # Initial guess
        initial_guess = [0.5, 0.3, 0.2]
        
        # Bounds for parameters
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        # Optimize (minimize negative of objective function)
        result = optimize.minimize(
            lambda params: -self.objective_function(params, components),
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_params = result.x
        optimal_quality = -result.fun  # Convert back from negative
        
        return {
            'method': 'scipy',
            'optimal_parameters': {
                'relevance_weight': optimal_params[0],
                'completeness_weight': optimal_params[1],
                'efficiency_weight': optimal_params[2]
            },
            'optimal_quality': optimal_quality,
            'optimization_success': result.success,
            'iterations': result.nit,
            'function_evaluations': result.nfev
        }
    
    def _grid_search_optimization(self, components: List[ContextComponent]) -> Dict:
        """Exhaustive grid search optimization"""
        
        # Define parameter grid
        param_values = np.linspace(0.1, 0.9, 10)
        
        best_quality = -float('inf')
        best_params = None
        evaluations = 0
        
        for r_weight in param_values:
            for c_weight in param_values:
                for e_weight in param_values:
                    params = [r_weight, c_weight, e_weight]
                    quality = self.objective_function(params, components)
                    evaluations += 1
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_params = params
        
        return {
            'method': 'grid_search',
            'optimal_parameters': {
                'relevance_weight': best_params[0],
                'completeness_weight': best_params[1],
                'efficiency_weight': best_params[2]
            },
            'optimal_quality': best_quality,
            'function_evaluations': evaluations
        }
    
    def _gradient_descent_optimization(self, components: List[ContextComponent]) -> Dict:
        """Simple gradient descent optimization"""
        
        # Starting point
        params = np.array([0.5, 0.3, 0.2])
        learning_rate = 0.01
        max_iterations = 1000
        tolerance = 1e-6
        
        history = []
        
        for iteration in range(max_iterations):
            # Calculate numerical gradient
            gradient = self._numerical_gradient(params, components)
            
            # Update parameters
            old_params = params.copy()
            params = params + learning_rate * gradient
            
            # Ensure non-negative and normalize
            params = np.maximum(params, 0.01)
            params = params / np.sum(params)
            
            # Calculate current quality
            quality = self.objective_function(params, components)
            history.append(quality)
            
            # Check convergence
            if np.linalg.norm(params - old_params) < tolerance:
                break
        
        return {
            'method': 'gradient_descent',
            'optimal_parameters': {
                'relevance_weight': params[0],
                'completeness_weight': params[1],
                'efficiency_weight': params[2]
            },
            'optimal_quality': quality,
            'iterations': iteration + 1,
            'convergence_history': history
        }
    
    def _numerical_gradient(self, params: np.ndarray, components: List[ContextComponent],
                           epsilon: float = 1e-6) -> np.ndarray:
        """Calculate numerical gradient for gradient descent"""
        
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            f_plus = self.objective_function(params_plus, components)
            f_minus = self.objective_function(params_minus, components)
            
            gradient[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return gradient

# ==============================================================================
# SECTION 3: INFORMATION THEORY - Quantifying Context Value
# ==============================================================================

class InformationAnalyzer:
    """Analyze information content and mutual information in context components"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
    def calculate_text_entropy(self, text: str, level: str = 'word') -> float:
        """
        Calculate Shannon entropy of text
        
        Args:
            text: Input text
            level: 'char' for character-level, 'word' for word-level
            
        Returns:
            Entropy in bits
        """
        
        if level == 'char':
            symbols = list(text.lower())
        elif level == 'word':
            symbols = text.lower().split()
        else:
            raise ValueError("Level must be 'char' or 'word'")
        
        if not symbols:
            return 0.0
        
        # Count symbol frequencies
        symbol_counts = {}
        for symbol in symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Calculate probabilities
        total_symbols = len(symbols)
        probabilities = [count / total_symbols for count in symbol_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return entropy
    
    def calculate_mutual_information(self, text1: str, text2: str) -> float:
        """
        Calculate mutual information between two text segments
        
        Uses TF-IDF vectors and cosine similarity as approximation
        """
        
        try:
            # Create TF-IDF vectors
            texts = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            if tfidf_matrix.shape[1] == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]
            
            # Convert similarity to mutual information estimate
            # This is a rough approximation: MI ≈ -log(1 - similarity)
            if similarity >= 0.999:
                return 10.0  # High mutual information
            
            mutual_info = -np.log(1 - similarity + 1e-10)
            
            return mutual_info
            
        except Exception:
            # Fallback to word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
            
            jaccard = overlap / union
            return -np.log(1 - jaccard + 1e-10)
    
    def analyze_context_information(self, components: List[ContextComponent], 
                                   query: str) -> Dict:
        """
        Analyze information content and relevance of context components
        
        Args:
            components: List of context components
            query: User query for relevance calculation
            
        Returns:
            Analysis results with entropy, mutual information, and relevance scores
        """
        
        results = {
            'component_analysis': [],
            'mutual_information_matrix': [],
            'query_relevance': [],
            'redundancy_analysis': {}
        }
        
        # Analyze each component
        for i, component in enumerate(components):
            # Calculate entropy
            word_entropy = self.calculate_text_entropy(component.content, 'word')
            char_entropy = self.calculate_text_entropy(component.content, 'char')
            
            # Calculate mutual information with query
            mi_with_query = self.calculate_mutual_information(component.content, query)
            
            component_analysis = {
                'component_type': component.component_type,
                'word_entropy': word_entropy,
                'char_entropy': char_entropy,
                'mutual_information_with_query': mi_with_query,
                'token_count': component.token_count,
                'information_density': word_entropy / (component.token_count + 1)
            }
            
            results['component_analysis'].append(component_analysis)
            results['query_relevance'].append(mi_with_query)
        
        # Calculate mutual information matrix between components
        n_components = len(components)
        mi_matrix = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            for j in range(i + 1, n_components):
                mi = self.calculate_mutual_information(
                    components[i].content, components[j].content
                )
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi  # Symmetric
        
        results['mutual_information_matrix'] = mi_matrix
        
        # Redundancy analysis
        redundancy_pairs = []
        for i in range(n_components):
            for j in range(i + 1, n_components):
                if mi_matrix[i, j] > 2.0:  # Threshold for high redundancy
                    redundancy_pairs.append({
                        'component1': components[i].component_type,
                        'component2': components[j].component_type,
                        'redundancy_score': mi_matrix[i, j]
                    })
        
        results['redundancy_analysis'] = {
            'high_redundancy_pairs': redundancy_pairs,
            'average_redundancy': np.mean(mi_matrix[mi_matrix > 0]),
            'max_redundancy': np.max(mi_matrix)
        }
        
        return results

# ==============================================================================
# SECTION 4: BAYESIAN INFERENCE - Learning Under Uncertainty
# ==============================================================================

class BayesianContextLearner:
    """
    Bayesian learning system for context strategy optimization
    
    Learns optimal strategies from user feedback using Bayesian inference
    """
    
    def __init__(self, strategies: List[str]):
        self.strategies = strategies
        
        # Initialize uniform priors
        self.strategy_beliefs = {strategy: 1.0 / len(strategies) for strategy in strategies}
        
        # Track feedback history
        self.feedback_history = []
        
        # Beta distributions for component relevance (alpha, beta parameters)
        self.component_relevance_beliefs = {}
        
    def update_strategy_beliefs(self, strategy_used: str, feedback_score: float):
        """
        Update strategy beliefs using Bayesian inference
        
        Args:
            strategy_used: Which strategy was employed
            feedback_score: User feedback (0-1 scale, 0.5 = neutral)
        """
        
        # Define likelihood model: P(feedback | strategy)
        # Assume different strategies have different success probabilities
        strategy_success_rates = {
            'technical_detailed': 0.7,
            'practical_concise': 0.85,
            'balanced_comprehensive': 0.8,
            'user_adapted': 0.9
        }
        
        # Calculate likelihoods for all strategies
        likelihoods = {}
        for strategy in self.strategies:
            if strategy == strategy_used:
                # Actually used strategy - calculate likelihood based on feedback
                success_rate = strategy_success_rates.get(strategy, 0.7)
                if feedback_score > 0.5:
                    # Positive feedback
                    likelihood = success_rate
                else:
                    # Negative feedback
                    likelihood = 1 - success_rate
            else:
                # Counterfactual strategy - estimate what would have happened
                success_rate = strategy_success_rates.get(strategy, 0.7)
                # Reduce likelihood since we're estimating
                if feedback_score > 0.5:
                    likelihood = success_rate * 0.5  # Uncertainty discount
                else:
                    likelihood = (1 - success_rate) * 0.5
            
            likelihoods[strategy] = likelihood
        
        # Calculate evidence (normalizing constant)
        evidence = sum(self.strategy_beliefs[s] * likelihoods[s] for s in self.strategies)
        
        # Update beliefs using Bayes' rule
        if evidence > 1e-10:  # Avoid division by zero
            for strategy in self.strategies:
                prior = self.strategy_beliefs[strategy]
                likelihood = likelihoods[strategy]
                posterior = (likelihood * prior) / evidence
                self.strategy_beliefs[strategy] = posterior
        
        # Record feedback
        self.feedback_history.append({
            'strategy_used': strategy_used,
            'feedback_score': feedback_score,
            'beliefs_after_update': self.strategy_beliefs.copy()
        })
    
    def update_component_relevance(self, component_id: str, relevance_evidence: float):
        """
        Update component relevance beliefs using Beta distribution
        
        Args:
            component_id: Identifier for the component
            relevance_evidence: Evidence of relevance (0-1 scale)
        """
        
        if component_id not in self.component_relevance_beliefs:
            # Initialize with uninformative prior (Beta(1,1) = uniform)
            self.component_relevance_beliefs[component_id] = [1.0, 1.0]  # [alpha, beta]
        
        alpha, beta = self.component_relevance_beliefs[component_id]
        
        # Update Beta parameters based on evidence
        if relevance_evidence > 0.5:
            # Evidence of relevance
            evidence_strength = (relevance_evidence - 0.5) * 2  # Scale to 0-1
            alpha += evidence_strength
        else:
            # Evidence of irrelevance
            evidence_strength = (0.5 - relevance_evidence) * 2  # Scale to 0-1
            beta += evidence_strength
        
        self.component_relevance_beliefs[component_id] = [alpha, beta]
    
    def select_best_strategy(self) -> Tuple[str, float]:
        """
        Select strategy with highest posterior probability
        
        Returns:
            Tuple of (strategy_name, confidence)
        """
        
        best_strategy = max(self.strategy_beliefs, key=self.strategy_beliefs.get)
        confidence = self.strategy_beliefs[best_strategy]
        
        return best_strategy, confidence
    
    def get_strategy_uncertainty(self) -> float:
        """
        Calculate uncertainty in strategy selection using entropy
        
        Returns:
            Entropy of strategy distribution (higher = more uncertain)
        """
        
        probs = list(self.strategy_beliefs.values())
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
        
        return entropy
    
    def get_component_relevance_estimate(self, component_id: str) -> Tuple[float, float]:
        """
        Get relevance estimate and confidence for component
        
        Returns:
            Tuple of (relevance_estimate, confidence_width)
        """
        
        if component_id not in self.component_relevance_beliefs:
            return 0.5, 1.0  # Neutral estimate, maximum uncertainty
        
        alpha, beta = self.component_relevance_beliefs[component_id]
        
        # Mean and variance of Beta distribution
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        
        # Confidence interval width (2 standard deviations)
        confidence_width = 2 * np.sqrt(variance)
        
        return mean, confidence_width

# ==============================================================================
# SECTION 5: INTEGRATED MATHEMATICAL FRAMEWORK
# ==============================================================================

class IntegratedContextEngineer:
    """
    Complete mathematical context engineering system integrating:
    1. Context Formalization: C = A(c₁, c₂, ..., c₆)
    2. Optimization Theory: F* = arg max E[Reward(C)]
    3. Information Theory: I(Context; Query) maximization
    4. Bayesian Inference: P(Strategy|Evidence) updating
    """
    
    def __init__(self, max_tokens: int = 1000):
        # Mathematical components
        self.context_assembler = ContextAssemblyFunction(max_tokens)
        self.optimizer = ContextOptimizer()
        self.info_analyzer = InformationAnalyzer()
        self.bayesian_learner = BayesianContextLearner([
            'relevance_optimized', 'completeness_focused', 'efficiency_maximized', 'adaptive_learning'
        ])
        
        # Integration state
        self.optimization_history = []
        self.learning_history = []
        
    def engineer_context(self, components: List[ContextComponent], 
                        query: str, user_feedback: Optional[float] = None) -> Dict:
        """
        Complete mathematical context engineering pipeline
        
        Args:
            components: Available context components
            query: User query
            user_feedback: Optional feedback from previous interaction
            
        Returns:
            Engineered context with mathematical analysis
        """
        
        print("\nIntegrated Context Engineering Pipeline")
        print("-" * 44)
        
        # Step 1: Context Formalization
        print("1. Context Formalization: C = A(c₁, c₂, ..., c₆)")
        
        # Select best assembly strategy using Bayesian learning
        if user_feedback is not None:
            best_strategy_name, _ = self.bayesian_learner.select_best_strategy()
            self.bayesian_learner.update_strategy_beliefs(best_strategy_name, user_feedback)
        else:
            best_strategy_name = 'weighted'  # Default
        
        # assembly_result = self.context_assembler.assemble_context(
        #     components, strategy=best_strategy_name.split('_')[0]
        # )
        strategy_map = {
            'relevance_optimized': 'weighted',
            'completeness_focused': 'hierarchical',
            'efficiency_maximized': 'linear',
            'adaptive_learning': 'weighted'
        }
        mapped_strategy = strategy_map.get(best_strategy_name, 'weighted')
        assembly_result = self.context_assembler.assemble_context(
            components, strategy=mapped_strategy
        )
        
        print(f"   Selected assembly strategy: {best_strategy_name}")
        print(f"   Components assembled: {assembly_result['included_components']}")
        print(f"   Token utilization: {assembly_result['utilization_rate']:.1%}")
        
        # Step 2: Information Theory Analysis
        print("\n2. Information Theory: I(Context; Query) analysis")
        
        info_analysis = self.info_analyzer.analyze_context_information(components, query)
        
        total_mi_with_query = sum(info_analysis['query_relevance'])
        avg_redundancy = info_analysis['redundancy_analysis']['average_redundancy']
        
        print(f"   Total mutual information with query: {total_mi_with_query:.3f}")
        print(f"   Average component redundancy: {avg_redundancy:.3f}")
        
        # Step 3: Optimization Theory Application
        print("\n3. Optimization: F* = arg max E[Reward(C)]")
        
        optimization_result = self.optimizer.optimize_assembly_strategy(components, method='scipy')
        
        optimal_params = optimization_result['optimal_parameters']
        print(f"   Optimal relevance weight: {optimal_params['relevance_weight']:.3f}")
        print(f"   Optimal completeness weight: {optimal_params['completeness_weight']:.3f}")
        print(f"   Optimal efficiency weight: {optimal_params['efficiency_weight']:.3f}")
        print(f"   Optimized quality score: {optimization_result['optimal_quality']:.3f}")
        
        # Step 4: Bayesian Inference Integration
        print("\n4. Bayesian Learning: P(Strategy|Evidence) updating")
        
        strategy_uncertainty = self.bayesian_learner.get_strategy_uncertainty()
        print(f"   Strategy selection uncertainty: {strategy_uncertainty:.3f}")
        
        best_strategy, confidence = self.bayesian_learner.select_best_strategy()
        print(f"   Recommended strategy: {best_strategy} (confidence: {confidence:.3f})")
        
        # Integrate all results
        integrated_result = {
            'formalized_context': assembly_result,
            'information_analysis': info_analysis,
            'optimization_results': optimization_result,
            'bayesian_insights': {
                'best_strategy': best_strategy,
                'strategy_confidence': confidence,
                'uncertainty': strategy_uncertainty
            },
            'mathematical_quality_score': self._calculate_integrated_quality(
                assembly_result, info_analysis, optimization_result
            )
        }
        
        # Record for learning
        self.optimization_history.append(optimization_result)
        self.learning_history.append(integrated_result)
        
        return integrated_result
    
    def _calculate_integrated_quality(self, assembly_result: Dict, 
                                    info_analysis: Dict, optimization_result: Dict) -> float:
        """Calculate integrated quality score using all mathematical frameworks"""
        
        # Assembly quality
        assembly_quality = assembly_result.get('utilization_rate', 0) * assembly_result.get('average_relevance', 0.5)
        
        # Information quality
        total_mi = sum(info_analysis['query_relevance'])
        avg_redundancy = info_analysis['redundancy_analysis']['average_redundancy']
        info_quality = total_mi / (1 + avg_redundancy)  # Penalize redundancy
        
        # Optimization quality
        opt_quality = optimization_result['optimal_quality']
        
        # Integrated score (weighted combination)
        integrated_quality = (
            0.3 * assembly_quality +
            0.4 * info_quality +
            0.3 * opt_quality
        )
        
        return integrated_quality


def example_01_restaurant_analogy() -> Dict:
    example_dir = create_example_dir("example_01_restaurant_analogy")
    log = get_example_logger("Example 01: Restaurant → Context Analogy", example_dir)

    restaurant = {
        'ambiance': ['cozy', 'elegant', 'casual', 'romantic'],
        'menu_variety': [0.3, 0.7, 0.9, 1.0],
        'chef_skill': [0.6, 0.8, 0.9, 0.95],
        'service_quality': [0.4, 0.7, 0.8, 0.9],
        'price_point': ['budget', 'moderate', 'upscale', 'luxury']
    }
    context = {
        'instructions': ['basic', 'detailed', 'expert', 'adaptive'],
        'knowledge_depth': [0.3, 0.7, 0.9, 1.0],
        'tool_availability': [0.6, 0.8, 0.9, 0.95],
        'memory_relevance': [0.4, 0.7, 0.8, 0.9],
        'query_complexity': ['simple', 'moderate', 'complex', 'expert']
    }

    mapping = [
        ("Ambiance", "Instructions"),
        ("Menu Variety", "Knowledge Depth"),
        ("Chef Skill", "Tool Availability"),
        ("Service Quality", "Memory Relevance"),
        ("Customer Craving", "Query Complexity"),
    ]

    log.info("Restaurant → Context Engineering Analogy Established")
    for r, c in mapping:
        log.info(f"  {r:18} → {c}")

    result = {"restaurant": restaurant, "context": context, "mapping": mapping}
    save_file(result, os.path.join(example_dir, "analogy_mapping.json"))
    save_file("\n".join([f"{r} → {c}" for r, c in mapping]),
              os.path.join(example_dir, "analogy.txt"))

    log.info("Example 01 completed")
    return result


def example_02_context_formalization() -> Dict:
    example_dir = create_example_dir("example_02_context_formalization")
    log = get_example_logger("Example 02: C = A(c₁,…,c₆) Assembly", example_dir)

    components = [
        ContextComponent(
            component_type='instructions',
            content='You are a helpful AI assistant specializing in data analysis.',
            relevance_score=0.9,
            token_count=50,
            quality_metrics={'clarity': 0.8, 'specificity': 0.7}
        ),
        ContextComponent(
            component_type='knowledge',
            content='Python pandas library provides powerful data manipulation tools including DataFrame operations, groupby functionality, and statistical analysis methods.',
            relevance_score=0.85,
            token_count=80,
            quality_metrics={'accuracy': 0.95, 'completeness': 0.8}
        ),
        ContextComponent(
            component_type='tools',
            content='Available functions: analyze_data(), create_visualization(), statistical_summary()',
            relevance_score=0.75,
            token_count=40,
            quality_metrics={'utility': 0.9, 'accessibility': 0.85}
        ),
        ContextComponent(
            component_type='memory',
            content='User previously asked about data cleaning techniques and expressed preference for visual explanations.',
            relevance_score=0.7,
            token_count=60,
            quality_metrics={'relevance': 0.8, 'recency': 0.9}
        ),
        ContextComponent(
            component_type='state',
            content='User is working on a dataset with 10,000 rows and experiencing performance issues.',
            relevance_score=0.8,
            token_count=45,
            quality_metrics={'accuracy': 0.9, 'timeliness': 0.95}
        ),
        ContextComponent(
            component_type='query',
            content='How can I optimize my pandas operations to handle large datasets more efficiently?',
            relevance_score=1.0,
            token_count=35,
            quality_metrics={'clarity': 0.9, 'specificity': 0.85}
        )
    ]

    assembler = ContextAssemblyFunction(max_tokens=250)
    results = {}
    for strategy in ['linear', 'weighted', 'hierarchical']:
        result = assembler.assemble_context(components, strategy)
        results[strategy] = result
        log.info(f"{strategy.capitalize()} → {result['included_components']} components, "
                 f"{result['total_tokens']} tokens ({result['utilization_rate']:.1%})")

    save_file(results, os.path.join(example_dir, "assembly_results.json"))
    for strat, res in results.items():
        with open(os.path.join(example_dir, f"context_{strat}.md"), "w") as f:
            f.write(f"# {strat.capitalize()} Assembly\n\n")
            f.write(res['assembled_context'])

    log.info("Example 02 completed – all assembly strategies tested")
    return {"components": components, "results": results}


def example_03_optimization_landscape() -> Dict:
    example_dir = create_example_dir("example_03_optimization_landscape")
    log = get_example_logger("Example 03: Optimization Landscape", example_dir)

    R = np.linspace(0, 1, 50)
    C = np.linspace(0, 1, 50)
    R, C = np.meshgrid(R, C)

    def quality(r, c):
        return r*(1-abs(r-0.6)) + c*(1-abs(c-0.4)) + 0.3*r*c + 0.05*np.sin(10*r)*np.cos(10*c)

    Quality = quality(R, C)
    max_idx = np.unravel_index(np.argmax(Quality), Quality.shape)
    opt_r, opt_c = R[max_idx], C[max_idx]
    opt_q = Quality[max_idx]

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(R, C, Quality, cmap='viridis', alpha=0.8)
    ax1.scatter([opt_r], [opt_c], [opt_q], color='red', s=100)
    ax1.set_xlabel('Relevance Weight')
    ax1.set_ylabel('Completeness Weight')
    ax1.set_zlabel('Quality')
    ax1.set_title('Context Optimization Landscape')

    ax2 = fig.add_subplot(122)
    contour = ax2.contour(R, C, Quality, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.scatter([opt_r], [opt_c], color='red', s=100)
    ax2.set_xlabel('Relevance Weight')
    ax2.set_ylabel('Completeness Weight')
    ax2.set_title('Quality Contours')

    plt.tight_layout()
    fig.savefig(os.path.join(example_dir, "optimization_landscape.png"), dpi=150)
    plt.close(fig)

    result = {"optimal_relevance": float(opt_r), "optimal_completeness": float(opt_c), "max_quality": float(opt_q)}
    save_file(result, os.path.join(example_dir, "optimal_params.json"))

    log.info(f"Optimal found → relevance={opt_r:.3f}, completeness={opt_c:.3f}, quality={opt_q:.3f}")
    log.info("Example 03 completed – landscape visualized")
    return result


def example_04_optimization_methods() -> Dict:
    example_dir = create_example_dir("example_04_optimization_methods")
    log = get_example_logger("Example 04: SciPy vs Grid Search vs GD", example_dir)

    # CORRECT WAY: Get the actual list of ContextComponent objects
    components = example_02_context_formalization()["components"]  # ← This is the real list

    optimizer = ContextOptimizer()
    results = {}
    for method in ['scipy', 'grid_search', 'gradient_descent']:
        res = optimizer.optimize_assembly_strategy(components, method)
        results[method] = res
        params = res['optimal_parameters']
        log.info(f"{method:15} → R:{params['relevance_weight']:.3f} "
                 f"C:{params['completeness_weight']:.3f} E:{params['efficiency_weight']:.3f} "
                 f"Quality:{res['optimal_quality']:.3f}")

    df = pd.DataFrame([{
        "method": k,
        **v['optimal_parameters'],
        "quality": v['optimal_quality']
    } for k, v in results.items()])

    # Visualize optimization comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameter comparison
    methods_list = list(results.keys())
    relevance_weights = [results[m]['optimal_parameters']['relevance_weight'] for m in methods_list]
    completeness_weights = [results[m]['optimal_parameters']['completeness_weight'] for m in methods_list]
    efficiency_weights = [results[m]['optimal_parameters']['efficiency_weight'] for m in methods_list]
    
    x = np.arange(len(methods_list))
    width = 0.25
    
    ax1.bar(x - width, relevance_weights, width, label='Relevance', alpha=0.8)
    ax1.bar(x, completeness_weights, width, label='Completeness', alpha=0.8)
    ax1.bar(x + width, efficiency_weights, width, label='Efficiency', alpha=0.8)
    
    ax1.set_xlabel('Optimization Method')
    ax1.set_ylabel('Weight Value')
    ax1.set_title('Optimal Parameter Weights by Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_list)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Quality comparison
    qualities = [results[m]['optimal_quality'] for m in methods_list]
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    bars = ax2.bar(methods_list, qualities, color=colors, alpha=0.8)
    ax2.set_xlabel('Optimization Method')
    ax2.set_ylabel('Optimal Quality Score')
    ax2.set_title('Optimization Quality Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, quality in zip(bars, qualities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{quality:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    fig.savefig(os.path.join(example_dir, "optimization_comparison.png"), dpi=150)
    plt.close(fig)

    save_file(results, os.path.join(example_dir, "optimization_results.json"))
    log.info("Example 04 completed – methods compared")
    
    return results


def example_05_information_theory_basics() -> Dict:
    example_dir = create_example_dir("example_05_information_theory_basics")
    log = get_example_logger("Example 05: Entropy & Mutual Information", example_dir)

    # Example: Information content of different events
    events = [
        ("Sun rises tomorrow", 0.9999),
        ("It rains today", 0.3),
        ("Coin flip is heads", 0.5),
        ("Win lottery", 0.0000001),
        ("AI becomes sentient", 0.001)
    ]
    
    log.info("Information Content Examples:")
    log.info("I(x) = -log₂(P(x)) [measured in bits]")
    log.info()
    
    information_contents = []
    
    for event, probability in events:
        if probability > 0:
            info_content = -np.log2(probability)
            information_contents.append(info_content)
            log.info(f"  {event:25} P={probability:8.1e} → I={info_content:6.2f} bits")
        else:
            log.info(f"  {event:25} P={probability:8.1e} → I=∞ bits")
    
    # Visualize information content vs probability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Information content curve
    probabilities = np.logspace(-6, 0, 1000)
    info_contents = -np.log2(probabilities)
    
    ax1.loglog(probabilities, info_contents)
    ax1.set_xlabel('Probability P(x)')
    ax1.set_ylabel('Information Content -log₂(P(x)) [bits]')
    ax1.set_title('Information Content vs Probability')
    ax1.grid(True, alpha=0.3)
    
    # Mark example events
    event_probs = [p for _, p in events if p > 0]
    event_info = [-np.log2(p) for p in event_probs]
    ax1.scatter(event_probs, event_info, color='red', s=50, zorder=5)
    
    # Entropy calculation example
    log.info("\nEntropy Calculation Example:")
    log.info("H(X) = -Σ P(x) × log₂(P(x))")
    
    # Simple distribution: coin flips
    fair_coin = [0.5, 0.5]
    biased_coin = [0.9, 0.1]
    certain_outcome = [1.0, 0.0]
    
    distributions = {
        'Fair coin': fair_coin,
        'Biased coin': biased_coin,
        'Certain outcome': certain_outcome
    }
    
    entropies = []
    
    for name, dist in distributions.items():
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in dist)
        entropies.append(entropy)
        log.info(f"  {name:15}: H = {entropy:.3f} bits")
    
    # Visualize entropy for different distributions
    ax2.bar(distributions.keys(), entropies, alpha=0.7, color=['blue', 'orange', 'green'])
    ax2.set_ylabel('Entropy H(X) [bits]')
    ax2.set_title('Entropy of Different Distributions')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, entropy in enumerate(entropies):
        ax2.text(i, entropy + 0.05, f'{entropy:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    fig.savefig(os.path.join(example_dir, "information_curve.png"), dpi=150)
    plt.close(fig)

    result = {"events": events}
    save_file(result, os.path.join(example_dir, "information_examples.json"))
    log.info("Example 05 completed – information theory visualized")
    return result


def example_06_context_information_analysis() -> Dict:
    """Demonstrate Information Theory analysis on real context components"""
    example_dir = create_example_dir("example_06_context_information_analysis")
    log = get_example_logger("Example 06: Context Information Theory Analysis", example_dir)

    # Reuse components from example_02
    components = example_02_context_formalization()["components"]
    query = "How can I optimize my pandas operations to handle large datasets more efficiently?"

    analyzer = InformationAnalyzer()
    analysis_results = analyzer.analyze_context_information(components, query)

    log.info("Information theory analysis completed")
    log.info(f"Components analyzed: {len(components)}")
    for item in analysis_results['component_analysis']:
        log.info(
            f"{item['component_type']:15} | "
            f"Entropy: {item['word_entropy']:.2f} | "
            f"MI(query): {item['mutual_information_with_query']:.3f} | "
            f"Density: {item['information_density']:.3f}"
        )

    # === Visualization (identical to the demonstration snippet) ===
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    component_types = [a['component_type'] for a in analysis_results['component_analysis']]
    word_entropies = [a['word_entropy'] for a in analysis_results['component_analysis']]
    query_relevance = analysis_results['query_relevance']
    info_densities = [a['information_density'] for a in analysis_results['component_analysis']]
    mi_matrix = analysis_results['mutual_information_matrix']

    # 1. Word entropy
    ax1.bar(component_types, word_entropies, alpha=0.7, color='skyblue')
    ax1.set_title('Information Content (Word Entropy)')
    ax1.set_ylabel('Entropy [bits]')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. Query relevance
    bars = ax2.bar(component_types, query_relevance, alpha=0.7, color='orange')
    ax2.set_title('Relevance to Query (Mutual Information)')
    ax2.set_ylabel('MI with Query')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    for bar, rel in zip(bars, query_relevance):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rel:.2f}', ha='center', va='bottom')

    # 3. Redundancy matrix
    im = ax3.imshow(mi_matrix, cmap='Blues')
    ax3.set_title('Component Redundancy Matrix')
    ax3.set_xticks(range(len(component_types)))
    ax3.set_yticks(range(len(component_types)))
    ax3.set_xticklabels(component_types, rotation=45)
    ax3.set_yticklabels(component_types)
    for i in range(len(component_types)):
        for j in range(len(component_types)):
            ax3.text(j, i, f'{mi_matrix[i,j]:.2f}',
                     ha="center", va="center",
                     color="black" if mi_matrix[i,j] < 1 else "white")
    plt.colorbar(im, ax=ax3, label='Mutual Information')

    # 4. Density scatter
    ax4.scatter(word_entropies, query_relevance,
                s=[d*1000 for d in info_densities], alpha=0.6, color='green')
    for i, ctype in enumerate(component_types):
        ax4.annotate(ctype, (word_entropies[i], query_relevance[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('Information Content (Word Entropy)')
    ax4.set_ylabel('Query Relevance (MI)')
    ax4.set_title('Content vs Relevance\n(Bubble size = Information Density)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(example_dir, "information_analysis_visualization.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Visualization saved: {fig_path}")

    # Save raw results
    result = {
        "query": query,
        "component_analysis": analysis_results['component_analysis'],
        "redundancy_analysis": analysis_results['redundancy_analysis'],
        "total_mutual_information_with_query": float(sum(query_relevance))
    }
    save_file(result, os.path.join(example_dir, "information_analysis_results.json"))

    log.info("Example 06 completed – Information Theory analysis with visualization")
    return result


def example_07_bayes_theorem_fundamentals() -> Dict:
    """Demonstrate Bayes' theorem fundamentals with context strategy selection"""
    example_dir = create_example_dir("example_07_bayes_theorem_fundamentals")
    log = get_example_logger("Example 07: Bayes' Theorem Fundamentals", example_dir)

    strategies = ['technical_detailed', 'practical_concise', 'balanced_comprehensive']

    priors = {
        'technical_detailed': 0.30,
        'practical_concise': 0.40,
        'balanced_comprehensive': 0.30
    }

    likelihoods = {  # P(positive_feedback | strategy)
        'technical_detailed': 0.70,
        'practical_concise': 0.90,
        'balanced_comprehensive': 0.80
    }

    evidence = sum(priors[s] * likelihoods[s] for s in strategies)

    posteriors = {
        s: (likelihoods[s] * priors[s]) / evidence
        for s in strategies
    }

    log.info("Bayes' Theorem applied to context strategy selection")
    log.info(f"Evidence P(positive_feedback) = {evidence:.4f}")
    for s in strategies:
        log.info(f"P({s} | positive) = {posteriors[s]:.4f}  (prior {priors[s]:.2f} → posterior)")

    # === Visualization ===
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    short_names = ['Technical', 'Practical', 'Balanced']
    x = np.arange(len(strategies))
    width = 0.35

    ax1.bar(x - width/2, [priors[s] for s in strategies], width, label='Prior', alpha=0.8)
    ax1.bar(x + width/2, [posteriors[s] for s in strategies], width, label='Posterior', alpha=0.8)
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Probability')
    ax1.set_title('Bayesian Updating: Prior → Posterior')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    prior_vals = [priors[s] for s in strategies]
    post_vals = [posteriors[s] for s in strategies]
    for i in range(len(strategies)):
        ax2.plot([0, 1], [prior_vals[i], post_vals[i]], 'o-', linewidth=2, markersize=8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Prior', 'Posterior'])
    ax2.set_ylabel('Probability')
    ax2.set_title('Belief Evolution After Positive Feedback')
    ax2.legend(short_names, loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(example_dir, "bayes_theorem_visualization.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Visualization saved: {fig_path}")

    result = {
        "strategies": strategies,
        "priors": priors,
        "likelihoods": likelihoods,
        "evidence": float(evidence),
        "posteriors": posteriors,
        "most_likely_strategy": max(posteriors, key=posteriors.get)
    }
    save_file(result, os.path.join(example_dir, "bayes_theorem_results.json"))

    log.info("Example 07 completed – Bayes' theorem fundamentals with visualization")
    return result


def example_08_bayesian_strategy_learning() -> Dict:
    example_dir = create_example_dir("example_08_bayesian_strategy_learning")
    log = get_example_logger("Example 08: Bayesian Strategy Adaptation", example_dir)

    # Initialize Bayesian learner
    strategies = ['technical_detailed', 'practical_concise', 'balanced_comprehensive', 'user_adapted']
    learner = BayesianContextLearner(strategies)
    
    log.info("Initial strategy beliefs (uniform prior):")
    for strategy, belief in learner.strategy_beliefs.items():
        log.info(f"  {strategy:25}: {belief:.3f}")
    
    # Simulate learning from feedback
    feedback_scenarios = [
        ('practical_concise', 0.9),      # Positive feedback
        ('technical_detailed', 0.3),    # Negative feedback
        ('practical_concise', 0.8),     # Positive feedback
        ('balanced_comprehensive', 0.7), # Positive feedback
        ('user_adapted', 0.95),         # Very positive feedback
        ('technical_detailed', 0.4),    # Negative feedback
        ('user_adapted', 0.9),          # Positive feedback
        ('practical_concise', 0.85),    # Positive feedback
    ]
    
    log.info(f"\nSimulating learning from {len(feedback_scenarios)} feedback instances...")
    
    # Track belief evolution
    belief_evolution = []
    uncertainties = []
    
    for i, (strategy, feedback) in enumerate(feedback_scenarios):
        learner.update_strategy_beliefs(strategy, feedback)
        belief_evolution.append(learner.strategy_beliefs.copy())
        uncertainties.append(learner.get_strategy_uncertainty())
        
        log.info(f"\nStep {i+1}: Used {strategy}, feedback = {feedback:.1f}")
        log.info("Updated beliefs:")
        for strat, belief in learner.strategy_beliefs.items():
            log.info(f"  {strat:25}: {belief:.3f}")
        log.info(f"Uncertainty (entropy): {uncertainties[-1]:.3f}")
    
    # Visualize learning evolution
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Belief evolution over time
    steps = range(len(belief_evolution))
    for strategy in strategies:
        beliefs = [beliefs_dict[strategy] for beliefs_dict in belief_evolution]
        ax1.plot(steps, beliefs, 'o-', label=strategy.replace('_', ' ').title(), linewidth=2)
    
    ax1.set_xlabel('Learning Step')
    ax1.set_ylabel('Belief Probability')
    ax1.set_title('Strategy Belief Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Uncertainty reduction over time
    ax2.plot(steps, uncertainties, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Learning Step')
    ax2.set_ylabel('Uncertainty (Entropy)')
    ax2.set_title('Learning Reduces Uncertainty')
    ax2.grid(True, alpha=0.3)
    
    # 3. Final strategy beliefs
    final_beliefs = learner.strategy_beliefs
    strategy_names = [s.replace('_', '\n') for s in strategies]
    belief_values = list(final_beliefs.values())
    
    bars = ax3.bar(strategy_names, belief_values, alpha=0.7, 
                   color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax3.set_ylabel('Final Belief Probability')
    ax3.set_title('Learned Strategy Preferences')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, belief_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Component relevance learning demo
    log.info("\nComponent Relevance Learning Demo:")
    
    components = ['technical_details', 'code_examples', 'conceptual_explanation', 'performance_tips']
    
    # Simulate relevance evidence
    relevance_evidence = [0.8, 0.9, 0.6, 0.95]  # Different evidence strengths
    
    relevance_estimates = []
    confidence_widths = []
    
    for component, evidence in zip(components, relevance_evidence):
        learner.update_component_relevance(component, evidence)
        estimate, width = learner.get_component_relevance_estimate(component)
        relevance_estimates.append(estimate)
        confidence_widths.append(width)
        
        log.info(f"  {component:20}: estimate = {estimate:.3f}, confidence width = {width:.3f}")
    
    # Plot component relevance estimates with confidence intervals
    x_pos = np.arange(len(components))
    ax4.bar(x_pos, relevance_estimates, alpha=0.7, color='lightblue')
    ax4.errorbar(x_pos, relevance_estimates, yerr=confidence_widths, 
                fmt='none', color='black', capsize=5)
    
    ax4.set_xlabel('Component')
    ax4.set_ylabel('Relevance Estimate')
    ax4.set_title('Component Relevance with Uncertainty')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([c.replace('_', '\n') for c in components])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(example_dir, "belief_evolution.png"), dpi=150)
    plt.close(fig)

    history = []
    for strat, score in feedback_scenarios:
        learner.update_strategy_beliefs(strat, score)
        history.append(learner.strategy_beliefs.copy())
        log.info(f"Feedback {strat}={score:.1f} → best: {learner.select_best_strategy()[0]}")

    best_strategy, confidence = learner.select_best_strategy()
    uncertainty = learner.get_strategy_uncertainty()
    uncertainty_desc = ""
    if uncertainty < 1.0:
        uncertainty_desc = "Low uncertainty: Confident in strategy selection"
    elif uncertainty < 2.0:
        uncertainty_desc = "Medium uncertainty: Some confidence in strategy selection"
    else:
        uncertainty_desc = "High uncertainty: Need more evidence for confident selection"

    result = {
        "best_strategy": best_strategy,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "uncertainty_desc": uncertainty_desc,
        "history": history,
    }
    save_file(result, os.path.join(example_dir, "bayesian_results.json"))

    log.info(f"Example 06 completed – learned strategy: {best_strategy} ({confidence:.3f} confidence)")
    return result


def example_09_integrated_mathematical_system() -> Dict:
    example_dir = create_example_dir("example_09_integrated_mathematical_system")
    log = get_example_logger("Example 07: Full Integrated Context Engineer", example_dir)

    # Initialize integrated system
    engineer = IntegratedContextEngineer(max_tokens=300)
    
    # Use sample components and query from previous demonstrations
    components = example_02_context_formalization()["components"]
    query = "How can I optimize my pandas operations to handle large datasets more efficiently?"
    
    # First iteration - no feedback
    log.info("=== FIRST ITERATION (No Prior Feedback) ===")
    result1 = engineer.engineer_context(components, query)
    
    log.info(f"\nIntegrated Quality Score: {result1['mathematical_quality_score']:.3f}")
    
    # Simulate user feedback and second iteration
    log.info("\n=== SECOND ITERATION (With User Feedback) ===")
    simulated_feedback = 0.8  # Positive feedback
    result2 = engineer.engineer_context(components, query, simulated_feedback)
    
    log.info(f"\nIntegrated Quality Score: {result2['mathematical_quality_score']:.3f}")

    improvement = result2['mathematical_quality_score'] - result1['mathematical_quality_score']
    log.info(f"Quality improved by {improvement:+.3f} after learning")
    
    # Visualize integrated results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Quality evolution
    quality_scores = [result1['mathematical_quality_score'], result2['mathematical_quality_score']]
    iterations = [1, 2]
    
    ax1.plot(iterations, quality_scores, 'bo-', linewidth=3, markersize=8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Integrated Quality Score')
    ax1.set_title('Mathematical Quality Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(quality_scores) * 1.1)
    
    # 2. Component contributions
    frameworks = ['Assembly', 'Information', 'Optimization']
    
    # Extract individual contributions for visualization
    assembly_scores = [0.4, 0.5]  # Simulated improvement
    info_scores = [0.6, 0.7]
    opt_scores = [0.5, 0.6]
    
    x = np.arange(len(frameworks))
    width = 0.35
    
    ax2.bar(x - width/2, [assembly_scores[0], info_scores[0], opt_scores[0]], 
           width, label='Iteration 1', alpha=0.7)
    ax2.bar(x + width/2, [assembly_scores[1], info_scores[1], opt_scores[1]], 
           width, label='Iteration 2', alpha=0.7)
    
    ax2.set_xlabel('Mathematical Framework')
    ax2.set_ylabel('Component Score')
    ax2.set_title('Framework Contribution Analysis')
    ax2.set_xticks(x)
    ax2.set_xticklabels(frameworks)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Bayesian learning progress
    strategy_beliefs = result2['bayesian_insights']
    
    # Show uncertainty reduction
    uncertainty_values = [2.0, strategy_beliefs['uncertainty']]  # Initial vs final
    
    ax3.bar(['Initial', 'After Learning'], uncertainty_values, 
           color=['red', 'green'], alpha=0.7)
    ax3.set_ylabel('Strategy Uncertainty (Entropy)')
    ax3.set_title('Bayesian Learning: Uncertainty Reduction')
    ax3.grid(True, alpha=0.3)
    
    # 4. Optimization landscape projection
    # Create a simple 2D projection of the optimization space
    x_vals = np.linspace(0, 1, 50)
    y_vals = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Simulated quality function
    Z = np.sin(np.pi * X) * np.cos(np.pi * Y) * 0.5 + 0.5
    
    contour = ax4.contour(X, Y, Z, levels=10, alpha=0.6)
    ax4.clabel(contour, inline=True, fontsize=8)
    
    # Mark optimization results
    opt_params1 = result1['optimization_results']['optimal_parameters']
    opt_params2 = result2['optimization_results']['optimal_parameters']
    
    ax4.scatter([opt_params1['relevance_weight']], [opt_params1['completeness_weight']], 
               color='red', s=100, label='Iteration 1', marker='o')
    ax4.scatter([opt_params2['relevance_weight']], [opt_params2['completeness_weight']], 
               color='blue', s=100, label='Iteration 2', marker='s')
    
    ax4.set_xlabel('Relevance Weight')
    ax4.set_ylabel('Completeness Weight')
    ax4.set_title('Optimization Trajectory')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(example_dir, "integrated_framework_results.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Visualization saved: integrated_framework_results.png")

    save_file({"iteration_1": result1, "iteration_2": result2},
              os.path.join(example_dir, "integrated_results.json"))

    log.info("Example 07 completed – all four pillars integrated + visualized")
    return {"results": [result1, result2], "improvement": improvement}

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    main_logger.info("Starting Mathematical Foundations Lab – Full Run")

    example_01_restaurant_analogy()
    example_02_context_formalization()
    example_03_optimization_landscape()
    example_04_optimization_methods()
    example_05_information_theory_basics()
    example_06_context_information_analysis()
    example_07_bayes_theorem_fundamentals()
    example_08_bayesian_strategy_learning()
    example_09_integrated_mathematical_system()

    main_logger.info("=" * 80)
    main_logger.info("LAB COMPLETED – All 9 examples saved")
    main_logger.info(f"Results: {BASE_OUTPUT_DIR}")
    main_logger.info("=" * 80)


if __name__ == "__main__":
    main()
