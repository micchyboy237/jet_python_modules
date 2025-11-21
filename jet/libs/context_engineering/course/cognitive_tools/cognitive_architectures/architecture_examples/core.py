import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json
import random
from datetime import datetime
import re

# =============================================================================
# CORE UTILITIES AND SHARED COMPONENTS
# =============================================================================

def generate_id() -> str:
    """Generate a unique identifier."""
    return f"id_{random.randint(10000, 99999)}_{int(datetime.now().timestamp())}"

def get_current_timestamp() -> str:
    """Get the current timestamp as a string."""
    return datetime.now().isoformat()

# Mock LLM executor for demonstration
def llm_executor(prompt: str) -> str:
    """
    Simulates execution of prompts through an LLM.
    
    In a real implementation, this would connect to an actual LLM API.
    """
    print(f"\n[LLM EXECUTOR] Processing prompt: {prompt[:100]}...")
    
    # Simulate different responses based on prompt content
    if "understand" in prompt.lower():
        return """{"understanding": {
            "problem_type": "algebraic equation",
            "variables": ["x"],
            "constraints": ["x must be a real number"],
            "goal": "find the value of x that satisfies the equation"
        }}"""
    elif "analyze" in prompt.lower():
        return """{"analysis": {
            "approach": "solve for x by isolating it on one side",
            "steps": ["combine like terms", "divide both sides by coefficient"],
            "expected_complexity": "low"
        }}"""
    elif "synthesize" in prompt.lower():
        return """{"synthesis": {
            "key_findings": ["concept A relates to concept B", "evidence supports hypothesis H1"],
            "patterns": ["temporal trend increasing", "correlation between X and Y"],
            "contradictions": ["study 1 and study 2 have conflicting results"],
            "gaps": ["no research on factor Z"]
        }}"""
    elif "hypothesis" in prompt.lower():
        return """{"hypothesis": {
            "statement": "Increased exposure to X leads to improved Y under conditions Z",
            "variables": {"independent": "X exposure", "dependent": "Y performance", "moderator": "Z conditions"},
            "testability": "high",
            "theoretical_grounding": "consistent with theory T"
        }}"""
    elif "explain" in prompt.lower():
        return """{"explanation": {
            "concept": "mathematical concept clearly explained",
            "examples": ["example 1", "example 2"],
            "analogies": ["real-world analogy that clarifies concept"],
            "potential_misconceptions": ["common misconception addressed"]
        }}"""
    
    # Generic fallback response
    return f"Simulated LLM response for: {prompt[:50]}..."

def execute_protocol(protocol: str) -> Dict[str, Any]:
    """
    Execute a protocol shell and parse the result.
    
    Args:
        protocol: Protocol shell to execute
        
    Returns:
        dict: Parsed protocol results
    """
    # Execute through LLM
    response = llm_executor(protocol)
    
    # Try to parse as JSON
    try:
        if isinstance(response, str):
            # Check if response looks like JSON
            if response.strip().startswith('{') and response.strip().endswith('}'):
                return json.loads(response)
        
        # If already a dict or parsing failed, return as is
        if isinstance(response, dict):
            return response
        
        # Create a simple wrapper if not parseable
        return {"raw_response": response}
        
    except Exception as e:
        print(f"[ERROR] Failed to parse protocol response: {e}")
        return {"error": str(e), "raw_response": response}

# =============================================================================
# PROTOCOL SHELL IMPLEMENTATION
# =============================================================================

class ProtocolShell:
    """Implementation of the protocol shell framework."""
    
    def __init__(self, intent: str, input_params: Dict[str, Any], 
                 process_steps: List[Dict[str, str]], output_spec: Dict[str, str]):
        """
        Initialize a protocol shell.
        
        Args:
            intent: Clear statement of purpose
            input_params: Input parameters
            process_steps: Ordered process steps
            output_spec: Expected output specification
        """
        self.intent = intent
        self.input_params = input_params
        self.process_steps = process_steps
        self.output_spec = output_spec
        self.execution_trace = []
    
    def to_prompt(self) -> str:
        """Convert protocol shell to structured prompt format."""
        # Generate a protocol name based on intent if not explicitly provided
        protocol_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.intent.lower().replace(' ', '_'))
        
        # Format input parameters
        input_params_str = ",\n        ".join([f"{k}={self._format_value(v)}" 
                                             for k, v in self.input_params.items()])
        
        # Format process steps
        process_steps_str = ",\n        ".join([f"/{step['action']}{{action=\"{step['description']}\"" +
                                              (f", tools={self._format_value(step.get('tools', []))}" 
                                               if 'tools' in step else "") + "}"
                                              for step in self.process_steps])
        
        # Format output specification
        output_spec_str = ",\n        ".join([f"{k}=\"{v}\"" 
                                            for k, v in self.output_spec.items()])
        
        # Construct the complete protocol prompt
        prompt = f"""
        /{protocol_name}{{
            intent="{self.intent}",
            input={{
                {input_params_str}
            }},
            process=[
                {process_steps_str}
            ],
            output={{
                {output_spec_str}
            }}
        }}
        """
        
        return prompt
    
    def _format_value(self, v: Any) -> str:
        """Format values appropriately based on type."""
        if isinstance(v, str):
            return f'"{v}"'
        elif isinstance(v, (list, tuple)):
            items = [self._format_value(item) for item in v]
            return f"[{', '.join(items)}]"
        elif isinstance(v, dict):
            items = [f"{k}: {self._format_value(v)}" for k, v in v.items()]
            return f"{{{', '.join(items)}}}"
        else:
            return str(v)
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the protocol shell.
        
        Returns:
            dict: Results of protocol execution
        """
        prompt = self.to_prompt()
        
        # Execute the protocol through LLM
        result = execute_protocol(prompt)
        
        # Record execution trace
        self.execution_trace.append({
            "timestamp": get_current_timestamp(),
            "prompt": prompt,
            "result": result
        })
        
        return result

# =============================================================================
# SEMANTIC FIELD IMPLEMENTATION
# =============================================================================

class SemanticField:
    """Base implementation of semantic field concepts for all architectures."""
    
    def __init__(self, dimensions: int = 128, name: str = "generic_field"):
        """
        Initialize a semantic field.
        
        Args:
            dimensions: Dimensionality of the field
            name: Name of the field
        """
        self.dimensions = dimensions
        self.name = name
        self.field_state = np.zeros((dimensions,))
        self.attractors = {}
        self.boundaries = {}
        self.trajectories = []
        self.residue = []
    
    def add_attractor(self, concept: str, position: np.ndarray = None, 
                     strength: float = 1.0, basin_shape: str = "gaussian") -> Dict[str, Any]:
        """
        Add an attractor to the field.
        
        Args:
            concept: Concept associated with the attractor
            position: Position in field space (random if None)
            strength: Attractor strength
            basin_shape: Shape of attractor basin
            
        Returns:
            dict: Attractor information
        """
        # Generate position if not provided
        if position is None:
            position = np.random.normal(0, 1, self.dimensions)
            position = position / np.linalg.norm(position)
        
        # Ensure position has correct dimensions
        if len(position) != self.dimensions:
            position = np.resize(position, (self.dimensions,))
        
        # Generate ID for attractor
        attractor_id = f"attr_{concept.replace(' ', '_')}_{generate_id()}"
        
        # Create attractor
        self.attractors[attractor_id] = {
            "concept": concept,
            "position": position,
            "strength": strength,
            "basin_shape": basin_shape,
            "created_at": get_current_timestamp()
        }
        
        # Update field state based on new attractor
        self._update_field_state()
        
        return self.attractors[attractor_id]
    
    def _update_field_state(self):
        """Update the field state based on attractors and boundaries."""
        # Start with zero field
        new_state = np.zeros((self.dimensions,))
        
        # Add influence of each attractor
        for attractor_id, attractor in self.attractors.items():
            position = attractor["position"]
            strength = attractor["strength"]
            basin_shape = attractor["basin_shape"]
            
            # Different basin shapes have different influence patterns
            if basin_shape == "gaussian":
                # Gaussian influence that falls off with distance
                for i in range(self.dimensions):
                    # Simplified: just add weighted position
                    new_state[i] += position[i] * strength
            
            # Other basin shapes could be implemented similarly
        
        # Normalize the field state
        if np.linalg.norm(new_state) > 0:
            new_state = new_state / np.linalg.norm(new_state)
        
        # Store the updated state
        self.field_state = new_state
    
    def calculate_trajectory(self, start_state: np.ndarray, steps: int = 10) -> List[np.ndarray]:
        """
        Calculate a trajectory through the field from a starting state.
        
        Args:
            start_state: Starting position in field space
            steps: Number of steps to simulate
            
        Returns:
            list: Sequence of states forming a trajectory
        """
        trajectory = [start_state]
        current_state = start_state.copy()
        
        for _ in range(steps):
            # Calculate the influence of all attractors
            next_state = current_state.copy()
            
            for attractor_id, attractor in self.attractors.items():
                position = attractor["position"]
                strength = attractor["strength"]
                
                # Vector from current state to attractor
                direction = position - current_state
                
                # Normalize
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                
                # Move towards attractor based on strength and distance
                # Simplified model: attraction decreases with square of distance
                distance = np.linalg.norm(position - current_state)
                if distance > 0:
                    attraction = strength / (distance * distance)
                    next_state += direction * attraction
            
            # Normalize the next state
            if np.linalg.norm(next_state) > 0:
                next_state = next_state / np.linalg.norm(next_state)
            
            # Add to trajectory and update current state
            trajectory.append(next_state)
            current_state = next_state
        
        # Record the trajectory
        self.trajectories.append({
            "start": start_state,
            "steps": trajectory,
            "created_at": get_current_timestamp()
        })
        
        return trajectory
    
    def detect_basins(self) -> List[Dict[str, Any]]:
        """
        Detect basin regions in the field.
        
        Returns:
            list: Detected basin regions
        """
        basins = []
        
        # For each attractor, identify its basin of attraction
        for attractor_id, attractor in self.attractors.items():
            # Basin properties would depend on attractor and field state
            basin = {
                "attractor_id": attractor_id,
                "concept": attractor["concept"],
                "center": attractor["position"],
                "radius": 0.2 + 0.3 * attractor["strength"],  # Simplified radius calculation
                "strength": attractor["strength"]
            }
            
            basins.append(basin)
        
        return basins
    
    def visualize(self, show_attractors: bool = True, show_trajectories: bool = True, 
                 reduced_dims: int = 2) -> plt.Figure:
        """
        Visualize the field in reduced dimensions.
        
        Args:
            show_attractors: Whether to show attractors
            show_trajectories: Whether to show trajectories
            reduced_dims: Dimensionality for visualization
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # For visualization, we'll reduce to 2D using a simple approach
        # In a real implementation, PCA or t-SNE would be more appropriate
        
        # Function to reduce dimensions
        def reduce_dims(vector):
            if reduced_dims == 2:
                return vector[:2] if len(vector) >= 2 else np.pad(vector, (0, 2 - len(vector)))
            else:
                return vector[:reduced_dims] if len(vector) >= reduced_dims else np.pad(vector, (0, reduced_dims - len(vector)))
        
        # Plot field boundaries (simplified as a circle for 2D)
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        ax.add_artist(circle)
        
        # Plot attractors
        if show_attractors and self.attractors:
            for attractor_id, attractor in self.attractors.items():
                pos = reduce_dims(attractor["position"])
                strength = attractor["strength"]
                
                # Plot attractor point
                ax.scatter(pos[0], pos[1], s=100 * strength, color='red', alpha=0.7)
                
                # Plot attractor label
                ax.text(pos[0], pos[1], attractor["concept"], fontsize=9, ha='center')
                
                # Plot basin of attraction (simplified as a circle)
                basin_circle = plt.Circle((pos[0], pos[1]), 0.2 * strength, fill=True, 
                                        color='red', alpha=0.1)
                ax.add_artist(basin_circle)
        
        # Plot trajectories
        if show_trajectories and self.trajectories:
            for trajectory in self.trajectories:
                points = [reduce_dims(step) for step in trajectory["steps"]]
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                
                # Plot trajectory line
                ax.plot(x_vals, y_vals, 'b-', alpha=0.5)
                
                # Plot start and end points
                ax.scatter(x_vals[0], y_vals[0], color='green', s=50, label='Start')
                ax.scatter(x_vals[-1], y_vals[-1], color='blue', s=50, label='End')
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # Set title and labels
        ax.set_title(f"Semantic Field: {self.name}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return fig
