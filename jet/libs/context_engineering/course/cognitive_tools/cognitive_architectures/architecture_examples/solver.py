import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any
import json
from jet.libs.context_engineering.course.cognitive_tools.cognitive_architectures.architecture_examples.core import ProtocolShell, SemanticField

# =============================================================================
# SOLVER ARCHITECTURE IMPLEMENTATION
# =============================================================================

class CognitiveToolsLibrary:
    """Implementation of cognitive tools for problem-solving."""
    
    @staticmethod
    def understand_question(question: str, domain: str = None) -> Dict[str, Any]:
        """
        Break down and comprehend a problem statement.
        
        Args:
            question: The problem to be understood
            domain: Optional domain context
            
        Returns:
            dict: Structured problem understanding
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Break down and comprehend the problem thoroughly",
            input_params={
                "question": question,
                "domain": domain if domain else "general"
            },
            process_steps=[
                {"action": "extract", "description": "Identify key components of the problem"},
                {"action": "identify", "description": "Detect variables, constants, and unknowns"},
                {"action": "determine", "description": "Identify goals and objectives"},
                {"action": "recognize", "description": "Identify constraints and conditions"},
                {"action": "classify", "description": "Classify problem type and domain"}
            ],
            output_spec={
                "components": "Identified key elements",
                "variables": "Detected variables and unknowns",
                "goals": "Primary objectives to achieve",
                "constraints": "Limitations and conditions",
                "problem_type": "Classification of problem"
            }
        )
        
        # Execute protocol
        return protocol.execute()
    
    @staticmethod
    def decompose_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompose a complex problem into simpler subproblems.
        
        Args:
            problem: Structured problem representation
            
        Returns:
            dict: Decomposed problem structure
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Decompose complex problem into manageable subproblems",
            input_params={"problem": problem},
            process_steps=[
                {"action": "analyze", "description": "Analyze problem structure"},
                {"action": "identify", "description": "Identify natural subproblems"},
                {"action": "organize", "description": "Determine subproblem dependencies"},
                {"action": "simplify", "description": "Reduce complexity of each subproblem"}
            ],
            output_spec={
                "subproblems": "List of identified subproblems",
                "dependencies": "Relationships between subproblems",
                "sequence": "Recommended solution sequence",
                "simplification": "How each subproblem is simplified"
            }
        )
        
        # Execute protocol
        return protocol.execute()
    
    @staticmethod
    def step_by_step(problem: Dict[str, Any], approach: str) -> Dict[str, Any]:
        """
        Generate a step-by-step solution plan.
        
        Args:
            problem: Structured problem representation
            approach: Solution approach to use
            
        Returns:
            dict: Step-by-step solution
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Generate detailed step-by-step solution",
            input_params={
                "problem": problem,
                "approach": approach
            },
            process_steps=[
                {"action": "plan", "description": "Plan solution steps"},
                {"action": "execute", "description": "Execute each step in sequence"},
                {"action": "track", "description": "Track progress and intermediate results"},
                {"action": "verify", "description": "Verify each step's correctness"}
            ],
            output_spec={
                "steps": "Ordered solution steps",
                "explanation": "Explanation for each step",
                "intermediate_results": "Results after each step",
                "final_solution": "Complete solution"
            }
        )
        
        # Execute protocol
        return protocol.execute()
    
    @staticmethod
    def verify_solution(problem: Dict[str, Any], solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the correctness of a solution.
        
        Args:
            problem: Structured problem representation
            solution: Proposed solution
            
        Returns:
            dict: Verification results
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Verify solution correctness and completeness",
            input_params={
                "problem": problem,
                "solution": solution
            },
            process_steps=[
                {"action": "check", "description": "Check solution against problem constraints"},
                {"action": "test", "description": "Test solution with examples or edge cases"},
                {"action": "analyze", "description": "Analyze for errors or inefficiencies"},
                {"action": "evaluate", "description": "Evaluate overall solution quality"}
            ],
            output_spec={
                "is_correct": "Whether the solution is correct",
                "verification_details": "Details of verification process",
                "errors": "Any identified errors",
                "improvements": "Potential improvements",
                "confidence": "Confidence in solution correctness"
            }
        )
        
        # Execute protocol
        return protocol.execute()

class MetaCognitiveController:
    """Implementation of metacognitive monitoring and regulation."""
    
    def __init__(self):
        """Initialize the metacognitive controller."""
        self.state = {
            "current_stage": None,
            "progress": {},
            "obstacles": [],
            "strategy_adjustments": [],
            "insights": []
        }
    
    def monitor(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor progress and detect obstacles.
        
        Args:
            phase_results: Results from current problem-solving phase
            
        Returns:
            dict: Monitoring assessment
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Track progress and identify obstacles",
            input_params={
                "phase": self.state["current_stage"],
                "results": phase_results
            },
            process_steps=[
                {"action": "assess", "description": "Evaluate progress against expected outcomes"},
                {"action": "detect", "description": "Identify obstacles, challenges, or limitations"},
                {"action": "identify", "description": "Identify uncertainty or knowledge gaps"},
                {"action": "measure", "description": "Measure confidence in current approach"}
            ],
            output_spec={
                "progress_assessment": "Evaluation of current progress",
                "obstacles": "Identified challenges or blockers",
                "uncertainty": "Areas of limited confidence",
                "recommendations": "Suggested adjustments"
            }
        )
        
        # Execute protocol
        monitoring_results = protocol.execute()
        
        # === FIX START ===
        # Safely update progress only if key exists
        progress_assessment = monitoring_results.get("progress_assessment")
        if progress_assessment is not None and self.state["current_stage"]:
            self.state["progress"][self.state["current_stage"]] = progress_assessment
        # === FIX END ===
        
        if "obstacles" in monitoring_results and isinstance(monitoring_results["obstacles"], list):
            self.state["obstacles"].extend(monitoring_results["obstacles"])
        
        return monitoring_results
    
    def regulate(self, monitoring_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust strategy based on monitoring.
        
        Args:
            monitoring_assessment: Results from monitoring
            
        Returns:
            dict: Strategy adjustments
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Adjust strategy to overcome obstacles",
            input_params={
                "current_phase": self.state["current_stage"],
                "assessment": monitoring_assessment,
                "history": self.state
            },
            process_steps=[
                {"action": "evaluate", "description": "Evaluate current strategy effectiveness"},
                {"action": "generate", "description": "Generate alternative approaches"},
                {"action": "select", "description": "Select most promising adjustments"},
                {"action": "formulate", "description": "Formulate implementation plan"}
            ],
            output_spec={
                "strategy_assessment": "Evaluation of current strategy",
                "adjustments": "Recommended strategy changes",
                "implementation": "How to apply adjustments",
                "expected_outcomes": "Anticipated improvements"
            }
        )
        
        # Execute protocol
        regulation_results = protocol.execute()
        
        # Update state with regulation results
        if "adjustments" in regulation_results:
            self.state["strategy_adjustments"].append(regulation_results["adjustments"])
        
        return regulation_results
    
    def reflect(self, complete_process: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the entire problem-solving process.
        
        Args:
            complete_process: The full problem-solving trace
            
        Returns:
            dict: Reflection insights and learning
        """
        # Create protocol shell
        protocol = ProtocolShell(
            intent="Extract insights and improve future problem-solving",
            input_params={
                "complete_process": complete_process
            },
            process_steps=[
                {"action": "analyze", "description": "Analyze effectiveness of overall approach"},
                {"action": "identify", "description": "Identify strengths and weaknesses"},
                {"action": "extract", "description": "Extract generalizable patterns and insights"},
                {"action": "formulate", "description": "Formulate lessons for future problems"}
            ],
            output_spec={
                "effectiveness": "Assessment of problem-solving approach",
                "strengths": "What worked particularly well",
                "weaknesses": "Areas for improvement",
                "patterns": "Identified recurring patterns",
                "insights": "Key learnings",
                "future_recommendations": "How to improve future problem-solving"
            }
        )
        
        # Execute protocol
        reflection_results = protocol.execute()
        
        # Update state with reflection results
        if "insights" in reflection_results:
            self.state["insights"] = reflection_results["insights"]
        
        return reflection_results

class SolverArchitecture:
    """Complete implementation of the Solver Architecture."""
    
    def __init__(self):
        """Initialize the solver architecture."""
        self.tools_library = CognitiveToolsLibrary()
        self.metacognitive_controller = MetaCognitiveController()
        # High-dim field used by the normal solve() workflow (embedding style)
        self.field = SemanticField(dimensions=128, name="solution_field")
        self.session_history = []
    
    def solve(self, problem: str, domain: str = None) -> Dict[str, Any]:
        """
        Solve a problem using the complete architecture.
        
        Args:
            problem: Problem statement
            domain: Optional domain context
            
        Returns:
            dict: Solution and reasoning trace
        """
        # Initialize session
        session = {
            "problem": problem,
            "domain": domain,
            "stages": {},
            "solution": None,
            "meta": {},
            "field_state": {}
        }
        
        # 1. UNDERSTAND stage
        self.metacognitive_controller.state["current_stage"] = "understand"
        understanding = self.tools_library.understand_question(problem, domain)
        session["stages"]["understand"] = understanding
        
        # Monitor understanding progress
        understanding_assessment = self.metacognitive_controller.monitor(understanding)
        
        # If obstacles detected, adjust strategy
        if understanding_assessment.get("obstacles"):
            understanding_adjustment = self.metacognitive_controller.regulate(understanding_assessment)
            # In a real implementation, would apply adjustments to understanding
        
        # 2. ANALYZE stage
        self.metacognitive_controller.state["current_stage"] = "analyze"
        analysis = self.tools_library.decompose_problem(understanding)
        session["stages"]["analyze"] = analysis
        
        # Monitor analysis progress
        analysis_assessment = self.metacognitive_controller.monitor(analysis)
        
        # If obstacles detected, adjust strategy
        if analysis_assessment.get("obstacles"):
            analysis_adjustment = self.metacognitive_controller.regulate(analysis_assessment)
            # In a real implementation, would apply adjustments to analysis
        
        # Create solution approach
        approach = analysis.get("approach", "step_by_step")
        
        # 3. SOLVE stage
        self.metacognitive_controller.state["current_stage"] = "solve"
        solution = self.tools_library.step_by_step(understanding, approach)
        session["stages"]["solve"] = solution
        
        # Monitor solution progress
        solution_assessment = self.metacognitive_controller.monitor(solution)
        
        # If obstacles detected, adjust strategy
        if solution_assessment.get("obstacles"):
            solution_adjustment = self.metacognitive_controller.regulate(solution_assessment)
            # In a real implementation, would apply adjustments to solution
        
        # 4. VERIFY stage
        self.metacognitive_controller.state["current_stage"] = "verify"
        verification = self.tools_library.verify_solution(understanding, solution)
        session["stages"]["verify"] = verification
        
        # Monitor verification progress
        verification_assessment = self.metacognitive_controller.monitor(verification)
        
        # Final solution
        session["solution"] = solution.get("final_solution", "Solution not found")
        
        # Meta-cognitive reflection
        reflection = self.metacognitive_controller.reflect({
            "understanding": understanding,
            "analysis": analysis,
            "solution": solution,
            "verification": verification
        })
        
        session["meta"] = {
            "progress": self.metacognitive_controller.state["progress"],
            "obstacles": self.metacognitive_controller.state["obstacles"],
            "strategy_adjustments": self.metacognitive_controller.state["strategy_adjustments"],
            "insights": reflection.get("insights", [])
        }
        
        # Update field state
        self.update_field_from_solution(understanding, solution)
        session["field_state"] = {
            "attractors": len(self.field.attractors),
            "trajectories": len(self.field.trajectories)
        }
        
        # Add to session history
        self.session_history.append(session)
        
        return session
    
    def update_field_from_solution(self, understanding: Dict[str, Any], solution: Dict[str, Any]):
        """
        Update the semantic field based on the problem and solution.
        
        Args:
            understanding: Problem understanding
            solution: Problem solution
        """
        # Add problem as attractor
        problem_type = understanding.get("problem_type", "unknown")
        self.field.add_attractor(f"Problem: {problem_type}", 
                               np.random.normal(0, 1, self.field.dimensions),
                               strength=0.8)
        
        # Add solution approach as attractor
        solution_approach = solution.get("approach", "unknown")
        self.field.add_attractor(f"Approach: {solution_approach}",
                               np.random.normal(0, 1, self.field.dimensions),
                               strength=1.0)
        
        # Simulate a solution trajectory
        start_state = np.random.normal(0, 1, self.field.dimensions)
        start_state = start_state / np.linalg.norm(start_state)
        self.field.calculate_trajectory(start_state, steps=10)
    
    def visualize_solution_process(self, session_index: int = -1) -> plt.Figure:
        """
        Visualize the solution process from a session.
        
        Args:
            session_index: Index of session to visualize
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        # Get the specified session
        if not self.session_history:
            raise ValueError("No solution sessions available for visualization")
        
        session = self.session_history[session_index]
        
        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Solution Process for Problem: {session['problem'][:50]}...", fontsize=16)
        
        # Plot 1: Problem understanding visualization (top left)
        understanding = session["stages"].get("understand", {})
        if understanding:
            # Create a simple graph representation of the problem components
            G = nx.DiGraph()
            
            # Add problem node
            G.add_node("Problem", pos=(0, 0))
            
            # Add component nodes
            components = understanding.get("components", [])
            if isinstance(components, list):
                for i, component in enumerate(components):
                    G.add_node(f"Component {i+1}: {component}", pos=(1, i - len(components)/2 + 0.5))
                    G.add_edge("Problem", f"Component {i+1}: {component}")
            
            # Add variable nodes
            variables = understanding.get("variables", [])
            if isinstance(variables, list):
                for i, variable in enumerate(variables):
                    G.add_node(f"Variable: {variable}", pos=(2, i - len(variables)/2 + 0.5))
                    G.add_edge("Problem", f"Variable: {variable}")
            
            # Draw the graph
            pos = nx.get_node_attributes(G, 'pos')
            nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', 
                   font_size=8, font_weight='bold', ax=axs[0, 0])
            
            axs[0, 0].set_title("Problem Understanding")
        else:
            axs[0, 0].text(0.5, 0.5, "No understanding data available", 
                          ha='center', va='center', fontsize=12)
        
        # Plot 2: Solution approach visualization (top right)
        analysis = session["stages"].get("analyze", {})
        if analysis:
            # Create a simple graph of the decomposed problem
            G = nx.DiGraph()
            
            # Add main problem node
            G.add_node("Main Problem", pos=(0, 0))
            
            # Add subproblem nodes
            subproblems = analysis.get("subproblems", [])
            if isinstance(subproblems, list):
                for i, subproblem in enumerate(subproblems):
                    G.add_node(f"Subproblem {i+1}", pos=(1, i - len(subproblems)/2 + 0.5))
                    G.add_edge("Main Problem", f"Subproblem {i+1}")
            
            # Draw the graph
            pos = nx.get_node_attributes(G, 'pos')
            nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightgreen', 
                   font_size=10, font_weight='bold', ax=axs[0, 1])
            
            axs[0, 1].set_title("Problem Decomposition")
        else:
            axs[0, 1].text(0.5, 0.5, "No analysis data available", 
                          ha='center', va='center', fontsize=12)
        
        # Plot 3: Solution steps visualization (bottom left)
        solution = session["stages"].get("solve", {})
        if solution:
            # Create a flowchart of solution steps
            steps = solution.get("steps", [])
            if isinstance(steps, list):
                G = nx.DiGraph()
                
                # Add step nodes in a vertical flow
                for i, step in enumerate(steps):
                    G.add_node(f"Step {i+1}", pos=(0, -i))
                    if i > 0:
                        G.add_edge(f"Step {i}", f"Step {i+1}")
                
                # Draw the graph
                pos = nx.get_node_attributes(G, 'pos')
                nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightsalmon', 
                       font_size=10, font_weight='bold', ax=axs[1, 0])
                
                # Add step descriptions as annotations
                for i, step in enumerate(steps):
                    if isinstance(step, str):
                        description = step
                    elif isinstance(step, dict) and "description" in step:
                        description = step["description"]
                    else:
                        description = f"Step {i+1}"
                    
                    axs[1, 0].annotate(description, xy=(0.2, -i), xycoords='data',
                                     fontsize=8, ha='left', va='center')
            
            axs[1, 0].set_title("Solution Steps")
        else:
            axs[1, 0].text(0.5, 0.5, "No solution steps available", 
                          ha='center', va='center', fontsize=12)
        
        # Plot 4: Metacognitive monitoring visualization (bottom right)
        meta = session.get("meta", {})
        if meta:
            # Create a grid to show metacognitive elements
            data = []
            labels = []
            
            # Process obstacles
            obstacles = meta.get("obstacles", [])
            if obstacles:
                for i, obstacle in enumerate(obstacles[:5]):  # Limit to 5 for clarity
                    data.append(0.7)  # Arbitrary value for visualization
                    if isinstance(obstacle, str):
                        labels.append(f"Obstacle: {obstacle}")
                    else:
                        labels.append(f"Obstacle {i+1}")
            
            # Process strategy adjustments
            adjustments = meta.get("strategy_adjustments", [])
            if adjustments:
                for i, adjustment in enumerate(adjustments[:5]):  # Limit to 5 for clarity
                    data.append(0.5)  # Arbitrary value for visualization
                    if isinstance(adjustment, str):
                        labels.append(f"Adjustment: {adjustment}")
                    else:
                        labels.append(f"Adjustment {i+1}")
            
            # Process insights
            insights = meta.get("insights", [])
            if insights:
                for i, insight in enumerate(insights[:5]):  # Limit to 5 for clarity
                    data.append(0.9)  # Arbitrary value for visualization
                    if isinstance(insight, str):
                        labels.append(f"Insight: {insight}")
                    else:
                        labels.append(f"Insight {i+1}")
            
            # Create horizontal bar chart
            if data and labels:
                y_pos = np.arange(len(labels))
                axs[1, 1].barh(y_pos, data, align='center')
                axs[1, 1].set_yticks(y_pos)
                axs[1, 1].set_yticklabels(labels, fontsize=8)
                axs[1, 1].invert_yaxis()  # Labels read top-to-bottom
            
            axs[1, 1].set_title("Metacognitive Monitoring")
        else:
            axs[1, 1].text(0.5, 0.5, "No metacognitive data available", 
                          ha='center', va='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        
        return fig

# Solver Example Functions

def solver_example_math_problem():
    """Example: Solving a complex mathematical problem."""
    print("\n===== SOLVER EXAMPLE: COMPLEX MATH PROBLEM =====")
    
    # Initialize the solver architecture
    solver = SolverArchitecture()
    
    # Define a complex math problem
    problem = "Find all values of x that satisfy the equation 2x^3 - 9x^2 + 12x - 5 = 0"
    
    # Solve the problem
    print(f"Solving problem: {problem}")
    solution = solver.solve(problem, domain="mathematics")
    
    # Print results
    print("\nProblem Understanding:")
    print(json.dumps(solution["stages"]["understand"], indent=2))
    
    print("\nProblem Analysis:")
    print(json.dumps(solution["stages"]["analyze"], indent=2))
    
    print("\nSolution Steps:")
    print(json.dumps(solution["stages"]["solve"], indent=2))
    
    print("\nVerification:")
    print(json.dumps(solution["stages"]["verify"], indent=2))
    
    print("\nMeta-cognitive Insights:")
    print(json.dumps(solution["meta"]["insights"], indent=2))
    
    # Visualize the solution process
    fig = solver.visualize_solution_process()
    plt.show()
    
    # Also visualize the field
    field_fig = solver.field.visualize()
    plt.show()
    
    return solution

def solver_example_algorithmic_design():
    """Example: Designing an algorithm for a complex problem."""
    print("\n===== SOLVER EXAMPLE: ALGORITHM DESIGN =====")
    
    # Initialize the solver architecture
    solver = SolverArchitecture()
    
    # Define an algorithm design problem
    problem = """
    Design an efficient algorithm to find the longest increasing subsequence in an array of integers.
    The algorithm should have a time complexity better than O(n²).
    """
    
    # Solve the problem
    print(f"Solving problem: {problem}")
    solution = solver.solve(problem, domain="computer_science")
    
    # Print results
    print("\nProblem Understanding:")
    print(json.dumps(solution["stages"]["understand"], indent=2))
    
    print("\nProblem Analysis:")
    print(json.dumps(solution["stages"]["analyze"], indent=2))
    
    print("\nSolution (Algorithm Design):")
    print(json.dumps(solution["stages"]["solve"], indent=2))
    
    print("\nVerification:")
    print(json.dumps(solution["stages"]["verify"], indent=2))
    
    print("\nMeta-cognitive Insights:")
    print(json.dumps(solution["meta"]["insights"], indent=2))
    
    # Visualize the solution process
    fig = solver.visualize_solution_process()
    plt.show()
    
    return solution

def solver_example_with_field_theory():
    """Example: Using field theory for solution space exploration."""
    print("\n===== SOLVER EXAMPLE: FIELD THEORY EXPLORATION =====")
    solver = SolverArchitecture()

    # Use a dedicated low-dimensional field for clear visualisation
    field = SemanticField(dimensions=3, name="tsp_solution_space")

    field.add_attractor("Greedy / Nearest Neighbour", np.array([0.8, 0.2, 0.1]), strength=0.7)
    field.add_attractor("Dynamic Programming", np.array([0.1, 0.9, 0.2]), strength=0.9)
    field.add_attractor("Divide & Conquer", np.array([0.4, 0.4, 0.8]), strength=0.6)
    field.add_attractor("Graph / Exact TSP", np.array([-0.7, 0.5, 0.1]), strength=0.8)

    problem = """
    Find the most efficient route for a delivery truck that must visit 20 locations
    and return to its starting point, minimizing the total distance traveled.
    """
    print(f"Solving problem: {problem}")
    # Still run the normal solver (uses its own 128-dim field)
    solution = solver.solve(problem, domain="optimization")

    print("\nProblem Understanding:")
    print(json.dumps(solution["stages"]["understand"], indent=2))
    print("\nProblem Analysis:")
    print(json.dumps(solution["stages"]["analyze"], indent=2))
    print("\nSolution Approach:")
    print(json.dumps(solution["stages"]["solve"], indent=2))

    start_positions = [
        np.array([0.9, 0.1, 0.2]),
        np.array([0.2, 0.8, 0.1]),
        np.array([0.3, 0.3, 0.9]),
        np.random.normal(0, 1, 3),
    ]

    print("\nExploring solution space through field trajectories...")
    for i, start_pos in enumerate(start_positions):
        start_pos = start_pos / np.linalg.norm(start_pos)  # normalise to unit sphere
        trajectory = field.calculate_trajectory(start_pos, steps=15)
        end_point = trajectory[-1]
        closest_attractor = None
        min_distance = float('inf')
        for attr_id, attr in field.attractors.items():
            pos = attr["position"]
            dist = np.linalg.norm(pos - end_point)
            if dist < min_distance:
                min_distance = dist
                closest_attractor = attr["concept"]
        print(f"Trajectory {i+1}: Converged to solution approach '{closest_attractor}'")

    field_fig = field.visualize(show_trajectories=True, reduced_dims=2)
    plt.show()
    return solution

if __name__ == "__main__":
    solver_example_math_problem()
    solver_example_algorithmic_design()
    solver_example_with_field_theory()
