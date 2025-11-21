"""
Cognitive Architecture Examples - Practical implementations of Solver, Tutor, and Research architectures.

This module demonstrates how the theoretical frameworks presented in the cognitive architecture 
documentation can be practically implemented and applied to real-world problems.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import random
from datetime import datetime
import math
import re
from collections import defaultdict
from jet.libs.context_engineering.course.cognitive_tools.cognitive_architectures.architecture_examples.core import ProtocolShell, SemanticField, get_current_timestamp, generate_id
from jet.libs.context_engineering.course.cognitive_tools.cognitive_architectures.architecture_examples.solver import solver_example_math_problem, solver_example_algorithmic_design, solver_example_with_field_theory
from jet.libs.context_engineering.course.cognitive_tools.cognitive_architectures.architecture_examples.tutor import tutor_example_math_concept, tutor_example_adaptive_scaffolding, tutor_example_misconception_remediation
from jet.libs.context_engineering.course.cognitive_tools.cognitive_architectures.architecture_examples.research import research_example_literature_review, research_example_hypothesis_development, research_example_interdisciplinary_research
from jet.libs.context_engineering.course.cognitive_tools.cognitive_architectures.architecture_examples.cross_architecture import cross_architecture_integration_example

def main():
    """Run architecture examples."""
    print("=" * 80)
    print("COGNITIVE ARCHITECTURE EXAMPLES")
    print("=" * 80)
    
    # Get example selection from user
    print("\nAvailable Examples:")
    print("  1. Solver: Math Problem")
    print("  2. Solver: Algorithm Design")
    print("  3. Solver: Field Theory")
    print("  4. Tutor: Math Concept")
    print("  5. Tutor: Adaptive Scaffolding")
    print("  6. Tutor: Misconception Remediation")
    print("  7. Research: Literature Review")
    print("  8. Research: Hypothesis Development")
    print("  9. Research: Interdisciplinary Research")
    print(" 10. Cross-Architecture Integration")
    print(" 11. Run All Examples")
    print("  0. Exit")
    
    try:
        choice = input("\nSelect an example to run (0-11): ")
        choice = int(choice.strip())
        
        if choice == 0:
            print("\nExiting...")
            return
        
        if choice == 1 or choice == 11:
            solver_example_math_problem()
        
        if choice == 2 or choice == 11:
            solver_example_algorithmic_design()
        
        if choice == 3 or choice == 11:
            solver_example_with_field_theory()
        
        if choice == 4 or choice == 11:
            tutor_example_math_concept()
        
        if choice == 5 or choice == 11:
            tutor_example_adaptive_scaffolding()
        
        if choice == 6 or choice == 11:
            tutor_example_misconception_remediation()
        
        if choice == 7 or choice == 11:
            research_example_literature_review()
        
        if choice == 8 or choice == 11:
            research_example_hypothesis_development()
        
        if choice == 9 or choice == 11:
            research_example_interdisciplinary_research()
        
        if choice == 10 or choice == 11:
            cross_architecture_integration_example()
        
    except ValueError:
        print("Invalid input. Please enter a number between 0 and 11.")
    
    print("\nExamples completed. Thank you for exploring the cognitive architectures!")

if __name__ == "__main__":
    main()
