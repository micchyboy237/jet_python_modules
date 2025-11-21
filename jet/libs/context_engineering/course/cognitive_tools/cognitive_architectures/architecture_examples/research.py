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

# =============================================================================
# RESEARCH ARCHITECTURE IMPLEMENTATION
# =============================================================================

class ResearchKnowledgeField(SemanticField):
    """Implementation of research domain knowledge field."""
    
    def __init__(self, domain: str, dimensions: int = 128):
        """
        Initialize the research knowledge field.
        
        Args:
            domain: Research domain
            dimensions: Dimensionality of the field
        """
        super().__init__(dimensions=dimensions, name=f"research_field_{domain}")
        self.domain = domain
        self.literature = {}
        self.research_questions = {}
        self.hypotheses = {}
        self.gaps = []
        self.contradictions = []
    
    def add_literature(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Integrate research literature into the knowledge field.
        
        Args:
            papers: Collection of research papers
            
        Returns:
            dict: Updated field state
        """
        # Protocol shell for literature integration
        protocol = ProtocolShell(
            intent="Integrate research literature into knowledge field",
            input_params={
                "papers": papers,
                "current_field": "field_state"
            },
            process_steps=[
                {"action": "extract", "description": "Identify key concepts and findings"},
                {"action": "map", "description": "Position concepts in field space"},
                {"action": "detect", "description": "Identify attractor basins"},
                {"action": "connect", "description": "Establish concept relationships"},
                {"action": "locate", "description": "Identify knowledge boundaries and gaps"}
            ],
            output_spec={
                "updated_field": "New field state with integrated literature",
                "new_concepts": "Newly added concepts",
                "new_attractors": "Newly identified attractor basins",
                "new_boundaries": "Updated knowledge boundaries",
                "new_gaps": "Newly detected knowledge gaps"
            }
        )
        
        # Execute protocol
        integration_results = protocol.execute()
        
        # Store papers in literature collection
        for paper in papers:
            paper_id = paper.get("id", generate_id())
            self.literature[paper_id] = paper
            
            # Add paper as attractor in field
            paper_title = paper.get("title", f"Paper {paper_id}")
            position = np.random.normal(0, 1, self.dimensions)
            position = position / np.linalg.norm(position)
            
            self.add_attractor(
                concept=f"Paper: {paper_title}",
                position=position,
                strength=0.7  # Moderate strength for literature attractors
            )
        
        # Extract potential gaps
        if "new_gaps" in integration_results and isinstance(integration_results["new_gaps"], list):
            for gap in integration_results["new_gaps"]:
                if gap not in self.gaps:
                    self.gaps.append(gap)
        
        # Extract potential contradictions
        contradictions = []
        for i, paper1 in enumerate(papers):
            for paper2 in papers[i+1:]:
                # Simulate contradiction detection (in real implementation would be more sophisticated)
                if random.random() < 0.2:  # 20% chance of contradiction
                    contradiction = {
                        "papers": [paper1.get("id", "unknown"), paper2.get("id", "unknown")],
                        "topic": "research_topic",
                        "description": "Contradictory findings on same phenomenon"
                    }
                    contradictions.append(contradiction)
        
        for contradiction in contradictions:
            if contradiction not in self.contradictions:
                self.contradictions.append(contradiction)
        
        return {
            "papers_added": len(papers),
            "new_gaps": len(self.gaps) - (len(self.gaps) - len(integration_results.get("new_gaps", []))),
            "new_contradictions": len(contradictions),
            "field_update": integration_results
        }
    
    def identify_research_opportunities(self, research_interests: List[str], 
                                      constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Identify promising research opportunities in the field.
        
        Args:
            research_interests: Areas of research interest
            constraints: Optional research constraints
            
        Returns:
            list: Promising research opportunities
        """
        # Protocol shell for opportunity identification
        protocol = ProtocolShell(
            intent="Identify promising research opportunities",
            input_params={
                "knowledge_field": "field_state",
                "research_interests": research_interests,
                "constraints": constraints if constraints else {}
            },
            process_steps=[
                {"action": "analyze", "description": "Examine knowledge gaps"},
                {"action": "explore", "description": "Identify boundary areas"},
                {"action": "evaluate", "description": "Assess attractor interactions"},
                {"action": "match", "description": "Align opportunities with interests"},
                {"action": "prioritize", "description": "Rank by promise and feasibility"}
            ],
            output_spec={
                "opportunities": "Prioritized research opportunities",
                "rationale": "Justification for each opportunity",
                "gap_alignment": "How opportunities address gaps",
                "impact_potential": "Potential research impact",
                "feasibility": "Implementation feasibility assessment"
            }
        )
        
        # Execute protocol
        opportunities = protocol.execute()
        
        # Generate simulated research opportunities
        simulated_opportunities = []
        
        # Create opportunities based on gaps and interests
        for i, interest in enumerate(research_interests[:3]):  # Limit to 3
            # Create a research opportunity
            opportunity = {
                "id": f"opportunity_{i+1}",
                "title": f"Research opportunity related to {interest}",
                "description": f"Investigate the relationship between {interest} and related factors",
                "gap_addressed": self.gaps[i % len(self.gaps)] if self.gaps else "Unknown gap",
                "alignment": random.uniform(0.6, 0.9),
                "feasibility": random.uniform(0.5, 0.9),
                "impact": random.uniform(0.4, 0.95)
            }
            
            simulated_opportunities.append(opportunity)
        
        return simulated_opportunities
    
    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Detect contradictions in the research literature.
        
        Returns:
            list: Detected contradictions
        """
        return self.contradictions
    
    def visualize_research_landscape(self, focus: str = "literature", include_gaps: bool = True) -> plt.Figure:
        """
        Visualize the research knowledge landscape.
        
        Args:
            focus: Focus of visualization (literature, gaps, opportunities)
            include_gaps: Whether to include knowledge gaps
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        # Create base visualization using parent class method
        fig = self.visualize(show_attractors=True, show_trajectories=False)
        
        # Get the current axes
        ax = plt.gca()
        
        # Add gap visualization if requested
        if include_gaps and self.gaps:
            # Visualize gaps as dashed boundaries
            for i, gap in enumerate(self.gaps[:5]):  # Limit to 5 for clarity
                # Create a dashed circle representing the gap
                gap_radius = random.uniform(0.1, 0.3)
                gap_x = random.uniform(-0.8, 0.8)
                gap_y = random.uniform(-0.8, 0.8)
                
                gap_circle = plt.Circle((gap_x, gap_y), gap_radius, fill=False, 
                                      color='red', linestyle='dashed', alpha=0.7)
                ax.add_artist(gap_circle)
                
                # Add gap label
                if isinstance(gap, str):
                    gap_label = gap
                else:
                    gap_label = f"Gap {i+1}"
                
                ax.text(gap_x, gap_y, gap_label, fontsize=8, ha='center', va='center', color='red')
        
        # Update title based on focus
        ax.set_title(f"Research Knowledge Landscape: {focus.capitalize()}")
        
        return fig

class ResearchInquiryModel:
    """Implementation of research question and hypothesis management."""
    
    def __init__(self):
        """Initialize the research inquiry model."""
        self.research_questions = {}
        self.hypotheses = {}
        self.evidence_mappings = {}
        self.inquiry_trajectories = []
    
    def develop_research_question(self, knowledge_field: ResearchKnowledgeField,
                               research_interest: str, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Develop well-formed research question from interest area.
        
        Args:
            knowledge_field: Research knowledge field
            research_interest: General area of interest
            constraints: Optional research constraints
            
        Returns:
            dict: Formulated research question
        """
        # Protocol shell for research question development
        protocol = ProtocolShell(
            intent="Formulate precise research question from interest area",
            input_params={
                "knowledge_field": "field_state",
                "research_interest": research_interest,
                "constraints": constraints if constraints else {}
            },
            process_steps=[
                {"action": "analyze", "description": "Examine knowledge field relevant to interest"},
                {"action": "identify", "description": "Locate knowledge gaps and boundaries"},
                {"action": "formulate", "description": "Craft potential research questions"},
                {"action": "evaluate", "description": "Assess question quality and feasibility"},
                {"action": "refine", "description": "Improve question precision and scope"}
            ],
            output_spec={
                "research_question": "Precisely formulated research question",
                "sub_questions": "Related sub-questions to explore",
                "rationale": "Justification and background",
                "relationship_to_gaps": "How question addresses knowledge gaps",
                "novelty_assessment": "Evaluation of question novelty"
            }
        )
        
        # Execute protocol
        question_results = protocol.execute()
        
        # Store the research question
        question_id = generate_id()
        self.research_questions[question_id] = {
            "question": question_results.get("research_question", f"Research question about {research_interest}"),
            "sub_questions": question_results.get("sub_questions", []),
            "rationale": question_results.get("rationale", ""),
            "gap_relationship": question_results.get("relationship_to_gaps", ""),
            "novelty": question_results.get("novelty_assessment", ""),
            "interest": research_interest,
            "constraints": constraints,
            "state": "active",
            "timestamp": get_current_timestamp()
        }
        
        return {
            "question_id": question_id,
            "question": self.research_questions[question_id]
        }
    
    def develop_hypothesis(self, knowledge_field: ResearchKnowledgeField, 
                         research_question_id: str, hypothesis_type: str = "explanatory") -> Dict[str, Any]:
        """
        Develop testable hypothesis for research question.
        
        Args:
            knowledge_field: Research knowledge field
            research_question_id: ID of the research question
            hypothesis_type: Type of hypothesis to develop
            
        Returns:
            dict: Formulated hypothesis
        """
        # Retrieve the research question
        if research_question_id not in self.research_questions:
            raise ValueError(f"Research question ID {research_question_id} not found")
            
        research_question = self.research_questions[research_question_id]
        
        # Protocol shell for hypothesis development
        protocol = ProtocolShell(
            intent="Formulate testable hypothesis for research question",
            input_params={
                "knowledge_field": "field_state",
                "research_question": research_question,
                "hypothesis_type": hypothesis_type
            },
            process_steps=[
                {"action": "analyze", "description": "Examine relevant theory and evidence"},
                {"action": "formulate", "description": "Craft potential hypotheses"},
                {"action": "evaluate", "description": "Assess testability and explanatory power"},
                {"action": "refine", "description": "Improve precision and falsifiability"},
                {"action": "connect", "description": "Link to existing knowledge"}
            ],
            output_spec={
                "hypothesis": "Precisely formulated hypothesis",
                "alternative_hypotheses": "Alternative explanations to consider",
                "testability": "Assessment of empirical testability",
                "variables": "Key variables and relationships",
                "predictions": "Specific predictions derived from hypothesis",
                "theoretical_grounding": "Connection to existing theory"
            }
        )
        
        # Execute protocol
        hypothesis_results = protocol.execute()
        
        # Store the hypothesis
        hypothesis_id = generate_id()
        self.hypotheses[hypothesis_id] = {
            "hypothesis": hypothesis_results.get("hypothesis", "Hypothesis statement"),
            "alternatives": hypothesis_results.get("alternative_hypotheses", []),
            "testability": hypothesis_results.get("testability", "medium"),
            "variables": hypothesis_results.get("variables", {}),
            "predictions": hypothesis_results.get("predictions", []),
            "theoretical_grounding": hypothesis_results.get("theoretical_grounding", ""),
            "research_question_id": research_question_id,
            "type": hypothesis_type,
            "state": "active",
            "timestamp": get_current_timestamp()
        }
        
        # Link hypothesis to research question
        if "hypotheses" not in self.research_questions[research_question_id]:
            self.research_questions[research_question_id]["hypotheses"] = []
        
        self.research_questions[research_question_id]["hypotheses"].append(hypothesis_id)
        
        return {
            "hypothesis_id": hypothesis_id,
            "hypothesis": self.hypotheses[hypothesis_id]
        }
    
    def refine_hypothesis(self, hypothesis_id: str, refinement_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine an existing hypothesis.
        
        Args:
            hypothesis_id: ID of the hypothesis to refine
            refinement_data: Data for refinement
            
        Returns:
            dict: Refined hypothesis
        """
        # Check if hypothesis exists
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis ID {hypothesis_id} not found")
        
        # Get the original hypothesis
        original_hypothesis = self.hypotheses[hypothesis_id]
        
        # Protocol shell for hypothesis refinement
        protocol = ProtocolShell(
            intent="Refine hypothesis for precision and testability",
            input_params={
                "original_hypothesis": original_hypothesis,
                "refinement_data": refinement_data
            },
            process_steps=[
                {"action": "analyze", "description": "Analyze refinement needs"},
                {"action": "improve", "description": "Improve precision and clarity"},
                {"action": "enhance", "description": "Enhance testability"},
                {"action": "update", "description": "Update variable relationships"},
                {"action": "revise", "description": "Revise predictions"}
            ],
            output_spec={
                "refined_hypothesis": "Improved hypothesis statement",
                "refinement_rationale": "Justification for changes",
                "improved_testability": "Assessment of enhanced testability",
                "updated_variables": "Updated variable definitions",
                "revised_predictions": "Revised empirical predictions"
            }
        )
        
        # Execute protocol
        refinement_results = protocol.execute()
        
        # Create new refined hypothesis
        refined_hypothesis_id = generate_id()
        self.hypotheses[refined_hypothesis_id] = {
            "hypothesis": refinement_results.get("refined_hypothesis", original_hypothesis["hypothesis"]),
            "alternatives": original_hypothesis["alternatives"],
            "testability": refinement_results.get("improved_testability", original_hypothesis["testability"]),
            "variables": refinement_results.get("updated_variables", original_hypothesis["variables"]),
            "predictions": refinement_results.get("revised_predictions", original_hypothesis["predictions"]),
            "theoretical_grounding": original_hypothesis["theoretical_grounding"],
            "research_question_id": original_hypothesis["research_question_id"],
            "refined_from": hypothesis_id,
            "refinement_rationale": refinement_results.get("refinement_rationale", ""),
            "type": original_hypothesis["type"],
            "state": "active",
            "timestamp": get_current_timestamp()
        }
        
        # Update original hypothesis state
        self.hypotheses[hypothesis_id]["state"] = "refined"
        self.hypotheses[hypothesis_id]["refined_to"] = refined_hypothesis_id
        
        # Update the research question to point to the new hypothesis
        research_question_id = original_hypothesis["research_question_id"]
        if research_question_id in self.research_questions:
            if "hypotheses" in self.research_questions[research_question_id]:
                # Replace the old hypothesis with the new one in the list
                hypotheses = self.research_questions[research_question_id]["hypotheses"]
                if hypothesis_id in hypotheses:
                    index = hypotheses.index(hypothesis_id)
                    hypotheses[index] = refined_hypothesis_id
        
        # Record trajectory
        self.inquiry_trajectories.append({
            "type": "hypothesis_refinement",
            "original": hypothesis_id,
            "refined": refined_hypothesis_id,
            "timestamp": get_current_timestamp()
        })
        
        return {
            "hypothesis_id": refined_hypothesis_id,
            "hypothesis": self.hypotheses[refined_hypothesis_id],
            "refinement": {
                "original_id": hypothesis_id,
                "changes": refinement_results
            }
        }

class ResearchSynthesisModel:
    """Implementation of research synthesis capabilities."""
    
    def __init__(self):
        """Initialize the research synthesis model."""
        self.evidence_collection = {}
        self.syntheses = {}
        self.theory_models = {}
        self.contradictions = []
        self.synthesis_trajectories = []
    
    def synthesize_findings(self, knowledge_field: ResearchKnowledgeField, evidence: List[Dict[str, Any]],
                          research_question_id: str = None, synthesis_type: str = "narrative") -> Dict[str, Any]:
        """
        Synthesize research findings into coherent understanding.
        
        Args:
            knowledge_field: Research knowledge field
            evidence: Collection of research findings
            research_question_id: Optional focus research question
            synthesis_type: Type of synthesis to perform
            
        Returns:
            dict: Research synthesis
        """
        # Protocol shell for synthesis
        protocol = ProtocolShell(
            intent="Synthesize research findings into coherent understanding",
            input_params={
                "knowledge_field": "field_state",
                "evidence": evidence,
                "research_question": research_question_id,
                "synthesis_type": synthesis_type
            },
            process_steps=[
                {"action": "organize", "description": "Structure evidence by themes and relationships"},
                {"action": "evaluate", "description": "Assess evidence quality and consistency"},
                {"action": "identify", "description": "Detect patterns and contradictions"},
                {"action": "integrate", "description": "Develop coherent understanding"},
                {"action": "contextualize", "description": "Position within broader knowledge"}
            ],
            output_spec={
                "synthesis": "Integrated understanding of findings",
                "evidence_evaluation": "Assessment of evidence quality",
                "patterns": "Identified patterns and relationships",
                "contradictions": "Unresolved contradictions",
                "gaps": "Remaining knowledge gaps",
                "implications": "Theoretical and practical implications"
            }
        )
        
        # Execute protocol
        synthesis_results = protocol.execute()
        
        # Store the synthesis
        synthesis_id = generate_id()
        self.syntheses[synthesis_id] = {
            "synthesis": synthesis_results.get("synthesis", "Synthesis of findings"),
            "evidence_evaluation": synthesis_results.get("evidence_evaluation", {}),
            "patterns": synthesis_results.get("patterns", []),
            "contradictions": synthesis_results.get("contradictions", []),
            "gaps": synthesis_results.get("gaps", []),
            "implications": synthesis_results.get("implications", []),
            "research_question_id": research_question_id,
            "evidence_ids": [e.get("id", "unknown") for e in evidence],
            "type": synthesis_type,
            "timestamp": get_current_timestamp()
        }
        
        # Update synthesis trajectories
        self.synthesis_trajectories.append({
            "synthesis_id": synthesis_id,
            "timestamp": get_current_timestamp(),
            "action": "creation",
            "type": synthesis_type
        })
        
        # Process contradictions
        if "contradictions" in synthesis_results and isinstance(synthesis_results["contradictions"], list):
            for contradiction in synthesis_results["contradictions"]:
                if contradiction not in self.contradictions:
                    self.contradictions.append(contradiction)
        
        return {
            "synthesis_id": synthesis_id,
            "synthesis": self.syntheses[synthesis_id]
        }
    
    def develop_theoretical_model(self, knowledge_field: ResearchKnowledgeField, 
                               synthesis_ids: List[str], model_type: str = "explanatory") -> Dict[str, Any]:
        """
        Develop theoretical model from research syntheses.
        
        Args:
            knowledge_field: Research knowledge field
            synthesis_ids: IDs of syntheses to incorporate
            model_type: Type of theoretical model
            
        Returns:
            dict: Theoretical model
        """
        # Retrieve syntheses
        syntheses = []
        for synthesis_id in synthesis_ids:
            if synthesis_id in self.syntheses:
                syntheses.append(self.syntheses[synthesis_id])
        
        if not syntheses:
            raise ValueError("No valid synthesis IDs provided")
        
        # Protocol shell for theoretical model development
        protocol = ProtocolShell(
            intent="Develop theoretical model from research syntheses",
            input_params={
                "knowledge_field": "field_state",
                "syntheses": syntheses,
                "model_type": model_type
            },
            process_steps=[
                {"action": "identify", "description": "Extract core concepts and relationships"},
                {"action": "structure", "description": "Organize into coherent theoretical framework"},
                {"action": "evaluate", "description": "Assess explanatory power and consistency"},
                {"action": "contextualize", "description": "Position within existing theory"},
                {"action": "extend", "description": "Generate novel implications and predictions"}
            ],
            output_spec={
                "theoretical_model": "Structured theoretical framework",
                "core_concepts": "Fundamental concepts and definitions",
                "relationships": "Proposed causal or structural relationships",
                "explanatory_power": "Assessment of explanatory scope",
                "falsifiability": "Potential ways to test the theory",
                "novelty": "Unique contributions to theoretical understanding",
                "implications": "Theoretical and practical implications"
            }
        )
        
        # Execute protocol
        model_results = protocol.execute()
        
        # Store the theoretical model
        model_id = generate_id()
        self.theory_models[model_id] = {
            "model": model_results.get("theoretical_model", "Theoretical model"),
            "core_concepts": model_results.get("core_concepts", []),
            "relationships": model_results.get("relationships", []),
            "explanatory_power": model_results.get("explanatory_power", "medium"),
            "falsifiability": model_results.get("falsifiability", []),
            "novelty": model_results.get("novelty", ""),
            "implications": model_results.get("implications", []),
            "synthesis_ids": synthesis_ids,
            "type": model_type,
            "timestamp": get_current_timestamp()
        }
        
        return {
            "model_id": model_id,
            "theoretical_model": self.theory_models[model_id]
        }

class ResearchCommunicationModel:
    """Implementation of research communication capabilities."""
    
    def __init__(self):
        """Initialize the research communication model."""
        self.communications = {}
        self.narratives = {}
        self.visualizations = {}
        self.communication_trajectories = []
    
    def develop_research_narrative(self, knowledge_field: ResearchKnowledgeField, synthesis_id: str,
                                 audience: str = "academic", narrative_type: str = "article") -> Dict[str, Any]:
        """
        Develop research narrative from synthesis.
        
        Args:
            knowledge_field: Research knowledge field
            synthesis_id: ID of the synthesis to communicate
            audience: Target audience
            narrative_type: Type of narrative to develop
            
        Returns:
            dict: Research narrative
        """
        # Retrieve synthesis
        if synthesis_id not in self.synthesis_model.syntheses:
            raise ValueError(f"Synthesis ID {synthesis_id} not found")
        synthesis = self.synthesis_model.syntheses[synthesis_id]
        
        # Protocol shell for narrative development
        protocol = ProtocolShell(
            intent="Develop compelling research narrative from synthesis",
            input_params={
                "knowledge_field": "field_state",
                "synthesis": synthesis,
                "audience": audience,
                "narrative_type": narrative_type
            },
            process_steps=[
                {"action": "structure", "description": "Organize content into narrative flow"},
                {"action": "frame", "description": "Establish framing and significance"},
                {"action": "develop", "description": "Elaborate key points with evidence"},
                {"action": "connect", "description": "Create narrative connections"},
                {"action": "refine", "description": "Enhance clarity and engagement"}
            ],
            output_spec={
                "narrative": "Complete research narrative",
                "structure": "Organizational structure",
                "key_points": "Central arguments and findings",
                "evidence_integration": "How evidence supports narrative",
                "framing": "Contextual framing of research",
                "significance": "Articulation of importance and implications"
            }
        )
        
        # Execute protocol
        narrative_results = protocol.execute()
        
        # Store the narrative
        narrative_id = generate_id()
        self.narratives[narrative_id] = {
            "narrative": narrative_results.get("narrative", "Research narrative"),
            "structure": narrative_results.get("structure", {}),
            "key_points": narrative_results.get("key_points", []),
            "evidence_integration": narrative_results.get("evidence_integration", {}),
            "framing": narrative_results.get("framing", ""),
            "significance": narrative_results.get("significance", ""),
            "synthesis_id": synthesis_id,
            "audience": audience,
            "type": narrative_type,
            "timestamp": get_current_timestamp()
        }
        
        return {
            "narrative_id": narrative_id,
            "narrative": self.narratives[narrative_id]
        }
    
    def create_research_visualization(self, knowledge_field: ResearchKnowledgeField, data: Dict[str, Any],
                                   visualization_type: str = "conceptual", purpose: str = "explanation") -> Dict[str, Any]:
        """
        Create research visualization.
        
        Args:
            knowledge_field: Research knowledge field
            data: Data to visualize
            visualization_type: Type of visualization
            purpose: Purpose of visualization
            
        Returns:
            dict: Research visualization
        """
        # Protocol shell for visualization creation
        protocol = ProtocolShell(
            intent="Create effective research visualization",
            input_params={
                "knowledge_field": "field_state",
                "data": data,
                "visualization_type": visualization_type,
                "purpose": purpose
            },
            process_steps=[
                {"action": "analyze", "description": "Determine appropriate visualization approach"},
                {"action": "structure", "description": "Organize visual elements for clarity"},
                {"action": "design", "description": "Create visualization with appropriate elements"},
                {"action": "annotate", "description": "Add necessary context and explanation"},
                {"action": "evaluate", "description": "Assess effectiveness and clarity"}
            ],
            output_spec={
                "visualization": "Complete visualization specification",
                "design_rationale": "Justification for design choices",
                "key_insights": "Central insights conveyed",
                "interpretation_guide": "How to interpret the visualization",
                "limitations": "Limitations of the visualization"
            }
        )
        
        # Execute protocol
        visualization_results = protocol.execute()
        
        # Store the visualization
        visualization_id = generate_id()
        self.visualizations[visualization_id] = {
            "visualization": visualization_results.get("visualization", {}),
            "design_rationale": visualization_results.get("design_rationale", ""),
            "key_insights": visualization_results.get("key_insights", []),
            "interpretation_guide": visualization_results.get("interpretation_guide", ""),
            "limitations": visualization_results.get("limitations", []),
            "data": data,
            "type": visualization_type,
            "purpose": purpose,
            "timestamp": get_current_timestamp()
        }
        
        return {
            "visualization_id": visualization_id,
            "visualization": self.visualizations[visualization_id]
        }

class ResearchArchitecture:
    """Complete implementation of the Research Architecture."""
    
    def __init__(self, domain: str = "general"):
        """
        Initialize the research architecture.
        
        Args:
            domain: Research domain
        """
        self.knowledge_field = ResearchKnowledgeField(domain=domain)
        self.inquiry_model = ResearchInquiryModel()
        self.synthesis_model = ResearchSynthesisModel()
        self.communication_model = ResearchCommunicationModel()
        self.session_history = []
        
        # Establish references between models
        self.synthesis_model.inquiry_model = self.inquiry_model
        self.communication_model.synthesis_model = self.synthesis_model
    
    def initialize_literature(self, papers: List[Dict[str, Any]]):
        """
        Initialize knowledge field with research literature.
        
        Args:
            papers: Research papers to add
        """
        self.knowledge_field.add_literature(papers)
    
    def conduct_literature_review(self, research_question: str, depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Conduct a literature review on a research question.
        
        Args:
            research_question: The research question
            depth: Depth of the literature review
            
        Returns:
            dict: Literature review results
        """
        # Extract domain from research question (simplified)
        domain = self.knowledge_field.domain
        
        # Create a session record
        session = {
            "type": "literature_review",
            "research_question": research_question,
            "depth": depth,
            "steps": [],
            "results": {},
            "field_updates": {}
        }
        
        # Step 1: Search for relevant literature
        search_results = {
            "query": research_question,
            "domain": domain,
            "sources": list(self.knowledge_field.literature.values())
        }
        session["steps"].append({
            "step": "search",
            "results": {
                "sources_found": len(search_results["sources"])
            }
        })
        
        # Step 2: Screen sources for relevance
        # Simulate screening by randomly selecting a subset
        screened_sources = random.sample(
            search_results["sources"], 
            min(len(search_results["sources"]), 5)
        )
        session["steps"].append({
            "step": "screen",
            "results": {
                "sources_screened": len(screened_sources)
            }
        })
        
        # Step 3: Extract information from sources
        extracted_information = []
        for source in screened_sources:
            # Simulate information extraction
            extracted_info = {
                "source_id": source.get("id", "unknown"),
                "key_findings": ["finding 1", "finding 2"],
                "methodology": "research methodology",
                "limitations": ["limitation 1"]
            }
            extracted_information.append(extracted_info)
        
        session["steps"].append({
            "step": "extract",
            "results": {
                "information_extracted": len(extracted_information)
            }
        })
        
        # Step 4: Analyze patterns across sources
        analysis_results = {
            "themes": ["theme 1", "theme 2"],
            "methodologies": ["methodology 1", "methodology 2"],
            "timeline": ["development 1", "development 2"],
            "contradictions": ["contradiction 1"] if random.random() < 0.3 else []
        }
        session["steps"].append({
            "step": "analyze",
            "results": {
                "themes_identified": len(analysis_results["themes"]),
                "contradictions_found": len(analysis_results["contradictions"])
            }
        })
        
        # Step 5: Synthesize findings
        synthesis_results = {
            "narrative": "Synthesis of literature findings...",
            "framework": {
                "components": ["component 1", "component 2"],
                "relationships": ["relationship 1"]
            }
        }
        session["steps"].append({
            "step": "synthesize",
            "results": {
                "synthesis_completed": True
            }
        })
        
        # Step 6: Identify gaps
        gap_results = {
            "gaps": ["gap 1", "gap 2"] if random.random() < 0.8 else [],
            "contradictions": analysis_results["contradictions"],
            "future_directions": ["direction 1", "direction 2"]
        }
        session["steps"].append({
            "step": "identify_gaps",
            "results": {
                "gaps_identified": len(gap_results["gaps"]),
                "future_directions": len(gap_results["future_directions"])
            }
        })
        
        # Compile literature review results
        review_results = {
            "literature_summary": synthesis_results["narrative"],
            "thematic_analysis": analysis_results["themes"],
            "methodological_assessment": analysis_results["methodologies"],
            "chronological_development": analysis_results["timeline"],
            "conceptual_framework": synthesis_results["framework"],
            "gaps": gap_results["gaps"],
            "contradictions": gap_results["contradictions"],
            "future_directions": gap_results["future_directions"],
            "sources": [s.get("id", "unknown") for s in screened_sources]
        }
        
        # Update session with results
        session["results"] = review_results
        
        # Add gaps to knowledge field
        for gap in gap_results["gaps"]:
            if gap not in self.knowledge_field.gaps:
                self.knowledge_field.gaps.append(gap)
        
        # Record field updates
        session["field_updates"] = {
            "gaps_added": len(gap_results["gaps"]),
            "contradictions_added": len(gap_results["contradictions"])
        }
        
        # Add to session history
        self.session_history.append(session)
        
        return review_results
    
    def develop_research_idea(self, research_interest: str, 
                            constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Develop a complete research idea from interest area.
        
        Args:
            research_interest: Research interest area
            constraints: Optional constraints
            
        Returns:
            dict: Complete research idea
        """
        # Create a session record
        session = {
            "type": "research_idea_development",
            "research_interest": research_interest,
            "constraints": constraints,
            "steps": [],
            "results": {},
            "field_updates": {}
        }
        
        # Step 1: Identify research opportunities
        opportunities = self.knowledge_field.identify_research_opportunities(
            research_interests=[research_interest],
            constraints=constraints
        )
        session["steps"].append({
            "step": "identify_opportunities",
            "results": {
                "opportunities_identified": len(opportunities)
            }
        })
        
        # Select the best opportunity (simplified)
        selected_opportunity = opportunities[0] if opportunities else {
            "id": generate_id(),
            "title": f"Research opportunity related to {research_interest}",
            "description": "Default opportunity"
        }
        
        # Step 2: Develop research question
        question_result = self.inquiry_model.develop_research_question(
            knowledge_field=self.knowledge_field,
            research_interest=selected_opportunity["title"],
            constraints=constraints
        )
        session["steps"].append({
            "step": "develop_question",
            "results": {
                "question_id": question_result["question_id"]
            }
        })
        
        # Step 3: Develop hypothesis
        hypothesis_result = self.inquiry_model.develop_hypothesis(
            knowledge_field=self.knowledge_field,
            research_question_id=question_result["question_id"]
        )
        session["steps"].append({
            "step": "develop_hypothesis",
            "results": {
                "hypothesis_id": hypothesis_result["hypothesis_id"]
            }
        })
        
        # Step 4: Create preliminary research design
        research_design = {
            "design_type": "experimental",
            "participants": {
                "sample_size": random.randint(30, 200),
                "characteristics": "target population"
            },
            "procedures": ["procedure 1", "procedure 2"],
            "measures": ["measure 1", "measure 2"],
            "analysis_plan": "statistical analysis approach"
        }
        session["steps"].append({
            "step": "create_research_design",
            "results": {
                "design_type": research_design["design_type"]
            }
        })
        
        # Compile research idea results
        idea_results = {
            "research_question": question_result["question"],
            "hypothesis": hypothesis_result["hypothesis"],           # content
            "hypothesis_id": hypothesis_result["hypothesis_id"],     # ← ADD THIS
            "research_design": research_design,
            "opportunity": selected_opportunity
        }
        
        # Update session with results
        session["results"] = idea_results
        
        # Add to session history
        self.session_history.append(session)
        
        return idea_results
    
    def analyze_interdisciplinary_potential(self, primary_domain: str, 
                                         secondary_domains: List[str]) -> Dict[str, Any]:
        """
        Analyze potential for interdisciplinary research.
        
        Args:
            primary_domain: Primary research domain
            secondary_domains: Secondary domains to consider
            
        Returns:
            dict: Interdisciplinary analysis
        """
        # Create a session record
        session = {
            "type": "interdisciplinary_analysis",
            "primary_domain": primary_domain,
            "secondary_domains": secondary_domains,
            "steps": [],
            "results": {},
            "field_updates": {}
        }
        
        # Step 1: Analyze domain characteristics
        domain_characteristics = {}
        for domain in [primary_domain] + secondary_domains:
            # Simulate domain analysis
            characteristics = {
                "key_concepts": [f"{domain} concept 1", f"{domain} concept 2"],
                "methodologies": [f"{domain} methodology 1", f"{domain} methodology 2"],
                "theoretical_frameworks": [f"{domain} framework 1", f"{domain} framework 2"]
            }
            domain_characteristics[domain] = characteristics
        
        session["steps"].append({
            "step": "analyze_domains",
            "results": {
                "domains_analyzed": len(domain_characteristics)
            }
        })
        
        # Step 2: Identify potential integration points
        integration_points = []
        for secondary_domain in secondary_domains:
            # Simulate integration points
            integration_point = {
                "domains": [primary_domain, secondary_domain],
                "conceptual_bridges": [f"Bridge between {primary_domain} and {secondary_domain}"],
                "methodological_synergies": [f"Synergy between methodologies"],
                "theoretical_integrations": [f"Integration of theories"]
            }
            integration_points.append(integration_point)
        
        session["steps"].append({
            "step": "identify_integration",
            "results": {
                "integration_points": len(integration_points)
            }
        })
        
        # Step 3: Evaluate research potential
        research_potential = []
        for integration_point in integration_points:
            # Simulate research potential
            potential = {
                "integration_point": integration_point,
                "research_questions": [f"Interdisciplinary question 1", f"Interdisciplinary question 2"],
                "novelty": random.uniform(0.6, 0.9),
                "feasibility": random.uniform(0.4, 0.8),
                "impact": random.uniform(0.5, 0.95)
            }
            research_potential.append(potential)
        
        session["steps"].append({
            "step": "evaluate_potential",
            "results": {
                "potential_areas": len(research_potential)
            }
        })
        
        # Step 4: Identify challenges and strategies
        challenges_strategies = {
            "conceptual_challenges": ["Challenge 1", "Challenge 2"],
            "methodological_challenges": ["Methodological challenge 1"],
            "practical_challenges": ["Practical challenge 1"],
            "mitigation_strategies": ["Strategy 1", "Strategy 2"]
        }
        
        session["steps"].append({
            "step": "identify_challenges",
            "results": {
                "challenges": len(challenges_strategies["conceptual_challenges"]) + 
                            len(challenges_strategies["methodological_challenges"]) + 
                            len(challenges_strategies["practical_challenges"]),
                "strategies": len(challenges_strategies["mitigation_strategies"])
            }
        })
        
        # Compile interdisciplinary analysis results
        analysis_results = {
            "domain_characteristics": domain_characteristics,
            "integration_points": integration_points,
            "research_potential": research_potential,
            "challenges_strategies": challenges_strategies,
            "recommended_approach": "Recommended interdisciplinary approach"
        }
        
        # Update session with results
        session["results"] = analysis_results
        
        # Add to session history
        self.session_history.append(session)
        
        return analysis_results
    
    def visualize_research_process(self, session_index: int = -1) -> plt.Figure:
        """
        Visualize the research process from a session.
        
        Args:
            session_index: Index of session to visualize
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        # Get the specified session
        if not self.session_history:
            raise ValueError("No research sessions available for visualization")
        
        session = self.session_history[session_index]
        session_type = session.get("type", "unknown")
        
        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Research Process: {session_type.replace('_', ' ').title()}", fontsize=16)
        
        # Plot 1: Process steps visualization (top left)
        steps = session.get("steps", [])
        if steps:
            # Create a flow diagram of steps
            G = nx.DiGraph()
            
            # Add step nodes
            for i, step in enumerate(steps):
                step_name = step.get("step", f"Step {i+1}")
                G.add_node(step_name, pos=(i, 0))
                
                # Connect steps
                if i > 0:
                    prev_step = steps[i-1].get("step", f"Step {i}")
                    G.add_edge(prev_step, step_name)
            
            # Draw the graph
            pos = nx.get_node_attributes(G, 'pos')
            nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', 
                   font_size=10, font_weight='bold', ax=axs[0, 0])
            
            # Add result annotations
            for i, step in enumerate(steps):
                step_name = step.get("step", f"Step {i+1}")
                results = step.get("results", {})
                
                # Create result text
                result_text = "\n".join([f"{k}: {v}" for k, v in results.items()])
                
                # Add annotation
                axs[0, 0].annotate(result_text, xy=(i, -0.3), xycoords='data',
                                 fontsize=8, ha='center', va='top')
            
            axs[0, 0].set_title("Research Process Steps")
        else:
            axs[0, 0].text(0.5, 0.5, "No process steps available", 
                          ha='center', va='center', fontsize=12)
        
        # Plot 2: Research content visualization (top right)
        if session_type == "literature_review":
            # Visualize literature review results
            results = session.get("results", {})
            
            if results:
                # Create a mind map style visualization of themes and gaps
                G = nx.Graph()
                
                # Add central node
                central_topic = "Literature Review"
                G.add_node(central_topic, pos=(0, 0))
                
                # Add theme nodes
                themes = results.get("thematic_analysis", [])
                for i, theme in enumerate(themes):
                    angle = i * 2 * np.pi / len(themes)
                    x = 1.5 * np.cos(angle)
                    y = 1.5 * np.sin(angle)
                    G.add_node(f"Theme: {theme}", pos=(x, y))
                    G.add_edge(central_topic, f"Theme: {theme}")
                
                # Add gap nodes
                gaps = results.get("gaps", [])
                for i, gap in enumerate(gaps):
                    angle = i * 2 * np.pi / len(gaps) if gaps else 0
                    x = 3 * np.cos(angle)
                    y = 3 * np.sin(angle)
                    G.add_node(f"Gap: {gap}", pos=(x, y))
                    
                    # Connect to most relevant theme (simplified)
                    if themes:
                        theme_index = i % len(themes)
                        theme = themes[theme_index]
                        G.add_edge(f"Theme: {theme}", f"Gap: {gap}")
                
                # Draw the graph
                pos = nx.get_node_attributes(G, 'pos')
                
                # Draw with different colors for different node types
                theme_nodes = [n for n in G.nodes if "Theme" in n]
                gap_nodes = [n for n in G.nodes if "Gap" in n]
                
                nx.draw_networkx_nodes(G, pos, nodelist=[central_topic], 
                                     node_color='lightgreen', node_size=3000, ax=axs[0, 1])
                nx.draw_networkx_nodes(G, pos, nodelist=theme_nodes, 
                                     node_color='lightblue', node_size=2000, ax=axs[0, 1])
                nx.draw_networkx_nodes(G, pos, nodelist=gap_nodes, 
                                     node_color='salmon', node_size=1500, ax=axs[0, 1])
                
                nx.draw_networkx_edges(G, pos, ax=axs[0, 1])
                nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=axs[0, 1])
                
                axs[0, 1].set_title("Literature Review Content Map")
            else:
                axs[0, 1].text(0.5, 0.5, "No literature review results available", 
                              ha='center', va='center', fontsize=12)
                
        elif session_type == "research_idea_development":
            # Visualize research idea
            results = session.get("results", {})
            
            if results:
                # Create a hierarchical diagram of research components
                G = nx.DiGraph()
                
                # Add central node for research question
                research_question = results.get("research_question", {}).get("question", "Research Question")
                G.add_node("Research Question", pos=(0, 0))
                
                # Add hypothesis
                hypothesis = results.get("hypothesis", {}).get("hypothesis", "Hypothesis")
                G.add_node("Hypothesis", pos=(0, -1))
                G.add_edge("Research Question", "Hypothesis")
                
                # Add research design components
                design = results.get("research_design", {})
                
                # Add design type
                design_type = design.get("design_type", "Design Type")
                G.add_node(f"Design: {design_type}", pos=(-2, -2))
                G.add_edge("Hypothesis", f"Design: {design_type}")
                
                # Add participants
                participants = design.get("participants", {})
                sample_size = participants.get("sample_size", "N/A")
                G.add_node(f"Participants: n={sample_size}", pos=(-1, -2))
                G.add_edge("Hypothesis", f"Participants: n={sample_size}")
                
                # Add measures
                measures = design.get("measures", [])
                for i, measure in enumerate(measures):
                    G.add_node(f"Measure: {measure}", pos=(1 + i*0.5, -2))
                    G.add_edge("Hypothesis", f"Measure: {measure}")
                
                # Add analysis
                analysis = design.get("analysis_plan", "Analysis Plan")
                G.add_node(f"Analysis: {analysis}", pos=(3, -2))
                G.add_edge("Hypothesis", f"Analysis: {analysis}")
                
                # Draw the graph
                pos = nx.get_node_attributes(G, 'pos')
                nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightgreen', 
                       font_size=8, font_weight='bold', ax=axs[0, 1])
                
                axs[0, 1].set_title("Research Idea Components")
            else:
                axs[0, 1].text(0.5, 0.5, "No research idea results available", 
                              ha='center', va='center', fontsize=12)
        
        elif session_type == "interdisciplinary_analysis":
            # Visualize interdisciplinary connections
            results = session.get("results", {})
        
            if results:
                # Create a network diagram of domain connections
                G = nx.Graph()
            
                # Add domain nodes
                primary_domain = session.get("primary_domain", "Primary")
                secondary_domains = session.get("secondary_domains", [])
            
                # Add primary domain at center
                G.add_node(primary_domain, pos=(0, 0))
            
                # Add secondary domains around it and connect
                for i, domain in enumerate(secondary_domains):
                    angle = i * 2 * np.pi / len(secondary_domains)
                    x = 2 * np.cos(angle)
                    y = 2 * np.sin(angle)
                    G.add_node(domain, pos=(x, y))
                    G.add_edge(primary_domain, domain)  # Ensure edge exists

                # Add integration point edges and labels
                integration_points = results.get("integration_points", [])
                edge_labels = {}
                for point in integration_points:
                    domains = point.get("domains", [])
                    bridges = point.get("conceptual_bridges", [])
                    if len(domains) >= 2 and bridges:
                        d1, d2 = domains[0], domains[1]
                        # Normalize edge tuple (undirected graph)
                        edge = tuple(sorted([d1, d2]))
                        # But NetworkX stores as (min, max) only if using add_edge with sorting — safer to check both
                        if G.has_edge(d1, d2):
                            edge_key = (d1, d2)
                        elif G.has_edge(d2, d1):
                            edge_key = (d2, d1)
                        else:
                            # Add missing edge
                            G.add_edge(d1, d2)
                            edge_key = (d1, d2)
                        edge_labels[edge_key] = bridges[0][:30] + "..." if len(bridges[0]) > 30 else bridges[0]

                # Draw the graph
                pos = nx.get_node_attributes(G, 'pos')
            
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, nodelist=[primary_domain],
                                    node_color='gold', node_size=3000, ax=axs[0, 1])
                nx.draw_networkx_nodes(G, pos, nodelist=secondary_domains,
                                    node_color='lightblue', node_size=2000, ax=axs[0, 1])
            
                nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, ax=axs[0, 1])
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=axs[0, 1])
                
                # Safely draw edge labels
                if edge_labels:
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                                font_size=8, ax=axs[0, 1])
            
                axs[0, 1].set_title("Interdisciplinary Connections")
            else:
                axs[0, 1].text(0.5, 0.5, "No interdisciplinary results available",
                            ha='center', va='center', fontsize=12)
        else:
            axs[0, 1].text(0.5, 0.5, f"No visualization for {session_type}", 
                          ha='center', va='center', fontsize=12)
        
        # Plot 3: Field visualization (bottom left)
        # Visualize knowledge field changes
        field_updates = session.get("field_updates", {})
        
        if field_updates:
            # Create a bar chart of field updates
            update_types = []
            update_values = []
            
            for update_type, value in field_updates.items():
                if isinstance(value, (int, float)):
                    update_types.append(update_type)
                    update_values.append(value)
            
            if update_types:
                y_pos = np.arange(len(update_types))
                axs[1, 0].barh(y_pos, update_values, align='center')
                axs[1, 0].set_yticks(y_pos)
                axs[1, 0].set_yticklabels(update_types)
                axs[1, 0].invert_yaxis()  # Labels read top-to-bottom
                
                axs[1, 0].set_title("Knowledge Field Updates")
            else:
                axs[1, 0].text(0.5, 0.5, "No field update data available", 
                              ha='center', va='center', fontsize=12)
        else:
            axs[1, 0].text(0.5, 0.5, "No field update data available", 
                          ha='center', va='center', fontsize=12)
        
        # Plot 4: Research field visualization (bottom right)
        # Visualize a simplified version of the research field
        
        # Create a circle representing the field
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        axs[1, 1].add_artist(circle)
        
        # Add attractor points for literature
        literature_count = len(self.knowledge_field.literature)
        
        if literature_count > 0:
            # Create points around a circle for literature
            angles = np.linspace(0, 2*np.pi, min(10, literature_count), endpoint=False)
            for i, angle in enumerate(angles):
                x = 0.7 * np.cos(angle)
                y = 0.7 * np.sin(angle)
                axs[1, 1].scatter(x, y, s=100, color='blue', alpha=0.7)
                axs[1, 1].text(x, y, f"Paper {i+1}", fontsize=8, ha='center', va='bottom')
        
        # Add points for research gaps
        gap_count = len(self.knowledge_field.gaps)
        
        if gap_count > 0:
            # Create points for gaps
            gap_angles = np.linspace(0, 2*np.pi, min(5, gap_count), endpoint=False)
            for i, angle in enumerate(gap_angles):
                # Position gaps at a different radius
                x = 0.4 * np.cos(angle)
                y = 0.4 * np.sin(angle)
                axs[1, 1].scatter(x, y, s=150, color='red', alpha=0.5, marker='*')
                axs[1, 1].text(x, y, f"Gap {i+1}", fontsize=8, ha='center', va='bottom')
        
        # Add research question
        if session_type in ["research_idea_development", "literature_review"]:
            research_question = (session.get("research_question", "") if session_type == "literature_review" else
                               session.get("results", {}).get("research_question", {}).get("question", ""))
            
            if research_question:
                # Position research question at center
                axs[1, 1].scatter(0, 0, s=200, color='green', alpha=0.7)
                axs[1, 1].text(0, 0, "Research\nQuestion", fontsize=9, ha='center', va='center')
        
        # Set equal aspect ratio and limits
        axs[1, 1].set_aspect('equal')
        axs[1, 1].set_xlim(-1.2, 1.2)
        axs[1, 1].set_ylim(-1.2, 1.2)
        axs[1, 1].set_title("Research Knowledge Field")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        
        return fig

# Research Example Functions

def research_example_literature_review():
    """Example: Conducting a systematic literature review."""
    print("\n===== RESEARCH EXAMPLE: SYSTEMATIC LITERATURE REVIEW =====")
    
    # Initialize the research architecture
    research = ResearchArchitecture(domain="cognitive_science")
    
    # Initialize with some sample papers
    sample_papers = [
        {
            "id": "paper1",
            "title": "Advances in Cognitive Processing",
            "authors": ["Author A", "Author B"],
            "year": 2023,
            "abstract": "This paper explores recent advances in cognitive processing..."
        },
        {
            "id": "paper2",
            "title": "Neural Mechanisms of Memory",
            "authors": ["Author C", "Author D"],
            "year": 2022,
            "abstract": "This study investigates the neural mechanisms underlying memory formation..."
        },
        {
            "id": "paper3",
            "title": "Cognitive Load Theory",
            "authors": ["Author E", "Author F"],
            "year": 2021,
            "abstract": "A comprehensive review of cognitive load theory and its applications..."
        },
        {
            "id": "paper4",
            "title": "Working Memory Capacity",
            "authors": ["Author G", "Author H"],
            "year": 2023,
            "abstract": "This research examines factors affecting working memory capacity..."
        },
        {
            "id": "paper5",
            "title": "Attention and Cognitive Control",
            "authors": ["Author I", "Author J"],
            "year": 2022,
            "abstract": "A study on the relationship between attention mechanisms and cognitive control..."
        }
    ]
    
    research.initialize_literature(sample_papers)
    
    # Define a research question
    research_question = "How do working memory capacity and cognitive load interact to affect learning outcomes?"
    
    # Conduct a literature review
    print(f"Conducting literature review on: {research_question}")
    review_results = research.conduct_literature_review(research_question)
    
    # Print results
    print("\nLiterature Review Results:")
    print(f"  Thematic Analysis: {review_results['thematic_analysis']}")
    print(f"  Gaps Identified: {review_results['gaps']}")
    print(f"  Future Directions: {review_results['future_directions']}")
    
    # Visualize the research process
    fig = research.visualize_research_process()
    plt.show()
    
    # Visualize the research landscape
    field_fig = research.knowledge_field.visualize_research_landscape(include_gaps=True)
    plt.show()
    
    return review_results

def research_example_hypothesis_development():
    """Example: Developing and refining research hypotheses."""
    print("\n===== RESEARCH EXAMPLE: HYPOTHESIS DEVELOPMENT =====")
    research = ResearchArchitecture(domain="psychology")
    
    # Initialize literature (required for realistic field state)
    sample_papers = [
        {
            "id": "paper1",
            "title": "Social Media and Mental Health",
            "authors": ["Author A", "Author B"],
            "year": 2023,
            "abstract": "This paper explores the relationship between social media use and mental health outcomes..."
        },
        {
            "id": "paper2",
            "title": "Screen Time Effects on Adolescents",
            "authors": ["Author C", "Author D"],
            "year": 2022,
            "abstract": "This study investigates how screen time affects adolescent development..."
        },
        {
            "id": "paper3",
            "title": "Digital Wellbeing Interventions",
            "authors": ["Author E", "Author F"],
            "year": 2021,
            "abstract": "A review of interventions designed to promote digital wellbeing..."
        }
    ]
    research.initialize_literature(sample_papers)

    research_interest = "The effects of social media usage patterns on psychological wellbeing"
    constraints = {
        "population": "young adults (18-25)",
        "timeframe": "longitudinal study",
        "resources": "limited budget"
    }

    print(f"Developing research idea on: {research_interest}")
    print(f"With constraints: {constraints}")

    # This now correctly returns hypothesis_id in the result dict
    idea_results = research.develop_research_idea(research_interest, constraints)

    print("\nInitial Research Idea:")
    if isinstance(idea_results.get("research_question"), dict):
        print(f"  Research Question: {idea_results['research_question'].get('question', 'N/A')}")
    else:
        print(f"  Research Question: {idea_results.get('research_question', 'N/A')}")

    if isinstance(idea_results.get("hypothesis"), dict):
        print(f"  Hypothesis: {idea_results['hypothesis'].get('hypothesis', 'N/A')}")
    else:
        print(f"  Hypothesis: {idea_results.get('hypothesis', 'N/A')}")

    print("\nRefining hypothesis through multiple iterations...")

    # CORRECTED: Now reliably gets the hypothesis_id from the returned dict
    hypothesis_id = idea_results.get("hypothesis_id")
    if not hypothesis_id:
        # Robust fallback (should never trigger with current fix in develop_research_idea)
        hypothesis_id = list(research.inquiry_model.hypotheses.keys())[-1]

    # First refinement
    refinement_data_1 = {
        "precision_improvement": "Add specific social media platforms",
        "variable_clarification": "Distinguish between active and passive usage",
        "measurement_specification": "Use validated wellbeing scales"
    }
    refined_1 = research.inquiry_model.refine_hypothesis(hypothesis_id, refinement_data_1)
    print(f"  Refinement 1: {refined_1['hypothesis']['hypothesis']}")

    # Second refinement (uses the new ID returned from refinement 1)
    refinement_data_2 = {
        "precision_improvement": "Specify usage frequency thresholds",
        "mediator_addition": "Include social comparison as mediator",
        "boundary_condition": "Limit to non-clinical population"
    }
    refined_2 = research.inquiry_model.refine_hypothesis(refined_1["hypothesis_id"], refinement_data_2)
    print(f"  Refinement 2: {refined_2['hypothesis']['hypothesis']}")

    # Final visualization
    fig = research.visualize_research_process()
    plt.show()

    return {
        "initial_idea": idea_results,
        "refinement_1": refined_1,
        "refinement_2": refined_2
    }

def research_example_interdisciplinary_research():
    """Example: Orchestrating interdisciplinary research."""
    print("\n===== RESEARCH EXAMPLE: INTERDISCIPLINARY RESEARCH =====")
    
    # Initialize the research architecture
    research = ResearchArchitecture(domain="human_computer_interaction")
    
    # Initialize with some sample papers
    sample_papers = [
        {
            "id": "paper1",
            "title": "User Experience Design Principles",
            "authors": ["Author A", "Author B"],
            "year": 2023,
            "domain": "human_computer_interaction",
            "abstract": "This paper explores foundational principles in UX design..."
        },
        {
            "id": "paper2",
            "title": "Cognitive Neuroscience of Decision Making",
            "authors": ["Author C", "Author D"],
            "year": 2022,
            "domain": "neuroscience",
            "abstract": "This study investigates neural mechanisms of decision making..."
        },
        {
            "id": "paper3",
            "title": "Behavioral Economics and Choice Architecture",
            "authors": ["Author E", "Author F"],
            "year": 2021,
            "domain": "behavioral_economics",
            "abstract": "A review of how choice architecture influences decision making..."
        },
        {
            "id": "paper4",
            "title": "AI Systems for Decision Support",
            "authors": ["Author G", "Author H"],
            "year": 2023,
            "domain": "artificial_intelligence",
            "abstract": "This research examines AI-based decision support systems..."
        }
    ]
    
    research.initialize_literature(sample_papers)
    
    # Define domains for interdisciplinary analysis
    primary_domain = "human_computer_interaction"
    secondary_domains = ["neuroscience", "behavioral_economics", "artificial_intelligence"]
    
    print(f"Analyzing interdisciplinary research potential:")
    print(f"  Primary Domain: {primary_domain}")
    print(f"  Secondary Domains: {', '.join(secondary_domains)}")
    
    # Conduct interdisciplinary analysis
    analysis_results = research.analyze_interdisciplinary_potential(
        primary_domain=primary_domain,
        secondary_domains=secondary_domains
    )
    
    # Print results
    print("\nInterdisciplinary Analysis Results:")
    print("  Integration Points:")
    for i, point in enumerate(analysis_results["integration_points"][:2]):  # Limit to 2 for clarity
        domains = point.get("domains", [])
        bridges = point.get("conceptual_bridges", [])
        print(f"    {i+1}. Between {' and '.join(domains)}: {bridges[0] if bridges else 'N/A'}")
    
    print("\n  Research Potential Areas:")
    for i, potential in enumerate(analysis_results["research_potential"][:2]):  # Limit to 2 for clarity
        questions = potential.get("research_questions", [])
        novelty = potential.get("novelty", 0)
        impact = potential.get("impact", 0)
        print(f"    {i+1}. Question: {questions[0] if questions else 'N/A'}")
        print(f"       Novelty: {novelty:.2f}, Impact: {impact:.2f}")
    
    print("\n  Challenges and Strategies:")
    challenges = analysis_results["challenges_strategies"]["conceptual_challenges"]
    strategies = analysis_results["challenges_strategies"]["mitigation_strategies"]
    for i, challenge in enumerate(challenges[:2]):  # Limit to 2 for clarity
        print(f"    Challenge {i+1}: {challenge}")
    for i, strategy in enumerate(strategies[:2]):  # Limit to 2 for clarity
        print(f"    Strategy {i+1}: {strategy}")
    
    # Visualize the research process
    fig = research.visualize_research_process()
    plt.show()
    
    return analysis_results

if __name__ == "__main__":
    research_example_literature_review()
    research_example_hypothesis_development()
    research_example_interdisciplinary_research()
