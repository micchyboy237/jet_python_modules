"""
Field Protocol Shells - Reusable templates for implementing field protocols

This module provides a framework for parsing, validating, and executing field protocols
defined in the Pareto-lang format. It includes base classes and utilities for implementing
the core protocols in the Context Engineering repository.

Basic usage:
    # Load a protocol shell
    protocol = ProtocolShell.from_file("path/to/attractor.co.emerge.shell")
    
    # Prepare input data
    input_data = {
        "current_field_state": field,
        "candidate_attractors": attractors
    }
    
    # Execute the protocol
    result = protocol.execute(input_data)
    
    # Use the output
    updated_field = result["updated_field_state"]
    co_emergent_attractors = result["co_emergent_attractors"]

Advanced usage:
    # Create a custom implementation of a protocol
    class MyCoEmergenceProtocol(ProtocolShell):
        def attractor_scan(self, field, **kwargs):
            # Custom implementation of attractor scanning
            return my_custom_attractor_scan(field, **kwargs)
        
        def residue_surface(self, field, **kwargs):
            # Custom implementation of residue surfacing
            return my_custom_residue_surface(field, **kwargs)
        
        # Implement other operations...
    
    # Load the shell but use custom implementation
    protocol = MyCoEmergenceProtocol.from_file("path/to/attractor.co.emerge.shell")
    result = protocol.execute(input_data)
"""

import json
import re
import os
import datetime
from typing import Dict, List, Any
import jsonschema

from jet.adapters.llama_cpp.llm import LlamacppLLM

# ============================================================================
# OUTPUT & LOGGING SETUP
# ============================================================================
from pathlib import Path
from jet.logger import CustomLogger
import shutil
# ----------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

main_logger = CustomLogger(
    name="main",
    filename=os.path.join(OUTPUT_DIR, "main.log"),
    console_level="INFO",
    level="DEBUG",
    overwrite=True
)

def create_example_dir(example_name: str) -> Path:
    from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
    base_dir = Path(get_entry_file_dir()) / "generated" / os.path.splitext(get_entry_file_name())[0]
    example_dir = base_dir / example_name
    # shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir

def get_example_logger(name: str, output_dir: Path) -> CustomLogger:
    log_file = output_dir / "run.log"
    return CustomLogger(name=name, filename=log_file, overwrite=True)

# Type aliases for clarity
Field = Dict[str, Any]  # Semantic field representation
Attractor = Dict[str, Any]  # Attractor representation
Residue = Dict[str, Any]  # Symbolic residue representation
Operation = Dict[str, Any]  # Operation representation

class ProtocolParser:
    """Parser for protocol shells in Pareto-lang format."""
    
    @staticmethod
    def parse_shell(shell_content: str) -> Dict[str, Any]:
        # 1. Strip comments
        clean = re.sub(r'#.*$', '', shell_content, flags=re.MULTILINE)
        
        # 2. Extract protocol name and entire block content
        #    Accepts any amount of whitespace around { } and inside
        match = re.search(r'(\w+(?:\.\w+)*)\s*\{([^}]*)\}', clean, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid protocol shell – cannot find name {{ block }}:\n{shell_content}")
        
        name, content = match.groups()
        result = {"name": name.strip()}

        # 3. Section patterns – very permissive
        sections = {
            "intent":  r'intent\s*:\s*"([^"]+)"',
            "input":   r'input\s*:\s*\{([^}]*)\}',
            "process": r'process\s*:\s*\[\s*([^]]*?)\s*\]',   # non-greedy for nested [] safety
            "output":  r'output\s*:\s*\{([^}]*)\}',
            "meta":    r'meta\s*:\s*\{([^}]*)\}',
        }

        for sec_name, pattern in sections.items():
            m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if m:
                raw = m.group(1)
                if sec_name == "process":
                    # Split operations by comma, ignore empty entries
                    ops = [op.strip() for op in re.split(r',(?!\s*[}\]])', raw) if op.strip()]
                    result[sec_name] = ops
                elif sec_name in ["input", "output", "meta"]:
                    result[sec_name] = ProtocolParser._parse_object_section(raw)
                else:
                    result[sec_name] = raw.strip().strip('"')

        return result

    @staticmethod
    def _parse_object_section(section_content: str) -> Dict[str, Any]:
        """Parse key: value pairs – works with quotes, <placeholders>, and commas."""
        result = {}
        # Match key: anything-until-comma-or-end
        for m in re.finditer(r'(\w+)\s*:\s*([^,]+)', section_content):
            k, v = m.groups()
            k = k.strip()
            v = v.strip().strip('"\'')
            result[k] = v
        return result

    @staticmethod
    def _parse_process_section(section_content: str) -> List[str]:
        """Parse the process section of the protocol shell."""
        # Split by commas and clean up each operation
        operations = [op.strip() for op in section_content.split(',')]
        # Filter out empty strings
        operations = [op for op in operations if op]
        return operations

    @staticmethod
    def serialize_shell(protocol_dict: Dict[str, Any]) -> str:
        """
        Serialize a protocol dictionary back to Pareto-lang format.
        
        Args:
            protocol_dict: Dictionary representation of the protocol
            
        Returns:
            String containing the protocol in Pareto-lang format
        """
        name = protocol_dict.get("name", "unnamed_protocol")
        
        sections = []
        
        # Add intent section
        if "intent" in protocol_dict:
            sections.append(f'  intent: "{protocol_dict["intent"]}",\n')
        
        # Add input section
        if "input" in protocol_dict:
            input_section = "  input: {\n"
            for key, value in protocol_dict["input"].items():
                input_section += f"    {key}: {value},\n"
            input_section += "  },\n"
            sections.append(input_section)
        
        # Add process section
        if "process" in protocol_dict:
            process_section = "  process: [\n"
            for operation in protocol_dict["process"]:
                process_section += f"    {operation},\n"
            process_section += "  ],\n"
            sections.append(process_section)
        
        # Add output section
        if "output" in protocol_dict:
            output_section = "  output: {\n"
            for key, value in protocol_dict["output"].items():
                output_section += f"    {key}: {value},\n"
            output_section += "  },\n"
            sections.append(output_section)
        
        # Add meta section
        if "meta" in protocol_dict:
            meta_section = "  meta: {\n"
            for key, value in protocol_dict["meta"].items():
                meta_section += f"    {key}: {value},\n"
            meta_section += "  }\n"
            sections.append(meta_section)
        
        # Combine all sections
        shell_content = f"{name} {{\n{''.join(sections)}}}"
        
        return shell_content


class ProtocolValidator:
    """Validator for protocol shells against JSON schemas."""
    
    @staticmethod
    def validate(protocol_dict: Dict[str, Any], schema_path: str) -> bool:
        """
        Validate a protocol dictionary against a JSON schema.
        
        Args:
            protocol_dict: Dictionary representation of the protocol
            schema_path: Path to the JSON schema file
            
        Returns:
            True if valid, raises jsonschema.ValidationError if invalid
        """
        # Load schema
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        # Validate protocol against schema
        jsonschema.validate(instance=protocol_dict, schema=schema)
        
        return True


class ProtocolShell:
    """Base class for protocol shells."""
    
    def __init__(self, protocol_dict: Dict[str, Any]):
        """
        Initialize a protocol shell from a dictionary representation.
        
        Args:
            protocol_dict: Dictionary representation of the protocol
        """
        self.protocol_dict = protocol_dict
        self.name = protocol_dict.get("name", "unnamed_protocol")
        self.intent = protocol_dict.get("intent", "")
        self.input_spec = protocol_dict.get("input", {})
        self.process = protocol_dict.get("process", [])
        self.output_spec = protocol_dict.get("output", {})
        self.meta = protocol_dict.get("meta", {})
        
        # Initialize operation registry
        self._init_operation_registry()
    
    @classmethod
    def from_file(cls, file_path: str) -> 'ProtocolShell':
        """
        Create a protocol shell from a file.
        
        Args:
            file_path: Path to the protocol shell file
            
        Returns:
            ProtocolShell instance
        """
        with open(file_path, 'r') as f:
            shell_content = f.read()
        
        protocol_dict = ProtocolParser.parse_shell(shell_content)
        return cls(protocol_dict)
    
    @classmethod
    def from_string(cls, shell_content: str) -> 'ProtocolShell':
        """
        Create a protocol shell from a string.
        
        Args:
            shell_content: String containing the protocol shell in Pareto-lang format
            
        Returns:
            ProtocolShell instance
        """
        protocol_dict = ProtocolParser.parse_shell(shell_content)
        return cls(protocol_dict)
    
    def _init_operation_registry(self):
        """Initialize the operation registry with implemented methods."""
        self.operation_registry = {}
        
        # Find all methods that match operation names
        for operation_name in self._extract_operation_names():
            method_name = self._operation_to_method_name(operation_name)
            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                self.operation_registry[operation_name] = getattr(self, method_name)
    
    def _extract_operation_names(self) -> List[str]:
        """Extract operation names from the process section."""
        operation_names = []
        for operation in self.process:
            # Extract name from format like "/operation.name{param='value'}"
            match = re.match(r'/(\w+\.\w+){', operation)
            if match:
                operation_names.append(match.group(1))
        return operation_names
    
    def _operation_to_method_name(self, operation_name: str) -> str:
        """Convert an operation name to a method name."""
        # Convert "namespace.operation" to "namespace_operation"
        return operation_name.replace('.', '_')
    
    def _extract_operation_params(self, operation: str) -> Dict[str, str]:
        """Extract parameters from an operation string."""
        # Extract content inside curly braces
        match = re.search(r'{(.*)}', operation)
        if not match:
            return {}
        
        params_str = match.group(1)
        params = {}
        
        # Parse parameters
        for param_match in re.finditer(r'(\w+)=([^,]+)(?:,|$)', params_str):
            key, value = param_match.groups()
            # Clean up value (remove quotes if string)
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            # Convert to appropriate type if possible
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif re.match(r'^-?\d+\.\d+$', value):
                value = float(value)
            
            params[key] = value
        
        return params
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the protocol with the provided input data.
        
        Args:
            input_data: Dictionary containing input data for the protocol
            
        Returns:
            Dictionary containing output data from the protocol
        """
        # Validate input data against input spec
        self._validate_input(input_data)
        
        # Initialize execution state with input data
        execution_state = input_data.copy()
        
        # Execute each operation in the process
        for operation in self.process:
            # Extract operation name and parameters
            match = re.match(r'/(\w+\.\w+){', operation)
            if not match:
                continue
            
            operation_name = match.group(1)
            params = self._extract_operation_params(operation)
            
            # Execute operation if implemented
            if operation_name in self.operation_registry:
                execution_state = self.operation_registry[operation_name](
                    execution_state, **params)
            else:
                print(f"Warning: Operation '{operation_name}' not implemented")
        
        # Prepare output based on output spec
        output = self._prepare_output(execution_state)
        
        # Add metadata
        if "meta" not in output:
            output["meta"] = {}
        output["meta"]["timestamp"] = datetime.datetime.now().isoformat()
        if "version" in self.meta:
            output["meta"]["version"] = self.meta["version"]
        
        return output
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate input data against input specification.
        
        Args:
            input_data: Dictionary containing input data for the protocol
            
        Raises:
            ValueError: If input data does not match specification
        """
        # This is a basic validation that just checks for required fields
        # In a real implementation, this would do more sophisticated validation
        for key in self.input_spec:
            if key not in input_data:
                # Check if the field has a default value placeholder
                if self.input_spec[key] == "<default>":
                    continue
                raise ValueError(f"Missing required input field: {key}")
    
    def _prepare_output(self, execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare output data based on output specification.
        
        Args:
            execution_state: Dictionary containing the current execution state
            
        Returns:
            Dictionary containing output data formatted according to output spec
        """
        output = {}
        
        # Extract fields specified in output spec
        for key in self.output_spec:
            if key in execution_state:
                output[key] = execution_state[key]
            else:
                # Include placeholder for missing fields
                output[key] = f"<{key} not generated>"
        
        return output


class AttractorCoEmergeProtocol(ProtocolShell):
    """Implementation of the attractor.co.emerge protocol."""
    
    def attractor_scan(self, state: Dict[str, Any], detect: str = 'attractors', 
                      filter_by: str = 'strength') -> Dict[str, Any]:
        """
        Scan the field for attractors and filter by the specified criterion.
        
        Args:
            state: Current execution state
            detect: What to detect ('attractors', 'patterns', etc.)
            filter_by: Criterion for filtering ('strength', 'coherence', etc.)
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would detect attractors based on field structure
        # This is a placeholder implementation
        attractors = self._detect_attractors(field, detect)
        
        # Filter attractors
        filtered_attractors = self._filter_attractors(attractors, filter_by)
        
        # Update state with detected attractors
        updated_state = state.copy()
        updated_state['detected_attractors'] = filtered_attractors
        
        return updated_state
    
    def residue_surface(self, state: Dict[str, Any], mode: str = 'recursive', 
                        integrate_residue: bool = True) -> Dict[str, Any]:
        """
        Surface symbolic residue in the field.
        
        Args:
            state: Current execution state
            mode: Method for surfacing residue ('recursive', 'echo', etc.)
            integrate_residue: Whether to integrate surfaced residue
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would detect symbolic residue based on field structure
        # This is a placeholder implementation
        residues = self._detect_residue(field, mode)
        
        # Integrate residue if requested
        if integrate_residue:
            field = self._integrate_residue(field, residues)
        
        # Update state with surfaced residues and potentially modified field
        updated_state = state.copy()
        updated_state['surfaced_residues'] = residues
        if integrate_residue:
            updated_state['current_field_state'] = field
        
        return updated_state
    
    def co_emergence_algorithms(self, state: Dict[str, Any], 
                               strategy: str = 'harmonic integration') -> Dict[str, Any]:
        """
        Apply co-emergence algorithms to facilitate attractor interaction.
        
        Args:
            state: Current execution state
            strategy: Strategy for co-emergence
            
        Returns:
            Updated execution state
        """
        # Extract field and attractors from state
        field = state.get('current_field_state', {})
        attractors = state.get('detected_attractors', [])
        
        # Implementation would apply co-emergence algorithms
        # This is a placeholder implementation
        if strategy == 'harmonic integration':
            field = self._apply_harmonic_integration(field, attractors)
        elif strategy == 'boundary dissolution':
            field = self._apply_boundary_dissolution(field, attractors)
        elif strategy == 'resonance amplification':
            field = self._apply_resonance_amplification(field, attractors)
        
        # Update state with modified field
        updated_state = state.copy()
        updated_state['current_field_state'] = field
        
        return updated_state
    
    def field_audit(self, state: Dict[str, Any], 
                   surface_new: str = 'attractor_basins') -> Dict[str, Any]:
        """
        Audit the field to identify new patterns or structures.
        
        Args:
            state: Current execution state
            surface_new: Type of patterns to surface
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would audit field for specified patterns
        # This is a placeholder implementation
        audit_results = {}
        
        if surface_new == 'attractor_basins':
            audit_results['attractor_basins'] = self._identify_attractor_basins(field)
        elif surface_new == 'field_coherence':
            audit_results['field_coherence'] = self._calculate_field_coherence(field)
        elif surface_new == 'emergent_patterns':
            audit_results['emergent_patterns'] = self._detect_emergent_patterns(field)
        
        # Update state with audit results
        updated_state = state.copy()
        updated_state['audit_results'] = audit_results
        
        return updated_state
    
    def agency_self_prompt(self, state: Dict[str, Any], 
                          trigger_condition: str = 'cycle interval') -> Dict[str, Any]:
        """
        Generate self-prompts for continued processing.
        
        Args:
            state: Current execution state
            trigger_condition: Condition for triggering self-prompts
            
        Returns:
            Updated execution state
        """
        # Extract field and audit results from state
        field = state.get('current_field_state', {})
        audit_results = state.get('audit_results', {})
        
        # Implementation would generate self-prompts based on trigger condition
        # This is a placeholder implementation
        self_prompts = []
        
        if trigger_condition == 'cycle interval':
            self_prompts.append(self._generate_cycle_prompt(field, audit_results))
        elif trigger_condition == 'emergent pattern':
            if 'emergent_patterns' in audit_results and audit_results['emergent_patterns']:
                self_prompts.append(self._generate_pattern_prompt(audit_results['emergent_patterns']))
        elif trigger_condition == 'coherence threshold':
            if 'field_coherence' in audit_results and audit_results['field_coherence'] > 0.8:
                self_prompts.append(self._generate_coherence_prompt(audit_results['field_coherence']))
        
        # Update state with self-prompts
        updated_state = state.copy()
        updated_state['self_prompts'] = self_prompts
        
        return updated_state
    
    def integration_protocol(self, state: Dict[str, Any], 
                            integrate: str = 'co_emergent_attractors') -> Dict[str, Any]:
        """
        Integrate specified elements back into the field.
        
        Args:
            state: Current execution state
            integrate: What to integrate
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would integrate specified elements
        # This is a placeholder implementation
        if integrate == 'co_emergent_attractors':
            # Detect co-emergent attractors
            co_emergent_attractors = self._detect_co_emergent_attractors(field)
            
            # Integrate them into the field
            field = self._integrate_attractors(field, co_emergent_attractors)
            
            # Update state
            updated_state = state.copy()
            updated_state['current_field_state'] = field
            updated_state['co_emergent_attractors'] = co_emergent_attractors
        else:
            # No integration performed
            updated_state = state.copy()
        
        return updated_state
    
    def boundary_collapse(self, state: Dict[str, Any], 
                         auto_collapse: str = 'field_boundaries') -> Dict[str, Any]:
        """
        Collapse boundaries in the field.
        
        Args:
            state: Current execution state
            auto_collapse: Type of boundaries to collapse
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would collapse specified boundaries
        # This is a placeholder implementation
        if auto_collapse == 'field_boundaries':
            field = self._collapse_all_boundaries(field)
        elif auto_collapse == 'selective':
            field = self._collapse_selected_boundaries(field)
        elif auto_collapse == 'gradient':
            field = self._create_gradient_boundaries(field)
        
        # Update state with modified field
        updated_state = state.copy()
        updated_state['current_field_state'] = field
        
        return updated_state
    
    # Helper methods (would be implemented in a real implementation)
    
    def _detect_attractors(self, field: Field, detect_type: str) -> List[Attractor]:
        """Detect attractors in the field."""
        # Placeholder implementation
        return [{"id": "attractor_1", "strength": 0.8, "pattern": "Example pattern"}]
    
    def _filter_attractors(self, attractors: List[Attractor], filter_by: str) -> List[Attractor]:
        """Filter attractors by the specified criterion."""
        # Placeholder implementation
        return attractors
    
    def _detect_residue(self, field: Field, mode: str) -> List[Residue]:
        """Detect symbolic residue in the field."""
        # Placeholder implementation
        return [{"id": "residue_1", "content": "Example residue", "strength": 0.6}]
    
    def _integrate_residue(self, field: Field, residues: List[Residue]) -> Field:
        """Integrate residue into the field."""
        # Placeholder implementation
        return field
    
    def _apply_harmonic_integration(self, field: Field, attractors: List[Attractor]) -> Field:
        """Apply harmonic integration to facilitate co-emergence."""
        # Placeholder implementation
        return field
    
    def _apply_boundary_dissolution(self, field: Field, attractors: List[Attractor]) -> Field:
        """Dissolve boundaries between attractors."""
        # Placeholder implementation
        return field
    
    def _apply_resonance_amplification(self, field: Field, attractors: List[Attractor]) -> Field:
        """Amplify resonance between attractors."""
        # Placeholder implementation
        return field
    
    def _identify_attractor_basins(self, field: Field) -> List[Dict[str, Any]]:
        """Identify basins of attraction in the field."""
        # Placeholder implementation
        return [{"id": "basin_1", "center": [0.5, 0.5], "radius": 0.3}]
    
    def _calculate_field_coherence(self, field: Field) -> float:
        """Calculate overall field coherence."""
        # Placeholder implementation
        return 0.85
    
    def _detect_emergent_patterns(self, field: Field) -> List[Dict[str, Any]]:
        """Detect emergent patterns in the field."""
        # Placeholder implementation
        return [{"id": "pattern_1", "type": "novel concept", "strength": 0.7}]
    
    def _generate_cycle_prompt(self, field: Field, audit_results: Dict[str, Any]) -> str:
        """Generate a prompt for the next cycle."""
        # Placeholder implementation
        return "Continue processing with focus on emerging patterns."
    
    def _generate_pattern_prompt(self, patterns: List[Dict[str, Any]]) -> str:
        """Generate a prompt based on emergent patterns."""
        # Placeholder implementation
        return f"Explore pattern {patterns[0]['id']} further."
    
    def _generate_coherence_prompt(self, coherence: float) -> str:
        """Generate a prompt based on field coherence."""
        # Placeholder implementation
        return f"Field coherence at {coherence:.2f}. Focus on integration."
    
    def _detect_co_emergent_attractors(self, field: Field) -> List[Attractor]:
        """Detect attractors that have co-emerged."""
        # Placeholder implementation
        return [{"id": "co_emergent_1", "strength": 0.9, "pattern": "Co-emergent pattern"}]
    
    def _integrate_attractors(self, field: Field, attractors: List[Attractor]) -> Field:
        """Integrate attractors into the field."""
        # Placeholder implementation
        return field
    
    def _collapse_all_boundaries(self, field: Field) -> Field:
        """Collapse all field boundaries."""
        # Placeholder implementation
        return field
    
    def _collapse_selected_boundaries(self, field: Field) -> Field:
        """Collapse selected boundaries."""
        # Placeholder implementation
        return field
    
    def _create_gradient_boundaries(self, field: Field) -> Field:
        """Create gradient boundaries."""
        # Placeholder implementation
        return field


class RecursiveEmergenceProtocol(ProtocolShell):
    """Implementation of the recursive.emergence protocol."""
    
    def self_prompt_loop(self, state: Dict[str, Any], 
                        trigger_condition: str = 'cycle_interval') -> Dict[str, Any]:
        """
        Initialize a self-prompting loop in the field.
        
        Args:
            state: Current execution state
            trigger_condition: When to trigger self-prompts
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('initial_field_state', {})
        
        # Implementation would initialize self-prompting mechanism
        # This is a placeholder implementation
        trigger = self._create_trigger(trigger_condition)
        self_prompt_mechanism = self._create_self_prompt_mechanism(trigger)
        field = self._integrate_mechanism(field, self_prompt_mechanism)
        
        # Update state with modified field
        updated_state = state.copy()
        updated_state['current_field_state'] = field
        updated_state['self_prompt_mechanism'] = self_prompt_mechanism
        
        return updated_state
    
    def agency_activate(self, state: Dict[str, Any], 
                       enable_field_agency: bool = True,
                       agency_level: float = 0.7) -> Dict[str, Any]:
        """
        Activate autonomous agency in the field.
        
        Args:
            state: Current execution state
            enable_field_agency: Whether to enable field agency
            agency_level: Level of autonomy (0.0 to 1.0)
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would activate field agency
        # This is a placeholder implementation
        if enable_field_agency:
            agency_mechanisms = self._create_agency_mechanisms(agency_level)
            field = self._integrate_agency(field, agency_mechanisms, agency_level)
        
        # Update state with modified field
        updated_state = state.copy()
        updated_state['current_field_state'] = field
        updated_state['agency_level'] = agency_level if enable_field_agency else 0.0
        
        return updated_state
    
    def residue_compress(self, state: Dict[str, Any],
                        integrate_residue_into_field: bool = True) -> Dict[str, Any]:
        """
        Compress and integrate symbolic residue.
        
        Args:
            state: Current execution state
            integrate_residue_into_field: Whether to integrate residue
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would compress and integrate residue
        # This is a placeholder implementation
        residue = self._detect_residue(field)
        compressed_residue = self._compress_residue(residue)
        
        if integrate_residue_into_field:
            field = self._integrate_residue(field, compressed_residue)
        
        # Update state with modified field and residue
        updated_state = state.copy()
        updated_state['current_field_state'] = field
        updated_state['integrated_residue'] = compressed_residue if integrate_residue_into_field else None
        updated_state['compressed_residue'] = compressed_residue
        
        return updated_state
    
    def boundary_collapse(self, state: Dict[str, Any],
                         monitor: str = 'field drift, coherence') -> Dict[str, Any]:
        """
        Manage field boundaries through controlled collapse.
        
        Args:
            state: Current execution state
            monitor: What aspects to monitor during collapse
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would monitor field and collapse boundaries
        # This is a placeholder implementation
        monitoring_results = self._monitor_field(field, monitor)
        
        if self._should_collapse_boundaries(monitoring_results):
            boundaries = self._identify_collapse_boundaries(field, monitoring_results)
            field = self._collapse_boundaries(field, boundaries)
        
        # Update state with modified field and monitoring results
        updated_state = state.copy()
        updated_state['current_field_state'] = field
        updated_state['monitoring_results'] = monitoring_results
        
        return updated_state
    
    def emergence_detect(self, state: Dict[str, Any],
                        pattern: str = 'recursive capability') -> Dict[str, Any]:
        """
        Detect emergent patterns in the field.
        
        Args:
            state: Current execution state
            pattern: Type of pattern to detect
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would detect emergent patterns
        # This is a placeholder implementation
        detector = self._create_pattern_detector(pattern)
        emergent_patterns = self._scan_for_patterns(field, detector)
        pattern_analysis = self._analyze_patterns(emergent_patterns)
        
        # Update state with detected patterns and analysis
        updated_state = state.copy()
        updated_state['emergent_patterns'] = emergent_patterns
        updated_state['pattern_analysis'] = pattern_analysis
        
        return updated_state
    
    def field_evolution(self, state: Dict[str, Any],
                       strategy: str = 'self_improving') -> Dict[str, Any]:
        """
        Guide field evolution according to the specified strategy.
        
        Args:
            state: Current execution state
            strategy: Evolution strategy
            
        Returns:
            Updated execution state
        """
        # Extract field from state
        field = state.get('current_field_state', {})
        
        # Implementation would guide field evolution
        # This is a placeholder implementation
        evolution_strategy = self._create_evolution_strategy(strategy)
        field = self._apply_evolution_strategy(field, evolution_strategy)
        evolution_metrics = self._measure_evolution(field)
        
        # Update state with evolved field and metrics
        updated_state = state.copy()
        updated_state['current_field_state'] = field
        updated_state['evolution_metrics'] = evolution_metrics
        
        return updated_state
    
    def halt_check(self, state: Dict[str, Any],
                  criteria: str = 'convergence || max_cycles') -> Dict[str, Any]:
        """
        Check whether the recursive process should halt.
        
        Args:
            state: Current execution state
            criteria: Halt criteria
            
        Returns:
            Updated execution state with halt flag
        """
        # Extract field and cycle count from state
        field = state.get('current_field_state', {})
        cycle_count = state.get('cycle_count', 0)
        max_cycles = state.get('max_cycles', 100)
        
        # Implementation would check halt criteria
        # This is a placeholder implementation
        should_halt = False
        
        if 'convergence' in criteria:
            convergence = self._measure_convergence(field)
            if convergence > 0.9:  # Convergence threshold
                should_halt = True
        
        if 'max_cycles' in criteria and cycle_count >= max_cycles:
            should_halt = True
        
        # Update state with halt flag
        updated_state = state.copy()
        updated_state['should_halt'] = should_halt
        updated_state['halt_reason'] = self._determine_halt_reason(should_halt, cycle_count, max_cycles, field)
        
        return updated_state
    
    # Helper methods (would be implemented in a real implementation)
    
    def _create_trigger(self, trigger_condition: str) -> Dict[str, Any]:
        """Create a trigger for self-prompting."""
        # Placeholder implementation
        return {"type": trigger_condition, "interval": 3}
    
    def _create_self_prompt_mechanism(self, trigger: Dict[str, Any]) -> Dict[str, Any]:
        """Create a self-prompting mechanism."""
        # Placeholder implementation
        return {"trigger": trigger, "templates": ["Template 1", "Template 2"]}
    
    def _integrate_mechanism(self, field: Field, mechanism: Dict[str, Any]) -> Field:
        """Integrate a mechanism into the field."""
        # Placeholder implementation
        return field
    
    def _create_agency_mechanisms(self, agency_level: float) -> List[Dict[str, Any]]:
        """Create agency mechanisms."""
        # Placeholder implementation
        return [
            {"type": "self_assessment", "strength": agency_level},
            {"type": "goal_setting", "strength": agency_level},
            {"type": "action_selection", "strength": agency_level}
        ]
    
    def _integrate_agency(self, field: Field, mechanisms: List[Dict[str, Any]], 
                        level: float) -> Field:
        """Integrate agency mechanisms into the field."""
        # Placeholder implementation
        return field
    
    def _detect_residue(self, field: Field) -> List[Residue]:
        """Detect symbolic residue in the field."""
        # Placeholder implementation
        return [{"id": "residue_1", "content": "Example residue", "strength": 0.6}]
    
    def _compress_residue(self, residue: List[Residue]) -> List[Residue]:
        """Compress symbolic residue."""
        # Placeholder implementation
        return residue
    
    def _integrate_residue(self, field: Field, residue: List[Residue]) -> Field:
        """Integrate residue into the field."""
        # Placeholder implementation
        return field
    
    def _monitor_field(self, field: Field, monitor: str) -> Dict[str, Any]:
        """Monitor specified aspects of the field."""
        # Placeholder implementation
        results = {}
        if 'field drift' in monitor:
            results['drift'] = 0.3  # Example drift value
        if 'coherence' in monitor:
            results['coherence'] = 0.8  # Example coherence value
        return results
    
    def _should_collapse_boundaries(self, monitoring_results: Dict[str, Any]) -> bool:
        """Determine if boundaries should be collapsed."""
        # Placeholder implementation
        return monitoring_results.get('drift', 0) > 0.5 or monitoring_results.get('coherence', 0) < 0.5
    
    def _identify_collapse_boundaries(self, field: Field, 
                                    monitoring_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify boundaries to collapse."""
        # Placeholder implementation
        return [{"id": "boundary_1", "type": "semantic", "strength": 0.7}]
    
    def _collapse_boundaries(self, field: Field, 
                           boundaries: List[Dict[str, Any]]) -> Field:
        """Collapse specified boundaries."""
        # Placeholder implementation
        return field
    
    def _create_pattern_detector(self, pattern: str) -> Dict[str, Any]:
        """Create a pattern detector."""
        # Placeholder implementation
        return {"type": pattern, "sensitivity": 0.7}
    
    def _scan_for_patterns(self, field: Field, 
                         detector: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan for patterns in the field."""
        # Placeholder implementation
        return [{"id": "pattern_1", "type": detector["type"], "strength": 0.8}]
    
    def _analyze_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze detected patterns."""
        # Placeholder implementation
        return {
            "count": len(patterns),
            "average_strength": sum(p["strength"] for p in patterns) / len(patterns) if patterns else 0,
            "recursion_depth": 2  # Example recursion depth
        }
    
    def _create_evolution_strategy(self, strategy: str) -> Dict[str, Any]:
        """Create an evolution strategy."""
        # Placeholder implementation
        return {"type": strategy, "rate": 0.5}
    
    def _apply_evolution_strategy(self, field: Field, 
                                strategy: Dict[str, Any]) -> Field:
        """Apply an evolution strategy to the field."""
        # Placeholder implementation
        return field
    
    def _measure_evolution(self, field: Field) -> Dict[str, Any]:
        """Measure evolution metrics."""
        # Placeholder implementation
        return {
            "improvement": 0.3,
            "complexity": 0.7,
            "agency_level": 0.8
        }
    
    def _measure_convergence(self, field: Field) -> float:
        """Measure field convergence."""
        # Placeholder implementation
        return 0.85
    
    def _determine_halt_reason(self, should_halt: bool, cycle_count: int, 
                             max_cycles: int, field: Field) -> str:
        """Determine the reason for halting."""
        # Placeholder implementation
        if not should_halt:
            return "not_halted"
        elif cycle_count >= max_cycles:
            return "max_cycles_reached"
        else:
            return "convergence_achieved"

# ----------------------------------------------------------------------------
# Example functions – each saves everything in its own folder
# ----------------------------------------------------------------------------

def example_01_basic_parsing_and_execution():
    ex_dir = create_example_dir("example_01_basic_parsing_and_execution")
    log = get_example_logger("ex01", ex_dir)

    shell_str = """
    test.protocol {
      intent: "Simple test protocol",
      input: { value: "<input_value>" },
      process: [
        /echo.message { message: "Hello from protocol!" }
      ],
      output: { greeting: "greeting" },
      meta: { version: "1.0" }
    }
    """.strip()

    class EchoProtocol(ProtocolShell):
        def echo_message(self, state, message: str):
            state["greeting"] = f"{message} Input was: {state.get('value', 'none')}"
            return state

    # This now works – parser handles the multi-line shell perfectly
    protocol = EchoProtocol.from_string(shell_str)
    result = protocol.execute({"value": "world"})

    (ex_dir / "protocol.shell").write_text(shell_str)
    (ex_dir / "input.json").write_text(json.dumps({"value": "world"}, indent=2))
    (ex_dir / "output.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (ex_dir / "protocol_pretty.md").write_text(
        f"# {protocol.name}\n\n{ProtocolParser.serialize_shell(protocol.protocol_dict)}"
    )

    log.info("example_01_basic_parsing_and_execution completed")
    log.info(f"Generated greeting → {result.get('greeting')}")

def example_02_attractor_co_emerge_protocol():
    ex_dir = create_example_dir("example_02_attractor_co_emerge_protocol")
    log = get_example_logger("ex02", ex_dir)

    shell_str = """
    attractor.co.emerge {
      intent: "Facilitate co-emergence of attractors",
      input: { current_field_state: "<field>" },
      process: [
        /attractor_scan { detect: "attractors", filter_by: "strength" },
        /residue_surface { mode: "recursive" },
        /co_emergence_algorithms { strategy: "harmonic integration" }
      ],
      output: { detected_attractors: "...", surfaced_residues: "...", current_field_state: "..." },
      meta: { version: "2.1" }
    }
    """
    protocol = AttractorCoEmergeProtocol.from_string(shell_str)
    fake_field = {"patterns": ["A", "B"], "strength": 0.8}
    result = protocol.execute({"current_field_state": fake_field})

    (ex_dir / "protocol.shell").write_text(shell_str)
    (ex_dir / "input_field.json").write_text(json.dumps(fake_field, indent=2))
    (ex_dir / "full_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("Attractor co-emergence protocol executed")

def example_03_recursive_emergence_with_llm_self_prompt():
    ex_dir = create_example_dir("example_03_recursive_emergence_with_llm_self_prompt")
    log = get_example_logger("ex03", ex_dir)

    llm = LlamacppLLM(model="qwen3-instruct-2507:4b", verbose=True)

    shell_str = """
    recursive.emergence {
      intent: "Enable recursive self-improvement via self-prompting",
      input: { initial_field_state: "<field>", cycle_count: 0 },
      process: [
        /self_prompt_loop { trigger_condition: "cycle_interval" },
        /agency_activate { enable_field_agency: true, agency_level: 0.8 },
        /residue_compress { integrate_residue_into_field: true },
        /emergence_detect { pattern: "recursive capability" }
      ],
      output: { current_field_state: "...", emergent_patterns: "...", agency_level: "..." },
      meta: { version: "3.0" }
    }
    """
    protocol = RecursiveEmergenceProtocol.from_string(shell_str)
    initial_field = {"concept": "self-awareness", "strength": 0.4}
    result = protocol.execute({"initial_field_state": initial_field, "cycle_count": 1})

    # Simulate LLM self-prompt streaming
    prompt = f"Field state:\n{initial_field}\nContinue recursive emergence..."
    log.info("Streaming LLM self-prompt...")
    response = ""
    for chunk in llm.generate(prompt, max_tokens=1200, stream=True):
        response += chunk
    (ex_dir / "llm_self_prompt.md").write_text(response)

    (ex_dir / "protocol.shell").write_text(shell_str)
    (ex_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("Recursive emergence example finished")

def example_04_protocol_with_real_llm_integration():
    ex_dir = create_example_dir("example_04_protocol_with_real_llm_integration")
    log = get_example_logger("ex04", ex_dir)

    llm = LlamacppLLM(model="qwen3-instruct-2507:4b", verbose=True)

    class LLMProtocol(ProtocolShell):
        def llm_generate(self, state, prompt_key: str = "prompt"):
            prompt = state.get(prompt_key, "Think step by step.")
            log.info(f"Generating with prompt: {prompt[:100]}...")
            full = ""
            for chunk in llm.generate(prompt, max_tokens=1200, stream=True):
                full += chunk
            state["llm_response"] = full
            return state

    shell = """
    llm.integration {
      intent: "Use real LLM inside protocol",
      input: { prompt: "Explain quantum entanglement simply" },
      process: [ /llm_generate { prompt_key: "prompt" } ],
      output: { llm_response: "..." },
      meta: { version: "1.0" }
    }
    """
    protocol = LLMProtocol.from_string(shell)
    result = protocol.execute({"prompt": "Explain quantum entanglement in simple terms."})

    (ex_dir / "protocol.shell").write_text(shell)
    (ex_dir / "llm_response.md").write_text(result["llm_response"])
    (ex_dir / "full_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("LLM-integrated protocol completed")

def example_05_full_cycle_with_field_simulation():
    ex_dir = create_example_dir("example_05_full_cycle_with_field_simulation")
    log = get_example_logger("ex05", ex_dir)

    # Simulate a very simple field dict
    field = {
        "attractors": [{"id": "a1", "pattern": "reasoning", "strength": 0.9}],
        "residues": [{"id": "r1", "content": "meta-cognition"}]
    }

    protocol = AttractorCoEmergeProtocol.from_string("""
    full.cycle {
      intent: "Run complete co-emergence cycle",
      input: { current_field_state: "<field>" },
      process: [
        /attractor_scan {},
        /residue_surface {},
        /co_emergence_algorithms { strategy: "resonance amplification" },
        /field_audit { surface_new: "attractor_basins" },
        /integration_protocol {}
      ],
      output: { detected_attractors: "...", current_field_state: "..." },
      meta: { version: "4.2" }
    }
    """)

    result = protocol.execute({"current_field_state": field})

    (ex_dir / "initial_field.json").write_text(json.dumps(field, indent=2))
    (ex_dir / "final_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("Full co-emergence cycle example completed")

# ----------------------------------------------------------------------------
# MAIN – Run all examples with beautiful live streaming
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main_logger.info("═" * 100)
    main_logger.info("FIELD PROTOCOL SHELLS – LIVE EXAMPLES WITH FULL ARTIFACT SAVING")
    main_logger.info("Watch streaming LLM chunks, per-example folders, and saved artifacts")
    main_logger.info("═" * 100)

    example_01_basic_parsing_and_execution()
    main_logger.info("\n" + "─" * 90 + "\n")

    example_02_attractor_co_emerge_protocol()
    main_logger.info("\n" + "─" * 90 + "\n")

    example_03_recursive_emergence_with_llm_self_prompt()
    main_logger.info("\n" + "─" * 90 + "\n")

    example_04_protocol_with_real_llm_integration()
    main_logger.info("\n" + "─" * 90 + "\n")

    example_05_full_cycle_with_field_simulation()

    main_logger.info(f"\nAll examples finished! Artifacts saved under: {OUTPUT_DIR}")
