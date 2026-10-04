"""
Demo: Display All OpenTelemetry Resource Attributes
Covers: Inspecting the current resource attributes set by initialize_telemetry.
"""

from jet.adapters.llama_cpp.config import PHOENIX_BASE_URL
from jet_telemetry import display_all_resources, initialize_telemetry

# Initialize telemetry to populate resources
initialize_telemetry(service_name="resource-demo", endpoint=PHOENIX_BASE_URL)

print("🔍 Current OpenTelemetry Resource Attributes:")
print("-" * 50)
display_all_resources()
