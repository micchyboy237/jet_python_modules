from jet.adapters.keybert import KeyLLM
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet_telemetry import get_trace_url, initialize_telemetry

initialize_telemetry(service_name="keyllm-demo")


# Define a simple LLM wrapper that uses jet's observed chat
class JetLLM:
    def __call__(self, prompt: str, **kwargs) -> str:
        # Use jet's observed chat function
        result = chat(prompt, project_name="keyllm-chat")
        return result.content


# Initialize KeyLLM with our Jet LLM wrapper
llm_wrapper = JetLLM()
kw_model = KeyLLM(llm=llm_wrapper)

doc = "The future of renewable energy looks promising with solar advancements."

# Extract keywords using LLM
print("--- KeyLLM Extraction ---")
keywords = kw_model.extract_keywords(doc)
print(keywords)

print(f"\nTrace URL: {get_trace_url()}")
