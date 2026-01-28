from typing import Optional
import datetime

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    tool,
    WebSearchTool,
    VisitWebpageTool,
    OpenAIModel,
    load_tool,
)

# ───────────────────────────────────────────────
#            Local model factory (as given)
# ───────────────────────────────────────────────
def create_local_model(
    temperature: float = 0.7,
    max_tokens: Optional[int] = 2048,
    model_id: str = "local-model",
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        base_url="http://shawn-pc.local:8080/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ───────────────────────────────────────────────
#        Example: Better-designed tool (with logging & clear format)
# ───────────────────────────────────────────────
@tool
def get_weather_report(location: str, date_time_str: str) -> str:
    """
    Get weather information for a surf spot.

    Args:
        location: Place name, preferably "spot, city, country" format
                  Example: "Anchor Point, Taghazout, Morocco"
        date_time_str: Date and time in format 'YYYY-MM-DD HH:MM'
                       (24-hour clock)

    Returns:
        Formatted weather string with temperature, rain risk, wave height
    """
    # Dummy implementation — replace with real API
    try:
        dt = datetime.datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")
    except ValueError as e:
        error_msg = (
            f"Date/time format error. Use 'YYYY-MM-DD HH:MM'. "
            f"Received: '{date_time_str}'. Original error: {str(e)}"
        )
        print(error_msg)
        raise ValueError(error_msg)

    print(f"Fetching weather for {location} at {dt}")

    # Dummy values
    temp_c = 27.8
    rain_prob = 0.22
    wave_m = 1.4

    result = (
        f"Weather report for {location} on {dt:%Y-%m-%d %H:%M}:\n"
        f"• Temperature: {temp_c} °C\n"
        f"• Rain probability: {rain_prob*100:.0f}%\n"
        f"• Wave height: {wave_m} m"
    )
    print(result)
    return result


# ───────────────────────────────────────────────
#                   DEMO FUNCTIONS
# ───────────────────────────────────────────────

def demo_01_simple_local_agent_with_guidance():
    """Demo 1: Basic local agent + custom instructions for clarity"""
    model = create_local_model(temperature=0.65, max_tokens=3072)

    agent = CodeAgent(
        tools=[],
        model=model,
        instructions=(
            "You are a precise calculator agent. "
            "Always show intermediate steps with print(). "
            "Use final_answer() only with the final numeric result."
        )
    )

    result = agent.run("What is (17**8 + 94321) mod 10000019 ? Show steps.")
    print("Result:", result)


def demo_02_local_agent_with_better_tool():
    """Demo 2: Using a well-designed tool (logging + strict format)"""
    model = create_local_model(temperature=0.7, max_tokens=4096)

    agent = CodeAgent(
        tools=[get_weather_report],
        model=model,
        instructions="Use the weather tool with correct date format 'YYYY-MM-DD HH:MM'"
    )

    result = agent.run(
        "What will be the wave height in Jeffreys Bay, South Africa tomorrow at 09:00?"
    )
    print("Final weather report:", result)


def demo_03_local_hierarchical_with_planning():
    """Demo 3: Manager + sub-agent + planning step every 4 actions"""
    model = create_local_model(temperature=0.68, max_tokens=4096)

    researcher = ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="researcher",
        description="Performs web searches and reads pages when needed"
    )

    manager = CodeAgent(
        model=model,
        tools=[],
        managed_agents=[researcher],
        planning_interval=4,           # ← activates planning every 4 steps
        instructions=(
            "You are a coordinating agent. "
            "Delegate research questions to the 'researcher' agent. "
            "Use planning steps to reflect on progress."
        )
    )

    result = manager.run(
        "If US real GDP grew 2.8% in 2024, how many years to double at that rate?"
    )
    print("Doubling time estimate:", result)


def demo_04_local_with_additional_args():
    """Demo 4: Passing extra context via additional_args"""
    model = create_local_model(temperature=0.7)

    agent = CodeAgent(
        tools=[],
        model=model,
        instructions="You can use variables passed via additional_args directly in your code."
    )

    extra_context = {
        "reference_mp3": "https://example.com/surf_conditions_2025.mp3",
        "preferred_spots": ["Supertubes", "Mundaka", "Pipeline"],
    }

    result = agent.run(
        "Which of the preferred spots would be best tomorrow based on typical winter conditions?",
        additional_args=extra_context
    )
    print(result)


def demo_05_local_with_image_generation_and_planning():
    """Demo 5: Planning + image generation tool from hub"""
    model = create_local_model(temperature=0.72, max_tokens=5120)

    # Example loading a community tool (text-to-image)
    try:
        image_gen_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
    except Exception as e:
        print("Could not load image tool:", e)
        image_gen_tool = None

    agent = CodeAgent(
        tools=[image_gen_tool] if image_gen_tool else [],
        model=model,
        planning_interval=3,  # Think about next steps every 3 actions
        instructions="When asked for images, use the image generation tool and return the path."
    )

    result = agent.run("Generate a picture of a perfect point break at sunset in Morocco")
    print("Execution result:", result)


def demo_06_minimal_langfuse_traced_local_agent():
    """Demo 6: Local + Langfuse tracing (minimal)"""
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor

    # Activate tracing
    SmolagentsInstrumentor().instrument()

    model = create_local_model(temperature=0.7)

    agent = ToolCallingAgent(
        model=model,
        tools=[WebSearchTool(), VisitWebpageTool()],
        name="traced-web-agent"
    )

    agent.run("What is the current world record for biggest wave ever surfed?")


# ───────────────────────────────────────────────
#                      Usage
# ───────────────────────────────────────────────

if __name__ == "__main__":
    # Uncomment the one(s) you want to run
    # demo_01_simple_local_agent_with_guidance()
    # demo_02_local_agent_with_better_tool()
    # demo_03_local_hierarchical_with_planning()
    # demo_04_local_with_additional_args()
    demo_05_local_with_image_generation_and_planning()
    # demo_06_minimal_langfuse_traced_local_agent()