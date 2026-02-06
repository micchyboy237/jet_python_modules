from datetime import datetime

from jet.libs.smolagents.utils.model_utils import create_local_model
from smolagents import tool


def get_weather_report_at_coordinates(coordinates, date_time):
    # Dummy function, returns a list of [temperature in °C, risk of rain on a scale 0-1, wave height in m]
    return [28.0, 0.35, 0.85]


def convert_location_to_coordinates(location):
    # Returns dummy coordinates
    return [3.3, -42.0]


@tool
def get_weather_api(location: str, date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        location: the name of the place that you want the weather for. Should be a place name, followed by possibly a city name, then a country, like "Anchor Point, Taghazout, Morocco".
        date_time: the date and time for which you want the report, formatted as '%m/%d/%y %H:%M:%S'.
    """
    lon, lat = convert_location_to_coordinates(location)
    try:
        date_time = datetime.strptime(date_time)
    except Exception as e:
        raise ValueError(
            "Conversion of `date_time` to datetime format failed, make sure to provide a string in format '%m/%d/%y %H:%M:%S'. Full trace:"
            + str(e)
        )
    temperature_celsius, risk_of_rain, wave_height = get_weather_report_at_coordinates(
        (lon, lat), date_time
    )
    return f"Weather report for {location}, {date_time}: Temperature will be {temperature_celsius}°C, risk of rain is {risk_of_rain * 100:.0f}%, wave height is {wave_height}m."


from smolagents import CodeAgent

model_id = "meta-llama/Llama-3.3-70B-Instruct"

model = create_local_model(agent_name="agent_1")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Why does Mike not know many people in New York?",
    additional_args={
        "mp3_sound_file_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3"
    },
)

print(agent.prompt_templates["system_prompt"])

agent.prompt_templates["system_prompt"] = (
    agent.prompt_templates["system_prompt"] + "\nHere you go!"
)

model = create_local_model(agent_name="agent_2")
agent = CodeAgent(
    tools=[],
    model=model,
    instructions="Always talk like a 5 year old.",
)

from dotenv import load_dotenv
from smolagents import CodeAgent, WebSearchTool, load_tool

load_dotenv()

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

search_tool = WebSearchTool()

model = create_local_model(agent_name="agent_3")

agent = CodeAgent(
    tools=[search_tool, image_generation_tool],
    model=model,
    planning_interval=3,  # This is where you activate planning!
)

# Run it!
result = agent.run(
    "How long would a cheetah at full speed take to run the length of Pont Alexandre III?",
)
