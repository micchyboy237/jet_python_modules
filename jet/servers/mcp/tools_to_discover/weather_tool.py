# jet_python_modules/jet/servers/mcp/tools_to_discover/weather_tool.py
# Demonstrates using session data for personalized weather tool
# Uses custom session to store user preferences for temperature units

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
import httpx

mcp = FastMCP(name="Weather")


@mcp.tool()
async def get_temperature(city: str, ctx: Context[ServerSession, None]) -> str:
    """Get the temperature for a city.
    Args:
        city: Name of the city
        ctx: Context object with session for preferences
    Returns:
        Formatted string with temperature in user-preferred units
    """
    # Store preferences in session state
    if not hasattr(ctx.session, "preferences"):
        ctx.session.preferences = {"unit": "metric"}
    unit = ctx.session.preferences.get("unit", "metric")
    api_key = "your_openweather_api_key"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units={unit}&appid={api_key}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        temp = data["main"]["temp"]
        return f"Temperature in {city}: {temp}°{'C' if unit == 'metric' else 'F'}"
