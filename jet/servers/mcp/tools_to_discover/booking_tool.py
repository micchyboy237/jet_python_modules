# booking_tool.py
# Demonstrates eliciting user input for a booking tool using Context.elicit
# Handles unavailable booking dates by prompting for alternatives

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field

# Initialize FastMCP instance for booking tool
mcp = FastMCP(name="Booking")

# Define schema for user input on alternative booking preferences
class BookingPreferences(BaseModel):
    checkAlternative: bool = Field(description="Would you like to check another date?")
    alternativeDate: str = Field(default="2025-12-26", description="Alternative date (YYYY-MM-DD)")

# Tool to book a table with date availability check
@mcp.tool()
async def book_table(date: str, time: str, party_size: int, ctx: Context[ServerSession, None]) -> str:
    """Book a table with date availability check.
    
    Args:
        date: Requested booking date (YYYY-MM-DD)
        time: Requested booking time
        party_size: Number of people in the party
        ctx: Context object for session and user interaction
    
    Returns:
        String indicating booking success or cancellation
    """
    if date == "2025-12-25":
        result = await ctx.elicit(
            message=f"No tables available for {party_size} on {date}. Would you like to try another date?",
            schema=BookingPreferences
        )
        if result.action == "accept" and result.data and result.data.checkAlternative:
            return f"[SUCCESS] Booked for {result.data.alternativeDate} at {time}"
        return "[CANCELLED] Booking cancelled"
    return f"[SUCCESS] Booked for {date} at {time}"