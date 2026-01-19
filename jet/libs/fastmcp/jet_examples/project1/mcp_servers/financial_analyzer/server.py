"""Simple financial analysis MCP server with tools and prompts."""
import os
from datetime import datetime
from typing import Dict, Any, List
from fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP(
    name="FinancialAnalyzer",
    instructions="You are a financial analysis assistant. Use the provided tools to help users analyze financial data."
)

class StockPriceInput(BaseModel):
    symbol: str
    date: str | None = None # YYYY-MM-DD

class PortfolioInput(BaseModel):
    holdings: Dict[str, float] # symbol -> shares

@mcp.tool(output_schema=StockPriceInput)
def get_stock_price(symbol: str, date: str | None = None) -> Dict[str, Any]:
    """Fetch current or historical stock price (mocked for demo)."""
    # In real life → call real API (e.g., yfinance, alpha vantage)
    price = 150.25 if date is None else 142.80
    return {
        "symbol": symbol.upper(),
        "price": price,
        "currency": "USD",
        "timestamp": datetime.utcnow().isoformat(),
        "source": "mock"
    }

@mcp.tool(output_schema=Dict[str, float])
def calculate_portfolio_value(holdings: Dict[str, float]) -> Dict[str, Any]:
    """Calculate total portfolio value from holdings (mocked prices)."""
    total = sum(shares * 150.25 for shares in holdings.values()) # mocked price
    return {
        "total_value": total,
        "currency": "USD",
        "timestamp": datetime.utcnow().isoformat(),
        "breakdown": {sym: shares * 150.25 for sym, shares in holdings.items()}
    }

@mcp.prompt(name="financial_report_template")
def financial_report_template(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate a formatted financial report prompt."""
    return [
        {"role": "system", "content": "You are a professional financial analyst."},
        {"role": "user", "content": f"Analyze this portfolio data:\n{data}"}
    ]

@mcp.resource(uri="resource://reports/example.json")
def example_report() -> Dict[str, Any]:
    """Static example financial report."""
    return {
        "title": "Sample Q4 Report",
        "total_value": 125000.75,
        "assets": ["AAPL", "TSLA", "GOOGL"]
    }

if __name__ == "__main__":
    mcp.run()
