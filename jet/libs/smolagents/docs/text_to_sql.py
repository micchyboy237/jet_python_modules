# demo_text_to_sql_local.py
"""
Text-to-SQL agent demos using smolagents + in-memory SQLite
Reuses create_local_model() → your local llama.cpp OpenAI-compatible endpoint
"""

import time

from jet.libs.smolagents.utils.model_utils import create_local_model
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from smolagents import CodeAgent, tool
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    inspect,
    text,
)

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Database setup (run once)
# ──────────────────────────────────────────────────────────────────────────────


def setup_receipts_database() -> tuple:
    """Creates in-memory SQLite DB with receipts and waiters tables."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()

    # Table 1: receipts
    receipts = Table(
        "receipts",
        metadata,
        Column("receipt_id", Integer, primary_key=True),
        Column("customer_name", String(16), primary_key=True),
        Column("price", Float),
        Column("tip", Float),
    )

    # Table 2: waiters
    waiters = Table(
        "waiters",
        metadata,
        Column("receipt_id", Integer, primary_key=True),
        Column("waiter_name", String(16), primary_key=True),
    )

    metadata.create_all(engine)

    # Insert data
    receipt_rows = [
        {"receipt_id": 1, "customer_name": "Alan Payne", "price": 12.06, "tip": 1.20},
        {"receipt_id": 2, "customer_name": "Alex Mason", "price": 23.86, "tip": 0.24},
        {
            "receipt_id": 3,
            "customer_name": "Woodrow Wilson",
            "price": 53.43,
            "tip": 5.43,
        },
        {
            "receipt_id": 4,
            "customer_name": "Margaret James",
            "price": 21.11,
            "tip": 1.00,
        },
    ]
    insert_rows_into_table(receipt_rows, receipts, engine)

    waiter_rows = [
        {"receipt_id": 1, "waiter_name": "Corey Johnson"},
        {"receipt_id": 2, "waiter_name": "Michael Watts"},
        {"receipt_id": 3, "waiter_name": "Michael Watts"},
        {"receipt_id": 4, "waiter_name": "Margaret James"},
    ]
    insert_rows_into_table(waiter_rows, waiters, engine)

    console.print(
        "[dim]In-memory SQLite database initialized with receipts & waiters[/dim]"
    )
    return engine, metadata


def insert_rows_into_table(rows, table, engine):
    for row in rows:
        stmt = insert(table).values(**row)
        with engine.begin() as conn:
            conn.execute(stmt)


# Global DB (created once)
ENGINE, METADATA = setup_receipts_database()


# ──────────────────────────────────────────────────────────────────────────────
# SQL Tool factory
# ──────────────────────────────────────────────────────────────────────────────


@tool
def sql_engine(query: str) -> str:
    """
    Execute SQL query against the receipts database.
    Returns string representation of results or error message.
    """
    output = []
    try:
        with ENGINE.connect() as conn:
            result = conn.execute(text(query))
            headers = result.keys()
            if headers:
                output.append(" | ".join(headers))
                output.append("-" * 40)
            for row in result:
                output.append(" | ".join(str(v) for v in row))
        return "\n".join(output) if output else "(no rows returned)"
    except Exception as e:
        return f"SQL Error: {str(e)}"


def update_sql_tool_description():
    """Dynamically builds tool description with current schema."""
    inspector = inspect(ENGINE)
    tables = ["receipts", "waiters"]

    desc = "Allows execution of SQL queries on the following tables:\n\n"

    for table_name in tables:
        columns = [
            (col["name"], str(col["type"])) for col in inspector.get_columns(table_name)
        ]
        desc += f"Table '{table_name}':\n"
        desc += "Columns:\n" + "\n".join(f"  - {name}: {typ}" for name, typ in columns)
        desc += "\n\n"

    desc += "Always write correct SQLite syntax. Use JOINs when needed."
    sql_engine.description = desc
    console.print("[dim]SQL tool description updated with current schema[/dim]")


# ──────────────────────────────────────────────────────────────────────────────
# Agent factory
# ──────────────────────────────────────────────────────────────────────────────


def create_text2sql_agent(max_steps: int = 8, verbosity_level: int = 1) -> CodeAgent:
    """Creates CodeAgent configured for text-to-SQL tasks."""
    update_sql_tool_description()  # refresh schema each time
    model = create_local_model(temperature=0.6)
    return CodeAgent(
        tools=[sql_engine],
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────


def demo_t2sql_1_single_table_max():
    """Demo 1: Find the most expensive receipt (single table)"""
    console.rule("Demo 1: Highest receipt amount", style="blue")

    agent = create_text2sql_agent(max_steps=6)

    question = "Who paid the highest amount (price) and how much was it?"
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")

    start = time.time()
    answer = agent.run(question)
    duration = time.time() - start

    console.print(Panel(answer, title="Agent Answer", border_style="green"))
    console.print(f"[dim]Completed in {duration:.1f}s[/dim]")


def demo_t2sql_2_aggregation_tips():
    """Demo 2: Total tips per waiter (aggregation + join)"""
    console.rule("Demo 2: Total tips per waiter", style="blue")

    agent = create_text2sql_agent(max_steps=8, verbosity_level=2)

    question = "Which waiter received the highest total tips? Show the amount."
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")

    start = time.time()
    answer = agent.run(question)
    duration = time.time() - start

    console.print(Panel(answer, title="Agent Answer", border_style="green"))
    console.print(f"[dim]Completed in {duration:.1f}s[/dim]")


def demo_t2sql_3_join_and_reasoning():
    """Demo 3: More complex join + reasoning"""
    console.rule("Demo 3: Who served the highest tipper", style="blue")

    agent = create_text2sql_agent(max_steps=10, verbosity_level=1)

    question = (
        "Which waiter served the customer who left the highest tip? "
        "Include customer name, tip amount, and waiter name."
    )
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")

    start = time.time()
    answer = agent.run(question)
    duration = time.time() - start

    console.print(Panel(answer, title="Agent Answer", border_style="green"))
    console.print(f"[dim]Completed in {duration:.1f}s[/dim]")


def main():
    console.rule("Text-to-SQL Agent Demos — LOCAL llama.cpp", style="bold magenta")

    console.print(
        "[dim]In-memory SQLite with receipts & waiters tables[/dim]\n"
        "[dim]Agent can query, join, aggregate and self-correct[/dim]\n"
    )

    # Select which demos to run
    demo_t2sql_1_single_table_max()
    demo_t2sql_2_aggregation_tips()
    # demo_t2sql_3_join_and_reasoning()

    console.rule("Done", style="bold green")


if __name__ == "__main__":
    main()
