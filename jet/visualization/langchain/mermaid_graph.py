import os
from pathlib import Path
from langchain_core.runnables.graph import MermaidDrawMethod

from jet.logger import logger

def render_mermaid_graph(
    agent,
    output_filename="graph_output.png",
    draw_method=MermaidDrawMethod.PYPPETEER,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    open_file: bool = False,
    **kwargs
):
    """
    Generates a Mermaid graph PNG from the agent and opens it using the system's default viewer on macOS.

    Args:
        agent: An object with a get_graph().draw_mermaid_png() method.
        output_filename (str): Name of the file to save the PNG to.
        draw_method: Drawing method (e.g., MermaidDrawMethod.API).
    """
    # Generate PNG bytes from the Mermaid graph
    png_bytes = agent.get_graph(**kwargs).draw_mermaid_png(
        draw_method=draw_method, max_retries=max_retries, retry_delay=retry_delay)

    # Define the output file path and ensure the directory exists
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)

    if open_file:
        # Open the PNG file with the default image viewer on macOS
        os.system(f"open {output_path}")

    logger.log("Saved graph to: ", output_filename, colors=["SUCCESS", "BRIGHT_SUCCESS"])
