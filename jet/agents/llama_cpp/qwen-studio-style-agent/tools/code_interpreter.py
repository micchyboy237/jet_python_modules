import asyncio
import logging
from typing import Optional

from agent.config import Config
from pyodide.code import run_js

logger = logging.getLogger(__name__)

CODE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": (
            "Execute Python code in a secure WebAssembly sandbox. "
            "Full stdlib + numpy/pandas available. Use print() for output. "
            "No file system or network access."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete, runnable Python code. Use print() for visible output.",
                }
            },
            "required": ["code"],
        },
    },
}

# Pyodide initialization is async; cache the runtime globally
_pyodide_runtime: Optional[object] = None
_init_lock = asyncio.Lock()


async def _get_pyodide():
    """Lazily initialize and cache the Pyodide WASM runtime."""
    global _pyodide_runtime
    if _pyodide_runtime is not None:
        return _pyodide_runtime

    async with _init_lock:
        if _pyodide_runtime is not None:
            return _pyodide_runtime

        logger.info(
            "Initializing Pyodide WASM runtime (first run may take a few seconds)..."
        )
        from pyodide import loadPyodide

        _pyodide_runtime = await loadPyodide()

        # Pre-load commonly needed packages for AI agent workloads
        await _pyodide_runtime.loadPackage(["numpy", "pandas"])
        logger.info("Pyodide runtime ready.")
        return _pyodide_runtime


async def _run_code_async(code: str) -> str:
    """Execute code inside the Pyodide WASM sandbox."""
    pyodide = await _get_pyodide()

    # Capture stdout/stderr via JS interop
    capture_setup = """
    const captured = [];
    const originalPrint = pyodide.globals.get('print');
    pyodide.setStdout({
        write: (text) => { captured.push(text); return text.length; }
    });
    pyodide.setStderr({
        write: (text) => { captured.push('[STDERR] ' + text); return text.length; }
    });
    """
    run_js(capture_setup.replace("pyodide", "globalThis.pyodide"))

    try:
        # Execute user code with timeout enforcement at the WASM level
        result = await asyncio.wait_for(
            pyodide.runPythonAsync(code), timeout=Config.CODE_TIMEOUT_SEC
        )

        # Retrieve captured output
        output = pyodide.globals.get("captured")
        if output and len(output) > 0:
            lines = [str(output[i]) for i in range(len(output))]
            return "\n".join(lines).strip()

        # Fallback: use repr of return value if nothing was printed
        if result is not None:
            return str(result)
        return "(Code executed successfully with no output)"

    except asyncio.TimeoutError:
        return f"Error: Code execution timed out after {Config.CODE_TIMEOUT_SEC}s"
    except Exception as e:
        error_msg = str(e)
        # Pyodide wraps JS errors; extract the Python traceback when available
        if hasattr(e, "message"):
            error_msg = e.message
        return f"Execution error: {error_msg[:500]}"
    finally:
        # Reset stdout/stderr to prevent leakage between executions
        reset_io = """
        pyodide.setStdout({write: (t) => t.length});
        pyodide.setStderr({write: (t) => t.length});
        """
        try:
            run_js(reset_io.replace("pyodide", "globalThis.pyodide"))
        except Exception:
            pass  # Best-effort cleanup


def code_interpreter(code: str) -> str:
    """Synchronous entry point compatible with ToolRegistry interface."""
    try:
        loop = asyncio.get_running_loop()
        # Already in an async context → create task and run synchronously
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run_code_async(code))
            return future.result(timeout=Config.CODE_TIMEOUT_SEC + 5)
    except RuntimeError:
        # No running loop → safe to use asyncio.run directly
        return asyncio.run(_run_code_async(code))
