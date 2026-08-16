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

_pyodide_runtime: Optional[object] = None
_init_lock = asyncio.Lock()


async def _get_pyodide():
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
        await _pyodide_runtime.loadPackage(["numpy", "pandas"])
        logger.info("Pyodide runtime ready.")
        return _pyodide_runtime


async def _run_code_async(code: str) -> str:
    pyodide = await _get_pyodide()

    capture_setup = """
    const captured = [];
    pyodide.setStdout({ write: (text) => { captured.push(text); return text.length; } });
    pyodide.setStderr({ write: (text) => { captured.push('[STDERR] ' + text); return text.length; } });
    """
    run_js(capture_setup.replace("pyodide", "globalThis.pyodide"))

    try:
        result = await asyncio.wait_for(
            pyodide.runPythonAsync(code), timeout=Config.CODE_TIMEOUT_SEC
        )
        output = pyodide.globals.get("captured")
        if output and len(output) > 0:
            lines = [str(output[i]) for i in range(len(output))]
            return "\n".join(lines).strip()
        if result is not None:
            return str(result)
        return "(Code executed successfully with no output)"
    except asyncio.TimeoutError:
        return f"Error: Code execution timed out after {Config.CODE_TIMEOUT_SEC}s"
    except Exception as e:
        error_msg = getattr(e, "message", str(e))
        return f"Execution error: {error_msg[:500]}"
    finally:
        reset_io = """
        pyodide.setStdout({write: (t) => t.length});
        pyodide.setStderr({write: (t) => t.length});
        """
        try:
            run_js(reset_io.replace("pyodide", "globalThis.pyodide"))
        except Exception:
            pass


def code_interpreter(code: str) -> str:
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _run_code_async(code))
            return future.result(timeout=Config.CODE_TIMEOUT_SEC + 5)
    except RuntimeError:
        return asyncio.run(_run_code_async(code))
