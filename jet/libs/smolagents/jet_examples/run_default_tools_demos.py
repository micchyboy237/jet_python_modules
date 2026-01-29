# demo_python_interpreter.py
from smolagents import PythonInterpreterTool


def demo_python_interpreter():
    tool = PythonInterpreterTool(
        authorized_imports=["math", "random", "datetime"], timeout_seconds=15
    )

    # Simple calculation
    result = tool("x = 42 * 13 + 7; print(x)")
    print("Result 1:", result)

    # Multi-line with import
    code = """
import math
print(math.sqrt(169))
print(math.pi)
    """.strip()
    result = tool(code)
    print("Result 2:", result)

    # Using state preservation (multiple calls)
    tool("items = ['apple', 'banana', 'cherry']")
    result = tool("print(sorted(items))")
    print("Result 3 (using previous state):", result)


# demo_final_answer.py
from smolagents import FinalAnswerTool


def demo_final_answer():
    tool = FinalAnswerTool()

    # These are the kinds of calls agents usually make
    print(tool(42))
    print(tool("The capital of Japan is Tokyo"))
    print(tool({"answer": "yes", "confidence": 0.92}))
    print(tool(["Paris", "London", "Berlin"]))


# demo_user_input.py
from smolagents import UserInputTool


def get_input_with_default(
    question: str,
    hint: str | None = None,
    default: str | None = None,
) -> str:
    """
    Helper to ask question with optional hint and default value when empty input.

    Returns stripped user input or default when empty.
    """
    prompt_parts = [question]

    if hint:
        prompt_parts.append(f"  [hint: {hint}]")

    if default is not None:
        prompt_parts.append(f"  (press Enter to use '{default}')")
    elif hint:
        prompt_parts.append("  (press Enter to skip)")

    full_prompt = "".join(prompt_parts) + " → "

    user_input = input(full_prompt).strip()

    # If user gave nothing (empty after strip) → use default or empty
    if not user_input:
        return default if default is not None else ""

    return user_input


def demo_user_input():
    tool = UserInputTool()

    print("\n=== Improved interactive input examples ===\n")

    # Example 1: with hint + default
    answer1 = get_input_with_default(
        question="What is your favorite color?",
        hint="e.g. blue, forest green, #FF69B4",
        default="not specified",
    )
    print(f"You answered: {answer1!r}\n")

    # Example 2: required feeling, no default
    answer2 = get_input_with_default(
        question="How was your day in one word?",
        hint="great / okay / rough / chaotic",
        # no default → empty is allowed
    )
    print(f"You answered: {answer2!r}\n")

    # Example 3: using the original tool directly (for comparison)
    print("Original tool style (no default/hint):")
    answer3 = tool("Quick: cats or dogs?")
    print(f"You answered: {answer3!r}\n")


# demo_duckduckgo_search.py
from smolagents import DuckDuckGoSearchTool


def demo_duckduckgo():
    tool = DuckDuckGoSearchTool(max_results=6, rate_limit=1.5)

    queries = [
        "best lightweight python web framework 2025",
        "current version of PyTorch",
        "smolagents huggingface",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        print(tool(q))
        print()


# demo_google_search_serpapi.py
# (requires SERPAPI_API_KEY environment variable)
from smolagents import GoogleSearchTool


def demo_google_serpapi():
    try:
        tool = GoogleSearchTool(provider="serpapi")
        result = tool("Python 3.13 new features", filter_year=2024)
        print(result)
    except ValueError as e:
        print("Google SerpApi demo skipped — missing/invalid API key:", e)


# demo_api_web_search_brave.py
# (requires BRAVE_API_KEY environment variable)
from smolagents import ApiWebSearchTool


def demo_brave_search():
    try:
        tool = ApiWebSearchTool(
            rate_limit=2.0,
            # endpoint and headers already default to Brave
        )
        print(tool("AI agents open source frameworks comparison 2025"))
    except Exception as e:
        print("Brave Search demo skipped:", str(e))


# demo_visit_webpage.py
from smolagents import VisitWebpageTool


def demo_visit_webpage():
    tool = VisitWebpageTool(max_output_length=12000)

    urls = [
        "https://huggingface.co/docs/hub/index",
        "https://peps.python.org/pep-0745/",
        "https://pytorch.org/blog/pytorch-2.6/",
    ]

    for url in urls:
        print(f"\n=== {url} ===\n")
        content = tool(url)
        print(content[:400], "..." if len(content) > 400 else "")
        print()


# demo_wikipedia_search.py
from smolagents import WikipediaSearchTool


def demo_wikipedia():
    tool = WikipediaSearchTool(
        user_agent="DemoBot (demo@example.com)",
        language="en",
        content_type="summary",  # or "text"
        extract_format="WIKI",
    )

    topics = [
        "Large language model",
        "Retrieval-augmented generation",
        "Mixture of Experts",
    ]

    for topic in topics:
        print(f"\nWikipedia → {topic}")
        print("-" * 50)
        print(tool(topic))
        print()


# demo_speech_to_text.py
from smolagents import SpeechToTextTool


def demo_transcriber():
    try:
        tool = SpeechToTextTool()

        # Examples of inputs the tool accepts:
        #   - local path
        #   - http/https url
        #   - torch tensor / numpy array (raw audio)

        result = tool(
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
        )
        print("Transcription:", result)

        # Local file example (uncomment if you have the file)
        # result = tool("./my_recording.mp3")
        # print(result)

    except ImportError:
        print("Transformers + torch not installed → skipping transcriber demo")
    except Exception as e:
        print("Transcriber demo failed:", str(e))


if __name__ == "__main__":
    print("═══════════════════════════════════════════════")
    print("   Smolagents default tools — demo usage")
    print("═══════════════════════════════════════════════\n")

    demos = [
        ("PythonInterpreterTool", demo_python_interpreter),
        ("FinalAnswerTool", demo_final_answer),
        ("UserInputTool", demo_user_input),
        ("DuckDuckGoSearchTool", demo_duckduckgo),
        # ("GoogleSearchTool (SerpApi)", demo_google_serpapi),        # needs API key
        # ("ApiWebSearchTool (Brave)", demo_brave_search),           # needs API key
        ("VisitWebpageTool", demo_visit_webpage),
        ("WikipediaSearchTool", demo_wikipedia),
        ("SpeechToTextTool (Whisper)", demo_transcriber),
    ]

    for name, func in demos:
        print(f"\n► {name}")
        print("─" * (len(name) + 4))
        try:
            func()
        except Exception as e:
            print(f"  → Demo failed: {type(e).__name__}: {e}")
        print()
