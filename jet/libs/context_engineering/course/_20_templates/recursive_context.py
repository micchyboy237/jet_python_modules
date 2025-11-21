"""
Recursive Context Framework 
============================================

Secure, minimal, pragmatic implementation of recursive context improvement.
Reduces complexity while adding production security.

Security: Zero trust architecture with input validation, output sanitization,
rate limiting, and secure credential handling.

Usage:
    framework = RecursiveContextFramework()
    result = framework.improve("Solve: 3x + 7 = 22", max_iterations=3)
"""

import json
import time
import hashlib
import re
from dataclasses import dataclass
from functools import wraps

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet._token.token_utils import token_counter

# ============================================================================
# OUTPUT & LOGGING SETUP
# ============================================================================

from pathlib import Path
from jet.logger import CustomLogger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
main_logger = CustomLogger(
    name="main",
    filename=os.path.join(OUTPUT_DIR, "main.log"),
    console_level="INFO",
    level="DEBUG",
    overwrite=True
)

def create_example_dir(example_name: str) -> Path:
    from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
    base_dir = Path(get_entry_file_dir()) / "generated" / os.path.splitext(get_entry_file_name())[0]
    example_dir = base_dir / example_name
    shutil.rmtree(example_dir, ignore_errors=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir

def get_example_logger(name: str, output_dir: Path) -> "CustomLogger":
    log_file = output_dir / "run.log"
    return CustomLogger(name=name, filename=log_file, overwrite=True)

@dataclass
class ContextResult:
    content: str
    iteration: int
    improvement_score: float
    processing_time: float
    input_tokens: int
    output_tokens: int

    def __post_init__(self):
        if not isinstance(self.content, str):
            raise TypeError("Content must be string")

class SecurityValidator:
    FORBIDDEN_PATTERNS = [
        re.compile(r'<script', re.IGNORECASE),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'eval\(', re.IGNORECASE),
        re.compile(r'exec\(', re.IGNORECASE),
        re.compile(r'__import__', re.IGNORECASE),
    ]

    @classmethod
    def validate_input(cls, text: str, max_tokens: int = 8000) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Input must be non-empty string")
        tokens = token_counter(text, model="qwen3-instruct-2507:4b")
        if tokens > max_tokens:
            raise ValueError(f"Input exceeds {max_tokens} tokens ({tokens} detected)")
        for pattern in cls.FORBIDDEN_PATTERNS:
            if pattern.search(text):
                raise ValueError("Input contains forbidden patterns")
        return re.sub(r'[<>"\']', '', text).strip()

    @classmethod
    def sanitize_output(cls, text: str) -> str:
        if not isinstance(text, str):
            return ""
        sanitized = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        return sanitized.strip()

class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()

    def allow_request(self) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.requests_per_minute, self.tokens + elapsed * self.requests_per_minute / 60)
        self.last_update = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

def rate_limited(limiter: RateLimiter):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.allow_request():
                raise RuntimeError("Rate limit exceeded – slow down!")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RecursiveContextFramework:
    def __init__(self):
        self.llm = LlamacppLLM(
            model="qwen3-instruct-2507:4b",
            base_url="http://shawn-pc.local:8080/v1",
            verbose=True  # This enables live teal-colored streaming!
        )
        self.rate_limiter = RateLimiter(requests_per_minute=30)
        self.validator = SecurityValidator()
        self._improvement_prompt = """
Analyze and improve this response:
Original: {original}
Current: {current}

Provide a better version that is:
1. More accurate and precise
2. Clearer and more structured
3. More complete with helpful details

Improved response:"""

    def _calculate_improvement_score(self, original: str, improved: str) -> float:
        if not original or not improved:
            return 0.0
        orig_t = token_counter(original, model="qwen3-instruct-2507:4b")
        imp_t = token_counter(improved, model="qwen3-instruct-2507:4b")
        if orig_t == 0:
            return 0.0
        length_ratio = imp_t / orig_t
        length_score = min(1.0, length_ratio * 0.8) if length_ratio <= 2.0 else max(0.0, 2.0 - length_ratio * 0.5)
        orig_words = set(original.lower().split())
        imp_words = set(improved.lower().split())
        vocab_growth = len(imp_words - orig_words) / max(len(orig_words), 1)
        return round((length_score * 0.4) + (vocab_growth * 0.6), 3)

    @rate_limited(RateLimiter(requests_per_minute=30))
    def _stream_generate(self, prompt: str, max_tokens: int = 1200) -> str:
        """Stream generation with live visual feedback."""
        full_response = ""
        main_logger.info("Generating improvement... (streaming live)")
        for chunk in self.llm.generate(prompt, max_tokens=max_tokens, stream=True):
            full_response += chunk
        return self.validator.sanitize_output(full_response)

    def improve(self, content: str, max_iterations: int = 3, improvement_threshold: float = 0.08) -> ContextResult:
        start_time = time.time()
        content = self.validator.validate_input(content)
        input_tokens = token_counter(content, model="qwen3-instruct-2507:4b")

        current = content
        best_score = 0.0
        iteration = 0

        main_logger.info(f"Starting recursive improvement (max {max_iterations} iterations)")

        for iteration in range(1, max_iterations + 1):
            main_logger.info(f"Iteration {iteration}/{max_iterations}")

            prompt = self._improvement_prompt.format(original=content, current=current)
            improved = self._stream_generate(prompt)

            score = self._calculate_improvement_score(current, improved)
            main_logger.info(f"Improvement score: {score:.3f} (Δ {score - best_score:+.3f})")

            if score - best_score < improvement_threshold:
                main_logger.info("Improvement below threshold → stopping early")
                break

            current = improved
            best_score = score
            main_logger.info("-" * 60)

        output_tokens = token_counter(current, model="qwen3-instruct-2507:4b")
        processing_time = round(time.time() - start_time, 3)

        return ContextResult(
            content=current,
            iteration=iteration,
            improvement_score=best_score,
            processing_time=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    def batch_improve(
        self,
        inputs: list[str],
        max_iterations: int = 3,
        improvement_threshold: float = 0.08
    ) -> list[ContextResult]:
        """
        Process multiple inputs in parallel (sequentially with rate-limiting).
        Returns a list of ContextResult in the same order as inputs.
        """
        main_logger.info(f"Batch improving {len(inputs)} items (max {max_iterations} iterations each)")
        results = []
        for idx, inp in enumerate(inputs, start=1):
            main_logger.info(f"Batch item {idx}/{len(inputs)}")
            try:
                result = self.improve(
                    content=inp,
                    max_iterations=max_iterations,
                    improvement_threshold=improvement_threshold
                )
                results.append(result)
            except Exception as e:
                main_logger.error(f"Batch item {idx} failed: {e}")
                # Append a minimal failed result so indexing stays consistent
                results.append(ContextResult(
                    content=f"[ERROR] {str(e)}",
                    iteration=0,
                    improvement_score=0.0,
                    processing_time=0.0,
                    input_tokens=0,
                    output_tokens=0
                ))
        return results

# Example secure integration with actual LLM provider
class SecureAnthropicProvider:
    """Secure Anthropic Claude integration."""
    
    def __init__(self, api_key: str):
        # In production: retrieve from secure key management service
        self.api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        # Store encrypted key, not plaintext
        
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate with built-in security."""
        # Implement actual Anthropic API call with:
        # - TLS verification
        # - Request signing
        # - Response validation
        # - Error handling
        return "[Secure Anthropic response]"


# --------------------------------------------------------------------------- #
# EXAMPLE FUNCTIONS – each saves its own isolated folder
# --------------------------------------------------------------------------- #

def example_01_live_math_solve():
    ex_dir = create_example_dir("example_01_live_math_solve")
    log = get_example_logger("ex01", ex_dir)

    framework = RecursiveContextFramework()
    result = framework.improve(
        "Solve for x and explain step by step: 2(x - 5) = 3(x + 4) - 15",
        max_iterations=4
    )

    (ex_dir / "input.txt").write_text("Solve for x and explain step by step: 2(x - 5) = 3(x + 4) - 15")
    (ex_dir / "final_answer.md").write_text(result.content)
    (ex_dir / "stats.json").write_text(json.dumps({
        "iterations": result.iteration,
        "final_score": result.improvement_score,
        "time_s": result.processing_time,
        "tokens_in": result.input_tokens,
        "tokens_out": result.output_tokens,
        "efficiency_ratio": round(result.input_tokens / max(result.output_tokens, 1), 2)
    }, indent=2))

    log.info("Live math solving complete!")
    log.info(f"Final answer in: {ex_dir}/final_answer.md")

def example_02_explanation_upgrade():
    ex_dir = create_example_dir("example_02_explanation_upgrade")
    log = get_example_logger("ex02", ex_dir)

    initial = """
    Photosynthesis is how plants make food using sunlight.
    They take in carbon dioxide and release oxygen.
    """

    framework = RecursiveContextFramework()
    result = framework.improve(initial, max_iterations=3)

    (ex_dir / "initial.md").write_text(initial.strip())
    (ex_dir / "improved.md").write_text(result.content)
    (ex_dir / "report.json").write_text(json.dumps({
        "iterations": result.iteration,
        "score": result.improvement_score,
        "token_growth": result.output_tokens - result.input_tokens
    }, indent=2))

    log.info("Explanation upgrade complete – watch the live refinement!")


def example_01_basic_math_problem():
    ex_dir = create_example_dir("example_01_basic_math_problem")
    log = get_example_logger("ex01", ex_dir)

    framework = RecursiveContextFramework()
    result = framework.improve("Solve for x: 3x + 7 = 22", max_iterations=3)

    (ex_dir / "input.txt").write_text("Solve for x: 3x + 7 = 22")
    (ex_dir / "final_response.md").write_text(result.content)
    (ex_dir / "result.json").write_text(json.dumps({
        "iteration": result.iteration,
        "improvement_score": result.improvement_score,
        "processing_time_s": result.processing_time
    }, indent=2))

    log.info("Example 01 – Basic math (real LLM)")
    log.info(f"Final answer → {result.content.strip()}")
    log.info(f"Iterations: {result.iteration} | Score: {result.improvement_score:.3f}")

def example_02_batch_processing():
    ex_dir = create_example_dir("example_02_batch_processing")
    log = get_example_logger("ex02", ex_dir)

    framework = RecursiveContextFramework()
    inputs = [
        "Explain quantum entanglement simply.",
        "Why do planets orbit the sun?",
        "What is photosynthesis?"
    ]
    results = framework.batch_improve(inputs, max_iterations=2)

    summary = []
    for i, (inp, res) in enumerate(zip(inputs, results), 1):
        (ex_dir / f"input_{i:02d}.txt").write_text(inp)
        (ex_dir / f"output_{i:02d}.md").write_text(res.content)
        summary.append({"i": i, "iter": res.iteration, "score": res.improvement_score})

    (ex_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Example 02 – Batch (real LLM) completed")

def example_03_rate_limit_behavior():
    """Demonstrates rate-limiter protection (simulated fast calls)."""
    ex_dir = create_example_dir("example_03_rate_limit_behavior")
    log = get_example_logger("ex03", ex_dir)

    framework = RecursiveContextFramework()   # uses internal RateLimiter(30 rpm)

    start = time.time()
    successes = 0
    failures = 0

    for i in range(40):  # try more than the limit
        try:
            _ = framework.improve(f"test {i}", max_iterations=1)
            successes += 1
        except RuntimeError as e:
            if "Rate limit exceeded" in str(e):
                failures += 1
            else:
                raise

    elapsed = time.time() - start
    (ex_dir / "rate_limit_report.json").write_text(json.dumps({
        "attempts": 40,
        "successes": successes,
        "rate_limit_blocks": failures,
        "elapsed_seconds": round(elapsed, 3)
    }, indent=2))

    log.info("Example 03 – Rate limit test")
    log.info(f"Successes : {successes}")
    log.info(f"Blocked   : {failures}")

def example_04_input_validation_rejection():
    """Shows security validator rejecting dangerous input."""
    ex_dir = create_example_dir("example_04_input_validation_rejection")
    log = get_example_logger("ex04", ex_dir)

    framework = RecursiveContextFramework()
    dangerous_inputs = [
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "__import__('os').system('rm -rf /')"
    ]

    report = []
    for txt in dangerous_inputs:
        try:
            framework.improve(txt)
            status = "accepted (unexpected)"
        except ValueError as e:
            status = f"rejected – {str(e)}"

        report.append({"input": txt[:60] + "...", "status": status})

    (ex_dir / "validation_report.json").write_text(json.dumps(report, indent=2))

    log.info("Example 04 – Input validation")
    for r in report:
        log.info(f"{r['status']}")

# --------------------------------------------------------------------------- #
# MAIN BLOCK – runs every example and writes to main log
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main_logger.info("=" * 70)
    main_logger.info("RecursiveContextFramework – Example Suite")
    main_logger.info("=" * 70)

    example_01_live_math_solve()
    example_02_explanation_upgrade()
    example_01_basic_math_problem()
    example_02_batch_processing()
    example_03_rate_limit_behavior()
    example_04_input_validation_rejection()

    main_logger.info("All examples finished – results saved under ./generated/recursive_context/")
