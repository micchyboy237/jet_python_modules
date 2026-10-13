# How to Run

## Prerequisites

Ensure your local llama.cpp servers and Phoenix are running:

```bash
# Verify servers
curl http://localhost:8080/v1/models  # LLM
curl http://localhost:8081/v1/models  # Embedding
curl http://localhost:8082/v1/models  # Reranker
curl http://localhost:6006/api/health # Phoenix
```

### Step 1: Index Your Repository

```bash
cd jet_python_modules/jet/agents/repo-agent-full
python main.py --repo /path/to/your/target/repo --reindex
```

### Step 2: Run Evaluation Harness

```bash
python main.py --repo /path/to/your/target/repo --eval
```

**Expected Output:**

```
 Starting Evaluation Harness...
Evaluating query 1/5: 'What does calculate_sum do?...'
  ✅ PASS Intent=FACTUAL, Score=0.92, Files=['main.py']
Evaluating query 2/5: 'Summarize this repository's purpose'...
  ✅ PASS Intent=SUMMARY, Score=0.85, Files=['README.md']
Evaluating query 3/5: 'List all API endpoints in the codebase'...
  ❌ FAIL Intent=FACTUAL, Score=0.65, Files=[]

======================================================================
📊 EVALUATION REPORT
======================================================================
Total Queries: 5
Passed: 4 (80.0%)
Average Relevance Score: 0.812
----------------------------------------------------------------------
✅ Query: 'What does calculate_sum do?'
   Expected Intent: FACTUAL | Actual: FACTUAL
   Relevance: 0.920 (min: 0.8)
   Sources Found: ['main.py']
...
======================================================================
```

### Step 3: View Traces in Phoenix

Open `http://localhost:6006` → Select project `repo-agent-eval` to see:

- Individual `llm-relevance-judge` spans with scores and reasoning
- Full agent traces for each golden query
- Performance metrics across all evaluation runs

## Key Design Decisions

| Decision                                    | Rationale                                                                                                                         |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **LLM-as-a-Judge over heuristics**          | Semantic relevance can't be measured by keyword matching; local Qwen3.5 provides consistent scoring without cloud dependency      |
| **`@evaluator` decorator**                  | Integrates natively with existing Phoenix telemetry; no separate logging infrastructure needed                                    |
| **Separate `repo-agent-eval` project name** | Keeps evaluation traces isolated from production chat traces for cleaner analysis                                                 |
| **JSONL format for golden dataset**         | Easy to version control, append new test cases, and parse incrementally                                                           |
| **Multi-criteria pass/fail**                | Single relevance score isn't enough; intent classification accuracy and source citation are equally important for agentic systems |

## Extending the Harness

To add new test cases, simply append lines to `golden_dataset.jsonl`:

```jsonl
{
  "query": "Your new test query",
  "intent": "FACTUAL",
  "expected_files": [
    "specific_file.py"
  ],
  "min_relevance_score": 0.8
}
```

To tune thresholds, adjust `min_relevance_score` per query based on observed baseline performance. The evaluation harness makes these thresholds explicit and measurable rather than implicit guesses.
