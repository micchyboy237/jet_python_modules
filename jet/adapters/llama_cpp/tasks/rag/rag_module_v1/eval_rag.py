# rag_module_v1/eval_rag.py

import argparse
import json
import time

from .search_knowledge import search_knowledge


def run_eval(dataset_path: str):
    results = []
    start = time.time()

    with open(dataset_path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    for ex in examples:
        pred = search_knowledge(ex["query"])

        pred_chunk_ids = [
            s["chunk_id"]
            for s in pred.get("sources", [])
            if isinstance(s, dict) and "chunk_id" in s
        ]

        expected = set(ex["expected_chunk_ids"])
        predicted = set(pred_chunk_ids[:5])

        if ex["should_abstain"]:
            recall_at_5 = 1.0 if pred["status"] == "abstained" else 0.0
        else:
            recall_at_5 = len(predicted & expected) / max(len(expected), 1)

        abstained = pred["status"] == "abstained"

        result = {
            "id": ex["id"],
            "query": ex["query"],
            "status": pred.get("status"),
            "expected_chunk_ids": ex["expected_chunk_ids"],
            "predicted_chunk_ids": pred_chunk_ids,
            "recall_at_5": recall_at_5,
            "should_abstain": ex["should_abstain"],
            "abstained": abstained,
            "abstention_correct": abstained == ex["should_abstain"],
            "parse_ok": isinstance(pred, dict)
            and "status" in pred
            and "sources" in pred
            and "answer_context" in pred,
            "latency_ms": pred.get("_latency_ms", 0),
        }

        results.append(result)

    elapsed = time.time() - start
    n = len(results)

    non_abstain_examples = [r for r in results if not r["should_abstain"]]
    abstain_examples = [r for r in results if r["should_abstain"]]

    false_positive_searches = [
        r for r in abstain_examples if r["status"] != "abstained"
    ]
    false_abstentions = [r for r in non_abstain_examples if r["status"] == "abstained"]

    latencies = sorted(r["latency_ms"] for r in results)
    p95_idx = min(int(0.95 * len(latencies)), len(latencies) - 1)

    metrics = {
        "n": n,
        "recall_at_5": sum(r["recall_at_5"] for r in non_abstain_examples)
        / max(len(non_abstain_examples), 1),
        "abstention_accuracy": sum(r["abstention_correct"] for r in results) / n,
        "negative_abstention_recall": sum(
            1 for r in abstain_examples if r["status"] == "abstained"
        )
        / max(len(abstain_examples), 1),
        "false_abstention_rate": len(false_abstentions)
        / max(len(non_abstain_examples), 1),
        "false_positive_search_rate": len(false_positive_searches)
        / max(len(abstain_examples), 1),
        "parse_rate": sum(r["parse_ok"] for r in results) / n,
        "p95_latency_ms": latencies[p95_idx],
        "total_time_s": round(elapsed, 2),
    }

    print(json.dumps(metrics, indent=2))

    failures = [
        r
        for r in results
        if r["recall_at_5"] < 0.8 or not r["abstention_correct"] or not r["parse_ok"]
    ]

    if failures:
        print(f"\n❌ {len(failures)} FAILURES:")
        for f in failures[:10]:
            print(
                f"  {f['id']}: status={f['status']} "
                f"recall={f['recall_at_5']:.2f} "
                f"abstain_ok={f['abstention_correct']} "
                f"expected={f['expected_chunk_ids']} "
                f"got={f['predicted_chunk_ids'][:5]}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run_eval(args.dataset)
