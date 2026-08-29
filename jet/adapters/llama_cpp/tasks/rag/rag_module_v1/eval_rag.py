# eval_rag.py — Run with: python eval_rag.py --dataset eval_v1.jsonl
import argparse
import json
import time

from your_rag_module import search_knowledge  # Your isolated RAG function


def run_eval(dataset_path):
    results = []
    start = time.time()

    with open(dataset_path) as f:
        examples = [json.loads(line) for line in f]

    for ex in examples:
        pred = search_knowledge(ex["query"])

        result = {
            "id": ex["id"],
            "recall_at_5": len(
                set(pred.get("sources", [])) & set(ex["expected_chunk_ids"])
            )
            / max(len(ex["expected_chunk_ids"]), 1),
            "abstention_correct": (pred["status"] == "abstained")
            == ex["should_abstain"],
            "parse_ok": isinstance(pred, dict) and "status" in pred,
            "latency_ms": pred.get(
                "_latency_ms", 0
            ),  # Instrument your RAG func to emit this
        }
        results.append(result)

    elapsed = time.time() - start

    # Aggregate metrics
    n = len(results)
    metrics = {
        "recall_at_5": sum(r["recall_at_5"] for r in results) / n,
        "abstention_precision": sum(
            1
            for r in results
            if r["abstention_correct"]
            and not any(e["should_abstain"] for e in examples if e["id"] == r["id"])
        )
        / max(
            sum(
                1
                for r in results
                if not any(e["should_abstain"] for e in examples if e["id"] == r["id"])
            ),
            1,
        ),
        "abstention_recall": sum(
            1
            for r in results
            if r["abstention_correct"]
            and any(e["should_abstain"] for e in examples if e["id"] == r["id"])
        )
        / max(sum(1 for e in examples if e["should_abstain"]), 1),
        "parse_rate": sum(r["parse_ok"] for r in results) / n,
        "p95_latency_ms": sorted(r["latency_ms"] for r in results)[int(0.95 * n)],
        "total_time_s": round(elapsed, 2),
    }

    print(json.dumps(metrics, indent=2))

    # Print failures for immediate inspection
    failures = [
        r for r in results if r["recall_at_5"] < 0.8 or not r["abstention_correct"]
    ]
    if failures:
        print(f"\n❌ {len(failures)} FAILURES:")
        for f in failures[:5]:  # Show top 5
            print(
                f"  {f['id']}: recall={f['recall_at_5']:.2f}, abstain_ok={f['abstention_correct']}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run_eval(args.dataset)
