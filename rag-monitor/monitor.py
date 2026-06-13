import time
import json
import os
import uuid
from datetime import datetime

# Token cost per 1M tokens (Groq LLaMA 3.1 8B pricing approximation)
COST_PER_1M_INPUT_TOKENS = 0.05   # $0.05 per 1M input tokens
COST_PER_1M_OUTPUT_TOKENS = 0.08  # $0.08 per 1M output tokens

LOGS_FILE = "query_logs.json"


def load_logs():
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, "r") as f:
            return json.load(f)
    return []


def save_logs(logs):
    with open(LOGS_FILE, "w") as f:
        json.dump(logs, f, indent=2)


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters"""
    return max(1, len(text) // 4)


def estimate_cost(input_text: str, output_text: str) -> float:
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT_TOKENS
    cost += (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS
    return round(cost, 8)


def log_query(query: str, answer: str, latency_ms: float, sources: list, success: bool):
    logs = load_logs()
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer_preview": answer[:120] + "..." if len(answer) > 120 else answer,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": estimate_tokens(query),
        "output_tokens": estimate_tokens(answer),
        "cost_usd": estimate_cost(query, answer),
        "sources_count": len(sources),
        "success": success,
    }
    logs.append(entry)
    save_logs(logs)
    return entry


def get_metrics():
    logs = load_logs()
    if not logs:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "p50_latency_ms": 0,
            "p95_latency_ms": 0,
            "total_cost_usd": 0,
            "avg_cost_per_query": 0,
            "success_rate": 0,
            "logs": [],
        }

    latencies = sorted([l["latency_ms"] for l in logs])
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[min(int(n * 0.95), n - 1)]
    total_cost = sum(l["cost_usd"] for l in logs)
    successes = sum(1 for l in logs if l["success"])

    return {
        "total_queries": n,
        "avg_latency_ms": round(sum(latencies) / n, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_query": round(total_cost / n, 8),
        "success_rate": round((successes / n) * 100, 1),
        "logs": list(reversed(logs[-50:])),  # last 50, newest first
    }
