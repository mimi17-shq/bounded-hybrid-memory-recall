from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bounded_recall import HybridRecallEngine, Memory  # noqa: E402
from bounded_recall.engine import cosine_similarity  # noqa: E402
from bounded_recall.text import literal_terms, longest_literal_match  # noqa: E402


class FixtureEmbedder:
    vectors = {
        "和教练吃饭": (1.0, 0.0, 0.0),
        "最近睡得不踏实": (0.0, 0.0, 1.0),
        "奶茶": (0.0, 1.0, 0.0),
        "量子焊接机": (0.0, 0.0, 0.0),
    }

    def __call__(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]


MEMORIES = (
    Memory(1, "奶茶偏好", "夜间选择低糖。", aliases=("奶茶",), vector=(1.0, 0.0, 0.0)),
    Memory(2, "周末训练安排", "时间变化时提前确认。", aliases=("教练",), vector=(0.0, 1.0, 0.0)),
    Memory(3, "夜间作息", "减少夜间咖啡因。", keywords=("睡眠",), vector=(0.0, 0.0, 1.0)),
)

CASES = (
    ("和教练吃饭", 2),
    ("最近睡得不踏实", 3),
    ("奶茶", 1),
    ("量子焊接机", None),
)


def literal_baseline(query: str) -> int | None:
    terms = literal_terms(query)
    scored = []
    for memory in MEMORIES:
        searchable = " ".join((memory.title, *memory.keywords, *memory.aliases))
        size, _ = longest_literal_match(terms, searchable)
        if size >= 2:
            scored.append((size, -memory.id, memory.id))
    return max(scored)[2] if scored else None


def nearest_neighbor_top1_no_threshold(query: str) -> int | None:
    vector = FixtureEmbedder()(query)
    scored = [(cosine_similarity(vector, memory.vector), -memory.id, memory.id) for memory in MEMORIES]
    return max(scored)[2] if scored else None


def main() -> None:
    engine = HybridRecallEngine(MEMORIES, embedder=FixtureEmbedder())
    rows = []
    methods = {
        "literal_only": literal_baseline,
        "nearest_neighbor_top1_no_threshold": nearest_neighbor_top1_no_threshold,
    }
    correct = {name: 0 for name in (*methods, "hybrid")}
    total_chars = 0
    for query, expected in CASES:
        result = engine.recall(query)
        hybrid = result.hits[0].memory_id if result.hits else None
        predictions = {name: method(query) for name, method in methods.items()}
        predictions["hybrid"] = hybrid
        for name, prediction in predictions.items():
            correct[name] += int(prediction == expected)
        total_chars += result.injection_chars
        rows.append({"query": query, "expected": expected, **predictions})
    output = {
        "label": "synthetic contract evaluation; not a retrieval benchmark",
        "cases": len(CASES),
        "accuracy": {name: correct[name] / len(CASES) for name in correct},
        "hybrid_average_injection_chars": total_chars / len(CASES),
        "rows": rows,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
