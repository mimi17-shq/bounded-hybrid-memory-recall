from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from hashlib import blake2b

from .models import Memory, RecallConfig, RecallHit, RecallResult
from .text import is_low_information, literal_terms, longest_literal_match, normalize_query


Vector = Sequence[float]
Embedder = Callable[[str], Vector]


def cosine_similarity(left: Vector, right: Vector) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class HashingEmbedder:
    """Dependency-free demo embedder. Replace with a semantic model in production."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def __call__(self, text: str) -> tuple[float, ...]:
        compact = "".join(normalize_query(text).lower().split())
        vector = [0.0] * self.dimensions
        if not compact:
            return tuple(vector)
        grams = [compact[index : index + 2] for index in range(max(1, len(compact) - 1))]
        for gram in grams:
            digest = blake2b(gram.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return tuple(value / norm for value in vector)


class HybridRecallEngine:
    def __init__(
        self,
        memories: Iterable[Memory],
        *,
        embedder: Embedder | None = None,
        config: RecallConfig | None = None,
    ) -> None:
        self.memories = tuple(memories)
        self.embedder = embedder
        self.config = config or RecallConfig()

    def recall(self, raw_query: str) -> RecallResult:
        query = normalize_query(raw_query)
        if is_low_information(query):
            return RecallResult(
                query=raw_query,
                normalized_query=query,
                skipped_reason="low_information",
                trace={"vector_called": False, "candidate_count": 0},
            )

        terms = literal_terms(query)
        query_vector: Vector = ()
        vector_called = False
        if self.embedder is not None:
            try:
                query_vector = tuple(float(value) for value in self.embedder(query))
                vector_called = True
            except Exception:
                query_vector = ()

        candidates: list[RecallHit] = []
        for memory in self.memories:
            title_space = " ".join((memory.title, *memory.keywords))
            literal_chars, literal_term = longest_literal_match(terms, title_space)
            alias_matches = [alias for alias in memory.aliases if len(alias) >= 2 and alias.lower() in query.lower()]
            entity_match = max(alias_matches, key=len, default="")
            if entity_match:
                literal_chars = max(literal_chars, len(entity_match))

            vector_score = cosine_similarity(query_vector, memory.vector) if query_vector else 0.0
            strong_literal = literal_chars >= self.config.min_literal_chars
            strong_vector = vector_score >= self.config.min_vector_score
            assisted_vector = strong_literal and vector_score >= self.config.assisted_vector_score
            if not (strong_literal or strong_vector or assisted_vector):
                continue

            evidence: list[str] = []
            if entity_match:
                evidence.append("entity_alias")
            elif literal_term:
                evidence.append("literal")
            if strong_vector or assisted_vector:
                evidence.append("vector")
            if len(evidence) > 1:
                evidence.append("cross_channel_agreement")

            compact, compact_source = memory.compact_text(self.config.per_hit_chars)
            compact = compact[: self.config.per_hit_chars]
            literal_score = min(1.0, literal_chars / 6.0)
            score = (0.58 * vector_score) + (0.42 * literal_score)
            if entity_match:
                # An explicit alias is deterministic evidence. A semantically
                # similar but unrelated vector hit must not outrank it.
                score = max(score, 0.78)
            elif strong_literal:
                score = max(score, 0.62)
            if len(evidence) > 1:
                score += 0.08
            candidates.append(
                RecallHit(
                    memory_id=memory.id,
                    title=memory.title,
                    text=compact,
                    score=round(score, 6),
                    evidence=tuple(evidence),
                    summary_source=compact_source,
                    vector_score=round(vector_score, 6),
                    literal_chars=literal_chars,
                )
            )

        ordered = sorted(candidates, key=lambda hit: (-hit.score, hit.memory_id))
        selected, injection = self._apply_budget(ordered)
        return RecallResult(
            query=raw_query,
            normalized_query=query,
            skipped_reason=None,
            hits=tuple(selected),
            injection=injection,
            injection_chars=len(injection),
            trace={
                "vector_called": vector_called,
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "max_hits": self.config.max_hits,
                "max_chars": self.config.max_chars,
            },
        )

    def _apply_budget(self, ordered: list[RecallHit]) -> tuple[list[RecallHit], str]:
        selected: list[RecallHit] = []
        blocks: list[str] = []
        used = 0
        for hit in ordered:
            if len(selected) >= self.config.max_hits:
                break
            prefix = f"#{hit.memory_id} {hit.title}\n"
            separator = "\n\n" if blocks else ""
            available = self.config.max_chars - used - len(separator) - len(prefix)
            if available <= 0:
                break
            text = hit.text[:available]
            if not text:
                break
            block = prefix + text
            selected.append(
                RecallHit(
                    memory_id=hit.memory_id,
                    title=hit.title,
                    text=text,
                    score=hit.score,
                    evidence=hit.evidence,
                    summary_source=hit.summary_source,
                    vector_score=hit.vector_score,
                    literal_chars=hit.literal_chars,
                )
            )
            blocks.append(block)
            used += len(separator) + len(block)
        return selected, "\n\n".join(blocks)
