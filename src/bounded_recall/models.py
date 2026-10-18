from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal


def content_digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Memory:
    id: int
    title: str
    body: str
    summary: str = ""
    summary_body_sha256: str = ""
    keywords: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    vector: tuple[float, ...] = ()

    @property
    def has_fresh_summary(self) -> bool:
        return bool(self.summary) and self.summary_body_sha256 == content_digest(self.body)

    def compact_text(self, fallback_chars: int = 240) -> tuple[str, str]:
        if self.has_fresh_summary:
            return self.summary, "summary"
        compact = " ".join(self.body.split())
        return compact[:fallback_chars], "body_fallback"


@dataclass(frozen=True)
class Message:
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: str


@dataclass(frozen=True)
class EvidenceBinding:
    memory_id: int
    message_id: int
    role: str
    message_created_at: str
    content_sha256: str
    evidence_kind: Literal["history", "persona"] = "history"
    selection_order: int = 0


@dataclass(frozen=True)
class RecallConfig:
    max_hits: int = 3
    max_chars: int = 1200
    per_hit_chars: int = 320
    min_vector_score: float = 0.58
    assisted_vector_score: float = 0.44
    min_literal_chars: int = 2


@dataclass(frozen=True)
class RecallHit:
    memory_id: int
    title: str
    text: str
    score: float
    evidence: tuple[str, ...]
    summary_source: str
    vector_score: float = 0.0
    literal_chars: int = 0


@dataclass(frozen=True)
class RecallResult:
    query: str
    normalized_query: str
    skipped_reason: str | None
    hits: tuple[RecallHit, ...] = ()
    injection: str = ""
    injection_chars: int = 0
    trace: dict[str, object] = field(default_factory=dict)
