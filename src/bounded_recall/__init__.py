"""Bounded hybrid recall with verifiable source evidence."""

from .engine import HashingEmbedder, HybridRecallEngine
from .evidence import EvidenceBinder, InMemoryConversationSource
from .models import (
    EvidenceBinding,
    Memory,
    Message,
    RecallConfig,
    RecallHit,
    RecallResult,
)

__all__ = [
    "EvidenceBinder",
    "EvidenceBinding",
    "HashingEmbedder",
    "HybridRecallEngine",
    "InMemoryConversationSource",
    "Memory",
    "Message",
    "RecallConfig",
    "RecallHit",
    "RecallResult",
]
