from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from .models import EvidenceBinding, Message, content_digest


class InMemoryConversationSource:
    """Synthetic adapter. Production can resolve the same IDs from an external archive."""

    def __init__(self, messages: Iterable[Message]) -> None:
        self._messages = {message.id: message for message in messages}

    def get(self, message_id: int) -> Message | None:
        return self._messages.get(message_id)

    def replace(self, message: Message) -> None:
        self._messages[message.id] = message

    def remove(self, message_id: int) -> None:
        self._messages.pop(message_id, None)

    def exact(self, quote: str) -> list[Message]:
        return [message for message in self._messages.values() if message.content == quote]

    def search_and(self, keywords: Iterable[str], limit: int = 10) -> list[Message]:
        terms = tuple(term.strip().lower() for term in keywords if term.strip())
        if not terms:
            return []
        matches = [
            message
            for message in self._messages.values()
            if all(term in message.content.lower() for term in terms)
        ]
        return sorted(matches, key=lambda item: item.id, reverse=True)[:limit]


class EvidenceBinder:
    def __init__(self, source: InMemoryConversationSource) -> None:
        self.source = source
        self._bindings: list[EvidenceBinding] = []

    def bind(
        self,
        memory_id: int,
        *,
        message_id: int | None = None,
        exact_quote: str | None = None,
        evidence_kind: str = "history",
    ) -> EvidenceBinding:
        if (message_id is None) == (exact_quote is None):
            raise ValueError("provide exactly one of message_id or exact_quote")
        if evidence_kind not in {"history", "persona"}:
            raise ValueError("unsupported evidence_kind")

        if message_id is not None:
            message = self.source.get(message_id)
            if message is None:
                raise LookupError("message_id not found")
        else:
            matches = self.source.exact(exact_quote or "")
            if not matches:
                raise LookupError("exact quote not found")
            if len(matches) > 1:
                raise ValueError("exact quote is ambiguous; bind an explicit message_id")
            message = matches[0]

        duplicate = next(
            (
                item
                for item in self._bindings
                if item.memory_id == memory_id and item.message_id == message.id
            ),
            None,
        )
        if duplicate:
            return duplicate
        binding = EvidenceBinding(
            memory_id=memory_id,
            message_id=message.id,
            role=message.role,
            message_created_at=message.created_at,
            content_sha256=content_digest(message.content),
            evidence_kind=evidence_kind,  # type: ignore[arg-type]
            selection_order=sum(1 for item in self._bindings if item.memory_id == memory_id),
        )
        self._bindings.append(binding)
        return binding

    def availability(self, memory_id: int) -> dict[str, int | bool]:
        count = sum(1 for item in self._bindings if item.memory_id == memory_id)
        return {"available": count > 0, "messages": count}

    def stored_metadata(self, memory_id: int) -> list[dict[str, object]]:
        """Return binding rows to demonstrate that message bodies are not duplicated."""
        return [asdict(item) for item in self._bindings if item.memory_id == memory_id]

    def read(self, memory_id: int) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for binding in sorted(
            (item for item in self._bindings if item.memory_id == memory_id),
            key=lambda item: (item.selection_order, item.message_id),
        ):
            message = self.source.get(binding.message_id)
            if message is None:
                results.append({"binding": asdict(binding), "integrity": "missing", "content": None})
                continue
            integrity = "ok" if content_digest(message.content) == binding.content_sha256 else "changed"
            results.append(
                {"binding": asdict(binding), "integrity": integrity, "content": message.content}
            )
        return results
