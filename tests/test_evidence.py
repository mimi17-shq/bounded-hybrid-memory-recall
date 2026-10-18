from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bounded_recall import EvidenceBinder, InMemoryConversationSource, Message  # noqa: E402


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = InMemoryConversationSource(
            [
                Message(10, "user", "我希望每轮召回都稳定。", "2026-01-01T10:00:00+08:00"),
                Message(11, "assistant", "会限制数量和字符预算。", "2026-01-01T10:00:03+08:00"),
                Message(12, "user", "好的", "2026-01-01T10:00:05+08:00"),
                Message(13, "user", "好的", "2026-01-02T10:00:05+08:00"),
            ]
        )
        self.binder = EvidenceBinder(self.source)

    def test_unique_exact_quote_can_bind(self) -> None:
        binding = self.binder.bind(1, exact_quote="我希望每轮召回都稳定。")
        self.assertEqual(binding.message_id, 10)
        self.assertEqual(self.binder.availability(1), {"available": True, "messages": 1})

    def test_ambiguous_quote_requires_explicit_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.binder.bind(1, exact_quote="好的")
        binding = self.binder.bind(1, message_id=12)
        self.assertEqual(binding.message_id, 12)

    def test_binding_metadata_does_not_duplicate_message_body(self) -> None:
        self.binder.bind(2, message_id=10)
        stored = self.binder.stored_metadata(2)[0]
        self.assertNotIn("content", stored)
        self.assertIn("content_sha256", stored)

    def test_on_demand_read_verifies_changed_and_missing_sources(self) -> None:
        self.binder.bind(3, message_id=10)
        self.assertEqual(self.binder.read(3)[0]["integrity"], "ok")
        self.source.replace(
            Message(10, "user", "内容已变化。", "2026-01-01T10:00:00+08:00")
        )
        self.assertEqual(self.binder.read(3)[0]["integrity"], "changed")
        self.source.remove(10)
        result = self.binder.read(3)[0]
        self.assertEqual(result["integrity"], "missing")
        self.assertIsNone(result["content"])

    def test_keyword_search_requires_all_terms_and_never_binds(self) -> None:
        hits = self.source.search_and(["召回", "稳定"])
        self.assertEqual([item.id for item in hits], [10])
        self.assertEqual(self.binder.availability(1), {"available": False, "messages": 0})


if __name__ == "__main__":
    unittest.main()
