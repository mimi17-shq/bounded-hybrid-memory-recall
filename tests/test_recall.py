from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bounded_recall import HybridRecallEngine, Memory, RecallConfig  # noqa: E402
from bounded_recall.models import content_digest  # noqa: E402


class ScriptedEmbedder:
    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def __call__(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        return self.mapping.get(text, (0.0, 0.0, 0.0))


class RecallTests(unittest.TestCase):
    def test_low_information_skips_everything(self) -> None:
        embedder = ScriptedEmbedder({})
        result = HybridRecallEngine([], embedder=embedder).recall("嗯嗯")
        self.assertEqual(result.skipped_reason, "low_information")
        self.assertEqual(embedder.calls, [])

    def test_transport_wrapper_is_removed_before_embedding(self) -> None:
        embedder = ScriptedEmbedder({"奶茶": (1.0, 0.0, 0.0)})
        memory = Memory(1, "奶茶偏好", "低糖", aliases=("奶茶",), vector=(1.0, 0.0, 0.0))
        result = HybridRecallEngine([memory], embedder=embedder).recall(
            '<channel chat="fake" message="42">奶茶</channel>'
        )
        self.assertEqual(result.normalized_query, "奶茶")
        self.assertEqual(embedder.calls, ["奶茶"])

    def test_two_character_chinese_alias_is_a_strong_signal(self) -> None:
        embedder = ScriptedEmbedder({"和教练出去吃饭": (0.0, 1.0, 0.0)})
        memories = [
            Memory(1, "训练计划", "周末训练", aliases=("教练",), vector=(1.0, 0.0, 0.0)),
            Memory(2, "设备维护", "检查风扇", vector=(0.0, 1.0, 0.0)),
        ]
        result = HybridRecallEngine(memories, embedder=embedder).recall("和教练出去吃饭")
        self.assertEqual(result.hits[0].memory_id, 1)
        self.assertIn("entity_alias", result.hits[0].evidence)

    def test_vector_recovers_paraphrase_without_literal_overlap(self) -> None:
        embedder = ScriptedEmbedder({"最近睡得不踏实": (1.0, 0.0, 0.0)})
        memory = Memory(3, "夜间作息", "减少夜间咖啡因。", vector=(0.95, 0.05, 0.0))
        result = HybridRecallEngine([memory], embedder=embedder).recall("最近睡得不踏实")
        self.assertEqual([hit.memory_id for hit in result.hits], [3])
        self.assertEqual(result.hits[0].evidence, ("vector",))

    def test_weak_vector_noise_is_rejected(self) -> None:
        embedder = ScriptedEmbedder({"量子焊接机": (1.0, 0.0, 0.0)})
        memory = Memory(4, "旅行计划", "去海边", vector=(0.4, 0.9165, 0.0))
        result = HybridRecallEngine([memory], embedder=embedder).recall("量子焊接机")
        self.assertEqual(result.hits, ())

    def test_embedder_failure_keeps_literal_recall_available(self) -> None:
        def broken_embedder(_: str) -> tuple[float, ...]:
            raise TimeoutError("synthetic timeout")

        memory = Memory(8, "奶茶偏好", "选低糖", aliases=("奶茶",))
        result = HybridRecallEngine([memory], embedder=broken_embedder).recall("奶茶")
        self.assertEqual([hit.memory_id for hit in result.hits], [8])
        self.assertFalse(result.trace["vector_called"])

    def test_cross_channel_agreement_is_exposed(self) -> None:
        embedder = ScriptedEmbedder({"低糖奶茶": (1.0, 0.0, 0.0)})
        memory = Memory(5, "奶茶偏好", "选低糖", aliases=("奶茶",), vector=(1.0, 0.0, 0.0))
        result = HybridRecallEngine([memory], embedder=embedder).recall("低糖奶茶")
        self.assertIn("cross_channel_agreement", result.hits[0].evidence)

    def test_summary_is_used_only_when_bound_to_current_body(self) -> None:
        body = "完整原文"
        embedder = ScriptedEmbedder({"主题词": (1.0, 0.0, 0.0)})
        fresh = Memory(
            6,
            "主题词",
            body,
            summary="短摘要",
            summary_body_sha256=content_digest(body),
            vector=(1.0, 0.0, 0.0),
        )
        stale = Memory(
            7,
            "主题词二",
            "已更新的完整原文",
            summary="过期摘要",
            summary_body_sha256=sha256(b"old").hexdigest(),
            vector=(1.0, 0.0, 0.0),
        )
        result = HybridRecallEngine([fresh, stale], embedder=embedder).recall("主题词")
        by_id = {hit.memory_id: hit for hit in result.hits}
        self.assertEqual(by_id[6].text, "短摘要")
        self.assertEqual(by_id[6].summary_source, "summary")
        self.assertNotIn("过期摘要", by_id[7].text)
        self.assertEqual(by_id[7].summary_source, "body_fallback")

    def test_hit_count_and_character_budget_are_hard_limits(self) -> None:
        embedder = ScriptedEmbedder({"共同主题": (1.0, 0.0, 0.0)})
        memories = [
            Memory(index, f"共同主题 {index}", "正文" * 300, vector=(1.0, 0.0, 0.0))
            for index in range(1, 8)
        ]
        config = RecallConfig(max_hits=3, max_chars=180, per_hit_chars=100)
        result = HybridRecallEngine(memories, embedder=embedder, config=config).recall("共同主题")
        self.assertLessEqual(len(result.hits), 3)
        self.assertLessEqual(result.injection_chars, 180)
        self.assertEqual(result.trace["max_hits"], 3)


if __name__ == "__main__":
    unittest.main()
