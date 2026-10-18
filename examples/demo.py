from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bounded_recall import (  # noqa: E402
    EvidenceBinder,
    HashingEmbedder,
    HybridRecallEngine,
    InMemoryConversationSource,
    Memory,
    Message,
)
from bounded_recall.models import content_digest  # noqa: E402


embedder = HashingEmbedder()
body = "用户喜欢桂花乌龙奶茶，但晚上会选择低糖，避免影响休息。"
memories = [
    Memory(
        id=101,
        title="奶茶偏好",
        body=body,
        summary="偏好桂花乌龙；夜间选择低糖。",
        summary_body_sha256=content_digest(body),
        keywords=("饮品", "低糖"),
        aliases=("奶茶", "桂花乌龙"),
        vector=embedder(body),
    ),
    Memory(
        id=102,
        title="周末训练安排",
        body="周六上午进行力量训练，时间变化时提前确认。",
        keywords=("训练", "周六"),
        aliases=("教练",),
        vector=embedder("周末和教练训练"),
    ),
]

engine = HybridRecallEngine(memories, embedder=embedder)
result = engine.recall('<channel source="mobile" message="synthetic">晚上想喝奶茶</channel>')
print(result.injection)
print(result.trace)

messages = InMemoryConversationSource(
    [Message(7, "user", "晚上喝的话帮我选低糖桂花乌龙。", "2026-01-02T20:30:00+08:00")]
)
binder = EvidenceBinder(messages)
binder.bind(101, message_id=7)
print(binder.availability(101))
print(binder.read(101))
