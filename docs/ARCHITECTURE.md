# Architecture and contracts

## Components

`HybridRecallEngine` is intentionally storage-agnostic. It accepts memory records whose vectors were already produced by a caller-provided embedder. The included `HashingEmbedder` is an offline fixture, not a semantic model.

The engine applies these contracts in order:

1. Normalize the transport payload before feature extraction.
2. Skip low-information input without calling the embedder.
3. Collect deterministic literal/entity evidence and vector similarity independently.
4. Admit a candidate only when at least one evidence gate passes.
5. Preserve evidence labels in every hit.
6. Prefer a summary only when its source-body digest still matches.
7. Apply hit-count and character budgets after ranking.

`EvidenceBinder` is separate from recall. Binding a source is an explicit write operation, while reading source text is an explicit, on-demand operation. A normal recall response needs only an availability tail.

## Production adapter shape

```text
conversation archive (append-only)
        ↑ explicit ID lookup / hash verification
memory_evidence (references only)
        ↓ availability count
curated memory ─ summary ─ vector
        ↓
bounded hybrid recall
        ↓
agent context
```

A production implementation should additionally provide:

- transactional memory writes and vector/summary invalidation;
- an embedding retry and backfill queue;
- ANN index monitoring and deterministic lexical fallback;
- per-session duplicate suppression;
- latency, empty-result, evidence-channel and budget metrics;
- access control around original-message reads;
- a migration and rollback path for every schema change.

The PostgreSQL example demonstrates storage separation and invalidation, but deliberately omits credentials, HTTP endpoints, model-provider adapters and deployment files.
