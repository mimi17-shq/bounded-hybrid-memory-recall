-- Standalone portfolio schema. It does not copy or require a production database.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS portfolio_memories (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    keywords TEXT[] NOT NULL DEFAULT '{}',
    aliases TEXT[] NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'current',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS portfolio_memory_summaries (
    memory_id BIGINT PRIMARY KEY REFERENCES portfolio_memories(id) ON DELETE CASCADE,
    body_sha256 CHAR(64) NOT NULL,
    summary TEXT NOT NULL,
    model_name TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS portfolio_memory_vectors (
    memory_id BIGINT PRIMARY KEY REFERENCES portfolio_memories(id) ON DELETE CASCADE,
    body_sha256 CHAR(64) NOT NULL,
    embedding halfvec(1024) NOT NULL,
    model_name TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- References only: the original message remains in the external conversation archive.
CREATE TABLE IF NOT EXISTS portfolio_memory_evidence (
    id BIGSERIAL PRIMARY KEY,
    memory_id BIGINT NOT NULL REFERENCES portfolio_memories(id) ON DELETE CASCADE,
    channel_key TEXT NOT NULL,
    external_message_id TEXT NOT NULL,
    role TEXT NOT NULL,
    message_created_at TIMESTAMPTZ,
    content_sha256 CHAR(64) NOT NULL,
    evidence_kind TEXT NOT NULL DEFAULT 'history'
        CHECK (evidence_kind IN ('history', 'persona')),
    selection_order INTEGER NOT NULL DEFAULT 0,
    UNIQUE (memory_id, channel_key, external_message_id)
);

CREATE INDEX IF NOT EXISTS portfolio_memory_evidence_memory_idx
    ON portfolio_memory_evidence(memory_id, selection_order, id);

CREATE OR REPLACE FUNCTION portfolio_invalidate_derived_memory_data()
RETURNS trigger AS $$
BEGIN
    IF NEW.body IS DISTINCT FROM OLD.body THEN
        DELETE FROM portfolio_memory_summaries WHERE memory_id = NEW.id;
        DELETE FROM portfolio_memory_vectors WHERE memory_id = NEW.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS portfolio_memory_body_changed ON portfolio_memories;
CREATE TRIGGER portfolio_memory_body_changed
AFTER UPDATE OF body ON portfolio_memories
FOR EACH ROW EXECUTE FUNCTION portfolio_invalidate_derived_memory_data();
