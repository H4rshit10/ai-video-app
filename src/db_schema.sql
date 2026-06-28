-- AI Video App — Postgres + pgvector schema (v2 backend)
--
-- Apply this to a Postgres 15+ instance with the pgvector extension available.
-- Cloud SQL for Postgres (recommended) or local Postgres for dev.
--
-- Local dev one-liner (Docker):
--   docker run -d --name aivideo-pg -p 5432:5432 \
--     -e POSTGRES_PASSWORD=devpass \
--     -e POSTGRES_DB=aivideo \
--     pgvector/pgvector:pg16
--   psql postgresql://postgres:devpass@localhost:5432/aivideo -f src/db_schema.sql
--
-- Cloud SQL setup:
--   gcloud sql instances create aivideo-pg --database-version=POSTGRES_16 ...
--   gcloud sql databases create aivideo --instance=aivideo-pg
--   In Cloud Console -> instance -> Flags -> add: cloudsql.iam_authentication = on
--   Enable pgvector via: CREATE EXTENSION vector;

CREATE EXTENSION IF NOT EXISTS vector;

-- Users — OAuth-linked accounts. oauth_sub is the stable Google subject ID.
CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    oauth_sub   TEXT UNIQUE,
    display_name TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Runs — every generation tracked with cost telemetry and the full plan.
CREATE TABLE IF NOT EXISTS runs (
    id                  SERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL UNIQUE,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    topic               TEXT NOT NULL,
    audience            TEXT NOT NULL,
    content_type        TEXT,
    cost_usd            NUMERIC(10, 4),
    voice_used          TEXT,
    allow_veo           BOOLEAN NOT NULL DEFAULT FALSE,
    veo_fallback_used   BOOLEAN NOT NULL DEFAULT FALSE,
    plan_json           JSONB,
    quiz_json           JSONB,
    final_video_path    TEXT,
    elapsed_seconds     NUMERIC(8, 2),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS runs_user_id_created_idx ON runs (user_id, created_at DESC);

-- Corpora — collections of uploaded curriculum docs per user (for RAG grounding).
CREATE TABLE IF NOT EXISTS corpora (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    source      TEXT,            -- e.g. "syllabus.pdf"
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS corpora_user_idx ON corpora (user_id);

-- Chunks — text fragments with their pgvector embeddings, used for RAG retrieval.
-- Dimension is 768 to match Vertex AI's text-embedding-005 default output.
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    corpus_id   INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
    chunk_text  TEXT NOT NULL,
    embedding   vector(768) NOT NULL,
    metadata    JSONB,           -- page number, position, source filename, etc.
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- IVFFlat index for cosine similarity. Adjust `lists` upward as the table grows;
-- 100 is the right starting point for tens of thousands of chunks.
CREATE INDEX IF NOT EXISTS chunks_embedding_cosine_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_corpus_idx ON chunks (corpus_id);
