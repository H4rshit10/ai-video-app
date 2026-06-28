"""Postgres + pgvector storage layer (v2 backend).

Optional — gated by ENABLE_POSTGRES env var. When disabled, the pipeline runs
exactly as before with local-filesystem outputs. When enabled, every run is
persisted to Postgres for history, and corpora + chunks tables back the RAG
flow for grounded video generation.

Setup (one-time):
  1. Bring up a Postgres 15+ instance with pgvector installed:
       docker run -d --name aivideo-pg -p 5432:5432 \\
         -e POSTGRES_PASSWORD=devpass -e POSTGRES_DB=aivideo \\
         pgvector/pgvector:pg16
  2. Apply schema:
       psql postgresql://postgres:devpass@localhost:5432/aivideo -f src/db_schema.sql
  3. Set in .env:
       ENABLE_POSTGRES=true
       DATABASE_URL=postgresql://postgres:devpass@localhost:5432/aivideo

Production: swap the URL for Cloud SQL Postgres with the same schema applied.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from . import config

logger = logging.getLogger(__name__)


# Deferred imports — psycopg/pgvector are only loaded when ENABLE_POSTGRES is true,
# so the rest of the app stays usable without the extra dependencies installed.

def _connect():
    """Return a fresh psycopg connection. Caller closes it (or uses `with`)."""
    if not config.ENABLE_POSTGRES:
        raise RuntimeError("Postgres backend is disabled (ENABLE_POSTGRES=false).")
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required when ENABLE_POSTGRES=true.")
    import psycopg
    from pgvector.psycopg import register_vector
    conn = psycopg.connect(config.DATABASE_URL)
    register_vector(conn)
    return conn


# --------------------------------------------------------------------------
# Runs — persistence of every generation
# --------------------------------------------------------------------------

@dataclass
class StoredRun:
    run_id: str
    user_id: int | None
    topic: str
    audience: str
    content_type: str | None
    cost_usd: float
    voice_used: str
    allow_veo: bool
    veo_fallback_used: bool
    plan_json: dict
    quiz_json: dict | None
    final_video_path: str
    elapsed_seconds: float


def save_run(run: StoredRun) -> int:
    """Persist a generated run. Returns the row id."""
    sql = """
        INSERT INTO runs (run_id, user_id, topic, audience, content_type, cost_usd,
                          voice_used, allow_veo, veo_fallback_used, plan_json,
                          quiz_json, final_video_path, elapsed_seconds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (
            run.run_id, run.user_id, run.topic, run.audience, run.content_type,
            run.cost_usd, run.voice_used, run.allow_veo, run.veo_fallback_used,
            json.dumps(run.plan_json),
            json.dumps(run.quiz_json) if run.quiz_json else None,
            run.final_video_path, run.elapsed_seconds,
        ))
        row_id = cur.fetchone()[0]
        conn.commit()
        return row_id


def list_recent_runs(user_id: int | None = None, limit: int = 20) -> list[dict]:
    """Return the most recent runs, optionally filtered by user_id."""
    if user_id is None:
        sql = """SELECT run_id, topic, audience, cost_usd, created_at, final_video_path
                 FROM runs ORDER BY created_at DESC LIMIT %s;"""
        params: tuple = (limit,)
    else:
        sql = """SELECT run_id, topic, audience, cost_usd, created_at, final_video_path
                 FROM runs WHERE user_id = %s ORDER BY created_at DESC LIMIT %s;"""
        params = (user_id, limit)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# --------------------------------------------------------------------------
# Users — minimal account model (OAuth integration is the caller's job)
# --------------------------------------------------------------------------

def upsert_user(email: str, oauth_sub: str | None = None, display_name: str | None = None) -> int:
    """Get or create a user row by email. Returns user id."""
    sql = """
        INSERT INTO users (email, oauth_sub, display_name)
        VALUES (%s, %s, %s)
        ON CONFLICT (email) DO UPDATE
            SET oauth_sub = COALESCE(EXCLUDED.oauth_sub, users.oauth_sub),
                display_name = COALESCE(EXCLUDED.display_name, users.display_name)
        RETURNING id;
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (email, oauth_sub, display_name))
        row_id = cur.fetchone()[0]
        conn.commit()
        return row_id


# --------------------------------------------------------------------------
# Corpora + chunks — RAG grounding (V2)
# --------------------------------------------------------------------------

def create_corpus(user_id: int, name: str, source: str | None = None) -> int:
    """Create a new corpus for a user. Returns corpus id."""
    sql = "INSERT INTO corpora (user_id, name, source) VALUES (%s, %s, %s) RETURNING id;"
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (user_id, name, source))
        row_id = cur.fetchone()[0]
        conn.commit()
        return row_id


def store_chunks(corpus_id: int, chunks: list[tuple[str, list[float], dict[str, Any] | None]]) -> int:
    """Insert pre-embedded chunks. Each tuple is (text, embedding_vector, metadata).

    Caller is responsible for chunking the source text and generating embeddings
    via Vertex AI's text-embedding-005 (or equivalent 768-dim model).

    Returns the number of rows inserted.
    """
    sql = """
        INSERT INTO chunks (corpus_id, chunk_text, embedding, metadata)
        VALUES (%s, %s, %s, %s);
    """
    with _connect() as conn, conn.cursor() as cur:
        for text, embedding, metadata in chunks:
            cur.execute(sql, (
                corpus_id, text, embedding,
                json.dumps(metadata) if metadata else None,
            ))
        conn.commit()
        return len(chunks)


def retrieve_similar_chunks(
    corpus_id: int,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """Cosine-similarity search across a corpus. Returns top_k chunks with text + distance.

    The Director can inject the returned text into its system prompt as grounding context.
    """
    sql = """
        SELECT chunk_text, embedding <=> %s::vector AS distance, metadata
        FROM chunks
        WHERE corpus_id = %s
        ORDER BY distance ASC
        LIMIT %s;
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (query_embedding, corpus_id, top_k))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# --------------------------------------------------------------------------
# Convenience: turn a pipeline.RunResult into a StoredRun for saving
# --------------------------------------------------------------------------

def stored_run_from_result(result, user_id: int | None = None) -> StoredRun:
    """Adapter that converts pipeline.RunResult -> StoredRun.

    Kept here so storage.py is the single place that knows about the persistence
    shape; pipeline.py stays decoupled from the DB layer.
    """
    return StoredRun(
        run_id=result.run_id,
        user_id=user_id,
        topic=result.plan.title,
        audience="",  # caller can pass the original audience if needed
        content_type=result.plan.content_type,
        cost_usd=float(result.cost.total_usd),
        voice_used=result.voice_used,
        allow_veo=result.veo_attempted,
        veo_fallback_used=result.veo_fallback_used,
        plan_json=result.plan.model_dump(),
        quiz_json=result.plan.end_quiz.model_dump() if result.plan.end_quiz else None,
        final_video_path=str(result.final_video),
        elapsed_seconds=float(result.elapsed_seconds),
    )
