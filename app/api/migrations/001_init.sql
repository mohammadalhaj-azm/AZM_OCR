-- Migration 001: the initial job store schema (PostgreSQL 13+).
-- Every statement is idempotent, so databases created before migrations existed take it as a no-op.
-- Document content (upload bytes, results, file names, caller metadata) lives only in the sealed bytea columns,
-- encrypted when RESULT_ENCRYPTION_KEY is set. Plain columns hold ids, sizes, states and timestamps only.

CREATE TABLE IF NOT EXISTS ocr_jobs (
    id           text PRIMARY KEY,
    owner        text NOT NULL,                 -- hash of the API key
    size         integer NOT NULL,
    status       text NOT NULL CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    callback_url text,
    request_id   text,
    private      bytea NOT NULL,                -- sealed JSON {filename, externalId, metadata, options}
    result       bytea,                         -- sealed JSON result
    error        jsonb,                         -- {code, message}: never document content
    webhook      jsonb,
    attempts     integer NOT NULL DEFAULT 0,
    locked_by    text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    started_at   timestamptz,
    heartbeat_at timestamptz,
    finished_at  timestamptz
);
CREATE INDEX IF NOT EXISTS ocr_jobs_queued ON ocr_jobs (created_at) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS ocr_jobs_processing ON ocr_jobs (heartbeat_at) WHERE status = 'processing';
CREATE INDEX IF NOT EXISTS ocr_jobs_finished ON ocr_jobs (finished_at) WHERE finished_at IS NOT NULL;

-- Upload bytes, deleted right after processing (also on failure). Separate table: job rows stay small.
CREATE TABLE IF NOT EXISTS ocr_job_uploads (
    job_id text PRIMARY KEY REFERENCES ocr_jobs (id) ON DELETE CASCADE,
    data   bytea NOT NULL                       -- sealed upload
);
-- PDFs/images are already compressed (or encrypted): don't let TOAST try again.
ALTER TABLE ocr_job_uploads ALTER COLUMN data SET STORAGE EXTERNAL;

CREATE TABLE IF NOT EXISTS ocr_idempotency_keys (
    owner       text NOT NULL,
    key         text NOT NULL,
    file_sha256 text NOT NULL,
    job_id      text NOT NULL REFERENCES ocr_jobs (id) ON DELETE CASCADE,  -- key expires with its job
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (owner, key)
);
CREATE INDEX IF NOT EXISTS ocr_idempotency_keys_job ON ocr_idempotency_keys (job_id);
