-- Migration 006: the admin dashboard (app/admin). Metadata only: no table here holds document content.

-- Dashboard aggregations filter the audit trail by event type and time.
CREATE INDEX IF NOT EXISTS ocr_audit_events_event_at ON ocr_audit_events (event, at);

-- Resource and health snapshot of every replica, every SAMPLE_PERSIST_S. The latest row of an instance is its
-- heartbeat. Deleted after SAMPLE_RETENTION_H.
CREATE TABLE IF NOT EXISTS ocr_instance_samples (
    id       bigserial PRIMARY KEY,
    instance text NOT NULL,
    at       timestamptz NOT NULL DEFAULT now(),
    data     jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS ocr_instance_samples_instance_at ON ocr_instance_samples (instance, at);
CREATE INDEX IF NOT EXISTS ocr_instance_samples_at ON ocr_instance_samples (at);

-- Alerts raised by the replicas' rule evaluation. One open alert per (rule, scope): replicas evaluating the same
-- rule update it instead of adding duplicates.
CREATE TABLE IF NOT EXISTS ocr_alerts (
    id           bigserial PRIMARY KEY,
    rule         text NOT NULL,
    scope        text NOT NULL DEFAULT '',   -- '' (global), an instance id, client:<id>, ...
    severity     text NOT NULL CHECK (severity IN ('critical', 'warning', 'info')),
    summary      text NOT NULL,
    details      jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at   timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    resolved_at  timestamptz,
    acked_by     text,
    acked_at     timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS ocr_alerts_open ON ocr_alerts (rule, scope) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS ocr_alerts_started ON ocr_alerts (started_at);

-- WARNING+ log records of every replica (log messages never hold document content). Deleted after
-- LOG_RETENTION_DAYS.
CREATE TABLE IF NOT EXISTS ocr_log_records (
    id         bigserial PRIMARY KEY,
    at         timestamptz NOT NULL,
    instance   text NOT NULL,
    level      text NOT NULL,
    logger     text NOT NULL,
    msg        text NOT NULL,
    request_id text,
    job_id     text,
    client_id  text,
    exc_type   text
);
CREATE INDEX IF NOT EXISTS ocr_log_records_at ON ocr_log_records (at);
CREATE INDEX IF NOT EXISTS ocr_log_records_request ON ocr_log_records (request_id) WHERE request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ocr_log_records_job ON ocr_log_records (job_id) WHERE job_id IS NOT NULL;

-- Configuration managed in the dashboard, append-only: the latest row of a kind is current, a restore adds a new
-- row. Kinds: settings (overrides of env settings), registry (clients + profiles, as in CLIENTS_FILE), rules:<name>.
-- Holds no secrets: API keys are SHA-256 hashes, webhook secrets live in ocr_client_secrets.
CREATE TABLE IF NOT EXISTS ocr_config_versions (
    id      bigserial PRIMARY KEY,
    at      timestamptz NOT NULL DEFAULT now(),
    kind    text NOT NULL,
    content jsonb NOT NULL,
    actor   text NOT NULL,
    comment text NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ocr_config_versions_kind ON ocr_config_versions (kind, id);

-- Webhook signing secrets generated in the dashboard, sealed (AES-256-GCM with RESULT_ENCRYPTION_KEY).
CREATE TABLE IF NOT EXISTS ocr_client_secrets (
    client_id  text PRIMARY KEY,
    sealed     bytea NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
