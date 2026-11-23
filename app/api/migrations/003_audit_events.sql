-- Migration 003: the audit trail (see app/api/audit.py for the events and their data).
-- Append-only, metadata only: event data never holds document text, file names, MRZ data or caller references in
-- clear. Rows outlive their job (no foreign key) and are deleted after AUDIT_RETENTION_DAYS.
CREATE TABLE IF NOT EXISTS ocr_audit_events (
    id         bigserial PRIMARY KEY,
    at         timestamptz NOT NULL DEFAULT now(),
    event      text NOT NULL,               -- e.g. job.received, job.done, request.error
    client_id  text,                        -- NULL: an unauthenticated request
    job_id     text,
    request_id text,                        -- X-Request-Id of the request that caused the event
    data       jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS ocr_audit_events_at ON ocr_audit_events (at);
CREATE INDEX IF NOT EXISTS ocr_audit_events_job ON ocr_audit_events (job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ocr_audit_events_request ON ocr_audit_events (request_id) WHERE request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ocr_audit_events_client ON ocr_audit_events (client_id, at);
