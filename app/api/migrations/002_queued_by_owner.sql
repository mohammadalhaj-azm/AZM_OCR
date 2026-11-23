-- Migration 002: per-client queue quotas (CLIENTS_FILE limits.max_queued_jobs) count a client's queued jobs.
-- The job table stays small (finished jobs are purged after RESULT_TTL_S), so a plain CREATE INDEX is quick.
CREATE INDEX IF NOT EXISTS ocr_jobs_queued_owner ON ocr_jobs (owner) WHERE status = 'queued';
