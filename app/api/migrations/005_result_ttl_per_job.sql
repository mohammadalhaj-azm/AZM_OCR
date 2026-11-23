-- Migration 005: a job can keep its result for its own time (its processing profile's result_ttl_s) instead of
-- RESULT_TTL_S. NULL = the global RESULT_TTL_S.
ALTER TABLE ocr_jobs ADD COLUMN IF NOT EXISTS result_ttl_s integer;
