-- Migration 004: webhooks are delivered from the job row (an outbox) instead of from memory, so retries survive
-- restarts and any replica sends them (see app/api/webhook.py).
ALTER TABLE ocr_jobs ADD COLUMN IF NOT EXISTS webhook_delivery_id text;       -- the same for every attempt
ALTER TABLE ocr_jobs ADD COLUMN IF NOT EXISTS webhook_next_at timestamptz;    -- next attempt due; NULL = none pending
ALTER TABLE ocr_jobs ADD COLUMN IF NOT EXISTS webhook_settled_at timestamptz; -- delivered/given up: the TTL restarts
CREATE INDEX IF NOT EXISTS ocr_jobs_webhook_due ON ocr_jobs (webhook_next_at) WHERE webhook_next_at IS NOT NULL;

-- Deliveries the previous release still held in memory, if it stopped before sending them, stay "pending" forever.
-- Send those again. Older releases gave up within a minute, so anything older than 5 minutes is really stuck
-- (newer ones may still be in flight on a running old replica).
UPDATE ocr_jobs
SET webhook_delivery_id = replace(gen_random_uuid()::text, '-', ''), webhook_next_at = now()
WHERE webhook->>'state' = 'pending' AND callback_url IS NOT NULL AND webhook_next_at IS NULL
  AND finished_at < now() - interval '5 minutes';
