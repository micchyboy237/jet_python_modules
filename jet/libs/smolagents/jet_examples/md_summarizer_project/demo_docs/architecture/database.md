# Database design

Nimbus uses a single PostgreSQL table, `jobs`, as its queue store.

## Schema

The `jobs` table has columns: `id` (uuid), `queue_name` (text), `payload`
(jsonb), `status` (enum: pending, running, done, failed, dead), `attempts`
(int), `idempotency_key` (text, unique), `available_at` (timestamptz), and
`created_at` (timestamptz).

## Indexing

A partial index on `(queue_name, available_at)` where `status = 'pending'`
keeps dequeue queries fast even as the table grows into the millions of rows.
Completed jobs older than 7 days are moved to a `jobs_archive` table by a
nightly job to keep the hot table small.

## Failure handling

A job is retried with exponential backoff up to 5 times. After the 5th
failure it is marked `dead` and moved to a dead-letter queue for manual
inspection. The `idempotency_key` uniqueness constraint prevents the same
logical job from being enqueued twice even if the enqueue call is retried by
the caller.
