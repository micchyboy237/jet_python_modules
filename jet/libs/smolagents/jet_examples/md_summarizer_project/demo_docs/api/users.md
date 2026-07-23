# Job submission API

## POST /jobs

Enqueues a new job. Required fields: `queue_name`, `payload`, and
`idempotency_key`. Optional fields: `available_at` (schedule for later,
defaults to now) and `max_attempts` (defaults to 5).

Returns HTTP 201 with the created job's `id` on success, or HTTP 409 if the
`idempotency_key` already exists, returning the existing job instead of
creating a duplicate.

## GET /jobs/{id}

Returns the current status, attempt count, and last error (if any) for a
single job. Requires an `admin`-scoped token.

## POST /jobs/{id}/retry

Forces an immediate retry of a `dead` job, resetting its attempt count to
zero. Requires an `admin`-scoped token. Intended for use after a downstream
outage has been resolved and dead-lettered jobs need to be replayed.
