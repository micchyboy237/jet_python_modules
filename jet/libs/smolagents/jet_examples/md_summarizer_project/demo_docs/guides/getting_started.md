# Getting started

This guide walks through enqueuing your first job with Nimbus.

## Prerequisites

You need a service token (see the authentication docs) and network access to
the Nimbus gateway at `https://nimbus.internal:8443`.

## Enqueue a job

Send a `POST /jobs` request with a JSON body containing `queue_name`,
`payload`, and a unique `idempotency_key`, for example one generated from your
own request ID so retries on your side stay safe.

## Check job status

Poll `GET /jobs/{id}` with an admin-scoped token to see whether the job is
still `pending`, currently `running`, finished as `done`, or ended up `dead`
after exhausting its retries.

## Local development

A `docker-compose.yml` in the repository root spins up Postgres, the gateway,
and two worker processes for local testing, with the gateway reachable at
`http://localhost:8080`.
