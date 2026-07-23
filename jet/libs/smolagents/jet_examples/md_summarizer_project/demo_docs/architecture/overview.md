# Architecture overview

Nimbus has three components: the API gateway, the queue store, and the worker
pool. Jobs flow from the gateway into the queue store, and workers pull from
the queue store in a loop.

## API gateway

The gateway exposes a small HTTP API (`POST /jobs`, `GET /jobs/{id}`) and is
stateless, so it can be scaled behind a load balancer with no session affinity.
It validates job payloads against a JSON schema before accepting them.

## Queue store

The queue store is backed by PostgreSQL using `SELECT ... FOR UPDATE SKIP LOCKED`
for safe concurrent dequeue. This avoids needing a separate broker like Redis
or RabbitMQ, at the cost of somewhat lower maximum throughput.

## Worker pool

Workers are stateless processes that long-poll the queue store. Each worker
handles one job at a time by default, though a `--concurrency` flag allows
running several job handlers per process for I/O-bound workloads.
