# Nimbus

Nimbus is a lightweight distributed task queue written in Go. It lets services
enqueue background jobs (emails, image processing, webhooks) and have a pool
of workers process them reliably, with retries and dead-letter handling.

## Goals

- At-least-once delivery with idempotency keys to make retries safe.
- Horizontal scaling of workers without a central coordinator.
- Sub-100ms enqueue latency at p99 under normal load.

## Non-goals

Nimbus is not a general-purpose message bus. It does not support pub/sub fan-out,
long-term event storage, or cross-region replication. Teams needing those
should use the existing Kafka cluster instead.
