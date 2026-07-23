# Authentication

All Nimbus API requests require a service token passed in the `Authorization`
header as `Bearer <token>`.

## Issuing tokens

Tokens are issued per-service by the platform team via the internal `nimbusctl
token create --service <name>` command. Tokens do not expire automatically but
can be revoked instantly through the same CLI.

## Scopes

Tokens carry one of two scopes: `enqueue` (can only call `POST /jobs`) and
`admin` (can also call `GET /jobs/{id}`, `POST /jobs/{id}/retry`, and
`DELETE /jobs/{id}`). Most services should only ever need an `enqueue`-scoped
token.

## Rate limits

Each token is rate limited to 500 requests per second, tracked in a sliding
1-second window. Exceeding the limit returns HTTP 429 with a `Retry-After`
header.
