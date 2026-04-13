# ADR 0005 - Public tunnel: Cloudflare Tunnel replaces ngrok

- Status: Accepted
- Date: 2026-04-13
- Deciders: @architect, @devops

## Context

Exposing the local dev environment publicly (for demos, webhook testing
and mobile QA) previously used ngrok. ngrok's free tier has:

- Random URLs that change on every restart.
- Hard connection caps per minute.
- Occasional blocks from Brazilian news providers (BrAPI 429s through
  certain ngrok egress IPs were observed during prototyping).

## Decision

Use **Cloudflare Tunnel** (`cloudflared`) as the default public-exposure
mechanism for SPP v3, documented in the README quick start.

Two modes are supported:

- *Quick tunnel* (no account): `cloudflared tunnel --url http://localhost:3000`
  for one-shot demos.
- *Named tunnel* (free Cloudflare account): fixed hostnames
  (`spp.example.com`, `api.spp.example.com`) with ingress rules in
  `~/.cloudflared/config.yml`.

## Consequences

Positive
- Free fixed hostname on `*.trycloudflare.com` or a custom domain.
- No connection-per-minute throttling.
- Lower latency from Brazilian ISPs (Cloudflare has POPs in GRU/GIG).
- Works as a zero-config egress even for WebSocket traffic to
  `/ws/live/{ticker}`.

Negative / trade-offs
- Requires `cloudflared` binary install outside the Docker stack.
- Named tunnels require a Cloudflare account (free) and DNS record.

## Alternatives considered

- **ngrok**: familiar, but limitations listed above.
- **Tailscale Funnel**: good for team-internal demos, but limited to
  HTTPS and requires Tailscale on the viewer side for best latency.
- **localtunnel / serveo**: unreliable uptime; not suitable for even
  demo-grade use.
