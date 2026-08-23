# PR Title

feat(memory): expose expert memory as an MCP server for external agents

---

## Summary

Adds a memory **MCP server** (Streamable HTTP at `/mcp/memory`) so external
agents (coding agents, bots, other AI tools) can directly **read / write /
update Octop expert memory**, aligned 1:1 with the in-process
`MemoryService` capabilities. Every write stamps a `source` marker that is
traceable on recall.

## Why

Octop experts accumulate rich memory (facts, conversations, decisions), but
today only the Octop dashboard / in-process agent can access it. External
agents that need to reuse that expertise (e.g. a coding agent asking a
business expert's accumulated knowledge) have no way in. This PR exposes the
same memory surface over the standard MCP protocol so any MCP-capable agent
can join the loop.

## What

- **New module** `src/octop/infra/agents/memory_mcp.py` — FastMCP server
  bound to one expert per connection, plus token auth and header routing.
- **Mount** in `api/app.py` (`build_app`) at `/mcp/memory`, with
  `streamable_http` task groups wired into the FastAPI lifespan.
- **Tests** `tests/unit/agents/test_memory_mcp.py` (13 tests).

### Tools

| Tool | Purpose | Backing API |
|------|---------|-------------|
| `memory_recall(query, limit=5)` | Recall memories (full pipeline: tokenize → FTS → rerank → dedupe); returns structured snippets + rendered markdown | `recall_for_prompt` |
| `memory_save(content, source, topic?)` | Persist a structured fact directly into the atom/tree (durable, no extraction) | `Memory.store` |
| `memory_capture(content, source, session_id?)` | Write an **L0 raw event** (goes through extraction); visible immediately via `memory_search_raw` | `Memory.add_raw` |
| `memory_search_raw(query, limit=10)` | FTS-search L0 raw events (capture visible before extraction) | `Memory.search_raw` |
| `memory_update(atom_id, new_content, source)` | Deprecate old atom + persist the new fact | `deprecate_atom` + `store` |

### Expert binding & auth

- **One connection binds one expert**: endpoint is a single `/mcp/memory`;
  the expert is selected at connect time via the `X-Octop-Agent-Id` header —
  callers never pass an agent id per tool call (they don't know the id list).
- **Auth**: independent token via `OCTOP_MEMORY_MCP_TOKEN` (fail-closed when
  unset). Authorization via `Authorization: Bearer` or `X-Octop-Memory-Token`.

### raw vs atom (for callers)

- `memory_capture` → **L0 raw event** (evidence layer), distilled later by
  the extraction pipeline (`extract → candidate → promote → atom`). Use it to
  record raw conversations/events; the record is visible immediately via
  `memory_search_raw` and recallable via `memory_recall` once promoted.
- `memory_save` → **atom/tree directly** (durable, no extraction). Use it
  when the fact is already known.

## Implementation notes

- Lives in `infra/agents/` with no api-layer dependency: opens the agent
  `Memory` instance via `open_memory_kwargs` + `Memory(...)` (workspace
  resolved from the agent registry).
- DNS rebinding protection disabled (`TransportSecuritySettings`) because
  Octop runs behind a reverse proxy (Host is the public domain, not localhost).
- `streamable_http_path` collapsed to `/` so the endpoint is exactly
  `/mcp/memory` (the SDK default `/mcp` would yield `/mcp/memory/mcp`).
- One `FastMCP` per expert, routed by an ASGI dispatcher on the
  `X-Octop-Agent-Id` header; missing/unknown agent → 404.

## Usage example

```bash
export OCTOP_MEMORY_MCP_TOKEN="<token>"
```

```json
{
  "mcpServers": {
    "octop-memory": {
      "type": "streamable_http",
      "url": "http://<host>/mcp/memory/",
      "headers": {
        "Authorization": "Bearer <token>",
        "X-Octop-Agent-Id": "<expert-id>"
      }
    }
  }
}
```

```text
memory_recall(query="what are the key project decisions?")
memory_save(content="the release window is every Tuesday", source="coding-agent", topic="release")
memory_capture(content="user reported: the report panel banner is not rendering", source="review-bot", session_id="review-2026-08-20")
memory_search_raw(query="report panel banner")
memory_update(atom_id="atom_xxx", new_content="updated fact", source="coding-agent")
```

## Testing

- `tests/unit/agents/test_memory_mcp.py` — 13 tests: tool registration,
  recall pipeline, capture (raw) semantics, search_raw, update, token
  middleware (401 / accept), header routing, 404 unknown agent, unified mount.
- Verified locally by booting the server and exercising the MCP endpoints:
  health, 401 without token, `initialize` (binds expert via header),
  `tools/list` (5 tools), `tools/call memory_recall`.

## Checklist

- [x] No internal/hard-coded environment-specific values in the diff
- [x] `make lint` clean (ruff)
- [x] Unit tests pass
