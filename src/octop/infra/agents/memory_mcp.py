"""Expose Octop expert memory as an MCP server for external agents.

External agents (coding agents, bots) can read/write Octop expert memory over
MCP (Streamable HTTP), aligned with the in-process ``MemoryService``
capabilities. Every write stamps a ``source`` marker that can be traced back
on recall.

Expert binding: the endpoint is a single ``/mcp/memory`` mount; the expert is
selected at connect time via the ``X-Octop-Agent-Id`` header (one connection
binds one expert — the caller never passes an agent id per tool call).

raw vs atom (aligned with ``MemoryService``):

* ``memory_capture`` -> ``add_raw``: writes an **L0 raw event**, which goes
  through the extraction pipeline (extract -> candidate -> promote -> atom).
  Use it to record raw conversations / events. The record is visible
  immediately via ``memory_search_raw``; ``memory_recall`` returns it only
  after extraction promotes it to an atom.
* ``memory_save`` -> ``store``: persists a structured fact directly into the
  canonical atom/tree (durable, no extraction). Use it when you already know
  the exact fact to remember.

Auth: independent token via ``OCTOP_MEMORY_MCP_TOKEN`` (fail-closed when
unset), enforced by the ASGI middleware in ``mount_memory_mcp``.
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from mcp.server.transport_security import TransportSecuritySettings

from octop.infra.agents.memory_backend import open_memory_kwargs
from octop.infra.server import OctopServer

logger = logging.getLogger(__name__)

# 当前 MCP HTTP 请求的调用者 user id（由 _AgentRouter 中间件写入，工具读取）。
# stateless streamable HTTP 下 mcp SDK 不提供 ctx.request_context，故用 contextvar
# 跨 ASGI 中间件 → 工具传递，供 memory_capture/save 做 per-user 追溯。
_current_caller_user: ContextVar[str] = ContextVar("octop_mcp_caller_user", default="")


def _open_memory(server: OctopServer, agent_id: str) -> Any:
    """Open the agent's ``Memory`` instance (sqlite by default, postgres opt-in).

    Mirrors ``api.common.memory_client._open_memory_for_agent`` but stays in
    ``infra/`` (no api dependency). Workspace is resolved from the agent
    registry, falling back to the Octop default layout.
    """
    from harness_memory.core import Memory  # noqa: PLC0415

    services = server.services
    assert services is not None, "server.services required for memory backend"
    runtime = getattr(server, "app_runtime", None)
    registry = getattr(runtime, "agent_registry", None) if runtime is not None else None
    if registry is not None and hasattr(registry, "resolve_workspace_dir"):
        workspace = registry.resolve_workspace_dir(agent_id)
    else:
        paths = getattr(server, "paths", None) or services.paths
        workspace = paths.ensure_agent_workspace(agent_id)

    row = services.agent_repo.get(agent_id)
    cfg: dict[str, Any] = {}
    if row is not None and row.config_json:
        import json  # noqa: PLC0415

        try:
            parsed = json.loads(row.config_json)
            if isinstance(parsed, dict):
                cfg = parsed
        except json.JSONDecodeError:
            cfg = {}

    ns, backend, backend_config = open_memory_kwargs(
        agent_id=agent_id,
        cfg=cfg,
        octop_config=services.config,
        workspace_dir=workspace,
    )
    return Memory(namespace=ns, backend=backend, backend_config=backend_config)


def build_memory_mcp(server: OctopServer, agent_id: str) -> FastMCP:
    """Build an MCP server bound to one expert (``agent_id`` captured in closure)."""
    mcp = FastMCP(
        f"octop-memory-{agent_id}",
        # Stateless streamable HTTP: every request gets a fresh transport, no
        # Mcp-Session-Id tracking. Session state is in-memory per process, so a
        # server restart silently orphans every client session id and the next
        # tool call fails with -32600 "Session not found". Stateless mode
        # eliminates that failure class entirely (clients re-initialize per
        # request); the cost is one extra initialize per tool call.
        stateless_http=True,
        # Octop runs behind a reverse proxy (Host is the public domain, forwarded
        # by nginx), not a localhost dev scenario — the mcp SDK's localhost
        # DNS-rebinding protection does not apply and would reject the Host
        # with 421 unless the domain is allow-listed.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    # Collapse the streamable-HTTP path to "/" so the endpoint is exactly
    # /mcp/memory (the default "/mcp" would make it /mcp/memory/mcp).
    mcp.settings.streamable_http_path = "/"

    def _memory() -> Any:
        return _open_memory(server, agent_id)

    def _caller_user(ctx: Any | None) -> str:
        """读取当前 MCP 请求的调用者 user id。

        优先级：显式 ``user`` 参数 → ``X-Octop-User-Id`` header（由
        ``_AgentRouter`` 中间件写入 contextvar）。stateless HTTP 下 mcp SDK
        不提供 ``ctx.request_context``，故不依赖它。
        """
        try:
            return _current_caller_user.get() or ""
        except Exception:  # noqa: BLE001
            return ""

    def _derive_session(source: str, user: str) -> str:
        """外部调用缺省 session_id 时派生稳定会话键。

        规则 ``ext:{source}:{user}``：同 source 同 user 的多次 capture 落入
        同一分组，harness 提取管线能聚合蒸馏成 atom；不同 source / 不同 user
        分开分组，避免混入彼此上下文。
        """
        return f"ext:{source or 'mcp'}:{user or 'anon'}"


    @mcp.tool()
    def memory_recall(
        query: str,
        limit: int = 5,
        user: str | None = None,
        ctx: Context | None = None,  # type: ignore[type-arg]
    ) -> dict[str, Any]:
        """Recall memories from this expert (aligned with the in-process recall_inject).

        **日常使用**：每次对话/任务开始前调用，把专家记忆中与 query 相关的
        atom 召回注入上下文。运行完整召回管线（tokenize -> FTS -> rerank ->
        dedupe），返回结构化片段 + 可注入 system prompt 的 markdown 块。

        调用者身份（``X-Octop-User-Id`` header 或 ``user`` 参数）会记录在
        返回的 ``caller`` 字段，供按调用者追溯召回来源；记忆本身是专家级
        共享，不按用户隔离。

        Args:
            query: free-form question / keywords (pass the whole sentence; the
                pipeline tokenizes CJK into n-grams internally).
            limit: max number of snippets to return.
            user: optional caller id (overrides the ``X-Octop-User-Id`` header).
            ctx: injected MCP context (reads ``X-Octop-User-Id`` header).
        """
        from harness_memory.pipeline.recall import recall_for_prompt  # noqa: PLC0415

        caller = user or _caller_user(ctx)
        memory = _memory()
        result = recall_for_prompt(memory, query, limit=limit)
        return {
            "memories": [
                {
                    "source_id": s.source_id,
                    "timestamp": s.timestamp_iso,
                    "layer": s.layer,
                    "text": s.text,
                }
                for s in result.snippets
            ],
            "count": len(result.snippets),
            "rendered": result.rendered,
            "caller": caller or None,
        }

    @mcp.tool()
    def memory_save(
        content: str,
        source: str,
        topic: str | None = None,
        user: str | None = None,
        ctx: Context | None = None,  # type: ignore[type-arg]
    ) -> dict[str, Any]:
        """Persist a structured fact directly (atom/tree, durable, no extraction).

        **显式记忆**（非日常）：仅当你知道一个明确的、需要长期记住的事实
        时才调用（如用户偏好、项目约定）。立即通过 ``memory_recall`` 可召回，
        不经过提取管线。日常对话内容请用 ``memory_capture`` 交给自动提取。
        The source marker is stored in ``metadata.source``; the caller id (from
        ``X-Octop-User-Id`` header or ``user`` arg) is stored in ``metadata.user``.

        Args:
            content: the fact to remember.
            source: who/what recorded it (e.g. "coding-agent"), for traceability.
            topic: optional topic label.
            user: optional caller id (overrides the ``X-Octop-User-Id`` header).
            ctx: injected MCP context (reads ``X-Octop-User-Id`` header).
        """
        caller = user or _caller_user(ctx)
        memory = _memory()
        node = memory.store(
            content,
            topic=topic,
            metadata={"source": source, **({"user": caller} if caller else {})},
        )
        return {
            "node_id": node.id,
            "content": node.content,
            "source": source,
            "user": caller or None,
        }

    @mcp.tool()
    def memory_capture(
        content: str,
        source: str,
        session_id: str | None = None,
        user: str | None = None,
        ctx: Context | None = None,  # type: ignore[type-arg]
    ) -> dict[str, Any]:
        """Record a raw event to L0 (goes through extraction: extract -> candidate -> atom).

        **日常使用**：把对话/事件原始内容记录下来，交给自动提取流水线
        （extract -> candidate -> promote -> atom），稍后经 ``memory_recall``
        可召回。记录后立即可用 ``memory_search_raw`` 查询。The source marker
        is stored in ``payload.source``; the caller id (from ``X-Octop-User-Id``
        header or ``user`` arg) is stored on the raw event for per-user
        traceability.

        If ``session_id`` is omitted it is derived as ``ext:{source}:{user}``
        so external callers without an Octop native session still get their raw
        events grouped and distilled into atoms (extraction groups by session).

        Example::

            memory_capture(
                content="user reported: the report panel banner is not rendering",
                source="review-bot",
            )
            # -> {"event_id": "...", "recorded": true, "extract_scheduled": true, ...}
            # later: memory_recall(query="report panel banner not rendering")

        Args:
            content: the raw conversation / event text.
            source: who/what recorded it, for traceability.
            session_id: optional stable session id (e.g. caller name) so the
                extraction pipeline can group events by session. When omitted,
                derived as ``ext:{source}:{user}``.
            user: optional caller id (overrides the ``X-Octop-User-Id`` header).
            ctx: injected MCP context (reads ``X-Octop-User-Id`` header).
        """
        caller = user or _caller_user(ctx)
        effective_session = session_id or _derive_session(source, caller)
        memory = _memory()
        raw = memory.add_raw(
            content,
            event_type="manual",
            host="mcp-external",
            session_id=effective_session,
            user=caller or None,
            payload={"source": source},
        )
        extract_scheduled = _trigger_extract(server, agent_id, effective_session)
        return {
            "event_id": raw.id,
            "source": source,
            "user": caller or None,
            "session_id": effective_session,
            "recorded": True,
            "extract_scheduled": extract_scheduled,
            "note": (
                "raw (L0) event recorded; visible now via memory_search_raw, "
                "recallable via memory_recall after the extraction pipeline "
                "promotes it to an atom"
            ),
        }

    @mcp.tool()
    def memory_search_raw(
        query: str,
        limit: int = 10,
        user: str | None = None,
        ctx: Context | None = None,  # type: ignore[type-arg]
    ) -> dict[str, Any]:
        """FTS-search L0 raw events of this expert (capture visible immediately).

        Unlike ``memory_recall`` (which reads atoms), this searches the raw
        event layer, so records written by ``memory_capture`` are visible right
        away, before extraction promotes them.

        Args:
            query: keywords to match against raw event content.
            limit: max number of events to return.
            user: optional caller id (overrides the ``X-Octop-User-Id`` header);
                returned in ``caller`` for per-caller traceability.
            ctx: injected MCP context (reads ``X-Octop-User-Id`` header).
        """
        caller = user or _caller_user(ctx)
        memory = _memory()
        events = memory.search_raw(query, limit=limit)
        return {
            "events": [
                {
                    "event_id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "session_id": e.session_id,
                    "user": e.user,
                    "source": (e.payload or {}).get("source") if e.payload else None,
                    "content": e.content,
                }
                for e in events
            ],
            "count": len(events),
            "caller": caller or None,
        }

    @mcp.tool()
    def memory_update(
        atom_id: str,
        new_content: str,
        source: str,
        note: str = "mcp update",
        user: str | None = None,
        ctx: Context | None = None,  # type: ignore[type-arg]
    ) -> dict[str, Any]:
        """Update a memory: deprecate the old atom and persist the new fact.

        **显式更新**（非日常）：仅当已知旧记忆已过时、需要替换时才调用
        （如用户纠正了一个事实）。旧 atom 标记 deprecated，新事实立即经
        ``memory_recall`` 可召回。日常纠错也可以走 ``memory_capture`` 让
        提取管线处理。

        Args:
            atom_id: id of the atom to supersede.
            new_content: the replacement fact.
            source: who/what updated it, for traceability.
            note: deprecation note.
            user: optional caller id (overrides the ``X-Octop-User-Id`` header).
            ctx: injected MCP context (reads ``X-Octop-User-Id`` header).
        """
        caller = user or _caller_user(ctx)
        memory = _memory()
        deprecated = memory.deprecate_atom(atom_id, actor="user", note=note)
        node = memory.store(
            new_content,
            metadata={
                "source": source,
                "supersedes": atom_id,
                **({"user": caller} if caller else {}),
            },
        )
        return {
            "deprecated": deprecated,
            "deprecated_atom_id": atom_id,
            "new_node_id": node.id,
            "source": source,
            "user": caller or None,
        }

    return mcp


def _trigger_extract(server: OctopServer, agent_id: str, session_id: str | None) -> bool:
    """Best-effort: asynchronously trigger the agent's memory extraction.

    Internal-network enhancement (not part of the community PR): raw events
    written by MCP capture are not in the harness-agent extractor's tracked
    sessions, so they would never be distilled into atoms. Reuse the agent's
    in-process ``MemoryService`` (with the agent's configured extraction LLM)
    via ``agent._memory_runtime.service`` (no public entrypoint; best-effort).
    Returns whether an extract task was scheduled.
    """
    import asyncio

    if not session_id:
        return False
    try:
        runtime_server = server.app_runtime
        assert runtime_server is not None, "app_runtime required for memory extract"
        agent = runtime_server.agent_registry.get_agent(agent_id)
        runtime = getattr(agent, "_memory_runtime", None)
        service = getattr(runtime, "service", None) if runtime else None
        if service is None:
            return False

        async def _extract() -> None:
            try:
                await asyncio.to_thread(
                    service.extract,
                    session_id,
                    incremental=True,
                    promote=True,
                    regen_pages=True,
                )
            except Exception:
                logger.warning(
                    "memory extract failed for session %s", session_id, exc_info=True
                )

        asyncio.create_task(_extract())
        return True
    except Exception:
        logger.debug("memory extract trigger skipped for agent %s", agent_id, exc_info=True)
        return False


def _memory_mcp_token() -> str | None:
    """Read the MCP auth token (empty string treated as unconfigured)."""
    return (os.environ.get("OCTOP_MEMORY_MCP_TOKEN") or "").strip() or None


class _TokenAuthMiddleware:
    """ASGI middleware enforcing ``Authorization: Bearer`` or ``X-Octop-Memory-Token``."""

    def __init__(self, app: Any, token: str) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        provided = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if not provided:
            provided = headers.get("x-octop-memory-token", "").strip()

        if provided != self._token:
            body = b'{"error":"unauthorized"}'
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        await self._app(scope, receive, send)


class _AgentRouter:
    """ASGI dispatcher routing to the per-expert MCP app by ``X-Octop-Agent-Id`` header."""

    def __init__(self, mcp_apps: dict[str, Any]) -> None:
        self._mcp_apps = mcp_apps

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return  # lifespan is wired into the host FastAPI manually; http only here

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        agent_id = headers.get("x-octop-agent-id", "").strip()
        target = self._mcp_apps.get(agent_id)
        if target is None:
            body = b'{"error":"missing or unknown agent_id (X-Octop-Agent-Id)"}'
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return
        # 把调用者 user id 写入 contextvar，供工具读取（stateless HTTP 下
        # mcp SDK 不提供 ctx.request_context）。
        user = headers.get("x-octop-user-id", "").strip()
        token_cv = _current_caller_user.set(user)
        try:
            await target(scope, receive, send)
        finally:
            _current_caller_user.reset(token_cv)


def mount_memory_mcp(app: Any, server: OctopServer) -> list[Any]:
    """Mount the memory MCP endpoint at ``/mcp/memory``; the expert is selected
    per connection via the ``X-Octop-Agent-Id`` header (one connection binds one
    expert; the URL stays uniform and does not leak expert ids).

    Does not mount when ``OCTOP_MEMORY_MCP_TOKEN`` is unset (fail-closed).
    Returns the session managers that must be initialized in the host FastAPI
    lifespan (``streamable_http_app`` task groups depend on it).
    """
    token = _memory_mcp_token()
    if token is None:
        return []

    managers: list[Any] = []
    mcp_apps: dict[str, Any] = {}
    services = server.services
    assert services is not None, "server.services required for memory MCP mount"
    rows = services.agent_repo.list_all(include_disabled=False)
    for row in rows:
        agent_id = row.agent_id
        mcp = build_memory_mcp(server, agent_id)
        mcp_apps[agent_id] = mcp.streamable_http_app()
        managers.append(mcp._session_manager)

    app.mount("/mcp/memory", _TokenAuthMiddleware(_AgentRouter(mcp_apps), token))
    return managers


__all__ = ["build_memory_mcp", "mount_memory_mcp"]
