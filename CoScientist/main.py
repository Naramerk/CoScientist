"""
CoScientist - Main entry point

Runs the multi-agent scientific discovery pipeline:
- Hypothesis generation
- Research
- Experimentation (FEDOT)
- Orchestration
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
from typing import Optional
import logging
from uuid import uuid4

from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.genai import types

from CoScientist.config import get_settings
from CoScientist.agents import orchestrator_agent, root_agent
from CoScientist.agents.callbacks import cleanup_uploaded_papers
from CoScientist.tools.fedot_artifact_handoff import fetch_artifact_table
from CoScientist.hitl.tool import hitl_toolset
from CoScientist.hitl import (
    AbstractHITLHandler,
    HITLRequest,
    HITLResponse,
)

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compaction_config():
    """Build the events-compaction config: when an agent's prompt grows past the
    token threshold, ADK summarizes older events (with the agent's own model),
    keeping the last N raw events. Disable with AGENT_CONTEXT_TOKEN_THRESHOLD=0.
    """
    try:
        from google.adk.apps.app import EventsCompactionConfig
        threshold = int(os.getenv("AGENT_CONTEXT_TOKEN_THRESHOLD", "150000"))
        if threshold <= 0:
            return None
        return EventsCompactionConfig(
            compaction_interval=int(os.getenv("AGENT_COMPACTION_INTERVAL", "15")),
            overlap_size=int(os.getenv("AGENT_COMPACTION_OVERLAP", "2")),
            token_threshold=threshold,
            event_retention_size=int(os.getenv("AGENT_CONTEXT_RETENTION", "12")),
        )
    except Exception:  # noqa: BLE001 — compaction is best-effort, never block startup
        return None


def reset_session_state(
    user_id: str,
    session_id: str,
    *,
    reset_research: Optional[bool] = None,
) -> None:
    """Explicitly reset graph state for one session only.

    TaskTracker state belongs to ADK session state and semantic memory is global
    across the installation, so neither is touched here.
    """
    try:
        from CoScientist.graph.memory import reset_knowledge_graph
        reset_knowledge_graph(user_id=user_id, session_id=session_id)
    except Exception:  # noqa: BLE001
        pass

    try:
        should_reset = (
            get_settings().research_graph.reset_on_session
            if reset_research is None
            else reset_research
        )
        if should_reset:
            from CoScientist.graph.research.store import get_research_graph
            get_research_graph(
                user_id=user_id,
                session_id=session_id,
            ).reset(archive=True)
    except Exception:  # noqa: BLE001
        pass


def _render_table_preview(table: dict) -> str:
    """Render a parsed artifact table (``{columns, rows}``) as a compact text preview.

    Column-agnostic: shows whatever headers the artifact actually has (rows are
    already bounded by the handoff parser), so the answer is built from the ACTUAL
    file contents — not a bare link or unverified prose — with no baked-in column
    list (F010.A6).
    """
    columns = [str(c) for c in (table.get("columns") or [])]
    rows = table.get("rows") or []
    if not columns or not rows:
        return ""
    lines = [" | ".join(columns)]
    for row in rows:
        lines.append(" | ".join(str(row.get(c, "")) for c in columns))
    return "\n".join(lines)


async def finalize_response_with_artifacts(
    *,
    session_service,
    app_name: str,
    user_id: str,
    session_id: str,
    final_response: str,
    run_error: Optional[Exception] = None,
) -> str:
    """Append any captured S3 FEDOT artifacts not already reflected in the answer.

    Deterministic finalizer (F010.A5/A6): the orchestrator LLM sometimes drops a
    successfully-generated result. The real molecules live behind a presigned S3
    URL that fedot_tool captured into state['fedot_artifacts']. Kept as a
    standalone function (rather than inlined into ``CoScientistManager.run``) so
    it stays unit-testable without spinning up the full agent stack; currently
    only the CLI calls it, but the same partial-delivery guarantee is available
    to any other entry point (e.g. a web socket handler) that adopts it later.
    """
    try:
        session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id,
        )
        state = (getattr(session, "state", None) or {}) if session else {}
    except Exception:
        state = {}

    # Prefer tables already parsed by fedot_tool; fall back to a one-off fetch.
    by_url = {
        t.get("url"): t
        for t in (state.get("fedot_artifact_tables") or [])
        if isinstance(t, dict) and t.get("url")
    }

    blocks = []
    for artifact in state.get("fedot_artifacts") or []:
        url = artifact.get("url")
        if not url:
            continue
        table = by_url.get(url) or await asyncio.to_thread(fetch_artifact_table, url)
        preview = _render_table_preview(table) if table else ""
        url_present = url in (final_response or "")
        preview_present = bool(preview) and preview in (final_response or "")
        # Nothing new to add: the link is already shown and either the rows are
        # too, or there are none to show.
        if url_present and (preview_present or not preview):
            continue
        count = artifact.get("generated_count")
        tag = f" ({count} molecules)" if count else ""
        head = (
            f"**Generated molecules{tag}**"
            if url_present
            else f"**Generated molecules{tag}** — [download full results]({url})"
        )
        parts = [head] + ([f"```\n{preview}\n```"] if preview and not preview_present else [])
        blocks.append("\n".join(parts))

    if blocks:
        final_response = (final_response or "").rstrip() + "\n\n---\n" + "\n\n".join(blocks)

    if not (final_response or "").strip():
        if run_error is not None:
            return (
                f"The run stopped early ({type(run_error).__name__}) before producing a result, "
                "and no partial artifacts were captured. This is usually a slow MCP tool hitting "
                "its timeout or a transient model/network error — please retry."
            )
        return (
            "I couldn't complete this request within the available steps — the orchestrator "
            "did not reach a tool that produced a result. Please retry or narrow the request."
        )
    return final_response


class CoScientistManager:
    """
    Main manager for CoScientist (ADK-based execution).
    """

    def __init__(
        self,
        app_name: str = "coscientist_app",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        hitl_handler: Optional[AbstractHITLHandler] = None,
        session_service: Optional[BaseSessionService] = None,
    ):
        self.app_name = app_name
        self.user_id = user_id or f"user_{uuid4().hex}"
        self.session_id = session_id or f"session_{uuid4().hex}"

        # Web mode injects one shared service so managers can reopen existing
        # sessions. CLI mode falls back to a private in-memory service.
        self.session_service: Optional[BaseSessionService] = session_service
        self.runner: Optional[Runner] = None
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

        # HITL setup
        self._hitl_handler = hitl_handler


    async def initialize(self):
        """Initialize session + runner."""
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return

            if self.session_service is None:
                self.session_service = InMemorySessionService()

            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id,
            )
            if session is None:
                from CoScientist.graph.session_scope import (
                    GRAPH_SCOPE_SESSION_KEY,
                    GRAPH_SCOPE_USER_KEY,
                )
                await self.session_service.create_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    state={
                        "active_tasks": [],
                        GRAPH_SCOPE_USER_KEY: self.user_id,
                        GRAPH_SCOPE_SESSION_KEY: self.session_id,
                    },
                )
            from google.adk.apps.app import App
            from CoScientist.logging.event_logger import EventLoggerPlugin
            from CoScientist.graph.plugin import GraphMemoryPlugin
            from CoScientist.graph.research.validator import BackgroundValidatorPlugin
            from CoScientist.agents.truncation_plugin import ToolResultTruncationPlugin

            app = App(
                name=self.app_name,
                root_agent=root_agent,
                plugins=[
                    EventLoggerPlugin(),
                    GraphMemoryPlugin(),
                    BackgroundValidatorPlugin(),
                    # Keep truncation last so observers receive full results.
                    ToolResultTruncationPlugin(),
                ],
                events_compaction_config=_compaction_config(),
            )
            self.runner = Runner(app=app, session_service=self.session_service)

            if self._hitl_handler:
                hitl_toolset._handler = self._hitl_handler

            self._initialized = True

    async def run(self, query: str, verbose: bool = True) -> str:
        """
        Execute a query through the orchestrator agent.

        Args:
            query: user query
            verbose: whether to print events

        Returns:
            Final agent response
        """
        await self.initialize()

        content = types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )

        final_response = "No response"
        run_error = None

        # Partial delivery (F015a.A4 #2): a mid-run failure — notably an MCP 300s
        # timeout / McpError on a slow tool — must NOT discard results already
        # captured at the tool boundary (state['fedot_artifacts']). Swallow it here
        # and fall through to the deterministic finalizer below, which surfaces those
        # artifacts so the user still gets the molecules produced before the stall.
        try:
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content,
                # ADK caps a run at 500 LLM calls by default, which a long
                # autonomous research run overruns mid-work; lift the ceiling so
                # one prompt can drive the whole job (finite, as a cost backstop).
                run_config=RunConfig(
                    max_llm_calls=get_settings().orchestrator.max_llm_calls
                ),
            ):
                if verbose:
                    print(
                        f"[Event] {event.author} | {type(event).__name__} | Final={event.is_final_response()}"
                    )

                if event.is_final_response():
                    if event.content and event.content.parts:
                        parts = event.content.parts
                        # Thinking models emit a separate `thought` part before the
                        # answer; parts[0] is often that reasoning. Prefer the
                        # non-thought answer text, falling back to any text so we
                        # never drop the response entirely.
                        answer = "\n".join(
                            p.text for p in parts
                            if getattr(p, "text", None) and not getattr(p, "thought", False)
                        )
                        final_response = answer or "\n".join(
                            p.text for p in parts if getattr(p, "text", None)
                        ) or ""
                    elif event.actions and event.actions.escalate:
                        final_response = f"Escalation: {getattr(event, 'error_message', None) or 'Unknown error'}"
        except Exception as exc:
            run_error = exc
            logger.error(
                f"run loop raised ({type(exc).__name__}: {str(exc)[:200]}); "
                "attempting partial delivery from captured S3 artifacts."
            )

        return await finalize_response_with_artifacts(
            session_service=self.session_service,
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
            final_response=final_response,
            run_error=run_error,
        )

    async def close(self):
        """Cleanup session-related resources and uploaded paper artifacts."""
        if self.runner is not None:
            try:
                await self.runner.close()
            except Exception as exc:  # noqa: BLE001 - continue local cleanup
                logger.error(
                    "Warning: failed to close runner for session %s: %s",
                    self.session_id,
                    exc,
                )
            finally:
                self.runner = None
                self._initialized = False
        try:
            await asyncio.to_thread(cleanup_uploaded_papers, self.user_id, self.session_id)
        except Exception as exc:
            logger.error(f"Warning: failed to cleanup uploaded papers for session {self.session_id}: {exc}")

# Convenience functions
async def create_manager() -> CoScientistManager:
    """Create and initialize a CoScientistManager."""
    manager = CoScientistManager()
    await manager.initialize()
    return manager


# Export public API
__all__ = [
    # Main classes
    "CoScientistManager",
    # Models
    # Functions
    "create_manager"
]

# CLI entrypoint — thin shim. Prefer: python -m CoScientist cli
if __name__ == "__main__":
    from CoScientist.cli import run_repl

    run_repl()
