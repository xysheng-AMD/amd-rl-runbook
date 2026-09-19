"""Event Interceptor — routes Harness events to collector callbacks.

Every handler is wrapped in try/except so that interceptor failures
NEVER propagate to the Harness or agent loop.  No data transformation
(JSON parsing, datetime conversion) happens here — raw data is passed
through as-is to keep the hot path minimal.
"""



import logging
import re
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_OPUS_FILE_PAT = re.compile(r"\.(opus|hpp)$")
_WRITE_CMD_PAT = re.compile(
    r"(?:cat\s*>|tee\s|echo\s.*>|write_file|>\s*)(.+?\.(?:opus|hpp))",
    re.IGNORECASE,
)
_HEREDOC_PAT = re.compile(
    r"<<\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\1",
    re.DOTALL,
)


class EventInterceptor:
    """Intercepts Harness events and routes them to collector callbacks.

    All handlers swallow exceptions — the agent loop must never be
    affected by collector failures.
    """

    def __init__(self):
        # type: () -> None
        self.on_analysis_trigger = None   # type: Optional[Callable]
        self.on_opus_trigger = None       # type: Optional[Callable]
        self.on_tool_call_cb = None       # type: Optional[Callable]
        self.on_tool_result_cb = None     # type: Optional[Callable]
        self.on_reasoning_cb = None       # type: Optional[Callable]
        self.on_inbox_cb = None           # type: Optional[Callable]
        self.on_file_read_cb = None       # type: Optional[Callable]
        self.on_file_write_cb = None      # type: Optional[Callable]
        self.on_session_start_cb = None   # type: Optional[Callable]
        self.on_session_end_cb = None     # type: Optional[Callable]

        self._pending_opus_write = None   # type: Optional[str]
        self._pending_read_path = None    # type: Optional[str]
        self._pending_write_content = None  # type: Optional[str]

    def register(self, harness):
        # type: (Any) -> None
        """Register event hooks with the Harness."""
        try:
            if hasattr(harness, "add_middleware"):
                harness.add_middleware(self)
            elif hasattr(harness, "on_event"):
                harness.on_event(self.handle_event)
            elif hasattr(harness, "register_hook"):
                harness.register_hook("tool_call", self._on_tool_call)
                harness.register_hook("tool_result", self._on_tool_result)
                harness.register_hook("reasoning", self._on_reasoning)
                harness.register_hook("present", self._on_present)
                harness.register_hook("inbox", self._on_inbox)
                harness.register_hook("session_start", self._on_session_start)
                harness.register_hook("session_end", self._on_session_end)
            else:
                logger.warning(
                    "Harness type %s has no known event API",
                    type(harness).__name__,
                )
        except Exception:
            logger.debug("register failed", exc_info=True)

    def handle_event(self, event_type, data):
        # type: (str, Dict[str, Any]) -> None
        """Unified event handler — routes by type."""
        handlers = {
            "tool_call": self._on_tool_call,
            "tool_use": self._on_tool_call,
            "tool_result": self._on_tool_result,
            "reasoning": self._on_reasoning,
            "thinking": self._on_reasoning,
            "present": self._on_present,
            "inbox": self._on_inbox,
            "agent/inbox/spliced": self._on_inbox,
            "session_start": self._on_session_start,
            "session_end": self._on_session_end,
        }
        handler = handlers.get(event_type)
        if handler:
            try:
                handler(data)
            except Exception:
                logger.debug("event %s handler error", event_type, exc_info=True)

    # ---- event handlers (all wrapped in try/except) ----

    def _on_tool_call(self, data):
        # type: (Dict[str, Any]) -> None
        try:
            name = data.get("name", "")
            args = data.get("arguments", data.get("input", {}))
            if not isinstance(args, dict):
                args = {"raw": args}

            if name == "read":
                path = args.get("file_path", args.get("path", ""))
                if path and _OPUS_FILE_PAT.search(path):
                    self._pending_read_path = path

            elif name in ("pwsh", "bash", "shell"):
                cmd = args.get("command", args.get("cmd", ""))
                if isinstance(cmd, str):
                    m = _WRITE_CMD_PAT.search(cmd)
                    if m:
                        self._pending_opus_write = m.group(1).strip()
                        hd = _HEREDOC_PAT.search(cmd)
                        self._pending_write_content = hd.group(2) if hd else None

            if self.on_tool_call_cb:
                self.on_tool_call_cb(name, args, data)
        except Exception:
            logger.debug("_on_tool_call error", exc_info=True)

    def _on_tool_result(self, data):
        # type: (Dict[str, Any]) -> None
        try:
            content = data.get("content", data.get("output", ""))
            if not isinstance(content, str):
                content = str(content)

            if self._pending_read_path:
                if self.on_file_read_cb:
                    self.on_file_read_cb(self._pending_read_path, content)
                self._pending_read_path = None

            if self._pending_opus_write:
                write_content = self._pending_write_content if self._pending_write_content else content
                if self.on_file_write_cb:
                    self.on_file_write_cb(self._pending_opus_write, write_content)
                if self.on_opus_trigger:
                    self.on_opus_trigger(self._pending_opus_write)
                self._pending_opus_write = None
                self._pending_write_content = None

            if self.on_tool_result_cb:
                self.on_tool_result_cb(content, data)
        except Exception:
            logger.debug("_on_tool_result error", exc_info=True)

    def _on_reasoning(self, data):
        # type: (Dict[str, Any]) -> None
        try:
            text = data.get("text", data.get("thinking", ""))
            if text and self.on_reasoning_cb:
                self.on_reasoning_cb(text, data)
        except Exception:
            logger.debug("_on_reasoning error", exc_info=True)

    def _on_present(self, data):
        # type: (Dict[str, Any]) -> None
        try:
            deliverable = data.get("deliverable", data.get("content", ""))
            if deliverable and self.on_analysis_trigger:
                self.on_analysis_trigger(deliverable)
        except Exception:
            logger.debug("_on_present error", exc_info=True)

    def _on_inbox(self, data):
        # type: (Dict[str, Any]) -> None
        try:
            sender = data.get("sender", data.get("from", ""))
            text = data.get("text", data.get("content", ""))
            timestamp = data.get("timestamp", 0)
            if not isinstance(timestamp, (int, float)):
                timestamp = 0
            if sender and text and self.on_inbox_cb:
                self.on_inbox_cb(sender, text, float(timestamp))
        except Exception:
            logger.debug("_on_inbox error", exc_info=True)

    def _on_session_start(self, data):
        # type: (Dict[str, Any]) -> None
        try:
            if self.on_session_start_cb:
                self.on_session_start_cb(data)
        except Exception:
            logger.debug("_on_session_start error", exc_info=True)

    def _on_session_end(self, data):
        # type: (Dict[str, Any]) -> None
        try:
            if self.on_session_end_cb:
                self.on_session_end_cb()
        except Exception:
            logger.debug("_on_session_end error", exc_info=True)
