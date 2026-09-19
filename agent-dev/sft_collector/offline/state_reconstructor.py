"""State Reconstructor — rebuild file states and context from event stream.

For offline extraction: we replay the event sequence through ShadowCopyManager
and ContextAccumulator to reconstruct the exact state at each point in time.
"""



import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.context_accumulator import ContextAccumulator
from ..core.schema import Event, EventKind, Segment
from ..core.shadow_copy import ShadowCopyManager


_OPUS_FILE_PAT = re.compile(r"\.(opus|hpp)$")
_WRITE_CMD_PAT = re.compile(
    r"(?:cat\s*>|tee\s|echo\s.*>|write_file|>\s*)(.+?\.(?:opus|hpp))",
    re.IGNORECASE,
)
_HEREDOC_PAT = re.compile(
    r"<<\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\1",
    re.DOTALL,
)


class StateReconstructor:
    """Replays events to reconstruct shadow copies and context."""

    def __init__(
        self,
        collection_mode: str = "direction_conditioned",
        harness_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        error_recovery = collection_mode == "error_recovery"
        self.shadow = ShadowCopyManager(error_recovery_mode=error_recovery)
        self.context = ContextAccumulator()
        self._config = harness_config or {}
        self._first_action_recorded = False

        if self._config:
            self.context.on_session_start(
                self._config,
                self._config.get("environment"),
            )

    def replay(self, events: List[Event]) -> Tuple[ShadowCopyManager, ContextAccumulator]:
        """Replay all events, building up shadow copy and context state.

        Returns (shadow_copy_manager, context_accumulator).
        """
        for ev in events:
            self._process_event(ev)
        return self.shadow, self.context

    def replay_segment(
        self, segment: Segment
    ) -> Tuple[ShadowCopyManager, ContextAccumulator]:
        """Replay events within a single segment."""
        for ev in segment.events:
            self._process_event(ev)
        return self.shadow, self.context

    def _process_event(self, ev: Event) -> None:
        if ev.kind == EventKind.SESSION_START:
            cfg = ev.data
            if cfg:
                self.context.on_session_start(cfg, cfg.get("environment"))

        elif ev.kind == EventKind.TOOL_CALL:
            self._on_tool_call(ev)

        elif ev.kind == EventKind.TOOL_RESULT:
            self._on_tool_result(ev)

        elif ev.kind == EventKind.REASONING:
            text = ev.data.get("text", "")
            if text:
                self.context.on_reasoning_block(text)

        elif ev.kind == EventKind.FILE_READ:
            path = ev.data.get("path", "")
            content = ev.data.get("content", "")
            if path and content:
                self.shadow.on_file_read(path, content)
                self.context.on_tool_result("read", path, content)

        elif ev.kind == EventKind.FILE_WRITE:
            path = ev.data.get("path", "")
            content = ev.data.get("content", "")
            if path and content:
                self.shadow.on_file_write(path, content)

        elif ev.kind == EventKind.INBOX:
            sender = ev.data.get("sender", "")
            text = ev.data.get("text", "")
            if sender and text:
                self.context.on_inbox_message(sender, text, ev.timestamp)

    def _on_tool_call(self, ev: Event) -> None:
        name = ev.data.get("name", "")
        args = ev.data.get("arguments", {})

        # Record first action for direction causality
        if not self._first_action_recorded:
            self.context.record_agent_first_action(ev.timestamp)
            self._first_action_recorded = True

        if name == "read":
            path = args.get("file_path", args.get("path", ""))
            if path and _OPUS_FILE_PAT.search(path):
                # We'll capture the content from the tool_result
                ev.data["_opus_read_path"] = path

        elif name in ("pwsh", "bash", "shell"):
            cmd = args.get("command", args.get("cmd", ""))
            m = _WRITE_CMD_PAT.search(cmd)
            if m:
                ev.data["_opus_write_path"] = m.group(1).strip()
                hd = _HEREDOC_PAT.search(cmd)
                if hd:
                    ev.data["_opus_write_content"] = hd.group(2)

    def _on_tool_result(self, ev: Event) -> None:
        content = ev.data.get("content", "")
        if isinstance(content, dict):
            content = str(content)
        if not isinstance(content, str):
            content = str(content)

        # Check if this is a response to an opus file read
        # We need to correlate with the preceding tool_call
        # For simplicity in offline mode, we check content patterns

        tool_name = ev.data.get("_from_tool", "")
        path = ev.data.get("_opus_read_path", "")

        if path:
            self.shadow.on_file_read(path, content)
            self.context.on_tool_result("read", path, content)
        else:
            # Generic tool result processing for context accumulation
            self.context.on_tool_result("", "", content)

        # Detect file write results — use extracted heredoc content if available,
        # skip if only shell stdout (unreliable for file writes)
        write_path = ev.data.get("_opus_write_path", "")
        if write_path:
            write_content = ev.data.get("_opus_write_content", "")
            if write_content:
                self.shadow.on_file_write(write_path, write_content)

        # Detect compile/correctness failures for error-recovery
        if self.shadow.error_recovery_mode:
            if _is_compile_error(content):
                for f in self.shadow.tracked_files():
                    self.shadow.on_compile_fail(f, content, content)
            elif _is_correctness_error(content):
                for f in self.shadow.tracked_files():
                    self.shadow.on_correctness_fail(f, content, content)


def correlate_tool_calls_results(events: List[Event]) -> List[Event]:
    """Pre-process events to link tool_call → tool_result pairs.

    Adds _opus_read_path / _opus_write_path / _from_tool to tool_results
    so the reconstructor can attribute them correctly.
    """
    result = list(events)
    pending_read_path: Optional[str] = None
    pending_write_path: Optional[str] = None
    pending_write_content: Optional[str] = None
    pending_tool_name: Optional[str] = None

    for i, ev in enumerate(result):
        if ev.kind == EventKind.TOOL_CALL:
            name = ev.data.get("name", "")
            args = ev.data.get("arguments", {})
            pending_tool_name = name

            if name == "read":
                path = args.get("file_path", args.get("path", ""))
                if path and _OPUS_FILE_PAT.search(path):
                    pending_read_path = path
                else:
                    pending_read_path = None
                pending_write_path = None
                pending_write_content = None
            elif name in ("pwsh", "bash", "shell"):
                cmd = args.get("command", args.get("cmd", ""))
                m = _WRITE_CMD_PAT.search(cmd)
                if m:
                    pending_write_path = m.group(1).strip()
                    hd = _HEREDOC_PAT.search(cmd)
                    pending_write_content = hd.group(2) if hd else None
                else:
                    pending_write_path = None
                    pending_write_content = None
                pending_read_path = None
            else:
                pending_read_path = None
                pending_write_path = None
                pending_write_content = None

        elif ev.kind == EventKind.TOOL_RESULT:
            if pending_read_path:
                ev.data["_opus_read_path"] = pending_read_path
            if pending_write_path:
                ev.data["_opus_write_path"] = pending_write_path
            if pending_write_content is not None:
                ev.data["_opus_write_content"] = pending_write_content
            if pending_tool_name:
                ev.data["_from_tool"] = pending_tool_name
            pending_read_path = None
            pending_write_path = None
            pending_write_content = None
            pending_tool_name = None

    return result


def _is_compile_error(text: str) -> bool:
    return bool(re.search(
        r"(error:|fatal error|compilation failed|undefined reference|cannot find)",
        text[:2000],
        re.IGNORECASE,
    ))


def _is_correctness_error(text: str) -> bool:
    return bool(re.search(
        r"(FAIL|mismatch|tolerance exceeded|correctness failed|assertion)",
        text[:2000],
        re.IGNORECASE,
    ))
