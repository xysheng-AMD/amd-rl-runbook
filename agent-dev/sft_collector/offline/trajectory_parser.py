"""Trajectory Parser — reads session.v3.jsonl and emits structured events.

session.v3.jsonl line format (DeepSeek Harness):
Each line is a JSON object with at least:
  - "type": event type string
  - "timestamp": unix timestamp or ISO string
  - "data": payload (varies by type)

Tool calls:  type="tool_call"   data={name, arguments, id}
Tool results: type="tool_result" data={content, tool_call_id}
Reasoning:   type="reasoning"   data={text}
Text output: type="text"        data={text}
Inbox:       type="inbox"       data={sender, text} or type="agent/inbox/spliced"
Present:     type="present"     data={deliverable}
Session:     type="session_start" / "session_end"
"""



import json
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional

from ..core.schema import Event, EventKind, Segment

logger = logging.getLogger(__name__)

# Map raw JSONL type strings to EventKind
_TYPE_MAP: Dict[str, EventKind] = {
    "tool_call": EventKind.TOOL_CALL,
    "tool_use": EventKind.TOOL_CALL,
    "tool_result": EventKind.TOOL_RESULT,
    "reasoning": EventKind.REASONING,
    "thinking": EventKind.REASONING,
    "text": EventKind.TEXT_OUTPUT,
    "text_output": EventKind.TEXT_OUTPUT,
    "inbox": EventKind.INBOX,
    "agent/inbox/spliced": EventKind.INBOX,
    "present": EventKind.PRESENT,
    "session_start": EventKind.SESSION_START,
    "session_end": EventKind.SESSION_END,
    "file_read": EventKind.FILE_READ,
    "file_write": EventKind.FILE_WRITE,
}


def parse_events(jsonl_path: str) -> List[Event]:
    """Parse a session.v3.jsonl file into a list of Events."""
    events: List[Event] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at %s:%d", jsonl_path, lineno)
                continue

            event = _parse_line(raw, lineno)
            if event:
                events.append(event)

    logger.info("Parsed %d events from %s", len(events), jsonl_path)
    return events


def _parse_line(raw: Dict[str, Any], lineno: int) -> Optional[Event]:
    """Convert a raw JSONL line to an Event."""
    # Try to determine event type
    raw_type = _detect_type(raw)
    if not raw_type:
        return None

    kind = _TYPE_MAP.get(raw_type)
    if kind is None:
        return None

    ts = _parse_timestamp(raw)
    data = _extract_data(raw, kind)

    return Event(kind=kind, timestamp=ts, data=data)


def _detect_type(raw: Dict[str, Any]) -> Optional[str]:
    """Detect event type from various JSONL formats."""
    # Direct "type" field
    if "type" in raw:
        return raw["type"]

    # Claude-style: role + content blocks
    if "role" in raw:
        role = raw["role"]
        if role == "assistant":
            content = raw.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            return "reasoning"
                        if block.get("type") == "tool_use":
                            return "tool_call"
                        if block.get("type") == "text":
                            return "text"
            if raw.get("tool_calls"):
                return "tool_call"
            return "text"
        elif role == "tool":
            return "tool_result"
        elif role == "user":
            return "inbox"

    # agent_inbox style
    if "sender" in raw and "text" in raw:
        return "inbox"

    return None


def _parse_timestamp(raw: Dict[str, Any]) -> float:
    """Extract timestamp, defaulting to 0."""
    ts = raw.get("timestamp", raw.get("ts", raw.get("created_at", 0)))
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            import datetime
            dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, AttributeError):
            return 0.0
    return 0.0


def _extract_data(raw: Dict[str, Any], kind: EventKind) -> Dict[str, Any]:
    """Extract relevant data fields based on event kind."""
    data = raw.get("data", {})
    if isinstance(data, dict) and data:
        return data

    # Build data from raw fields
    if kind == EventKind.TOOL_CALL:
        # Claude format: content blocks with tool_use
        content = raw.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return {
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                        "id": block.get("id", ""),
                    }
        # Separate tool_calls field
        tool_calls = raw.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
            return {
                "name": tc.get("name", tc.get("function", {}).get("name", "")),
                "arguments": tc.get("arguments", tc.get("function", {}).get("arguments", {})),
                "id": tc.get("id", ""),
            }
        return {"name": raw.get("name", ""), "arguments": raw.get("arguments", {})}

    if kind == EventKind.TOOL_RESULT:
        return {"content": raw.get("content", raw.get("output", "")),
                "tool_call_id": raw.get("tool_call_id", raw.get("tool_use_id", ""))}

    if kind == EventKind.REASONING:
        content = raw.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    return {"text": block.get("text", "")}
        return {"text": raw.get("text", raw.get("thinking", ""))}

    if kind == EventKind.TEXT_OUTPUT:
        content = raw.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return {"text": block.get("text", "")}
        return {"text": raw.get("text", raw.get("content", ""))}

    if kind == EventKind.INBOX:
        return {
            "sender": raw.get("sender", raw.get("from", "")),
            "text": raw.get("text", raw.get("content", "")),
        }

    if kind == EventKind.PRESENT:
        return {"deliverable": raw.get("deliverable", raw.get("content", ""))}

    if kind in (EventKind.FILE_READ, EventKind.FILE_WRITE):
        return {"path": raw.get("path", ""), "content": raw.get("content", "")}

    return raw


def segment_by_present(events: List[Event]) -> List[Segment]:
    """Split events into segments by present() boundaries."""
    segments: List[Segment] = []
    current_events: List[Event] = []
    start_time = 0.0

    for ev in events:
        if ev.kind == EventKind.SESSION_START and not current_events:
            start_time = ev.timestamp
            continue

        current_events.append(ev)

        if ev.kind == EventKind.PRESENT:
            seg = Segment(
                events=current_events,
                start_time=start_time or (current_events[0].timestamp if current_events else 0),
                end_time=ev.timestamp,
                deliverable=ev.data.get("deliverable", ""),
            )
            segments.append(seg)
            start_time = ev.timestamp
            current_events = []

    # Remaining events after last present()
    if current_events:
        # Check if there's substantial content (not just session_end)
        has_content = any(
            e.kind in (EventKind.TOOL_CALL, EventKind.REASONING, EventKind.TEXT_OUTPUT)
            for e in current_events
        )
        if has_content:
            seg = Segment(
                events=current_events,
                start_time=start_time or (current_events[0].timestamp if current_events else 0),
                end_time=current_events[-1].timestamp,
                deliverable=None,
            )
            segments.append(seg)

    logger.info("Created %d segments from %d events", len(segments), len(events))
    return segments


def find_trajectory_files(directory: str) -> List[str]:
    """Find all .jsonl trajectory files in a directory tree.

    Accepts any .jsonl file — session*.jsonl, trajectory*.jsonl,
    online_buffer_*.jsonl, or other naming conventions.
    """
    if os.path.isfile(directory):
        return [directory] if directory.endswith(".jsonl") else []
    results: List[str] = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".jsonl"):
                results.append(os.path.join(root, fname))
    results.sort()
    return results
