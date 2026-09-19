"""Online Middleware — Pipeline B: real-time SFT data collection.

NON-INTRUSIVE design:
  - During the agent loop, callbacks only append raw events to an in-memory
    buffer.  No disk I/O, no CPU-intensive processing, no exception propagation.
  - Large content (file reads, tool results) is truncated to bound memory.
  - Session-end flush is best-effort and exception-isolated.
  - All heavy processing is deferred to explicit post-session process_buffer().
"""



import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_CONTENT_BYTES = 50 * 1024    # truncate individual content fields to 50 KB
MAX_SOURCE_BYTES = 512 * 1024    # source files get a higher limit (512 KB)
MAX_EVENTS = 100000              # drop events beyond this count


class SFTCollectorMiddleware:
    """Lightweight event recorder that never blocks the agent loop.

    During the session it only appends events to a bounded in-memory list.
    On session end it dumps the buffer to a JSONL file (best-effort),
    which can then be processed by the offline pipeline.
    """

    def __init__(
        self,
        run_dir,   # type: str
        config,    # type: Dict[str, Any]
    ):
        # type: (...) -> None
        self.run_dir = run_dir
        self.config = config
        self._events = []           # type: List[Dict[str, Any]]
        self._session_cfg = {}      # type: Dict[str, Any]
        self._started = 0.0
        self._flushed = False
        self._dropped = 0

    # ---- attach ----

    def attach(self, harness):
        # type: (Any) -> None
        """Register with the harness event system."""
        from .event_interceptor import EventInterceptor

        interceptor = EventInterceptor()
        interceptor.on_session_start_cb = self._on_start
        interceptor.on_session_end_cb = self._on_end
        interceptor.on_tool_call_cb = self._on_tool_call
        interceptor.on_tool_result_cb = self._on_tool_result
        interceptor.on_reasoning_cb = self._on_reasoning
        interceptor.on_inbox_cb = self._on_inbox
        interceptor.on_file_read_cb = self._on_file_read
        interceptor.on_file_write_cb = self._on_file_write
        interceptor.on_analysis_trigger = self._on_present
        interceptor.on_opus_trigger = self._on_opus_write

        interceptor.register(harness)
        logger.info("SFT Collector attached (buffer-only mode)")

    # ---- callbacks: append-only, no processing ----

    def _on_start(self, config):
        # type: (Dict[str, Any]) -> None
        try:
            self._session_cfg = config
            self._started = time.time()
            self._append("session_start", config)
        except Exception:
            pass

    def _on_end(self):
        # type: () -> None
        try:
            self._append("session_end", {})
        except Exception:
            pass
        # flush is best-effort, fully isolated
        try:
            self._flush()
        except Exception:
            logger.debug("flush on session_end failed", exc_info=True)

    def _on_tool_call(self, name, args, raw):
        # type: (str, Dict[str, Any], Dict[str, Any]) -> None
        try:
            self._append("tool_call", {"name": name, "arguments": args})
        except Exception:
            pass

    def _on_tool_result(self, content, raw):
        # type: (str, Dict[str, Any]) -> None
        try:
            self._append("tool_result", {
                "content": _truncate(content),
            })
        except Exception:
            pass

    def _on_reasoning(self, text, raw):
        # type: (str, Dict[str, Any]) -> None
        try:
            self._append("reasoning", {"text": _truncate(text)})
        except Exception:
            pass

    def _on_inbox(self, sender, text, timestamp):
        # type: (str, str, float) -> None
        try:
            self._append("inbox", {
                "sender": sender,
                "text": _truncate(text),
                "timestamp": timestamp,
            })
        except Exception:
            pass

    def _on_file_read(self, path, content):
        # type: (str, str) -> None
        try:
            self._append("file_read", {
                "path": path,
                "content": _truncate(content, MAX_SOURCE_BYTES),
            })
        except Exception:
            pass

    def _on_file_write(self, path, content):
        # type: (str, str) -> None
        try:
            self._append("file_write", {
                "path": path,
                "content": _truncate(content, MAX_SOURCE_BYTES),
            })
        except Exception:
            pass

    def _on_present(self, deliverable):
        # type: (str) -> None
        try:
            self._append("present", {"deliverable": _truncate(deliverable)})
        except Exception:
            pass

    def _on_opus_write(self, path):
        # type: (str) -> None
        try:
            self._append("opus_write", {"path": path})
        except Exception:
            pass

    # ---- internal helpers ----

    def _append(self, kind, data):
        # type: (str, Dict[str, Any]) -> None
        if len(self._events) >= MAX_EVENTS:
            self._dropped += 1
            return
        self._events.append({
            "type": kind,
            "timestamp": time.time(),
            "data": data,
        })

    def _flush(self):
        # type: () -> None
        """Dump buffered events to a JSONL file for offline processing.

        Best-effort: failure is logged and silently ignored.
        """
        if self._flushed or not self._events:
            return
        self._flushed = True

        import json

        out_dir = os.path.join(self.run_dir, "sft_collected")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            logger.debug("cannot create output dir %s", out_dir, exc_info=True)
            return

        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        out_path = os.path.join(out_dir, "online_buffer_{}.jsonl".format(ts))

        try:
            with open(out_path, "w") as f:
                for ev in self._events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            if self._dropped:
                logger.warning(
                    "Flushed %d events (%d dropped due to buffer limit) to %s",
                    len(self._events), self._dropped, out_path,
                )
            else:
                logger.info(
                    "Flushed %d events to %s", len(self._events), out_path,
                )
        except OSError:
            logger.debug("failed to write buffer to %s", out_path, exc_info=True)

    # ---- post-session batch processing (call explicitly) ----

    def process_buffer(self, skip_verify=False):
        # type: (bool) -> Dict[str, int]
        """Run the full offline extraction pipeline on the buffered events.

        Call this AFTER the agent loop has finished — never during.
        """
        from ..offline.offline_agent import OfflineAgent, OfflineExtractionConfig

        self._flush()

        out_dir = os.path.join(self.run_dir, "sft_collected")
        if not os.path.isdir(out_dir):
            return {"analysis": 0, "opus": 0, "concept": 0,
                    "dead_end": 0, "rejected": 0}

        jsonl_files = [
            os.path.join(out_dir, f)
            for f in os.listdir(out_dir)
            if f.startswith("online_buffer_") and f.endswith(".jsonl")
        ]
        if not jsonl_files:
            return {"analysis": 0, "opus": 0, "concept": 0,
                    "dead_end": 0, "rejected": 0}

        config = OfflineExtractionConfig(
            collection_mode=self.config.get("collection_mode", "direction_conditioned"),
            harness_config=self.config,
            workspace_dir=self.config.get("workspace_dir", ""),
            skip_verify=skip_verify,
        )
        agent = OfflineAgent(run_dir=self.run_dir, config=config)

        totals = {"analysis": 0, "opus": 0, "concept": 0,
                  "dead_end": 0, "rejected": 0}
        for path in jsonl_files:
            result = agent.process_trajectory(path)
            for k in totals:
                totals[k] += result.get(k, 0)

        return totals


def _truncate(text, limit=MAX_CONTENT_BYTES):
    # type: (str, int) -> str
    if not isinstance(text, str):
        text = str(text)
    if len(text) > limit:
        return text[:limit] + "\n... [truncated, {} bytes total]".format(len(text))
    return text
