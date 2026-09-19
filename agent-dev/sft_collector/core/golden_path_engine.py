"""Golden Path Engine — §4.4: heuristic-based golden/dead-end classification.

Five rules applied to each tool call in a segment:
1. Reference detection — reasoning cites tool result
2. Retry detection — consecutive calls to same target, first errors
3. Empty result detection — tool result empty / error / "not found"
4. Deliverable reference — tool result content appears in deliverable
5. Minimum retention — keep at least 2 golden steps
"""



import re
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    Event,
    EventKind,
    GoldenLabel,
    GoldenPathResult,
    GoldenStep,
    Segment,
)


_EMPTY_PATTERNS = re.compile(
    r"(page not found|404|file not found|no such file|permission denied"
    r"|error:|not found|empty|timed? ?out)",
    re.IGNORECASE,
)


def _extract_key_tokens(text: str, max_tokens: int = 50) -> List[str]:
    """Extract significant tokens from text for substring matching."""
    words = re.findall(r"[A-Za-z_][\w.]*", text)
    # filter noise
    return [w for w in words if len(w) > 3][:max_tokens]


def _overlap_score(result_text: str, reasoning_text: str) -> float:
    """Fraction of result key tokens found in reasoning."""
    tokens = _extract_key_tokens(result_text)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in reasoning_text)
    return hits / len(tokens)


class GoldenPathEngine:
    """Classifies tool calls within a segment as golden or dead end."""

    def __init__(self, reference_threshold: float = 0.1) -> None:
        self.reference_threshold = reference_threshold

    def classify(self, segment: Segment) -> GoldenPathResult:
        # Collect tool call / result pairs and reasoning blocks
        tool_pairs = self._pair_tool_calls(segment.events)
        reasoning_blocks = [
            e.data.get("text", "")
            for e in segment.events
            if e.kind == EventKind.REASONING
        ]
        all_reasoning = "\n".join(reasoning_blocks)
        deliverable = segment.deliverable or ""

        labels: List[Tuple[int, GoldenLabel, Dict[str, Any], str, str]] = []

        for i, (idx, tc, tr) in enumerate(tool_pairs):
            result_text = tr

            # Rule 2 — retry detection (must precede Rule 3)
            # If the next call targets the same resource, this call was a
            # failed attempt that the agent retried — mark as dead end.
            if i + 1 < len(tool_pairs):
                next_tc = tool_pairs[i + 1][1]
                if (
                    tc.get("name") == next_tc.get("name")
                    and _same_target(tc, next_tc)
                ):
                    labels.append((idx, GoldenLabel.DEAD_END, tc, tr, "retry_superseded"))
                    continue

            # Rule 3 — empty / error result
            if len(result_text.strip()) < 20 or _EMPTY_PATTERNS.search(result_text[:500]):
                labels.append((idx, GoldenLabel.DEAD_END, tc, tr, "empty_or_error"))
                continue

            # Rule 1 — reference detection in subsequent reasoning
            subsequent_reasoning = self._reasoning_after(segment.events, idx)
            score = _overlap_score(result_text, subsequent_reasoning)
            if score >= self.reference_threshold:
                labels.append((idx, GoldenLabel.GOLDEN, tc, tr, "referenced"))
                continue

            # Rule 4 — deliverable reference
            if deliverable and _overlap_score(result_text, deliverable) >= self.reference_threshold:
                labels.append((idx, GoldenLabel.GOLDEN, tc, tr, "deliverable_ref"))
                continue

            labels.append((idx, GoldenLabel.DEAD_END, tc, tr, "unreferenced"))

        # Rule 5 — minimum retention: at least 2 golden steps
        golden_count = sum(1 for _, lbl, _, _, _ in labels if lbl == GoldenLabel.GOLDEN)
        if golden_count < 2 and len(labels) >= 2:
            # promote top candidates by reasoning overlap
            dead_ends_with_score = []
            for entry in labels:
                idx, lbl, tc, tr, reason = entry
                if lbl == GoldenLabel.DEAD_END:
                    s = _overlap_score(tr, all_reasoning)
                    dead_ends_with_score.append((s, idx, tc, tr))
            dead_ends_with_score.sort(reverse=True)
            promote_count = 2 - golden_count
            promoted_indices = {
                de[1] for de in dead_ends_with_score[:promote_count]
            }
            labels = [
                (idx, GoldenLabel.GOLDEN if idx in promoted_indices else lbl, tc, tr, reason)
                for idx, lbl, tc, tr, reason in labels
            ]

        golden_steps = []
        dead_ends = []
        for idx, lbl, tc, tr, _reason in labels:
            step = GoldenStep(
                event_index=idx,
                tool_call=tc,
                tool_result=tr,
                related_reasoning=self._reasoning_after(segment.events, idx)[:2048],
            )
            if lbl == GoldenLabel.GOLDEN:
                golden_steps.append(step)
            else:
                dead_ends.append(step)

        return GoldenPathResult(golden_steps=golden_steps, dead_ends=dead_ends)

    def _pair_tool_calls(
        self, events: List[Event]
    ) -> List[Tuple[int, Dict[str, Any], str]]:
        """Pair tool_call events with their corresponding tool_result."""
        pairs: List[Tuple[int, Dict[str, Any], str]] = []
        for i, ev in enumerate(events):
            if ev.kind == EventKind.TOOL_CALL:
                result = ""
                for j in range(i + 1, min(i + 5, len(events))):
                    if events[j].kind == EventKind.TOOL_RESULT:
                        result = events[j].data.get("content", "")
                        break
                pairs.append((i, ev.data, result))
        return pairs

    def _reasoning_after(self, events: List[Event], start_idx: int) -> str:
        """Collect reasoning text after a given event index."""
        parts = []
        for i in range(start_idx + 1, len(events)):
            if events[i].kind == EventKind.REASONING:
                parts.append(events[i].data.get("text", ""))
            elif events[i].kind == EventKind.TOOL_CALL:
                break  # stop at next tool call
        return "\n".join(parts)


def _same_target(tc1: Dict[str, Any], tc2: Dict[str, Any]) -> bool:
    """Check if two tool calls target the same resource."""
    args1 = tc1.get("arguments", tc1.get("args", {}))
    args2 = tc2.get("arguments", tc2.get("args", {}))
    if isinstance(args1, dict) and isinstance(args2, dict):
        for key in ("file_path", "path", "url", "query"):
            if key in args1 and key in args2:
                return args1[key] == args2[key]
    return False
