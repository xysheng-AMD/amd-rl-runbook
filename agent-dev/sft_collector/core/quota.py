"""Collection Quota — stop collecting when targets are met."""

import time
from typing import Any, Dict, Optional


class CollectionQuota:
    """Tracks per-type sample counts against configurable targets.

    When all targets are met, is_all_full() returns True and the caller
    should stop processing further trajectories.
    """

    def __init__(
        self,
        analysis=200,    # type: int
        opus=400,        # type: int
        concept=400,     # type: int
    ):
        # type: (...) -> None
        self.targets = {
            "analysis_trajectory": analysis,
            "opus_kernel": opus,
            "concept_snapshot": concept,
        }
        self.counts = {
            "analysis_trajectory": 0,
            "opus_kernel": 0,
            "concept_snapshot": 0,
            "dead_end": 0,
            "rejected": 0,
            "pending": 0,
        }

    def record(self, sample_type):
        # type: (str) -> bool
        """Increment count. Returns True if this type just reached its target."""
        self.counts[sample_type] = self.counts.get(sample_type, 0) + 1
        return self.is_type_full(sample_type)

    def is_type_full(self, sample_type):
        # type: (str) -> bool
        target = self.targets.get(sample_type)
        if target is None:
            return False
        return self.counts.get(sample_type, 0) >= target

    def is_all_full(self):
        # type: () -> bool
        for st, target in self.targets.items():
            if self.counts.get(st, 0) < target:
                return False
        return True

    def total_positive(self):
        # type: () -> int
        return sum(
            self.counts.get(st, 0) for st in self.targets
        )

    def total_target(self):
        # type: () -> int
        return sum(self.targets.values())

    def progress(self):
        # type: () -> Dict[str, Dict[str, Any]]
        result = {}
        for st, target in self.targets.items():
            current = self.counts.get(st, 0)
            pct = (current / target * 100) if target > 0 else 0.0
            result[st] = {
                "current": current,
                "target": target,
                "pct": round(pct, 1),
                "full": current >= target,
            }
        return result

    def write_status(self, path):
        # type: (str) -> None
        """Write human-readable status report to sft-collect-status.txt."""
        prog = self.progress()
        total_pos = self.total_positive()
        total_tgt = self.total_target()
        overall_pct = (total_pos / total_tgt * 100) if total_tgt > 0 else 0.0

        lines = []
        lines.append("=== SFT Collection Status ===")
        lines.append("Generated: {}".format(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ))
        lines.append("")
        lines.append("{:<25s} {:>8s} {:>8s} {:>10s}".format(
            "Type", "Current", "Target", "Progress"
        ))
        lines.append("-" * 55)

        not_ready = []
        for st in ("analysis_trajectory", "opus_kernel", "concept_snapshot"):
            p = prog[st]
            mark = "  DONE" if p["full"] else ""
            lines.append("{:<25s} {:>8d} {:>8d} {:>9.1f}%{}".format(
                st, p["current"], p["target"], p["pct"], mark,
            ))
            if not p["full"]:
                need = p["target"] - p["current"]
                not_ready.append("{} (need {} more)".format(st, need))

        lines.append("")
        lines.append("Dead ends (RL neg):     {:>6d}".format(
            self.counts.get("dead_end", 0)
        ))
        lines.append("Rejected:               {:>6d}".format(
            self.counts.get("rejected", 0)
        ))
        lines.append("Pending (await verify):  {:>6d}".format(
            self.counts.get("pending", 0)
        ))
        lines.append("")
        lines.append("Overall: {} / {} ({:.1f}%)".format(
            total_pos, total_tgt, overall_pct,
        ))

        if not_ready:
            lines.append("Status: NOT READY")
            for item in not_ready:
                lines.append("  - {}".format(item))
        else:
            lines.append("Status: READY FOR TRAINING")

        lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))
