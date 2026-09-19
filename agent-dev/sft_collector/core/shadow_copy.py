"""File Shadow Copy — §4.6 + §4.6.1: track parent→candidate for OPUS files.

Handles normal mode and error-recovery mode where parent resets to
the failed version on compile/correctness failure.
"""



import glob
import hashlib
import os
import time
from typing import Dict, List, Optional, Tuple

from .schema import ShadowSnapshot


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _is_opus_file(path: str) -> bool:
    return path.endswith(".opus") or path.endswith(".hpp")


class ShadowCopyManager:
    """Maintains shadow copies for OPUS files throughout a session."""

    def __init__(self, error_recovery_mode: bool = False) -> None:
        self._copies: Dict[str, List[ShadowSnapshot]] = {}
        self._error_snapshots: Dict[str, Dict] = {}
        self.error_recovery_mode = error_recovery_mode

    def init_workspace_snapshot(self, workspace_dir: str) -> None:
        """§4.10 session_start: snapshot all *.opus files in workspace."""
        for ext in ("*.opus", "*.hpp"):
            for filepath in glob.glob(
                os.path.join(workspace_dir, "**", ext), recursive=True
            ):
                try:
                    with open(filepath, "r") as f:
                        content = f.read()
                    self._copies[filepath] = [
                        ShadowSnapshot(
                            content=content,
                            timestamp=time.time(),
                            snapshot_type="parent",
                        )
                    ]
                except (OSError, UnicodeDecodeError):
                    pass

    def on_file_read(self, path: str, content: str) -> None:
        """Record first read as parent shadow copy."""
        if not _is_opus_file(path):
            return
        if path not in self._copies:
            self._copies[path] = [
                ShadowSnapshot(
                    content=content,
                    timestamp=time.time(),
                    snapshot_type="parent",
                )
            ]

    def on_file_write(self, path: str, content: str) -> None:
        """Record write as candidate snapshot."""
        if not _is_opus_file(path):
            return
        snap = ShadowSnapshot(
            content=content,
            timestamp=time.time(),
            snapshot_type="candidate",
        )
        if path in self._copies:
            self._copies[path].append(snap)
        else:
            self._copies[path] = [snap]

    # §4.6.1 error-recovery handling
    def on_compile_fail(self, path: str, content: str, stderr: str) -> None:
        """Error-recovery: reset parent to the failed version."""
        if not _is_opus_file(path):
            return
        error_parent = ShadowSnapshot(
            content=content,
            timestamp=time.time(),
            snapshot_type="error_parent",
        )
        if path not in self._copies:
            self._copies[path] = []
        self._copies[path] = [error_parent]
        self._error_snapshots[path] = {
            "failed_source_hash": _sha256(content),
            "compile_error": stderr[:4096],
            "correctness_error": None,
        }

    def on_correctness_fail(
        self, path: str, content: str, failure_info: str
    ) -> None:
        """Error-recovery: reset parent to the failed version."""
        if not _is_opus_file(path):
            return
        error_parent = ShadowSnapshot(
            content=content,
            timestamp=time.time(),
            snapshot_type="error_parent",
        )
        if path not in self._copies:
            self._copies[path] = []
        self._copies[path] = [error_parent]
        self._error_snapshots[path] = {
            "failed_source_hash": _sha256(content),
            "compile_error": None,
            "correctness_error": failure_info[:4096],
        }

    def collect(self, path: str) -> Optional[Tuple[str, str, str]]:
        """Collect parent, candidate, and unified diff for a file.

        Returns (parent_content, candidate_content, unified_diff) or None.
        """
        if path not in self._copies or len(self._copies[path]) < 2:
            return None

        snapshots = self._copies[path]

        # Find parent: error_parent takes precedence in error_recovery
        parent = snapshots[0]
        for s in snapshots:
            if s.snapshot_type == "error_parent":
                parent = s
                break

        # Candidate: last write
        candidate = snapshots[-1]
        if candidate.snapshot_type == "parent" or candidate.snapshot_type == "error_parent":
            return None  # no actual modification

        if parent.content == candidate.content:
            return None

        diff = _unified_diff(parent.content, candidate.content, path)
        return (parent.content, candidate.content, diff)

    def get_error_snapshot(self, path: str) -> Optional[Dict]:
        return self._error_snapshots.get(path)

    def get_parent_hash(self, path: str) -> str:
        if path in self._copies and self._copies[path]:
            return self._copies[path][0].sha256
        return ""

    def has_file(self, path: str) -> bool:
        return path in self._copies

    def tracked_files(self) -> List[str]:
        return list(self._copies.keys())

    def clear_file(self, path: str) -> None:
        self._copies.pop(path, None)
        self._error_snapshots.pop(path, None)


def _unified_diff(old: str, new: str, filename: str) -> str:
    """Generate unified diff between old and new content."""
    import difflib

    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{os.path.basename(filename)}",
        tofile=f"b/{os.path.basename(filename)}",
    )
    return "".join(diff)
