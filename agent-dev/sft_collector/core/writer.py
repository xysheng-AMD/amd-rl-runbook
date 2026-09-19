"""Writer — atomic writes to run directory with manifest tracking."""



import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from .schema import SFTSample

logger = logging.getLogger(__name__)


class Writer:
    """Writes SFT samples to <run_dir>/sft_collected/ with integrity guarantees."""

    def __init__(self, run_dir: str) -> None:
        self.base_dir = os.path.join(run_dir, "sft_collected")
        self.samples_dir = os.path.join(self.base_dir, "samples")
        self.artifacts_dir = os.path.join(self.base_dir, "artifacts")
        self.dead_ends_dir = os.path.join(self.base_dir, "dead_ends")
        self.rejected_dir = os.path.join(self.base_dir, "rejected")
        self.pending_dir = os.path.join(self.base_dir, "pending")
        self.manifest_path = os.path.join(self.base_dir, "manifest.jsonl")

        for d in (
            self.samples_dir, self.artifacts_dir,
            self.dead_ends_dir, self.rejected_dir, self.pending_dir,
        ):
            os.makedirs(d, exist_ok=True)

    def write_sample(
        self,
        sample: SFTSample,
        parent_source: Optional[str] = None,
        candidate_source: Optional[str] = None,
    ) -> str:
        """Write a positive sample. Returns the file path."""
        data = sample.to_dict()
        content = json.dumps(data, ensure_ascii=False, indent=2)
        sha = hashlib.sha256(content.encode()).hexdigest()

        filepath = os.path.join(self.samples_dir, f"{sample.sample_id}.json")
        _atomic_write(filepath, content)

        # Save artifacts if provided
        if parent_source or candidate_source:
            art_dir = os.path.join(self.artifacts_dir, sample.sample_id)
            if parent_source:
                parent_dir = os.path.join(art_dir, "parent")
                os.makedirs(parent_dir, exist_ok=True)
                _atomic_write(
                    os.path.join(parent_dir, "kernel.opus"),
                    parent_source,
                )
            if candidate_source:
                cand_dir = os.path.join(art_dir, "candidate")
                os.makedirs(cand_dir, exist_ok=True)
                _atomic_write(
                    os.path.join(cand_dir, "kernel.opus"),
                    candidate_source,
                )

        self._append_manifest(sample.sample_id, sha, "samples")
        logger.info("Wrote sample %s → %s", sample.sample_id, filepath)
        return filepath

    def write_rejected(
        self, sample: SFTSample, reasons: List[str]
    ) -> str:
        """Write a rejected sample with rejection reasons."""
        data = sample.to_dict()
        data["rejection_reasons"] = reasons
        content = json.dumps(data, ensure_ascii=False, indent=2)
        sha = hashlib.sha256(content.encode()).hexdigest()

        filepath = os.path.join(self.rejected_dir, f"{sample.sample_id}.json")
        _atomic_write(filepath, content)
        self._append_manifest(sample.sample_id, sha, "rejected")
        logger.info("Rejected %s: %s", sample.sample_id, reasons)
        return filepath

    def write_pending(self, sample: SFTSample) -> str:
        """Write a pending sample (awaiting verify)."""
        data = sample.to_dict()
        data["verify_status"] = "pending_verify"
        content = json.dumps(data, ensure_ascii=False, indent=2)
        sha = hashlib.sha256(content.encode()).hexdigest()

        filepath = os.path.join(self.pending_dir, f"{sample.sample_id}.json")
        _atomic_write(filepath, content)
        self._append_manifest(sample.sample_id, sha, "pending")
        logger.info("Pending %s", sample.sample_id)
        return filepath

    def write_dead_end(self, sample_id: str, data: Dict[str, Any]) -> str:
        """Write dead-end steps as RL negatives."""
        content = json.dumps(data, ensure_ascii=False, indent=2)
        sha = hashlib.sha256(content.encode()).hexdigest()

        filepath = os.path.join(self.dead_ends_dir, f"{sample_id}.json")
        _atomic_write(filepath, content)
        self._append_manifest(sample_id, sha, "dead_ends")
        return filepath

    def write_collector_meta(self, meta: Dict[str, Any]) -> None:
        """Write collector metadata (version, config, stats)."""
        filepath = os.path.join(self.base_dir, "collector_meta.json")
        content = json.dumps(meta, ensure_ascii=False, indent=2)
        _atomic_write(filepath, content)

    def _append_manifest(self, sample_id: str, sha256: str, category: str) -> None:
        entry = {
            "sample_id": sample_id,
            "sha256": sha256,
            "category": category,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        try:
            with open(self.manifest_path, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            logger.error("Failed to append manifest: %s", e)


def _atomic_write(filepath: str, content: str) -> None:
    """Write via temp file → fsync → atomic rename."""
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, filepath)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
