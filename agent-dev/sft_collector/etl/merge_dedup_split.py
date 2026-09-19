"""Offline ETL — §4.9: merge runs, dedup, split, balance, output."""



import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def merge_runs(run_dirs: List[str], output_dir: str) -> List[Dict[str, Any]]:
    """Merge samples from multiple run directories.

    Validates SHA256 from manifest, collects all samples into a unified list.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_samples: List[Dict[str, Any]] = []
    integrity_errors = 0

    for run_dir in run_dirs:
        collected = os.path.join(run_dir, "sft_collected")
        if not os.path.isdir(collected):
            logger.warning("No sft_collected in %s, skipping", run_dir)
            continue

        manifest_path = os.path.join(collected, "manifest.jsonl")
        manifest_entries: Dict[str, str] = {}
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("category") == "samples":
                        manifest_entries[entry["sample_id"]] = entry["sha256"]

        samples_dir = os.path.join(collected, "samples")
        if not os.path.isdir(samples_dir):
            continue

        for fname in os.listdir(samples_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(samples_dir, fname)
            with open(fpath) as f:
                content = f.read()

            sample = json.loads(content)
            sample_id = sample.get("sample_id", fname.replace(".json", ""))

            # SHA256 integrity check
            actual_sha = hashlib.sha256(content.encode()).hexdigest()
            expected_sha = manifest_entries.get(sample_id)
            if expected_sha is None:
                logger.error("Missing manifest entry for %s, skipping", sample_id)
                integrity_errors += 1
                continue
            if actual_sha != expected_sha:
                logger.error(
                    "SHA256 mismatch for %s: expected %s, got %s",
                    sample_id, expected_sha[:16], actual_sha[:16],
                )
                integrity_errors += 1
                continue

            sample["_source_run"] = run_dir
            all_samples.append(sample)

    logger.info(
        "Merged %d samples from %d runs (%d integrity errors)",
        len(all_samples), len(run_dirs), integrity_errors,
    )
    return all_samples


def dedup(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by exact hash and fuzzy text similarity."""
    # Phase 1: exact dedup by source_hash + patch_hash
    seen_exact: Set[str] = set()
    phase1: List[Dict[str, Any]] = []

    for s in samples:
        prov = s.get("provenance", {})
        key = f"{prov.get('source_hash', '')}:{prov.get('patch_hash', '')}"
        if key and key != ":" and key in seen_exact:
            logger.debug("Exact dedup: %s", s.get("sample_id"))
            continue
        if key != ":":
            seen_exact.add(key)
        phase1.append(s)

    # Phase 2: fuzzy dedup by normalized content hash
    seen_fuzzy: Set[str] = set()
    phase2: List[Dict[str, Any]] = []

    for s in phase1:
        norm = _normalize_for_dedup(s)
        norm_hash = hashlib.sha256(norm.encode()).hexdigest()
        if norm_hash in seen_fuzzy:
            logger.debug("Fuzzy dedup: %s", s.get("sample_id"))
            continue
        seen_fuzzy.add(norm_hash)
        phase2.append(s)

    logger.info("Dedup: %d → %d (exact) → %d (fuzzy)", len(samples), len(phase1), len(phase2))
    return phase2


def assign_splits(
    samples: List[Dict[str, Any]],
    dev_ratio: float = 0.1,
    held_out_ratio: float = 0.05,
) -> List[Dict[str, Any]]:
    """Assign train/dev/held_out splits by source lineage grouping.

    Same lineage → same split (no cross-split leakage).
    """
    # Group by lineage
    lineage_groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        lineage = (
            s.get("provenance", {}).get("source_lineage_id")
            or s.get("provenance", {}).get("implementation_family_id")
            or s.get("sample_id", str(i))
        )
        lineage_groups[lineage].append(i)

    lineages = list(lineage_groups.keys())
    total_lineages = len(lineages)
    dev_count = max(1, int(total_lineages * dev_ratio))
    held_out_count = max(1, int(total_lineages * held_out_ratio))
    train_count = total_lineages - dev_count - held_out_count

    # Deterministic assignment based on lineage hash
    lineages_sorted = sorted(lineages, key=lambda l: hashlib.md5(l.encode()).hexdigest())
    dev_lineages = set(lineages_sorted[:dev_count])
    held_out_lineages = set(lineages_sorted[dev_count:dev_count + held_out_count])

    for s in samples:
        lineage = (
            s.get("provenance", {}).get("source_lineage_id")
            or s.get("provenance", {}).get("implementation_family_id")
            or s.get("sample_id", "")
        )
        if lineage in dev_lineages:
            s["split"] = "dev"
        elif lineage in held_out_lineages:
            s["split"] = "held_out"
        else:
            s["split"] = "train"

    split_counts = Counter(s["split"] for s in samples)
    logger.info(
        "Split assignment: train=%d, dev=%d, held_out=%d",
        split_counts["train"], split_counts["dev"], split_counts["held_out"],
    )
    return samples


def balance_check(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check data balance: sample_type distribution and language distribution."""
    type_counts = Counter(s.get("sample_type", "unknown") for s in samples)
    lang_counts = Counter(
        s.get("input", {}).get("contract", {}).get("language", "unknown")
        for s in samples
    )
    task_type_counts = Counter(s.get("task_type", "unknown") for s in samples)

    total = len(samples) or 1
    report = {
        "total_samples": len(samples),
        "by_sample_type": dict(type_counts),
        "by_language": dict(lang_counts),
        "by_task_type": dict(task_type_counts),
        "sample_type_ratios": {k: round(v / total, 3) for k, v in type_counts.items()},
        "language_ratios": {k: round(v / total, 3) for k, v in lang_counts.items()},
        "task_type_ratios": {k: round(v / total, 3) for k, v in task_type_counts.items()},
    }

    # Warnings
    warnings = []
    at_ratio = type_counts.get("analysis_trajectory", 0) / total
    cs_ratio = type_counts.get("concept_snapshot", 0) / total
    ok_ratio = type_counts.get("opus_kernel", 0) / total

    if at_ratio > 0 and abs(at_ratio - 0.4) > 0.15:
        warnings.append(f"analysis_trajectory ratio {at_ratio:.1%} deviates from target 40%")
    if cs_ratio > 0 and abs(cs_ratio - 0.4) > 0.15:
        warnings.append(f"concept_snapshot ratio {cs_ratio:.1%} deviates from target 40%")

    report["warnings"] = warnings
    return report


def output_files(
    samples: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, str]:
    """Write final output files."""
    os.makedirs(output_dir, exist_ok=True)

    analysis = [s for s in samples if s.get("sample_type") == "analysis_trajectory"]
    coding = [s for s in samples if s.get("sample_type") in ("opus_kernel",)]
    concepts = [s for s in samples if s.get("sample_type") == "concept_snapshot"]

    paths = {}

    # Analysis trajectories
    at_path = os.path.join(output_dir, "analysis_trajectories.jsonl")
    _write_jsonl(at_path, analysis)
    paths["analysis_trajectories"] = at_path

    # Kernel coding samples
    kc_path = os.path.join(output_dir, "kernel_coding_samples.jsonl")
    _write_jsonl(kc_path, coding)
    paths["kernel_coding_samples"] = kc_path

    # Concept snapshots
    cs_path = os.path.join(output_dir, "concept_snapshots.jsonl")
    _write_jsonl(cs_path, concepts)
    paths["concept_snapshots"] = cs_path

    # Combined manifest
    manifest = {
        "total": len(samples),
        "analysis_trajectories": len(analysis),
        "kernel_coding_samples": len(coding),
        "concept_snapshots": len(concepts),
        "splits": dict(Counter(s.get("split", "train") for s in samples)),
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    paths["manifest"] = manifest_path

    # Checksums
    checksum_path = os.path.join(output_dir, "checksums.sha256")
    with open(checksum_path, "w") as f:
        for name, path in paths.items():
            if path != checksum_path and os.path.exists(path):
                sha = _file_sha256(path)
                f.write(f"{sha}  {os.path.basename(path)}\n")
    paths["checksums"] = checksum_path

    logger.info("Output written to %s: %s", output_dir, manifest)
    return paths


def run_etl(
    run_dirs: List[str],
    output_dir: str,
    dev_ratio: float = 0.1,
    held_out_ratio: float = 0.05,
) -> Dict[str, Any]:
    """Full ETL pipeline: merge → dedup → split → balance → output."""
    samples = merge_runs(run_dirs, output_dir)
    samples = dedup(samples)
    samples = assign_splits(samples, dev_ratio, held_out_ratio)
    balance = balance_check(samples)
    paths = output_files(samples, output_dir)

    return {
        "balance_report": balance,
        "output_paths": paths,
    }


# ----- helpers -----

def _normalize_for_dedup(sample: Dict[str, Any]) -> str:
    """Normalize sample content for fuzzy dedup."""
    parts = []
    if "messages" in sample:
        for m in sample["messages"]:
            content = m.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", ""))
            elif isinstance(content, str):
                parts.append(content)
    if "output" in sample:
        parts.append(str(sample["output"].get("patch", "")))
    text = " ".join(parts)
    # Normalize whitespace
    import re
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            # Remove internal fields
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
