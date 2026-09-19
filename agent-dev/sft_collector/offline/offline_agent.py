"""Offline Agent — Pipeline A: extract SFT data from saved trajectories.

Reads saved trajectory files, reconstructs file states and context,
runs golden path engine, formats to GEAK schema, and writes to run directory.
Stops when collection quota is met.
"""



import hashlib
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set

from ..core.concept_extractor import ConceptExtractor
from ..core.context_accumulator import ContextAccumulator
from ..core.formatter import Formatter
from ..core.golden_path_engine import GoldenPathEngine
from ..core.independent_verifier import IndependentVerifier
from ..core.quality_gates import (
    GateResult,
    check_analysis_trajectory,
    check_common,
    check_concept_snapshot,
    check_field_completeness,
    check_opus_kernel,
)
from ..core.quota import CollectionQuota
from ..core.schema import (
    EventKind,
    SFTSample,
    Segment,
    VerifyStatus,
)
from ..core.shadow_copy import ShadowCopyManager
from ..core.writer import Writer
from .state_reconstructor import StateReconstructor, correlate_tool_calls_results
from .trajectory_parser import (
    find_trajectory_files,
    parse_events,
    segment_by_present,
)

logger = logging.getLogger(__name__)


class OfflineExtractionConfig:
    """Configuration for offline extraction."""

    def __init__(
        self,
        collection_mode="direction_conditioned",  # type: str
        harness_config=None,                       # type: Optional[Dict[str, Any]]
        workspace_dir="",                          # type: str
        compiler_cmd="opus-compile",               # type: str
        test_runner_cmd="opus-test",               # type: str
        benchmark_cmd="opus-bench",                # type: Optional[str]
        verify_timeout=300,                        # type: int
        skip_verify=False,                         # type: bool
        extract_concepts=True,                     # type: bool
        cap_analysis=200,                          # type: int
        cap_opus=400,                              # type: int
        cap_concept=400,                           # type: int
    ):
        # type: (...) -> None
        self.collection_mode = collection_mode
        self.harness_config = harness_config or {}
        self.workspace_dir = workspace_dir
        self.compiler_cmd = compiler_cmd
        self.test_runner_cmd = test_runner_cmd
        self.benchmark_cmd = benchmark_cmd
        self.verify_timeout = verify_timeout
        self.skip_verify = skip_verify
        self.extract_concepts = extract_concepts
        self.cap_analysis = cap_analysis
        self.cap_opus = cap_opus
        self.cap_concept = cap_concept


class OfflineAgent:
    """Processes saved trajectory files and extracts SFT data."""

    def __init__(self, run_dir, config):
        # type: (str, OfflineExtractionConfig) -> None
        self.run_dir = run_dir
        self.config = config
        self.writer = Writer(run_dir)
        self.formatter = Formatter(run_id=os.path.basename(run_dir))
        self.golden_engine = GoldenPathEngine()
        self.concept_extractor = ConceptExtractor()
        self.verifier = IndependentVerifier(
            compiler_cmd=config.compiler_cmd,
            test_runner_cmd=config.test_runner_cmd,
            benchmark_cmd=config.benchmark_cmd,
            timeout=config.verify_timeout,
        )
        self.quota = CollectionQuota(
            analysis=config.cap_analysis,
            opus=config.cap_opus,
            concept=config.cap_concept,
        )
        self._concept_hashes = set()  # type: Set[str]

    def process_trajectory(self, jsonl_path):
        # type: (str) -> Dict[str, int]
        """Process a single trajectory file. Returns counts by type."""
        if self.quota.is_all_full():
            logger.info("Quota full, skipping %s", jsonl_path)
            return {"analysis": 0, "opus": 0, "concept": 0, "dead_end": 0, "rejected": 0}

        logger.info("Processing trajectory: %s", jsonl_path)
        counts = {"analysis": 0, "opus": 0, "concept": 0, "dead_end": 0, "rejected": 0}

        events = parse_events(jsonl_path)
        if not events:
            logger.warning("No events found in %s", jsonl_path)
            return counts

        events = correlate_tool_calls_results(events)
        segments = segment_by_present(events)

        reconstructor = StateReconstructor(
            collection_mode=self.config.collection_mode,
            harness_config=self.config.harness_config,
        )

        for segment in segments:
            if self.quota.is_all_full():
                break

            reconstructor.replay_segment(segment)
            shadow = reconstructor.shadow
            context = reconstructor.context

            # Analysis trajectory
            if segment.deliverable and not self.quota.is_type_full("analysis_trajectory"):
                c = self._extract_analysis(segment, context)
                counts["analysis"] += c["ok"]
                counts["dead_end"] += c["dead"]
                counts["rejected"] += c["reject"]

            # OPUS kernel
            if not self.quota.is_type_full("opus_kernel"):
                for filepath in shadow.tracked_files():
                    result = shadow.collect(filepath)
                    if result:
                        parent, candidate, diff = result
                        c = self._extract_opus(
                            filepath, parent, candidate, diff, context
                        )
                        counts["opus"] += c["ok"]
                        counts["rejected"] += c["reject"]
                        shadow.clear_file(filepath)

            # Concept snapshots
            if self.config.extract_concepts and not self.quota.is_type_full("concept_snapshot"):
                for ev in segment.events:
                    if self.quota.is_type_full("concept_snapshot"):
                        break
                    if ev.kind == EventKind.REASONING:
                        text = ev.data.get("text", "")
                        c = self._extract_concept(text, context)
                        counts["concept"] += c

        logger.info(
            "Trajectory %s: %d analysis, %d opus, %d concept, %d dead_end, %d rejected",
            os.path.basename(jsonl_path),
            counts["analysis"], counts["opus"], counts["concept"],
            counts["dead_end"], counts["rejected"],
        )
        return counts

    def process_batch(self, trajectory_dir):
        # type: (str) -> Dict[str, int]
        """Process all trajectory files in a directory.

        Stops early when collection quota is met.
        """
        files = find_trajectory_files(trajectory_dir)
        logger.info("Found %d trajectory files in %s", len(files), trajectory_dir)

        totals = {"analysis": 0, "opus": 0, "concept": 0, "dead_end": 0, "rejected": 0}
        processed = 0

        for fpath in files:
            if self.quota.is_all_full():
                logger.info("Collection quota met, stopping early (%d/%d files processed)",
                            processed, len(files))
                break
            try:
                counts = self.process_trajectory(fpath)
                for k in totals:
                    totals[k] += counts.get(k, 0)
                processed += 1
            except Exception:
                logger.exception("Failed to process %s", fpath)

        # Write collector metadata
        self.writer.write_collector_meta({
            "version": "1.0.0",
            "extraction_method": "offline_agent",
            "collection_mode": self.config.collection_mode,
            "trajectory_dir": trajectory_dir,
            "trajectory_count": len(files),
            "trajectories_processed": processed,
            "quota_targets": self.quota.targets,
            "quota_counts": self.quota.counts,
            "quota_met": self.quota.is_all_full(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        # Write status report
        status_path = os.path.join(self.run_dir, "sft-collect-status.txt")
        self.quota.write_status(status_path)
        logger.info("Status report: %s", status_path)

        logger.info("Batch complete: %s", totals)
        return totals

    # ---- internal extraction methods ----

    def _extract_analysis(self, segment, context):
        # type: (Segment, ContextAccumulator) -> Dict[str, int]
        result = {"ok": 0, "dead": 0, "reject": 0}

        ctx_snap = context.snapshot()
        golden = self.golden_engine.classify(segment)

        sample = self.formatter.format_analysis_trajectory(
            golden=golden,
            context=ctx_snap,
            deliverable=segment.deliverable or "",
        )

        gate_common = check_common(sample)
        gate_analysis = check_analysis_trajectory(sample)

        if not gate_common or not gate_analysis:
            reasons = gate_common.reasons + gate_analysis.reasons
            self.writer.write_rejected(sample, reasons)
            self.quota.record("rejected")
            result["reject"] = 1
            return result

        self.writer.write_sample(sample)
        self.quota.record("analysis_trajectory")
        result["ok"] = 1

        if golden.dead_ends:
            for de in golden.dead_ends:
                de_id = "de-{}-{}".format(sample.sample_id, de.event_index)
                self.writer.write_dead_end(de_id, {
                    "source_sample_id": sample.sample_id,
                    "event_index": de.event_index,
                    "tool_call": de.tool_call,
                    "tool_result": de.tool_result[:2000],
                    "reasoning": de.related_reasoning[:2000],
                })
                self.quota.record("dead_end")
                result["dead"] += 1

        return result

    def _extract_opus(self, filepath, parent, candidate, diff, context):
        # type: (str, str, str, str, ContextAccumulator) -> Dict[str, int]
        result = {"ok": 0, "reject": 0}
        ctx_snap = context.snapshot()

        verify_receipt = None
        if not self.config.skip_verify:
            ws_config = {
                "workspace_dir": self.config.workspace_dir,
                "kernel_file": os.path.basename(filepath),
                "gpu_identity": ctx_snap.architecture.gpu_sku,
            }
            diff_hash = hashlib.sha256(diff.encode()).hexdigest()[:12]
            verify_receipt = self.verifier.verify(
                parent, candidate, diff, ws_config, self.run_dir,
                "opus-verify-{}".format(diff_hash),
            )

        sample = self.formatter.format_opus_kernel(
            parent_source=parent,
            candidate_source=candidate,
            diff=diff,
            context=ctx_snap,
            verify_receipt=verify_receipt,
        )

        gate_common = check_common(sample)
        gate_opus = check_opus_kernel(sample, verify_receipt, skip_verify=self.config.skip_verify)
        gate_fields = check_field_completeness(sample) if not self.config.skip_verify else GateResult(True, [])

        if verify_receipt and verify_receipt.status == VerifyStatus.PENDING:
            self.writer.write_pending(sample)
            self.quota.record("pending")
            return result

        if not gate_common or not gate_opus or not gate_fields:
            reasons = gate_common.reasons + gate_opus.reasons + gate_fields.reasons
            self.writer.write_rejected(sample, reasons)
            self.quota.record("rejected")
            result["reject"] = 1
            return result

        self.writer.write_sample(sample, parent, candidate)
        self.quota.record("opus_kernel")
        result["ok"] = 1
        return result

    def _extract_concept(self, reasoning_text, context):
        # type: (str, ContextAccumulator) -> int
        snapshot = self.concept_extractor.extract(reasoning_text)
        if snapshot is None:
            return 0
        if snapshot.has_unverified_constants:
            return 0

        ctx_snap = context.snapshot()
        sample = self.formatter.format_concept_snapshot(
            concept_text=snapshot.concept_text,
            inferred_question=snapshot.inferred_question,
            concept_count=snapshot.concept_count,
            has_unverified_constants=snapshot.has_unverified_constants,
            context=ctx_snap,
        )

        gate_common = check_common(sample)
        gate = check_concept_snapshot(sample, self._concept_hashes)
        if not gate_common or not gate:
            reasons = gate_common.reasons + gate.reasons
            self.writer.write_rejected(sample, reasons)
            self.quota.record("rejected")
            return 0

        self._concept_hashes.add(sample.content_hash())
        self.writer.write_sample(sample)
        self.quota.record("concept_snapshot")
        return 1
