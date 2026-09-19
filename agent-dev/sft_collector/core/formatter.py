"""Formatter — converts raw extracted data into GEAK-compatible SFT samples."""



import hashlib
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .schema import (
    ContextState,
    GoldenPathResult,
    GoldenStep,
    SFTSample,
    VerifyReceipt,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Formatter:
    """Produces GEAK-schema SFT samples from extracted components."""

    def __init__(self, run_id: str = "") -> None:
        self.run_id = run_id
        self._seq = 0

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self.run_id}-{self._seq:04d}"

    # ----- analysis trajectory -----

    def format_analysis_trajectory(
        self,
        golden: GoldenPathResult,
        context: ContextState,
        deliverable: str,
        system_prompt: str = "You are a GPU kernel performance analyst for AMD CDNA4/gfx950...",
    ) -> SFTSample:
        """Format golden path steps + deliverable as multi-turn messages."""
        sample_id = self._next_id("at")

        messages: List[Dict[str, Any]] = []

        # System
        messages.append({"role": "system", "content": system_prompt})

        # Build multi-turn from golden steps
        for step in golden.golden_steps:
            # Assistant with thinking + tool call
            assistant_content: List[Dict[str, Any]] = []
            if step.related_reasoning:
                assistant_content.append({
                    "type": "thinking",
                    "text": step.related_reasoning[:4096],
                })
            msg: Dict[str, Any] = {"role": "assistant", "content": assistant_content}
            if step.tool_call:
                msg["tool_calls"] = [step.tool_call]
            messages.append(msg)

            # Tool result (compressed to ≤ 2000 chars)
            if step.tool_result:
                messages.append({
                    "role": "tool",
                    "content": _compress_tool_result(step.tool_result),
                })

        # Final deliverable
        if deliverable:
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": deliverable},
                ],
            })

        si = context.session_identity
        return SFTSample(
            sample_id=sample_id,
            split="train",
            sample_domain="kernel",
            sample_type="analysis_trajectory",
            task_type=si.collection_mode or "direction_conditioned",
            messages=messages,
            labels={
                "golden_step_count": len(golden.golden_steps),
                "dead_end_count": len(golden.dead_ends),
            },
            provenance=_build_provenance(context, sample_id, "harness_middleware"),
        )

    # ----- OPUS kernel -----

    def format_opus_kernel(
        self,
        parent_source: str,
        candidate_source: str,
        diff: str,
        context: ContextState,
        verify_receipt: Optional[VerifyReceipt] = None,
    ) -> SFTSample:
        sample_id = self._next_id("opus")

        contract = context.contract
        arch = context.architecture

        input_data: Dict[str, Any] = {
            "contract": {
                "operator": contract.operator,
                "language": contract.language or "opus",
                "architecture": arch.target_gpu,
                "dimensions": contract.dimensions,
                "input_dtype": contract.input_dtype,
                "output_dtype": contract.output_dtype,
                "layout": contract.layout,
                "optimization_target": contract.optimization_target,
                "correctness_tolerance": contract.correctness_tol,
                "regression_constraints": contract.regression_constraints,
                "language_version": contract.language_version,
                "backend_version": contract.backend_version,
                "modifiable_files": contract.modifiable_files,
            },
            "parent_source": parent_source,
            "architecture_info": {
                "target_gpu": arch.target_gpu,
                "gpu_sku": arch.gpu_sku,
                "cache_hierarchy": arch.cache_hierarchy,
                "compute_caps": arch.compute_caps,
                "memory_subsystem": arch.memory_subsystem,
                "source_refs": arch.source_refs,
            },
            "baseline": {
                "latency_us": context.baseline.baseline_latency,
                "roofline": context.baseline.roofline_position,
            },
        }

        # Conditional fields
        if context.profile:
            input_data["profile"] = {
                "counters": {c.get("counter_name", ""): c.get("value") for c in context.profile.raw_counters},
                "derived": context.profile.derived_metrics,
                "hotspot": context.profile.hotspot,
                "profiler": context.profile.profiler_tool,
            }
        else:
            input_data["profile"] = None

        if context.direction and context.direction.frozen:
            input_data["direction"] = {
                "strategy": context.direction.optimization_strategy,
                "techlead_direction": context.direction.direction_prompt,
                "constraints": context.direction.key_constraints,
            }
        else:
            input_data["direction"] = None

        input_data["error_feedback"] = None
        if context.error:
            input_data["error_feedback"] = {
                "compile_error": context.error.compile_error or None,
                "correctness_error": context.error.correctness_error or None,
                "failed_source_hash": context.error.failed_source_hash,
            }

        input_data["per_case_benchmark"] = None
        if context.per_case_benchmark:
            pcb = context.per_case_benchmark
            input_data["per_case_benchmark"] = {
                "cases": pcb.cases,
                "aggregate_speedup": pcb.aggregate_speedup,
                "regressed_cases": pcb.regressed_cases,
                "regression_feedback": pcb.regression_feedback,
            }

        # Labels from verify receipt
        labels: Dict[str, Any] = {}
        if verify_receipt:
            labels = {
                "patch_applies": verify_receipt.patch_applies,
                "compile_pass": verify_receipt.compile_pass,
                "correctness_pass": verify_receipt.correctness_pass,
                "benchmark_valid": verify_receipt.benchmark_valid,
                "verified_speedup": verify_receipt.verified_speedup,
            }

        prov = _build_provenance(context, sample_id, "harness_middleware")
        prov["source_hash"] = _hash(parent_source)
        prov["patch_hash"] = _hash(diff)
        prov["baseline_source_hash"] = context.baseline.baseline_source_hash
        if verify_receipt:
            prov["verify_receipt"] = "opus_verify/{}/receipt.json".format(sample_id)
            prov["cheating_checks"] = verify_receipt.cheating_checks
            prov["verify_source"] = "independent_verify"
        else:
            prov["verify_source"] = "skipped"

        # Direction provenance
        if context.direction:
            d = context.direction
            prov["direction_source"] = d.direction_source
            prov["direction_id"] = d.direction_id
            prov["direction_created_at"] = d.direction_created_at
            prov["skill_corpus_hash"] = d.skill_corpus_hash

        si = context.session_identity
        return SFTSample(
            sample_id=sample_id,
            split="train",
            sample_domain="kernel",
            sample_type="opus_kernel",
            task_type=si.collection_mode or "direction_conditioned",
            input_data=input_data,
            output_data={"patch": diff},
            labels=labels,
            provenance=prov,
        )

    # ----- concept snapshot -----

    def format_concept_snapshot(
        self,
        concept_text: str,
        inferred_question: str,
        concept_count: int,
        has_unverified_constants: bool,
        context: ContextState,
    ) -> SFTSample:
        sample_id = self._next_id("cs")

        messages = [
            {
                "role": "system",
                "content": "You are a GPU kernel performance analyst for AMD CDNA4/gfx950...",
            },
            {"role": "user", "content": inferred_question},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": concept_text}],
            },
        ]

        return SFTSample(
            sample_id=sample_id,
            split="train",
            sample_domain="kernel",
            sample_type="concept_snapshot",
            task_type=context.session_identity.collection_mode or "direction_conditioned",
            messages=messages,
            labels={
                "concept_count": concept_count,
                "has_unverified_constants": has_unverified_constants,
            },
            provenance=_build_provenance(context, sample_id, "harness_middleware"),
        )


def _build_provenance(
    ctx: ContextState, sample_id: str, extraction_method: str
) -> Dict[str, Any]:
    si = ctx.session_identity
    env = ctx.environment
    mdl = ctx.model
    orch = ctx.orchestrator

    return {
        "schema_version": si.schema_version,
        "dataset_version": si.dataset_version,
        "run_id": si.run_id,
        "eval_dir": si.eval_dir,
        "round": si.round,
        "engineer_id": si.engineer_id,
        "kernel_name": si.kernel_name,
        "workspace_head": si.workspace_head,
        "extraction_method": extraction_method,
        "creation_timestamp": _ts(),
        "gpu": ctx.architecture.target_gpu,
        "gpu_sku": ctx.architecture.gpu_sku,
        "rocm_version": env.rocm_version,
        "compiler_version": env.hipcc_version,
        "container_digest": env.container_digest,
        "lumen_git_sha": env.lumen_git_sha,
        "geak_git_sha": env.geak_git_sha,
        "model_checkpoint": mdl.model_checkpoint,
        "orchestrator_budget_total": orch.budget_total,
        "orchestrator_budget_used": orch.budget_used,
    }


def _compress_tool_result(text: str, max_chars: int = 2000) -> str:
    """Compress tool result to keep only relevant lines."""
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    if len(lines) <= 10:
        return text[:max_chars] + "\n... [truncated]"
    # Keep first and last portions
    head = "\n".join(lines[:5])
    tail = "\n".join(lines[-5:])
    return f"{head}\n... [{len(lines) - 10} lines omitted] ...\n{tail}"
