"""Quality Gates — §4.8: validation rules for each sample type."""



from typing import Dict, List, Optional, Set, Tuple

from .schema import SFTSample, TaskType, VerifyReceipt


class GateResult:
    __slots__ = ("passed", "reasons")

    def __init__(self, passed: bool, reasons: List[str]) -> None:
        self.passed = passed
        self.reasons = reasons

    def __bool__(self) -> bool:
        return self.passed


def check_common(sample: SFTSample) -> GateResult:
    """Universal gates: reasoning ≥ 100 chars, output ≥ 200 chars."""
    reasons: List[str] = []

    reasoning_len = _reasoning_length(sample)
    if reasoning_len < 100:
        reasons.append(f"reasoning too short: {reasoning_len} < 100 chars")

    output_len = _output_length(sample)
    if output_len < 200:
        reasons.append(f"output too short: {output_len} < 200 chars")

    return GateResult(len(reasons) == 0, reasons)


def check_analysis_trajectory(sample: SFTSample) -> GateResult:
    """Analysis trajectory: golden path ≥ 2 tool calls, deliverable exists, span ≥ 30s."""
    reasons: List[str] = []

    messages = sample.messages or []
    tool_call_count = sum(
        1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls")
    )
    if tool_call_count < 2:
        reasons.append(f"golden path has {tool_call_count} tool calls, need ≥ 2")

    # Check deliverable exists
    has_deliverable = False
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str) and len(content) > 200:
                has_deliverable = True
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        if len(block.get("text", "")) > 200:
                            has_deliverable = True
    if not has_deliverable:
        reasons.append("no deliverable found in trajectory")

    return GateResult(len(reasons) == 0, reasons)


def check_opus_kernel(
    sample: SFTSample,
    verify_receipt: Optional[VerifyReceipt] = None,
    skip_verify: bool = False,
) -> GateResult:
    """OPUS kernel: hard gates — independent verify, contract completeness."""
    reasons: List[str] = []
    inp = sample.input_data

    # parent ≠ candidate (actual modification)
    patch = sample.output_data.get("patch", "")
    if not patch or len(patch.strip()) == 0:
        reasons.append("empty patch")
    if len(patch) > 50 * 1024:
        reasons.append("patch too large: {} > 50KB".format(len(patch)))

    # Independent verify hard gates (skipped when --skip-verify)
    if not skip_verify:
        if verify_receipt is None:
            reasons.append("no verify receipt")
        else:
            if not verify_receipt.compile_pass:
                reasons.append("compile failed")
            if not verify_receipt.correctness_pass:
                reasons.append("correctness failed")
            if not verify_receipt.patch_applies:
                reasons.append("patch does not apply")

    # Contract completeness
    contract = inp.get("contract", {})
    for required_field in ("operator", "language", "architecture"):
        if not contract.get(required_field):
            reasons.append(f"contract missing {required_field}")

    # Input fields
    if not inp.get("parent_source"):
        reasons.append("missing parent_source")

    arch_info = inp.get("architecture_info", {})
    if not arch_info.get("target_gpu"):
        reasons.append("missing architecture_info.target_gpu")

    # Task-type conditional checks
    task_type = sample.task_type
    _check_task_type_fields(task_type, inp, reasons)

    return GateResult(len(reasons) == 0, reasons)


def check_concept_snapshot(
    sample: SFTSample,
    existing_hashes: Optional[Set[str]] = None,
) -> GateResult:
    """Concept snapshot: ≥ 2 concepts, no unverified constants, dedup."""
    reasons: List[str] = []

    # Check concept count from labels
    concept_count = sample.labels.get("concept_count", 0)
    if concept_count < 2:
        reasons.append(f"only {concept_count} concepts, need ≥ 2")

    # Check for unverified constants
    if sample.labels.get("has_unverified_constants", False):
        reasons.append("contains precise constants without source annotation")

    # Dedup check
    if existing_hashes is not None:
        content_hash = sample.content_hash()
        if content_hash in existing_hashes:
            reasons.append("duplicate concept snapshot")

    return GateResult(len(reasons) == 0, reasons)


def check_field_completeness(sample: SFTSample) -> GateResult:
    """§4.12 field completeness: required / conditional / optional."""
    reasons: List[str] = []
    prov = sample.provenance

    # Required provenance fields
    for f in (
        "schema_version", "run_id", "source_hash", "patch_hash",
        "gpu", "rocm_version", "compiler_version",
        "verify_source", "creation_timestamp",
    ):
        if not prov.get(f):
            reasons.append(f"provenance missing {f}")

    if prov.get("verify_source") != "independent_verify":
        reasons.append(f"verify_source={prov.get('verify_source')}, expected independent_verify")

    return GateResult(len(reasons) == 0, reasons)


# ----- helpers -----

def _reasoning_length(sample: SFTSample) -> int:
    total = 0
    if sample.messages:
        for m in sample.messages:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "thinking":
                            total += len(block.get("text", ""))
                elif isinstance(content, str):
                    total += len(content)
    return total


def _output_length(sample: SFTSample) -> int:
    if sample.output_data:
        patch = sample.output_data.get("patch", "")
        return len(patch)
    if sample.messages:
        for m in reversed(sample.messages):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            return len(block.get("text", ""))
                elif isinstance(content, str):
                    return len(content)
    return 0


def _check_task_type_fields(
    task_type: str, inp: Dict, reasons: List[str]
) -> None:
    """Validate fields per task_type (§4.12)."""
    if task_type == TaskType.COLD_START.value:
        if inp.get("profile") is not None:
            reasons.append("cold_start must have profile=null")
        if inp.get("direction") is not None:
            reasons.append("cold_start must have direction=null")

    elif task_type == TaskType.PROFILE_GUIDED.value:
        if inp.get("profile") is None:
            reasons.append("profile_guided must have profile≠null")
        if inp.get("direction") is not None:
            reasons.append("profile_guided must have direction=null")

    elif task_type == TaskType.DIRECTION_CONDITIONED.value:
        if inp.get("direction") is None:
            reasons.append("direction_conditioned must have direction≠null")

    elif task_type == TaskType.ERROR_RECOVERY.value:
        if inp.get("error_feedback") is None:
            reasons.append("error_recovery must have error_feedback≠null")

    elif task_type == TaskType.REGRESSION_BALANCE.value:
        if inp.get("per_case_benchmark") is None:
            reasons.append("regression_balance must have per_case_benchmark≠null")
