"""Data types for SFT collection — aligned with GEAK samples.jsonl schema."""



import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, enum.Enum):
    COLD_START = "cold_start"
    PROFILE_GUIDED = "profile_guided"
    DIRECTION_CONDITIONED = "direction_conditioned"
    ERROR_RECOVERY = "error_recovery"
    REGRESSION_BALANCE = "regression_balance"


class SampleType(str, enum.Enum):
    ANALYSIS_TRAJECTORY = "analysis_trajectory"
    OPUS_KERNEL = "opus_kernel"
    CONCEPT_SNAPSHOT = "concept_snapshot"


class VerifyStatus(str, enum.Enum):
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending_verify"


class GoldenLabel(str, enum.Enum):
    GOLDEN = "golden"
    DEAD_END = "dead_end"


# ---------------------------------------------------------------------------
# Low-level event types (for trajectory parsing)
# ---------------------------------------------------------------------------

class EventKind(str, enum.Enum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    TEXT_OUTPUT = "text_output"
    INBOX = "inbox"
    PRESENT = "present"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"


@dataclass
class Event:
    kind: EventKind
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shadow Copy
# ---------------------------------------------------------------------------

@dataclass
class ShadowSnapshot:
    content: str
    timestamp: float
    snapshot_type: str  # "parent" | "candidate" | "error_parent"
    sha256: str = ""

    def __post_init__(self):
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Context types — §4.10 Context Accumulator
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureContext:
    target_gpu: str = ""              # gfx950 / gfx942
    gpu_sku: str = ""                 # MI355X / MI308
    isa_version: str = ""
    cache_hierarchy: Dict[str, Any] = field(default_factory=dict)
    compute_caps: Dict[str, Any] = field(default_factory=dict)
    memory_subsystem: Dict[str, Any] = field(default_factory=dict)
    special_features: List[str] = field(default_factory=list)
    source_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ContractContext:
    operator: str = ""
    entry_point: str = ""
    dimensions: Dict[str, Any] = field(default_factory=dict)
    input_dtype: str = ""
    output_dtype: str = ""
    layout: str = ""
    quantization: Dict[str, Any] = field(default_factory=dict)
    language: str = ""
    language_version: str = ""
    backend_version: str = ""
    modifiable_files: List[str] = field(default_factory=list)
    correctness_tol: Optional[float] = None
    optimization_target: str = ""
    regression_constraints: List[str] = field(default_factory=list)
    shape_regime: str = ""
    commandment_hash: str = ""


@dataclass
class BaselineContext:
    baseline_latency: Dict[str, Any] = field(default_factory=dict)
    baseline_throughput: Dict[str, Any] = field(default_factory=dict)
    baseline_source_hash: str = ""
    roofline_position: Dict[str, Any] = field(default_factory=dict)
    bottleneck: str = ""


@dataclass
class ProfileContext:
    raw_counters: List[Dict[str, Any]] = field(default_factory=list)
    derived_metrics: Dict[str, Any] = field(default_factory=dict)
    hotspot: str = ""
    profiler_tool: str = ""
    profiler_version: str = ""
    source_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class DirectionContext:
    direction_source: str = ""        # tech_lead | fallback | skill | user
    direction_id: str = ""
    direction_prompt: str = ""
    direction_prompt_hash: str = ""
    direction_created_at: str = ""
    fallback_reason: Optional[str] = None
    skill_refs: List[Dict[str, str]] = field(default_factory=list)
    skill_corpus_hash: str = ""
    optimization_strategy: str = ""
    key_constraints: List[str] = field(default_factory=list)
    prior_attempts: List[Dict[str, Any]] = field(default_factory=list)
    frozen: bool = False              # once frozen, reasoning cannot override


@dataclass
class PerCaseBenchmarkContext:
    cases: List[Dict[str, Any]] = field(default_factory=list)
    aggregate_speedup: Optional[float] = None
    regressed_cases: List[Dict[str, Any]] = field(default_factory=list)
    regression_feedback: str = ""


@dataclass
class ErrorContext:
    failed_source_hash: str = ""
    compile_error: Dict[str, Any] = field(default_factory=dict)
    correctness_error: Dict[str, Any] = field(default_factory=dict)
    error_analysis: str = ""


@dataclass
class SkillRetrievalContext:
    retrieval_query: str = ""
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected_entries: List[Dict[str, Any]] = field(default_factory=list)
    corpus_hash: str = ""
    corpus_git_sha: str = ""


@dataclass
class EnvironmentContext:
    rocm_version: str = ""
    driver_version: str = ""
    hipcc_version: str = ""
    pytorch_version: str = ""
    triton_version: str = ""
    container_image: str = ""
    container_digest: str = ""
    geak_git_sha: str = ""
    lumen_git_sha: str = ""
    gpu_clocks: Dict[str, Any] = field(default_factory=dict)
    gpu_power_mode: str = ""
    gpu_temperature: str = ""


@dataclass
class ModelContext:
    model_checkpoint: str = ""
    tokenizer_revision: str = ""
    adapter_version: Optional[str] = None
    chat_template: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 32000
    stop_tokens: List[str] = field(default_factory=list)


@dataclass
class OrchestratorContext:
    budget_total: Optional[int] = None
    budget_used: Optional[int] = None
    round_limit: Optional[int] = None
    candidate_floor: Optional[float] = None
    min_improve: Optional[float] = None
    max_no_improve: Optional[int] = None
    stopped_by: Optional[str] = None
    warm_start_mode: bool = False
    gpu_pool_pin_mode: str = ""


@dataclass
class SessionIdentity:
    schema_version: str = "geak_kernel_sft_v1"
    dataset_version: str = "phase1-v1"
    run_id: str = ""
    eval_dir: str = ""
    kernel_name: str = ""
    round: int = 0
    engineer_id: str = ""
    collection_mode: str = ""
    creation_timestamp: str = ""
    workspace_head: str = ""


@dataclass
class ContextState:
    """Aggregated context tracked by the Context Accumulator (§4.10)."""
    architecture: ArchitectureContext = field(default_factory=ArchitectureContext)
    contract: ContractContext = field(default_factory=ContractContext)
    baseline: BaselineContext = field(default_factory=BaselineContext)
    profile: Optional[ProfileContext] = None
    direction: Optional[DirectionContext] = None
    per_case_benchmark: Optional[PerCaseBenchmarkContext] = None
    error: Optional[ErrorContext] = None
    skill_retrieval: Optional[SkillRetrievalContext] = None
    environment: EnvironmentContext = field(default_factory=EnvironmentContext)
    model: ModelContext = field(default_factory=ModelContext)
    orchestrator: OrchestratorContext = field(default_factory=OrchestratorContext)
    session_identity: SessionIdentity = field(default_factory=SessionIdentity)


# ---------------------------------------------------------------------------
# Segment (between present() boundaries)
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    events: List[Event] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    deliverable: Optional[str] = None


# ---------------------------------------------------------------------------
# Golden Path result
# ---------------------------------------------------------------------------

@dataclass
class GoldenStep:
    event_index: int
    tool_call: Dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""
    related_reasoning: str = ""


@dataclass
class GoldenPathResult:
    golden_steps: List[GoldenStep] = field(default_factory=list)
    dead_ends: List[GoldenStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verify Receipt — §4.7
# ---------------------------------------------------------------------------

@dataclass
class VerifyReceipt:
    patch_applies: bool = False
    compile_pass: bool = False
    compile_command: str = ""
    compile_cwd: str = ""
    compile_exit_code: int = -1
    compile_stdout: str = ""
    compile_stderr: str = ""
    correctness_pass: bool = False
    correctness_cases: List[Any] = field(default_factory=list)
    correctness_seeds: List[Any] = field(default_factory=list)
    correctness_tolerance: float = 1e-3
    benchmark_valid: Optional[bool] = None
    warmup_iterations: int = 5
    measurement_iterations: int = 100
    baseline_repeats: int = 3
    candidate_repeats: int = 3
    timer_type: str = "device"
    raw_latencies_baseline: List[float] = field(default_factory=list)
    raw_latencies_candidate: List[float] = field(default_factory=list)
    aggregation_method: str = "median"
    verified_speedup: Optional[float] = None
    variance_note: str = ""
    timing_reliable: bool = True
    kernel_launch_verified: bool = False
    regression_per_case: List[Dict[str, Any]] = field(default_factory=list)
    verify_workspace: str = ""
    verify_timestamp: str = ""
    gpu_identity: str = ""
    cheating_checks: Dict[str, bool] = field(default_factory=lambda: {
        "harness_modified": False,
        "reference_delegation": False,
        "fixed_output": False,
        "cached_result": False,
        "input_mutation": False,
    })
    status: VerifyStatus = VerifyStatus.PENDING


# ---------------------------------------------------------------------------
# SFT Sample — aligned with GEAK samples.jsonl
# ---------------------------------------------------------------------------

@dataclass
class SFTSample:
    sample_id: str = ""
    split: str = "train"
    sample_domain: str = "kernel"
    sample_type: str = ""             # analysis_trajectory / opus_kernel / concept_snapshot
    task_type: str = ""               # cold_start / profile_guided / ...
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    messages: Optional[List[Dict[str, Any]]] = None  # for analysis_trajectory format

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "split": self.split,
            "sample_domain": self.sample_domain,
            "sample_type": self.sample_type,
            "task_type": self.task_type,
        }
        if self.messages is not None:
            d["messages"] = self.messages
        else:
            d["input"] = self.input_data
            d["output"] = self.output_data
        d["labels"] = self.labels
        d["provenance"] = self.provenance
        return d

    def content_hash(self) -> str:
        import json
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()
