"""Context Accumulator — §4.10: tracks all kernel-coding-relevant context.

Maintains 11 context types. When OPUS collection triggers, the current snapshot
becomes the sample's `input` field.
"""



import copy
import hashlib
import re
import time
from typing import Any, Dict, List, Optional

from .schema import (
    ArchitectureContext,
    BaselineContext,
    ContractContext,
    ContextState,
    DirectionContext,
    EnvironmentContext,
    ErrorContext,
    ModelContext,
    OrchestratorContext,
    PerCaseBenchmarkContext,
    ProfileContext,
    SessionIdentity,
    SkillRetrievalContext,
)

# Patterns for auto-detection
_ISA_PATH_PAT = re.compile(r"(ISA|isa|hardware.spec|arch)", re.IGNORECASE)
_ROCPROF_PAT = re.compile(r"(rocprof|SQ_WAVES|TCC_HIT|TCC_MISS|GRBM_COUNT|SPI_)", re.IGNORECASE)
_BASELINE_PAT = re.compile(r"(baseline.latency|baseline_us|baseline_ms|baseline.throughput)", re.IGNORECASE)
_PER_CASE_PAT = re.compile(r"(case_id|regression_budget|budget_exceeded|per.case)", re.IGNORECASE)
_COMPILE_ERROR_PAT = re.compile(r"(error:|fatal error|compilation failed|undefined reference)", re.IGNORECASE)
_CORRECTNESS_ERROR_PAT = re.compile(r"(FAIL|mismatch|tolerance exceeded|correctness failed)", re.IGNORECASE)
_DIRECTION_PAT = re.compile(r"(优化方向|strategy|应该.*替代|建议.*使用|focus on)", re.IGNORECASE)
_SKILL_RETRIEVAL_PAT = re.compile(r"(skill_retrieval|retrieval_query|selected_entries|corpus_hash)")


class ContextAccumulator:
    """Incrementally builds ContextState from session events."""

    def __init__(self) -> None:
        self.state = ContextState()
        self._agent_first_action_ts: Optional[float] = None

    # ----- session start -----

    def on_session_start(
        self,
        harness_config: Dict[str, Any],
        env_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = harness_config

        # collection_mode / task_type
        cm = cfg.get("collection_mode", {})
        task_type = cm.get("task_type", "") if isinstance(cm, dict) else str(cm)
        self.state.session_identity.collection_mode = task_type

        # orchestrator
        orch = cfg.get("orchestrator", {})
        o = self.state.orchestrator
        o.budget_total = orch.get("budget_total") or orch.get("budget")
        o.budget_used = orch.get("budget_used", 0)
        o.round_limit = orch.get("round_limit")
        o.candidate_floor = orch.get("candidate_floor")
        o.min_improve = orch.get("min_improve")
        o.max_no_improve = orch.get("max_no_improve")
        o.warm_start_mode = orch.get("warm_start_mode", False)
        o.gpu_pool_pin_mode = orch.get("gpu_pool_pin_mode", "")

        # model
        mdl = cfg.get("model", {})
        m = self.state.model
        m.model_checkpoint = mdl.get("model_checkpoint", mdl.get("model", ""))
        m.tokenizer_revision = mdl.get("tokenizer_revision", "")
        m.adapter_version = mdl.get("adapter_version")
        m.chat_template = mdl.get("chat_template", "")
        m.temperature = mdl.get("temperature", 0.7)
        m.top_p = mdl.get("top_p", 0.9)
        m.max_output_tokens = mdl.get("max_output_tokens", 32000)
        m.stop_tokens = mdl.get("stop_tokens", [])

        # session identity
        si = self.state.session_identity
        si.run_id = cfg.get("run_id", "")
        si.eval_dir = cfg.get("eval_dir", "")
        si.kernel_name = cfg.get("kernel_name", "")
        si.round = cfg.get("round", 0)
        si.engineer_id = cfg.get("engineer_id", "")
        si.creation_timestamp = cfg.get("creation_timestamp", "")
        si.workspace_head = cfg.get("workspace_head", "")

        # environment
        if env_snapshot:
            e = self.state.environment
            e.rocm_version = env_snapshot.get("rocm_version", "")
            e.driver_version = env_snapshot.get("driver_version", "")
            e.hipcc_version = env_snapshot.get("hipcc_version", "")
            e.pytorch_version = env_snapshot.get("pytorch_version", "")
            e.triton_version = env_snapshot.get("triton_version", "")
            e.container_image = env_snapshot.get("container_image", "")
            e.container_digest = env_snapshot.get("container_digest", "")
            e.geak_git_sha = env_snapshot.get("geak_git_sha", "")
            e.lumen_git_sha = env_snapshot.get("lumen_git_sha", "")
            e.gpu_clocks = env_snapshot.get("gpu_clocks", {})
            e.gpu_power_mode = env_snapshot.get("gpu_power_mode", "")
            e.gpu_temperature = env_snapshot.get("gpu_temperature", "")

        # contract from config
        contract_cfg = cfg.get("contract", {})
        if contract_cfg:
            c = self.state.contract
            c.operator = contract_cfg.get("operator", "")
            c.language = contract_cfg.get("language", "")
            c.language_version = contract_cfg.get("language_version", "")
            c.backend_version = contract_cfg.get("backend_version", "")
            c.modifiable_files = contract_cfg.get("modifiable_files", [])
            c.layout = contract_cfg.get("layout", "")
            c.input_dtype = contract_cfg.get("input_dtype", "")
            c.output_dtype = contract_cfg.get("output_dtype", "")
            c.optimization_target = contract_cfg.get("optimization_target", "")
            c.correctness_tol = contract_cfg.get("correctness_tolerance")
            c.regression_constraints = contract_cfg.get("regression_constraints", [])
            if contract_cfg.get("dimensions"):
                c.dimensions = contract_cfg["dimensions"]

        # architecture from config
        arch_cfg = cfg.get("architecture", {})
        if arch_cfg:
            a = self.state.architecture
            a.target_gpu = arch_cfg.get("target_gpu", arch_cfg.get("gfx", ""))
            a.gpu_sku = arch_cfg.get("gpu_sku", "")

    def record_agent_first_action(self, ts: float) -> None:
        if self._agent_first_action_ts is None:
            self._agent_first_action_ts = ts

    # ----- tool results -----

    def on_tool_result(
        self,
        tool_name: str,
        path: str,
        content: str,
        tool_call_id: Optional[int] = None,
    ) -> None:
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # architecture from ISA / hardware spec reads
        if tool_name == "read" and _ISA_PATH_PAT.search(path):
            self.state.architecture.source_refs.append(
                {"file": path, "content_hash": content_hash}
            )
            self._extract_architecture_details(content)

        # profile from rocprof output
        if _ROCPROF_PAT.search(content):
            if self.state.profile is None:
                self.state.profile = ProfileContext()
            self._extract_profile(content, tool_call_id, content_hash)

        # baseline
        if _BASELINE_PAT.search(content):
            self._extract_baseline(content)

        # per-case benchmark
        if _PER_CASE_PAT.search(content):
            if self.state.per_case_benchmark is None:
                self.state.per_case_benchmark = PerCaseBenchmarkContext()
            self._extract_per_case_benchmark(content)

        # compile / correctness errors
        if _COMPILE_ERROR_PAT.search(content):
            if self.state.error is None:
                self.state.error = ErrorContext()
            self.state.error.compile_error = {
                "stderr": content[:4096],
                "error_category": "compile",
            }

        if _CORRECTNESS_ERROR_PAT.search(content):
            if self.state.error is None:
                self.state.error = ErrorContext()
            self.state.error.correctness_error = {
                "failed_output": content[:4096],
                "error_category": "correctness",
            }

    # ----- reasoning blocks -----

    def on_reasoning_block(self, text: str) -> None:
        # Direction: only extract from reasoning if not already frozen via TechLead
        if self.state.direction is None or not self.state.direction.frozen:
            if _DIRECTION_PAT.search(text):
                if self.state.direction is None:
                    self.state.direction = DirectionContext()
                if not self.state.direction.frozen:
                    self.state.direction.optimization_strategy = text[:2048]

        # Contract understanding
        self._try_extract_contract_from_text(text)

    # ----- inbox messages -----

    def on_inbox_message(self, sender: str, text: str, timestamp: float) -> None:
        sender_lower = sender.lower()

        # TechLead direction → FREEZE immediately (causality guarantee)
        if "techlead" in sender_lower or "tech_lead" in sender_lower:
            d = DirectionContext()
            d.direction_source = "tech_lead"
            d.direction_prompt = text
            d.direction_prompt_hash = hashlib.sha256(text.encode()).hexdigest()
            d.direction_created_at = str(timestamp)
            d.frozen = True
            self.state.direction = d

        # Task assignment
        if "task" in sender_lower or "orchestrator" in sender_lower:
            self._try_extract_contract_from_text(text)

    def on_skill_retrieval(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
        corpus_hash: str = "",
        corpus_git_sha: str = "",
    ) -> None:
        self.state.skill_retrieval = SkillRetrievalContext(
            retrieval_query=query,
            candidates=candidates,
            selected_entries=selected,
            corpus_hash=corpus_hash,
            corpus_git_sha=corpus_git_sha,
        )

    # ----- snapshot -----

    def snapshot(self) -> ContextState:
        return copy.deepcopy(self.state)

    def direction_causality_ok(self) -> bool:
        """Check direction_created_at < agent_first_action."""
        d = self.state.direction
        if d is None or not d.direction_created_at:
            return True
        if self._agent_first_action_ts is None:
            return True
        try:
            created = float(d.direction_created_at)
            return created < self._agent_first_action_ts
        except ValueError:
            return True

    # ----- helpers -----

    def _extract_architecture_details(self, content: str) -> None:
        a = self.state.architecture
        if not a.target_gpu:
            m = re.search(r"gfx\d+", content)
            if m:
                a.target_gpu = m.group(0)

    def _extract_profile(
        self, content: str, tool_call_id: Optional[int], content_hash: str
    ) -> None:
        assert self.state.profile is not None
        p = self.state.profile
        # Try to detect profiler tool
        if "rocprof-compute" in content.lower():
            p.profiler_tool = "rocprof-compute"
        elif "rocprof" in content.lower():
            p.profiler_tool = "rocprof"
        p.source_refs.append(
            {"tool_call_id": str(tool_call_id), "content_hash": content_hash}
        )

    def _extract_baseline(self, content: str) -> None:
        b = self.state.baseline
        m = re.search(r"baseline[_\s]*latency[:\s]*([\d.]+)", content, re.IGNORECASE)
        if m:
            b.baseline_latency = {"mean": float(m.group(1))}

    def _extract_per_case_benchmark(self, content: str) -> None:
        assert self.state.per_case_benchmark is not None
        pcb = self.state.per_case_benchmark
        if not pcb.regression_feedback and "退化" in content:
            pcb.regression_feedback = content[:2048]

    def _try_extract_contract_from_text(self, text: str) -> None:
        c = self.state.contract
        if not c.operator:
            m = re.search(
                r"operator[:\s]+([\w_]+)", text, re.IGNORECASE
            )
            if m:
                c.operator = m.group(1)
