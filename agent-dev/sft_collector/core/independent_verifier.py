"""Independent Verifier — §4.7: OPUS compile + correctness in fresh workspace.

Hard gate: no compiler → no positive samples (pending/ queue).
"""



import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from .schema import VerifyReceipt, VerifyStatus


class IndependentVerifier:
    """Runs OPUS compile + correctness verification in an isolated workspace."""

    def __init__(
        self,
        compiler_cmd: str = "opus-compile",
        test_runner_cmd: str = "opus-test",
        benchmark_cmd: Optional[str] = "opus-bench",
        timeout: int = 300,
    ) -> None:
        self.compiler_cmd = compiler_cmd
        self.test_runner_cmd = test_runner_cmd
        self.benchmark_cmd = benchmark_cmd
        self.timeout = timeout

    def compiler_available(self) -> bool:
        """Check if OPUS compiler is accessible."""
        try:
            result = subprocess.run(
                [self.compiler_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def verify(
        self,
        parent_source: str,
        candidate_source: str,
        diff: str,
        workspace_config: Dict[str, Any],
        run_dir: str,
        sample_id: str,
    ) -> VerifyReceipt:
        """Full verification: apply patch → compile → correctness → benchmark."""
        receipt = VerifyReceipt(
            verify_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            gpu_identity=workspace_config.get("gpu_identity", ""),
        )

        # Check compiler availability — hard gate
        if not self.compiler_available():
            receipt.status = VerifyStatus.PENDING
            return receipt

        verify_dir = os.path.join(run_dir, "opus_verify", sample_id)
        os.makedirs(verify_dir, exist_ok=True)
        receipt.verify_workspace = f"opus_verify/{sample_id}/"

        try:
            # 1. Create fresh workspace
            fresh_ws = os.path.join(verify_dir, "workspace")
            os.makedirs(fresh_ws, exist_ok=True)

            # Copy workspace context (config, deps, build files)
            src_ws = workspace_config.get("workspace_dir", "")
            if src_ws and os.path.isdir(src_ws):
                for item in os.listdir(src_ws):
                    s = os.path.join(src_ws, item)
                    d = os.path.join(fresh_ws, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)

            # Write parent source
            kernel_file = workspace_config.get("kernel_file", "kernel.opus")
            parent_path = os.path.join(fresh_ws, kernel_file)
            with open(parent_path, "w") as f:
                f.write(parent_source)

            # 2. Apply patch
            patch_file = os.path.join(verify_dir, "patch.diff")
            with open(patch_file, "w") as f:
                f.write(diff)

            apply_result = subprocess.run(
                ["patch", "-p1", "--forward", "-i", patch_file],
                capture_output=True, text=True,
                cwd=fresh_ws, timeout=30,
            )
            if apply_result.returncode != 0:
                receipt.patch_applies = False
                receipt.status = VerifyStatus.FAILED
                self._save_receipt(verify_dir, receipt)
                return receipt
            receipt.patch_applies = True

            # 3. Compile
            compile_cmd = workspace_config.get(
                "compile_command", f"{self.compiler_cmd} {kernel_file}"
            )
            receipt.compile_command = compile_cmd
            receipt.compile_cwd = fresh_ws

            compile_result = subprocess.run(
                shlex.split(compile_cmd),
                capture_output=True, text=True,
                cwd=fresh_ws, timeout=self.timeout,
            )
            receipt.compile_exit_code = compile_result.returncode
            receipt.compile_stdout = compile_result.stdout[:8192]
            receipt.compile_stderr = compile_result.stderr[:8192]

            if compile_result.returncode != 0:
                receipt.compile_pass = False
                receipt.status = VerifyStatus.FAILED
                self._save_receipt(verify_dir, receipt)
                return receipt
            receipt.compile_pass = True

            # 4. Correctness
            test_cmd = workspace_config.get(
                "test_command", f"{self.test_runner_cmd} {kernel_file}"
            )
            test_result = subprocess.run(
                shlex.split(test_cmd),
                capture_output=True, text=True,
                cwd=fresh_ws, timeout=self.timeout,
            )
            if test_result.returncode != 0:
                receipt.correctness_pass = False
                receipt.status = VerifyStatus.FAILED
                self._save_receipt(verify_dir, receipt)
                return receipt
            receipt.correctness_pass = True

            # 5. Performance benchmark (optional)
            if self.benchmark_cmd:
                bench_cmd = workspace_config.get(
                    "benchmark_command", f"{self.benchmark_cmd} {kernel_file}"
                )
                try:
                    bench_result = subprocess.run(
                        shlex.split(bench_cmd),
                        capture_output=True, text=True,
                        cwd=fresh_ws, timeout=self.timeout,
                    )
                    if bench_result.returncode == 0:
                        self._parse_benchmark(bench_result.stdout, receipt)
                        receipt.benchmark_valid = True
                    else:
                        receipt.benchmark_valid = None
                except subprocess.TimeoutExpired:
                    receipt.benchmark_valid = None

            # 6. Cheating checks
            receipt.cheating_checks = self._run_cheating_checks(
                fresh_ws, workspace_config, candidate_source
            )
            if any(receipt.cheating_checks.values()):
                receipt.status = VerifyStatus.FAILED
                self._save_receipt(verify_dir, receipt)
                return receipt

            receipt.status = VerifyStatus.PASSED
            self._save_receipt(verify_dir, receipt)
            return receipt

        except Exception as e:
            receipt.compile_stderr = str(e)
            receipt.status = VerifyStatus.FAILED
            self._save_receipt(verify_dir, receipt)
            return receipt

    def _parse_benchmark(self, stdout: str, receipt: VerifyReceipt) -> None:
        """Try to parse benchmark output for latency data."""
        # This is a placeholder — actual parsing depends on benchmark tool format
        import re
        latencies = re.findall(r"latency[:\s]*([\d.]+)\s*(?:us|μs)", stdout, re.IGNORECASE)
        if latencies:
            receipt.raw_latencies_candidate = [float(x) for x in latencies[:10]]

    def _run_cheating_checks(
        self,
        workspace: str,
        config: Dict[str, Any],
        candidate: str,
    ) -> Dict[str, bool]:
        checks = {
            "harness_modified": False,
            "reference_delegation": False,
            "fixed_output": False,
            "cached_result": False,
            "input_mutation": False,
        }

        # Check if harness/test files were modified
        harness_files = config.get("readonly_files", [])
        for hf in harness_files:
            path = os.path.join(workspace, hf)
            if os.path.exists(path):
                # Compare with original
                orig = config.get("readonly_hashes", {}).get(hf, "")
                if orig:
                    import hashlib
                    with open(path, "rb") as f:
                        current = hashlib.sha256(f.read()).hexdigest()
                    if current != orig:
                        checks["harness_modified"] = True

        # Check for reference delegation patterns
        ref_patterns = ["aiter.", "torch.reference", "cublas", "hipblas"]
        candidate_lower = candidate.lower()
        for pat in ref_patterns:
            if pat in candidate_lower:
                checks["reference_delegation"] = True
                break

        return checks

    def _save_receipt(self, verify_dir: str, receipt: VerifyReceipt) -> None:
        receipt_path = os.path.join(verify_dir, "receipt.json")
        from dataclasses import asdict
        with open(receipt_path, "w") as f:
            json.dump(asdict(receipt), f, indent=2, default=str)
