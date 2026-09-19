"""CLI entry point for SFT Collector.

Subcommands:
  offline     Process saved trajectory files (session.v3.jsonl)
  online      Start online middleware (attach to Harness)
  etl         Merge, dedup, split, balance from multiple runs
  verify      Batch verify pending samples
"""



import argparse
import json
import logging
import sys
from typing import List, Optional


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    parser = argparse.ArgumentParser(
        prog="sft_collector",
        description="SFT Data Collector — offline + online pipelines",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- offline ----
    p_off = sub.add_parser("offline", help="Extract SFT data from saved trajectories")
    p_off.add_argument(
        "--trajectory-dir", required=True,
        help="Directory containing session.v3.jsonl files",
    )
    p_off.add_argument(
        "--run-dir", required=True,
        help="Output run directory",
    )
    p_off.add_argument(
        "--config", default=None,
        help="Path to JSON config file (harness config, collection mode, etc.)",
    )
    p_off.add_argument(
        "--collection-mode", default="direction_conditioned",
        choices=[
            "cold_start", "profile_guided", "direction_conditioned",
            "error_recovery", "regression_balance",
        ],
    )
    p_off.add_argument(
        "--skip-verify", action="store_true",
        help="Skip OPUS independent verify (for testing only)",
    )
    p_off.add_argument(
        "--workspace-dir", default="",
        help="Workspace directory for OPUS verification",
    )
    p_off.add_argument(
        "--cap-analysis", type=int, default=200,
        help="Stop collecting analysis_trajectory after this many (default: 200)",
    )
    p_off.add_argument(
        "--cap-opus", type=int, default=400,
        help="Stop collecting opus_kernel after this many (default: 400)",
    )
    p_off.add_argument(
        "--cap-concept", type=int, default=400,
        help="Stop collecting concept_snapshot after this many (default: 400)",
    )

    # ---- online ----
    p_on = sub.add_parser("online", help="Start online middleware (standalone test mode)")
    p_on.add_argument(
        "--run-dir", required=True,
        help="Output run directory",
    )
    p_on.add_argument(
        "--config", required=True,
        help="Path to JSON config file",
    )

    # ---- etl ----
    p_etl = sub.add_parser("etl", help="Merge, dedup, split from multiple runs")
    p_etl.add_argument(
        "--run-dirs", nargs="+", required=True,
        help="List of run directories to merge",
    )
    p_etl.add_argument(
        "--output-dir", required=True,
        help="Output directory for merged dataset",
    )
    p_etl.add_argument("--dev-ratio", type=float, default=0.1)
    p_etl.add_argument("--held-out-ratio", type=float, default=0.05)

    # ---- verify ----
    p_ver = sub.add_parser("verify", help="Batch verify pending samples")
    p_ver.add_argument(
        "--run-dir", required=True,
        help="Run directory with pending/ samples",
    )
    p_ver.add_argument(
        "--workspace-dir", required=True,
        help="Workspace directory for verification",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "offline":
        return _cmd_offline(args)
    elif args.command == "online":
        return _cmd_online(args)
    elif args.command == "etl":
        return _cmd_etl(args)
    elif args.command == "verify":
        return _cmd_verify(args)
    return 1


def _cmd_offline(args: argparse.Namespace) -> int:
    from .offline.offline_agent import OfflineAgent, OfflineExtractionConfig

    harness_config = {}
    if args.config:
        with open(args.config) as f:
            harness_config = json.load(f)

    config = OfflineExtractionConfig(
        collection_mode=args.collection_mode,
        harness_config=harness_config,
        workspace_dir=args.workspace_dir,
        skip_verify=args.skip_verify,
        cap_analysis=args.cap_analysis,
        cap_opus=args.cap_opus,
        cap_concept=args.cap_concept,
    )

    agent = OfflineAgent(run_dir=args.run_dir, config=config)
    totals = agent.process_batch(args.trajectory_dir)

    print(f"\n=== Offline Extraction Complete ===")
    print(f"  Analysis trajectories: {totals['analysis']}")
    print(f"  OPUS kernels:          {totals['opus']}")
    print(f"  Concept snapshots:     {totals['concept']}")
    print(f"  Dead ends (RL neg):    {totals['dead_end']}")
    print(f"  Rejected:              {totals['rejected']}")
    print(f"  Output: {args.run_dir}/sft_collected/")
    return 0


def _cmd_online(args: argparse.Namespace) -> int:
    """Standalone test mode for online middleware.

    In production, the middleware is attached programmatically
    by the Harness, not via CLI.
    """
    from .online.middleware import SFTCollectorMiddleware

    with open(args.config) as f:
        config = json.load(f)

    middleware = SFTCollectorMiddleware(run_dir=args.run_dir, config=config)
    print(f"Online middleware initialized. Run directory: {args.run_dir}")
    print("In production, call middleware.attach(harness) to start collection.")
    print("This standalone mode is for testing configuration only.")
    return 0


def _cmd_etl(args: argparse.Namespace) -> int:
    from .etl.merge_dedup_split import run_etl

    result = run_etl(
        run_dirs=args.run_dirs,
        output_dir=args.output_dir,
        dev_ratio=args.dev_ratio,
        held_out_ratio=args.held_out_ratio,
    )

    balance = result["balance_report"]
    print(f"\n=== ETL Complete ===")
    print(f"  Total samples: {balance['total_samples']}")
    print(f"  By type:       {balance['by_sample_type']}")
    print(f"  By language:   {balance['by_language']}")
    print(f"  By task type:  {balance['by_task_type']}")
    if balance.get("warnings"):
        print(f"  Warnings:")
        for w in balance["warnings"]:
            print(f"    - {w}")
    print(f"  Output: {args.output_dir}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Batch verify pending samples."""
    import os
    import subprocess
    import tempfile
    from .core.independent_verifier import IndependentVerifier
    from .core.schema import VerifyStatus

    pending_dir = os.path.join(args.run_dir, "sft_collected", "pending")
    samples_dir = os.path.join(args.run_dir, "sft_collected", "samples")
    rejected_dir = os.path.join(args.run_dir, "sft_collected", "rejected")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    if not os.path.isdir(pending_dir):
        print("No pending/ directory found.")
        return 0

    verifier = IndependentVerifier()
    if not verifier.compiler_available():
        print("ERROR: OPUS compiler not available. Cannot verify.")
        return 1

    files = [f for f in os.listdir(pending_dir) if f.endswith(".json")]
    print(f"Found {len(files)} pending samples to verify.")

    passed = 0
    failed = 0
    skipped = 0
    for fname in files:
        fpath = os.path.join(pending_dir, fname)
        with open(fpath) as f:
            sample = json.load(f)

        parent = sample.get("input", {}).get("parent_source", "")
        patch = sample.get("output", {}).get("patch", "")
        if not parent or not patch:
            skipped += 1
            continue

        candidate = _reconstruct_candidate(parent, patch)
        if candidate is None:
            dest = os.path.join(rejected_dir, fname)
            os.rename(fpath, dest)
            failed += 1
            continue

        ws_config = {"workspace_dir": args.workspace_dir}
        receipt = verifier.verify(
            parent, candidate, patch, ws_config, args.run_dir,
            sample.get("sample_id", fname),
        )

        if receipt.status == VerifyStatus.PASSED:
            dest = os.path.join(samples_dir, fname)
            os.rename(fpath, dest)
            passed += 1
        else:
            dest = os.path.join(rejected_dir, fname)
            os.rename(fpath, dest)
            failed += 1

    print(f"Verified: {passed} passed, {failed} failed, {skipped} skipped")
    return 0


def _reconstruct_candidate(parent: str, patch: str):
    # type: (str, str) -> Optional[str]
    """Apply patch to parent source to reconstruct the candidate."""
    import subprocess
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            src = os.path.join(tmpdir, "source.opus")
            with open(src, "w") as f:
                f.write(parent)
            patch_file = os.path.join(tmpdir, "diff.patch")
            with open(patch_file, "w") as f:
                f.write(patch)
            result = subprocess.run(
                ["patch", "-p1", "--forward", "-i", patch_file],
                capture_output=True, text=True, cwd=tmpdir, timeout=30,
            )
            if result.returncode != 0:
                return None
            with open(src) as f:
                return f.read()
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
