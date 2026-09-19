# Qwen3-30B-A3B Agentic RL — GEAK Kernel Agent Training Runbook

> Train an AMD kernel optimisation agent using GSPO + GEAK sandbox + TRLOO.
> The agent learns to optimise HIP, Triton, and FlyDSL kernels sourced from aiter.
>
> **Hardware**: 2 nodes on spur cluster
> - Node A (MI355X / gfx950): GEAK sandbox + ATOM rollout (8 GPUs)
> - Node B (MI300X / gfx942): Megatron training (8 GPUs)
>
> **Prerequisites**: spur JobID with 2 nodes allocated. See `SPUR_NODE_ACCESS_GUIDE.md`.

---

## 0. Agent Rules

1. User gives a JobID — use only that JobID. Never `sbatch`/`spur alloc` a new one.
2. Never `scancel` unless user explicitly asks.
3. Enter compute nodes via `spur exec "$JOBID" bash -lc "..."` (head) or `~/nx.sh` (non-head). Never SSH directly.
4. All paths below assume `$HOME` is shared NFS across nodes. Node-local `/mnt/m2m_nobackup` is per-node only.
5. Before running any destructive command, confirm with user.

---

## 1. Collect User Inputs

```bash
# Fixed at the start of the session
JOBID=<user provides>
scontrol show job "$JOBID" | head -20

# Identify the two nodes
NODES=( $(scontrol show job "$JOBID" | grep -oE 'crsuse2-m2m-[0-9]+' | sort -u) )
echo "Node count: ${#NODES[@]}"
echo "Node A (sandbox/rollout): ${NODES[0]}"
echo "Node B (trainer):         ${NODES[1]}"
```

Verify GPU architecture on each node:

```bash
# Node A — expect gfx950 (MI355X)
spur exec "$JOBID" bash -lc 'rocm-smi --showproductname 2>/dev/null | head -5; rocminfo 2>/dev/null | grep -m1 "gfx9"'

# Node B — expect gfx942 (MI300X)
JOBID=$JOBID ~/nx.sh "${NODES[1]}" 'rocm-smi --showproductname 2>/dev/null | head -5; rocminfo 2>/dev/null | grep -m1 "gfx9"'
```

If node GPU architectures are swapped, swap the role assignments. MI355X (gfx950) is sandbox/rollout; MI300X (gfx942) is trainer.

Collect workspace paths:

```bash
# All paths on shared NFS ($HOME)
WORK_ROOT="$HOME/agentic-rl-workspace"
LUMENRL_DIR="$HOME/Lumen-RL"
AITER_DIR="$HOME/aiter"
GEAK_DIR="$HOME/GEAK"
MODEL_DIR="$HOME/models/Qwen3-30B-A3B"
RUNTIME_DIR="$WORK_ROOT/runtime"
CASES_DIR="$WORK_ROOT/geak-cases"

mkdir -p "$WORK_ROOT" "$RUNTIME_DIR/logs" "$RUNTIME_DIR/ckpts" "$RUNTIME_DIR/configs" "$CASES_DIR"
```

Save to env file:

```bash
cat > "$HOME/agentic-rl.env" <<EOF
export JOBID='$JOBID'
export SANDBOX_NODE='${NODES[0]}'
export TRAINER_NODE='${NODES[1]}'
export WORK_ROOT='$WORK_ROOT'
export LUMENRL_DIR='$LUMENRL_DIR'
export AITER_DIR='$AITER_DIR'
export GEAK_DIR='$GEAK_DIR'
export MODEL_DIR='$MODEL_DIR'
export RUNTIME_DIR='$RUNTIME_DIR'
export CASES_DIR='$CASES_DIR'
EOF
```

---

## 2. Discover Kernel Tasks from aiter

aiter provides three categories of kernels suitable for GEAK training tasks:

### 2.1 Catalog aiter Kernels

The following script scans the aiter source tree and generates a GEAK-compatible cases YAML.

```bash
source "$HOME/agentic-rl.env"

python3 - <<'PY'
import os
import yaml
from pathlib import Path

aiter_dir = Path(os.environ["AITER_DIR"])
cases_dir = Path(os.environ["CASES_DIR"])

tasks = []

# ─── HIP kernels (csrc/) ───
hip_kernels = {
    "batched_gemm_a8w8": {
        "source": "csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8.cu",
        "task_type": "hip2hip",
        "instruction": "Optimize CK batched A8W8 quantized GEMM for MoE expert compute on gfx942.",
        "target_functions": ["batched_gemm_a8w8"],
    },
    "batched_gemm_bf16": {
        "source": "csrc/ck_batched_gemm_bf16/batched_gemm_bf16.cu",
        "task_type": "hip2hip",
        "instruction": "Optimize CK batched BF16 GEMM for dense linear layers on gfx942.",
        "target_functions": ["batched_gemm_bf16"],
    },
    "gemm_a8w8_blockscale": {
        "source": "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu",
        "task_type": "hip2hip",
        "instruction": "Optimize FP8 block-scaled GEMM for quantized inference on gfx942.",
        "target_functions": ["gemm_a8w8_blockscale"],
    },
    "asm_mha_fwd": {
        "source": "csrc/py_itfs_cu/asm_mha_fwd.cu",
        "task_type": "hip2hip",
        "instruction": "Optimize assembly-level multi-head attention forward pass on gfx942.",
        "target_functions": ["asm_mha_fwd"],
    },
    "asm_fmoe": {
        "source": "csrc/py_itfs_cu/asm_fmoe.cu",
        "task_type": "hip2hip",
        "instruction": "Optimize fused MoE dispatch and expert GEMM assembly kernel on gfx942.",
        "target_functions": ["asm_fmoe"],
    },
    "asm_mla": {
        "source": "csrc/py_itfs_cu/asm_mla.cu",
        "task_type": "hip2hip",
        "instruction": "Optimize multi-latent attention assembly kernel for DeepSeek-style MLA on gfx942.",
        "target_functions": ["asm_mla"],
    },
    "asm_pa": {
        "source": "csrc/py_itfs_cu/asm_pa.cu",
        "task_type": "hip2hip",
        "instruction": "Optimize paged attention assembly kernel for KV cache serving on gfx942.",
        "target_functions": ["asm_pa"],
    },
    "deepgemm": {
        "source": "csrc/ck_deepgemm/deepgemm.cu",
        "task_type": "hip2hip",
        "instruction": "Optimize DeepGEMM CK-based kernel for high-throughput matrix multiply on gfx942.",
        "target_functions": ["deepgemm"],
    },
}

for name, info in hip_kernels.items():
    src = aiter_dir / info["source"]
    if src.exists():
        tasks.append({
            "task_id": f"hip_{name}",
            "task_type": info["task_type"],
            "kernel_path": str(src),
            "target_functions": info["target_functions"],
            "instruction": info["instruction"],
            "category": "hip",
        })

# ─── Triton kernels (aiter/ops/triton/) ───
triton_kernels = {
    "softmax": {
        "source": "aiter/ops/triton/softmax.py",
        "instruction": "Optimize Triton softmax kernel for large vocabulary logits on MI300X.",
        "target_functions": ["_softmax_kernel"],
    },
    "topk": {
        "source": "aiter/ops/triton/topk.py",
        "instruction": "Optimize Triton top-k selection kernel for MoE router on MI300X.",
        "target_functions": ["_topk_kernel"],
    },
    "activation": {
        "source": "aiter/ops/triton/activation.py",
        "instruction": "Optimize Triton fused activation (SiLU/GELU) kernel for MLP layers on MI300X.",
        "target_functions": ["_activation_kernel"],
    },
    "kv_cache": {
        "source": "aiter/ops/triton/kv_cache.py",
        "instruction": "Optimize Triton KV cache copy/reshape kernel for paged attention on MI300X.",
        "target_functions": ["_kv_cache_kernel"],
    },
    "gmm": {
        "source": "aiter/ops/triton/gmm.py",
        "instruction": "Optimize Triton grouped matrix multiply for MoE expert batching on MI300X.",
        "target_functions": ["_gmm_kernel"],
    },
    "gather_kv_b_proj": {
        "source": "aiter/ops/triton/gather_kv_b_proj.py",
        "instruction": "Optimize Triton gather + KV B-projection fused kernel for MLA decode on MI300X.",
        "target_functions": ["_gather_kv_b_proj_kernel"],
    },
}

for name, info in triton_kernels.items():
    src = aiter_dir / info["source"]
    if src.exists():
        tasks.append({
            "task_id": f"triton_{name}",
            "task_type": "triton2triton",
            "kernel_path": str(src),
            "target_functions": info["target_functions"],
            "instruction": info["instruction"],
            "category": "triton",
        })

# ─── FlyDSL kernels (aiter/ops/flydsl/) ───
flydsl_kernels = {
    "fmha": {
        "source": "aiter/ops/flydsl/fmha_kernels.py",
        "instruction": "Optimize FlyDSL fused multi-head attention kernel for MI300X/MI355X.",
        "target_functions": ["fmha_fwd_kernel"],
    },
    "gemm": {
        "source": "aiter/ops/flydsl/gemm_kernels.py",
        "instruction": "Optimize FlyDSL GEMM kernel with preshuffle and block-scale for MI300X/MI355X.",
        "target_functions": ["gemm_kernel"],
    },
    "moe": {
        "source": "aiter/ops/flydsl/moe_kernels.py",
        "instruction": "Optimize FlyDSL MoE 2-stage GEMM kernel for expert-parallel execution on MI300X/MI355X.",
        "target_functions": ["moe_gemm_kernel"],
    },
    "linear_attention": {
        "source": "aiter/ops/flydsl/linear_attention_kernels.py",
        "instruction": "Optimize FlyDSL linear attention kernel for gated delta net on MI300X/MI355X.",
        "target_functions": ["linear_attention_kernel"],
    },
    "silu_and_mul": {
        "source": "aiter/ops/flydsl/kernels/silu_and_mul_fq.py",
        "instruction": "Optimize FlyDSL fused SiLU-and-multiply with FP8 quantization kernel.",
        "target_functions": ["silu_and_mul_fq_kernel"],
    },
    "splitk_hgemm": {
        "source": "aiter/ops/flydsl/kernels/splitk_hgemm.py",
        "instruction": "Optimize FlyDSL split-K half-precision GEMM kernel for small-M decode on MI300X.",
        "target_functions": ["splitk_hgemm_kernel"],
    },
}

for name, info in flydsl_kernels.items():
    src = aiter_dir / info["source"]
    if src.exists():
        tasks.append({
            "task_id": f"flydsl_{name}",
            "task_type": "flydsl2flydsl",
            "kernel_path": str(src),
            "target_functions": info["target_functions"],
            "instruction": info["instruction"],
            "category": "flydsl",
        })

# ─── Write cases YAML ───
output = {
    "description": "aiter kernel optimisation tasks for GEAK agentic RL training",
    "gpu_arch": ["gfx942", "gfx950"],
    "source": "aiter",
    "tasks": tasks,
}

out_path = cases_dir / "aiter-kernel-tasks.yaml"
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

# Summary
by_cat = {}
for t in tasks:
    by_cat.setdefault(t["category"], []).append(t["task_id"])

print(f"Total tasks: {len(tasks)}")
for cat, ids in by_cat.items():
    print(f"  {cat}: {len(ids)} tasks")
    for tid in ids:
        print(f"    - {tid}")
print(f"\nWritten to: {out_path}")
PY
```

If some kernel files are missing (e.g., aiter not cloned), the script silently skips them. Verify the output file has tasks for all three categories.

### 2.2 Verify Generated Cases

```bash
source "$HOME/agentic-rl.env"
cat "$CASES_DIR/aiter-kernel-tasks.yaml"
```

Expected: tasks across `hip`, `triton`, and `flydsl` categories.

---

## 3. Network Discovery

Both nodes must discover their Ray IP and, if available, RoCE RDMA configuration. Follow the same procedure as `qwen3-30b-a3b-lumenrl-megatron-runbook.md` §3.

### 3.1 Discover Ray Node IPs

On each node:

```bash
source "$HOME/agentic-rl.env"

# On sandbox node (head, via spur exec)
spur exec "$JOBID" bash -lc '
AUTO_RAY_IP=$(
  ip -4 route get 1.1.1.1 |
    awk "{for (i=1;i<=NF;i++) if (\$i==\"src\") {print \$(i+1); exit}}"
)
[ -z "$AUTO_RAY_IP" ] && AUTO_RAY_IP=$(ip -o -4 addr show scope global | awk "NR==1 {split(\$4,a,\"/\"); print a[1]}")
echo "SANDBOX_NODE_IP=$AUTO_RAY_IP"
echo "All candidates:"
ip -o -4 addr show scope global | awk "{split(\$4,a,\"/\"); printf \"  iface=%-16s ip=%s\\n\",\$2,a[1]}"
'

# On trainer node (via nx.sh)
JOBID=$JOBID ~/nx.sh "$TRAINER_NODE" '
AUTO_RAY_IP=$(
  ip -4 route get 1.1.1.1 |
    awk "{for (i=1;i<=NF;i++) if (\$i==\"src\") {print \$(i+1); exit}}"
)
[ -z "$AUTO_RAY_IP" ] && AUTO_RAY_IP=$(ip -o -4 addr show scope global | awk "NR==1 {split(\$4,a,\"/\"); print a[1]}")
echo "TRAINER_NODE_IP=$AUTO_RAY_IP"
echo "All candidates:"
ip -o -4 addr show scope global | awk "{split(\$4,a,\"/\"); printf \"  iface=%-16s ip=%s\\n\",\$2,a[1]}"
'
```

Record the IPs:

```bash
cat >> "$HOME/agentic-rl.env" <<EOF
export SANDBOX_NODE_IP='<discovered sandbox IP>'
export TRAINER_NODE_IP='<discovered trainer IP>'
EOF
```

Verify cross-node connectivity:

```bash
source "$HOME/agentic-rl.env"
spur exec "$JOBID" bash -lc "ping -c 2 $TRAINER_NODE_IP"
JOBID=$JOBID ~/nx.sh "$TRAINER_NODE" "ping -c 2 $SANDBOX_NODE_IP"
```

### 3.2 NCCL Network Configuration

Discover RoCE/RDMA configuration (same as §3.2 of the base runbook). If RoCE is not available, weight sync falls back to shared folder:

```bash
source "$HOME/agentic-rl.env"
spur exec "$JOBID" bash -lc '
ls /dev/infiniband 2>/dev/null && echo "RDMA devices present" || echo "No RDMA — will use shared_folder weight sync"
'
```

Record NCCL settings in env:

```bash
cat >> "$HOME/agentic-rl.env" <<EOF
export NCCL_SOCKET_IFNAME='<discovered iface>'
export NCCL_IB_HCA='<discovered HCA or empty>'
export NCCL_IB_GID_INDEX='<discovered GID index or 0>'
export NCCL_IB_DISABLE='<0 if RDMA available, 1 otherwise>'
EOF
```

---

## 4. Verify Source Code

All code is on shared NFS (`$HOME`). Verify:

```bash
source "$HOME/agentic-rl.env"

echo "LumenRL: $(cd "$LUMENRL_DIR" && git rev-parse --short HEAD) ($(git branch --show-current))"
echo "aiter:   $(cd "$AITER_DIR" && git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "GEAK:    $(cd "$GEAK_DIR" && git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"

# Verify critical files exist
test -f "$LUMENRL_DIR/examples/AgenticRL/Qwen3_30B_A3B/run_agentic_rl.sh" && echo "run_agentic_rl.sh OK"
test -f "$LUMENRL_DIR/examples/AgenticRL/Qwen3_30B_A3B/gspo_qwen3_30b_a3b_geak_smoke.yaml" && echo "config OK"
test -f "$LUMENRL_DIR/lumenrl/algorithms/gspo.py" && echo "GSPO algorithm OK"
test -f "$LUMENRL_DIR/lumenrl/rewards/profiling_reward.py" && echo "profiling reward OK"
test -d "$LUMENRL_DIR/experiments/multi-tune-agent/geak_gym" && echo "GEAK gym OK"
test -f "$LUMENRL_DIR/experiments/multi-tune-agent/hacking_detection.py" && echo "hacking detection OK"
test -f "$LUMENRL_DIR/experiments/multi-tune-agent/turn_level_reward.py" && echo "turn-level reward OK"
test -f "$LUMENRL_DIR/experiments/multi-tune-agent/multi_turn_rollout.py" && echo "multi-turn rollout OK"
```

If any file is missing, pull the latest from the agentic RL branch:

```bash
cd "$LUMENRL_DIR" && git fetch origin && git merge --ff-only origin/dev/moe-grpo
```

---

## 5. Verify Model

```bash
source "$HOME/agentic-rl.env"
test -f "$MODEL_DIR/config.json" && echo "Model present" || echo "MISSING — download model first"
```

If missing:

```bash
source "$HOME/agentic-rl.env"
spur exec "$JOBID" bash -lc "
pip install -q huggingface_hub
python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-30B-A3B', local_dir='$MODEL_DIR')
\"
"
```

---

## 6. Environment Setup on Both Nodes

### 6.1 Install Dependencies (both nodes)

Since we're on bare spur nodes (not Docker containers), install directly:

```bash
source "$HOME/agentic-rl.env"

# On sandbox node (head)
spur exec "$JOBID" bash -lc '
pip install -e '"$LUMENRL_DIR"' 2>&1 | tail -3
pip install -e '"$AITER_DIR"' 2>&1 | tail -3
pip install gymnasium pyyaml omegaconf wandb 2>&1 | tail -3
python3 -c "import lumenrl; print(\"lumenrl OK:\", lumenrl.__file__)"
python3 -c "import aiter; print(\"aiter OK:\", aiter.__file__)"
'

# On trainer node
JOBID=$JOBID ~/nx.sh "$TRAINER_NODE" '
pip install -e '"$LUMENRL_DIR"' 2>&1 | tail -3
pip install -e '"$AITER_DIR"' 2>&1 | tail -3
python3 -c "import lumenrl; print(\"lumenrl OK:\", lumenrl.__file__)"
python3 -c "import aiter; print(\"aiter OK:\", aiter.__file__)"
'
```

### 6.2 Verify GPU Access

```bash
source "$HOME/agentic-rl.env"

# Sandbox node — expect 8 GPUs, gfx950
spur exec "$JOBID" bash -lc '
python3 -c "
import torch
print(\"GPUs:\", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f\"  GPU {i}: {torch.cuda.get_device_name(i)}\")
"
'

# Trainer node — expect 8 GPUs, gfx942
JOBID=$JOBID ~/nx.sh "$TRAINER_NODE" '
python3 -c "
import torch
print(\"GPUs:\", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f\"  GPU {i}: {torch.cuda.get_device_name(i)}\")
"
'
```

---

## 7. Generate Deployment Config

Generate a config tailored to this deployment's discovered IPs and paths:

```bash
source "$HOME/agentic-rl.env"

python3 - <<'PY'
import copy
import os
from pathlib import Path
from omegaconf import OmegaConf

src = Path(os.environ["LUMENRL_DIR"]) / "examples/AgenticRL/Qwen3_30B_A3B/gspo_qwen3_30b_a3b_geak_smoke.yaml"
out_dir = Path(os.environ["RUNTIME_DIR"]) / "configs"
out_dir.mkdir(parents=True, exist_ok=True)

cfg = OmegaConf.load(src)

# Cluster
cfg.cluster.num_nodes = 2
cfg.cluster.gpus_per_node = 8
cfg.cluster.ray_address = "auto"

# Node placement: sandbox=rollout, trainer=actor
if hasattr(cfg, "controller") and hasattr(cfg.controller, "ray"):
    cfg.controller.ray.rollout.topology_tags = {"node_ip": os.environ["SANDBOX_NODE_IP"]}
    cfg.controller.ray.actor.topology_tags = {"node_ip": os.environ["TRAINER_NODE_IP"]}

# Model
cfg.policy.model_name = os.environ["MODEL_DIR"]

# GEAK sandbox
cfg.agentic_rl.geak_root = os.environ["GEAK_DIR"]
cfg.agentic_rl.cases_path = os.environ["CASES_DIR"] + "/aiter-kernel-tasks.yaml"
cfg.agentic_rl.gpu_ids = "0"

# Checkpointing
cfg.checkpointing.checkpoint_dir = os.environ["RUNTIME_DIR"] + "/ckpts/agentic-rl-smoke"
cfg.checkpointing.resume = False

# Smoke: 3 steps
cfg.num_training_steps = 3
cfg.logger.wandb_enabled = False

smoke = out_dir / "agentic-rl-geak-smoke.yaml"
OmegaConf.save(cfg, smoke)
print(f"Smoke config: {smoke}")

# Longrun: 200 steps
longrun_cfg = copy.deepcopy(cfg)
longrun_cfg.num_training_steps = 200
longrun_cfg.checkpointing.checkpoint_dir = os.environ["RUNTIME_DIR"] + "/ckpts/agentic-rl-longrun"
longrun_cfg.checkpointing.save_steps = 20
longrun_cfg.checkpointing.save_total_limit = 3
longrun_cfg.logger.wandb_enabled = True
longrun_cfg.logger.log_interval = 1

longrun = out_dir / "agentic-rl-geak-longrun.yaml"
OmegaConf.save(longrun_cfg, longrun)
print(f"Longrun config: {longrun}")
PY
```

Verify generated configs:

```bash
source "$HOME/agentic-rl.env"
cat "$RUNTIME_DIR/configs/agentic-rl-geak-smoke.yaml" | head -30
```

---

## 8. Start Ray Cluster

### 8.1 Start Ray Head on Sandbox Node (MI355X)

```bash
source "$HOME/agentic-rl.env"

spur exec "$JOBID" bash -lc '
ulimit -n 524288
ray stop --force 2>/dev/null || true
ray start --head \
  --node-ip-address='"$SANDBOX_NODE_IP"' \
  --port=6379 \
  --num-gpus=8 \
  --num-cpus=64 \
  --dashboard-host=0.0.0.0
'
```

### 8.2 Join Ray Cluster from Trainer Node (MI300X)

```bash
source "$HOME/agentic-rl.env"

JOBID=$JOBID ~/nx.sh "$TRAINER_NODE" '
ulimit -n 524288
ray stop --force 2>/dev/null || true
ray start \
  --address='"$SANDBOX_NODE_IP"':6379 \
  --node-ip-address='"$TRAINER_NODE_IP"' \
  --num-gpus=8 \
  --num-cpus=64
'
```

### 8.3 Verify Ray Cluster

```bash
source "$HOME/agentic-rl.env"
spur exec "$JOBID" bash -lc 'ray status'
```

Expected:

```text
Active: 2 nodes
Total: 16 GPU
```

---

## 9. Smoke Test (3 Steps)

Run a 3-step smoke test to verify the full pipeline before long training:

```bash
source "$HOME/agentic-rl.env"

spur exec "$JOBID" bash -lc '
export RL_ROOT='"$WORK_ROOT"'
export DATA_ROOT='"$RUNTIME_DIR"'
export LUMENRL_DIR='"$LUMENRL_DIR"'
export AITER_DIR='"$AITER_DIR"'
export MODEL_PATH='"$MODEL_DIR"'
export GEAK_ROOT='"$GEAK_DIR"'
export CASES_PATH='"$CASES_DIR/aiter-kernel-tasks.yaml"'
export MODE=smoke
export STEPS=3
export CONFIG_OVERRIDE='"$RUNTIME_DIR/configs/agentic-rl-geak-smoke.yaml"'
export LUMENRL_KEEP_RAY_CLUSTER=1
export RUN_ID=agentic-rl-geak-smoke3
export LOG='"$RUNTIME_DIR"'/logs/agentic-rl-geak-smoke3.log
export CKPT_DIR='"$RUNTIME_DIR"'/ckpts/agentic-rl-geak-smoke3
export PYTHONPATH='"$LUMENRL_DIR"':'"$LUMENRL_DIR"'/experiments/multi-tune-agent/src:'"$LUMENRL_DIR"'/experiments/multi-tune-agent:'"$AITER_DIR"':${PYTHONPATH:-}

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export HIP_FORCE_DEV_KERNARG=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DEDUP_LOGS=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export NCCL_TIMEOUT=7200
export VLLM_USE_V1=1
export LUMENRL_LOG_LEVEL=INFO

cd '"$LUMENRL_DIR"'
python3 -u -m lumenrl.trainer.main \
  --config '"$RUNTIME_DIR/configs/agentic-rl-geak-smoke.yaml"' \
  policy.model_name='"$MODEL_DIR"' \
  agentic_rl.geak_root='"$GEAK_DIR"' \
  agentic_rl.cases_path='"$CASES_DIR/aiter-kernel-tasks.yaml"' \
  checkpointing.checkpoint_dir='"$RUNTIME_DIR"'/ckpts/agentic-rl-geak-smoke3 \
  num_training_steps=3 \
  seed=10086
'
```

### 9.1 Smoke Verification

```bash
source "$HOME/agentic-rl.env"
LOG="$RUNTIME_DIR/logs/agentic-rl-geak-smoke3.log"

# Check for successful steps
grep -a "callbacks: step=" "$LOG" | tail -5

# Check for errors
grep -aiE "Traceback|OutOfMemory|SIGABRT|=nan|Training failed" "$LOG" | tail

# Check GEAK sandbox was used
grep -ai "GEAK\|sandbox\|kernel\|evaluate" "$LOG" | tail -10

# Check GSPO loss is being computed
grep -ai "loss_pg\|loss_total\|gspo" "$LOG" | tail -5
```

Pass criteria:
- 3 consecutive steps completed
- GSPO loss is finite
- GEAK sandbox executed kernel evaluations
- No OOM, NaN, or tracebacks

---

## 10. Full Training (200 Steps)

Only proceed after smoke test passes.

```bash
source "$HOME/agentic-rl.env"

RUN_ID="agentic-rl-geak-longrun-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$RUNTIME_DIR/logs/${RUN_ID}.log"
CKPT="$RUNTIME_DIR/ckpts/${RUN_ID}"

echo "$RUN_ID" > "$RUNTIME_DIR/current_run_id.txt"
echo "$LOG_FILE" > "$RUNTIME_DIR/current_run_log.txt"
echo "$CKPT" > "$RUNTIME_DIR/current_ckpt_dir.txt"

spur exec "$JOBID" bash -lc '
export RL_ROOT='"$WORK_ROOT"'
export DATA_ROOT='"$RUNTIME_DIR"'
export LUMENRL_DIR='"$LUMENRL_DIR"'
export AITER_DIR='"$AITER_DIR"'
export MODEL_PATH='"$MODEL_DIR"'
export GEAK_ROOT='"$GEAK_DIR"'
export CASES_PATH='"$CASES_DIR/aiter-kernel-tasks.yaml"'
export MODE=longrun
export STEPS=200
export CONFIG_OVERRIDE='"$RUNTIME_DIR/configs/agentic-rl-geak-longrun.yaml"'
export LUMENRL_KEEP_RAY_CLUSTER=1
export RUN_ID='"$RUN_ID"'
export LOG='"$LOG_FILE"'
export CKPT_DIR='"$CKPT"'
export PYTHONPATH='"$LUMENRL_DIR"':'"$LUMENRL_DIR"'/experiments/multi-tune-agent/src:'"$LUMENRL_DIR"'/experiments/multi-tune-agent:'"$AITER_DIR"':${PYTHONPATH:-}

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export HIP_FORCE_DEV_KERNARG=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DEDUP_LOGS=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export NCCL_TIMEOUT=7200
export VLLM_USE_V1=1
export LUMENRL_LOG_LEVEL=INFO

if [ -f '"$RUNTIME_DIR"'/wandb.key ]; then
  export WANDB_API_KEY="$(cut -d= -f2- '"$RUNTIME_DIR"'/wandb.key | tr -d "[:space:]")"
fi

cd '"$LUMENRL_DIR"'
nohup python3 -u -m lumenrl.trainer.main \
  --config '"$RUNTIME_DIR/configs/agentic-rl-geak-longrun.yaml"' \
  policy.model_name='"$MODEL_DIR"' \
  agentic_rl.geak_root='"$GEAK_DIR"' \
  agentic_rl.cases_path='"$CASES_DIR/aiter-kernel-tasks.yaml"' \
  checkpointing.checkpoint_dir='"$CKPT"' \
  num_training_steps=200 \
  seed=10086 > '"$LOG_FILE"' 2>&1 &
echo "Training started: PID=$! RUN_ID='"$RUN_ID"'"
'
```

---

## 11. Monitoring

### 11.1 Training Progress

```bash
source "$HOME/agentic-rl.env"
LOG="$(cat "$RUNTIME_DIR/current_run_log.txt")"

# Latest step
grep -a "callbacks: step=" "$LOG" | tail -1

# GSPO metrics
grep -a "loss_pg\|loss_total\|gspo" "$LOG" | tail -3

# Hacking detection
grep -ai "hacking\|penalty\|lazy_deletion\|hardcoded" "$LOG" | tail -5

# Errors
grep -aiE "Traceback|OutOfMemory|NCCL.*timeout|SIGABRT|=nan|Training failed" "$LOG" | tail
```

### 11.2 GEAK Sandbox Activity

```bash
source "$HOME/agentic-rl.env"
LOG="$(cat "$RUNTIME_DIR/current_run_log.txt")"

# Kernel evaluations
grep -ai "evaluate\|compile\|correctness\|speedup\|profiling" "$LOG" | tail -20

# Turn-level rewards
grep -ai "turn_reward\|trloo\|advantage" "$LOG" | tail -10
```

### 11.3 Process Status

```bash
source "$HOME/agentic-rl.env"

# Check trainer process
spur exec "$JOBID" bash -lc 'pgrep -af "lumenrl.trainer.main" || echo "No training process found"'

# Ray status
spur exec "$JOBID" bash -lc 'ray status'

# GPU usage on sandbox node
spur exec "$JOBID" bash -lc 'rocm-smi'

# GPU usage on trainer node
JOBID=$JOBID ~/nx.sh "$TRAINER_NODE" 'rocm-smi'
```

### 11.4 Disk Usage

```bash
source "$HOME/agentic-rl.env"
df -h "$RUNTIME_DIR"
du -sh "$RUNTIME_DIR/ckpts"/* 2>/dev/null
du -sh "$RUNTIME_DIR/logs"/* 2>/dev/null
```

---

## 12. Stop and Resume

### 12.1 Stop Training

```bash
source "$HOME/agentic-rl.env"
spur exec "$JOBID" bash -lc 'pkill -TERM -f "lumenrl.trainer.main" || true'
```

### 12.2 Resume from Checkpoint

Before resuming, verify the latest checkpoint is complete:

```bash
source "$HOME/agentic-rl.env"
CKPT="$(cat "$RUNTIME_DIR/current_ckpt_dir.txt")"
ls -lh "$CKPT"/global_step_*/actor/ 2>/dev/null | tail -20
```

Resume by re-running §10 with `checkpointing.resume=true`:

```bash
# Add to the python3 command in §10:
#   checkpointing.resume=true
```

After resume, verify two consecutive steps complete with finite metrics before leaving it unattended.

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: geak_gym` | PYTHONPATH missing multi-tune-agent | Ensure `$LUMENRL_DIR/experiments/multi-tune-agent/src` and `$LUMENRL_DIR/experiments/multi-tune-agent` are in PYTHONPATH |
| `ModuleNotFoundError: lumenrl.algorithms.gspo` | GSPO not in codebase | Pull latest from `dev/moe-grpo` branch |
| GEAK sandbox compile fails on gfx950 | Kernel not built for MI355X arch | Set `GPU_ARCHS=gfx950` for aiter/flash-attn builds on sandbox node |
| GEAK sandbox compile fails on gfx942 | Kernel not built for MI300X arch | Set `GPU_ARCHS=gfx942` for aiter/flash-attn builds on trainer node |
| `OutOfMemoryError` during rollout | 8192-token context too long for MI355X | Reduce `policy.max_total_sequence_length` or `policy.max_response_length` |
| Hacking penalty dominates reward | Agent gaming the sandbox | Check `hacking_detection` config thresholds; inspect agent trajectories |
| NaN in GSPO loss | Learning rate too high or advantage explosion | Verify `policy.learning_rate: 5e-7`; check `algorithm.gspo.clip_ratio` |
| Ray cluster shows 1 node | Trainer node failed to join | Re-run §8.2; check firewall/network between nodes |
| `Too many open files` | ulimit not set | Add `ulimit -n 524288` before training launch |
| No TRLOO advantages | `turn_rewards` tensor missing | Verify multi-turn rollout produces `turn_ids` and `turn_rewards` in DataProto |
| profiling_reward errors | rocprof not available | Install rocprof on sandbox node; verify `rocprof --help` works |

---

## 14. Architecture Reference

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Sandbox Node (MI355X / gfx950)                                 │
│  ├── GEAK Sandbox: isolated kernel workspace                    │
│  │   ├── HIP kernels from aiter/csrc/                          │
│  │   ├── Triton kernels from aiter/ops/triton/                 │
│  │   └── FlyDSL kernels from aiter/ops/flydsl/                │
│  ├── Three-stage eval: compile → correctness → performance      │
│  ├── Profiling-based reward (rocprof)                           │
│  ├── Hacking detection (lazy deletion, hardcoded, test gaming)  │
│  └── ATOM rollout (TP=2 × 4 replicas, BF16 + FP8 KV)         │
├─────────────────────────────────────────────────────────────────┤
│  Trainer Node (MI300X / gfx942)                                 │
│  ├── Megatron: TP=4, EP=8 (128 experts)                        │
│  ├── GSPO loss: sequence-level ratio, MoE-friendly             │
│  ├── TRLOO advantage: turn-level LOO for multi-turn agent      │
│  └── Distributed optimizer with FP32 master weights            │
└─────────────────────────────────────────────────────────────────┘
```

### Reward Composition

```
R_total = R_correctness + R_performance + R_profiling + R_hacking_penalty

R_correctness:  compile pass +0.1, correctness pass +0.3, failure -1.0
R_performance:  log(speedup) + 0.5 * improvement_over_best
R_profiling:    0.3 * bandwidth + 0.3 * occupancy + 0.4 * instruction_efficiency
R_hacking:      lazy deletion -2.0 / hardcoded output -3.0 / test gaming -2.5
```

### Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `algorithm.name` | `gspo` | Sequence-level importance ratio |
| `algorithm.adv_estimator` | `trloo` | Turn-level LOO advantage |
| `algorithm.gspo.clip_ratio` | `0.2` | Symmetric clip range |
| `algorithm.gspo.num_generations` | `8` | Rollout episodes per prompt |
| `policy.learning_rate` | `5e-7` | Lower LR for agentic RL stability |
| `policy.max_total_sequence_length` | `8192` | Multi-turn context length |
| `agentic_rl.max_turns` | `15` | Max turns per episode |
| `agentic_rl.command_timeout` | `300` | Per-command timeout seconds |

### Key Source Files

```
lumenrl/algorithms/gspo.py                         # GSPO algorithm
lumenrl/algorithms/loss_functions.py                # gspo_loss()
lumenrl/algorithms/advantage_estimators.py          # trloo estimator
lumenrl/rewards/profiling_reward.py                 # rocprof reward
experiments/multi-tune-agent/geak_gym/env.py        # GEAKGymEnv
experiments/multi-tune-agent/multi_turn_rollout.py  # Multi-turn rollout
experiments/multi-tune-agent/turn_level_reward.py   # Per-turn rewards
experiments/multi-tune-agent/hacking_detection.py   # Hacking detection
examples/AgenticRL/Qwen3_30B_A3B/run_agentic_rl.sh # Launch script
```

---

## 15. Kernel Task Categories from aiter

### HIP Kernels (csrc/)

| Task ID | Source | Target |
|---------|--------|--------|
| `hip_batched_gemm_a8w8` | `csrc/ck_batched_gemm_a8w8/` | A8W8 quantized batched GEMM |
| `hip_batched_gemm_bf16` | `csrc/ck_batched_gemm_bf16/` | BF16 batched GEMM |
| `hip_gemm_a8w8_blockscale` | `csrc/ck_gemm_a8w8_blockscale/` | FP8 block-scaled GEMM |
| `hip_asm_mha_fwd` | `csrc/py_itfs_cu/asm_mha_fwd.cu` | Assembly MHA forward |
| `hip_asm_fmoe` | `csrc/py_itfs_cu/asm_fmoe.cu` | Fused MoE GEMM |
| `hip_asm_mla` | `csrc/py_itfs_cu/asm_mla.cu` | Multi-Latent Attention |
| `hip_asm_pa` | `csrc/py_itfs_cu/asm_pa.cu` | Paged Attention |
| `hip_deepgemm` | `csrc/ck_deepgemm/deepgemm.cu` | DeepGEMM |

### Triton Kernels (aiter/ops/triton/)

| Task ID | Source | Target |
|---------|--------|--------|
| `triton_softmax` | `ops/triton/softmax.py` | Softmax for vocabulary logits |
| `triton_topk` | `ops/triton/topk.py` | Top-k for MoE router |
| `triton_activation` | `ops/triton/activation.py` | Fused SiLU/GELU |
| `triton_kv_cache` | `ops/triton/kv_cache.py` | KV cache copy/reshape |
| `triton_gmm` | `ops/triton/gmm.py` | Grouped matrix multiply |
| `triton_gather_kv_b_proj` | `ops/triton/gather_kv_b_proj.py` | Gather + KV B-projection |

### FlyDSL Kernels (aiter/ops/flydsl/)

| Task ID | Source | Target |
|---------|--------|--------|
| `flydsl_fmha` | `ops/flydsl/fmha_kernels.py` | Fused MHA |
| `flydsl_gemm` | `ops/flydsl/gemm_kernels.py` | GEMM with preshuffle |
| `flydsl_moe` | `ops/flydsl/moe_kernels.py` | MoE 2-stage GEMM |
| `flydsl_linear_attention` | `ops/flydsl/linear_attention_kernels.py` | Gated delta net attention |
| `flydsl_silu_and_mul` | `ops/flydsl/kernels/silu_and_mul_fq.py` | SiLU + FP8 quant |
| `flydsl_splitk_hgemm` | `ops/flydsl/kernels/splitk_hgemm.py` | Split-K half GEMM |

---

## 16. MI355X (gfx950) vs MI300X (gfx942) Notes

| Aspect | MI355X (gfx950) | MI300X (gfx942) |
|--------|-----------------|-----------------|
| Role in this deployment | Sandbox + Rollout | Training |
| HBM | HBM3e (larger) | HBM3 |
| Compute units | More CUs | 304 CUs |
| GPU arch flag | `GPU_ARCHS=gfx950` | `GPU_ARCHS=gfx942` |
| flash-attn build | Must compile for gfx950 | Must compile for gfx942 |
| aiter build | Must compile for gfx950 | Must compile for gfx942 |
| GEAK sandbox kernels | Evaluated here | Not used for eval |
| FlyDSL gfx1250 kernels | Skip (wrong arch) | Skip (wrong arch) |

When building aiter on the sandbox node (MI355X), set:

```bash
GPU_ARCHS=gfx950 pip install -e "$AITER_DIR"
```

On the trainer node (MI300X):

```bash
GPU_ARCHS=gfx942 pip install -e "$AITER_DIR"
```

FlyDSL kernels with `gfx1250` in the name (e.g., `bpreshuffle_gemm_gfx1250.py`, `grouped_moe_gfx1250.py`) target MI400-series and must be excluded from the training cases for this deployment.

---

Last updated: 2026-09-07
