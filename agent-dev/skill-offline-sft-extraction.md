# Skill: Offline SFT Data Extraction

## 目标

从已保存的 agent 轨迹中提取 SFT 训练数据。
这是**存量数据处理**——你将读取历史轨迹，重建文件状态和上下文，提取三类样本。

## 你是谁

你是数据提取 agent。你的任务是处理已保存的 agent session 轨迹（可能是单个 JSONL 文件，
也可能是包含多个轨迹文件的目录树），从中提取高质量的 SFT 训练数据，用于训练
Qwen3-30B-A3B 的 kernel architect 能力。

## 输入

- **轨迹路径**: 可以是以下任意形式：
  - 单个 `.jsonl` 轨迹文件
  - 包含多个 `.jsonl` 文件的目录（递归扫描子目录）
  - 包含多个 session 子目录的顶层目录（每个子目录内含轨迹文件）
  - 支持的文件名模式：`session*.jsonl`、`trajectory*.jsonl`、以及任何 `.jsonl` 文件
- **输出目录**: `<run_dir>` — 采集结果写入 `<run_dir>/sft_collected/`
- **配置文件** (可选): JSON 格式，包含 collection_mode、harness_config 等

## 提取的三类数据

### 1. Analysis Trajectory（分析轨迹）
从 `present()` 事件分割的 segment 中提取 golden path：
- 用 Golden Path Engine 的 5 条启发式规则自动分类 golden / dead end
- Golden steps → SFT 正样本（多轮 messages 格式）
- Dead ends → RL negatives（保存到 `dead_ends/`）

### 2. OPUS Kernel（代码样本）
从文件 read/write 事件重建 shadow copy，生成 parent → candidate diff：
- 对齐 GEAK `samples.jsonl` schema（含完整 input context）
- **必须 independent verify**：在隔离 workspace 中独立编译 + correctness 测试
- 无 compiler → 样本进入 `pending/` 等待补验，**不降级**

### 3. Concept Snapshot（概念快照）
从 reasoning blocks 自动提取概念框架：
- 包含 ≥ 2 个独立概念定义的段落
- 排除含有未标注来源的精确数值（hex、bit position 等）
- 自动推断对应的 user question

## 执行步骤

### Step 1: 环境检查

```bash
# 确认 sft_collector 包可用
python -c "from sft_collector.offline.offline_agent import OfflineAgent; print('OK')"

# 检查轨迹文件
ls <trajectory_dir>/*.jsonl | wc -l

# 检查 OPUS compiler (可选)
opus-compile --version 2>/dev/null && echo "compiler OK" || echo "compiler unavailable - OPUS samples will be pending"
```

### Step 2: 运行离线提取

```bash
python -m sft_collector offline \
  --trajectory-dir <trajectory_dir> \
  --run-dir <run_dir> \
  --collection-mode direction_conditioned \
  --workspace-dir <workspace_dir>  # OPUS verify 用
```

如果只想快速测试（跳过 OPUS verify）：
```bash
python -m sft_collector offline \
  --trajectory-dir <trajectory_dir> \
  --run-dir <run_dir> \
  --skip-verify \
  -v
```

### Step 3: 检查结果

```bash
# 查看输出结构
find <run_dir>/sft_collected/ -type f | head -20

# 统计各类样本
echo "=== Samples ==="
ls <run_dir>/sft_collected/samples/ | wc -l

echo "=== Dead Ends ==="
ls <run_dir>/sft_collected/dead_ends/ | wc -l

echo "=== Rejected ==="
ls <run_dir>/sft_collected/rejected/ | wc -l

echo "=== Pending ==="
ls <run_dir>/sft_collected/pending/ | wc -l

# 查看 manifest
cat <run_dir>/sft_collected/collector_meta.json
```

### Step 4: 补验 pending 样本 (如果有)

当 OPUS compiler 可用时：
```bash
python -m sft_collector verify \
  --run-dir <run_dir> \
  --workspace-dir <workspace_dir>
```

### Step 5: ETL 合并（多个 run 时）

```bash
python -m sft_collector etl \
  --run-dirs <run_dir_1> <run_dir_2> <run_dir_3> \
  --output-dir <final_dataset_dir>
```

## 配置文件格式

```json
{
  "collection_mode": {
    "task_type": "direction_conditioned"
  },
  "run_id": "offline-extraction-001",
  "kernel_name": "flash_attention_fwd",
  "orchestrator": {
    "budget_total": 10,
    "round_limit": 3
  },
  "contract": {
    "operator": "flash_attention_fwd",
    "language": "opus",
    "language_version": "0.1.0",
    "backend_version": "rocm-6.2"
  },
  "architecture": {
    "target_gpu": "gfx950",
    "gpu_sku": "MI355X"
  },
  "environment": {
    "rocm_version": "6.2.0",
    "container_image": "..."
  }
}
```

## 质量门禁

所有样本必须通过以下门禁才能写入 `samples/`：

| 门禁 | 规则 |
|------|------|
| 通用 | reasoning ≥ 100 chars, output ≥ 200 chars |
| analysis_trajectory | golden path ≥ 2 tool calls, deliverable 非空 |
| opus_kernel | independent verify pass (compile + correctness), contract 完整, parent_source 非空 |
| concept_snapshot | ≥ 2 个独立概念, 无未标注精确值, 文本相似度 < 0.8 |

## 关键约束

1. **Direction 因果性**: `direction_created_at` 必须 < agent 首次动作时间
2. **Error-recovery parent**: parent_source 指向失败版本，不是原始版本
3. **Five task types**: 不能事后重分类，必须从 Orchestrator 配置读取
4. **Independent verify 硬门禁**: 无 compiler = 无正样本，只有 pending
5. **GEAK schema 对齐**: 输出必须兼容 `runbook-GEAK-SFT-Dataset.md` 定义的 `samples.jsonl`

## 代码位置

```
/home/danyzhan/sft_collector/
├── core/           # 共享组件 (context_accumulator, golden_path, shadow_copy, ...)
├── offline/        # 离线管线 (trajectory_parser, state_reconstructor, offline_agent)
├── online/         # 在线管线 (event_interceptor, middleware)
├── etl/            # 后处理 (merge_dedup_split)
└── cli.py          # CLI 入口
```

## 输出目录结构

```
<run_dir>/sft_collected/
├── samples/          # 正样本 (通过所有门禁)
│   └── <sample_id>.json
├── artifacts/        # OPUS parent/candidate 文件
│   └── <sample_id>/
│       ├── parent/kernel.opus
│       └── candidate/kernel.opus
├── dead_ends/        # RL negatives
├── rejected/         # 未通过门禁
├── pending/          # 等待补验
├── manifest.jsonl    # append-only 索引 (每条带 SHA256)
└── collector_meta.json
```
