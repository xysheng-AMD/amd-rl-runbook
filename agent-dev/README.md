# SFT Data Collection for Kernel Agent Training

为 Qwen3-30B-A3B（MoE, 3B active params）训练采集 SFT 数据。
核心思路：用 Claude Opus 5 的优质轨迹"蒸馏"出一个轻量级但在 kernel optimization 领域专精的 agent。

## 训练什么能力

### 1. 分析推理能力（analysis_trajectory）

- 读 ISA 文档、profile 数据、baseline 性能数据后，能做出正确的分析判断
- 知道该调用什么工具（read/grep/rocprof）、按什么顺序调用
- 能区分有效探索和无效探索（golden path vs dead end）
- 最终能输出有质量的分析 deliverable

### 2. 代码编写能力（opus_kernel）

- 在给定 contract（operator、dtype、layout）和硬件约束（gfx950/MI355X）下，能写出正确且高性能的 OPUS kernel 代码
- 理解 parent->candidate 的迭代优化过程
- 五种任务场景：冷启动、profile 引导、direction 引导、错误修复、回归修复
- 写出的代码必须能通过独立编译和正确性验证

### 3. 概念理解能力（concept_snapshot）

- 掌握 GPU 架构知识（cache hierarchy、MFMA 指令、LDS、occupancy 等）
- 能把这些概念用结构化的方式解释清楚
- 教概念不教事实 -- 这是 MoE 3B active params 的关键设计：模型容量有限，所以教它理解原理和框架，不教它记精确的 hex 常数

## 训练数据量目标

| 类型 | 目标数量 | 占比 |
|------|---------|------|
| `analysis_trajectory` | 200 | ~20% |
| `opus_kernel` | 400 | ~40% |
| `concept_snapshot` | 400 | ~40% |
| **总计** | **1000** | |

收集到上限后自动停止。查看进度：

```bash
cat <run_dir>/sft-collect-status.txt
```

输出示例：

```
=== SFT Collection Status ===
Type                       Current   Target   Progress
analysis_trajectory            120      200      60.0%
opus_kernel                    400      400     100.0%  DONE
concept_snapshot               280      400      70.0%

Overall: 800 / 1000 (80.0%)
Status: NOT READY
  - analysis_trajectory (need 80 more)
  - concept_snapshot (need 120 more)
```

## 两个 Skill

- **`skill-offline-sft-extraction.md`** -- 从已保存的轨迹文件批量提取
- **`skill-online-sft-collection.md`** -- 在 agent loop 运行时录制事件（buffer-only，不干扰 agent）

## 数据输出位置

```
<run_dir>/
├── sft-collect-status.txt        # 采集进度和是否达标
└── sft_collected/
    ├── samples/                  # 通过门禁的正样本
    ├── dead_ends/                # RL 负样本
    ├── rejected/                 # 未通过门禁
    ├── pending/                  # 等待 OPUS 编译验证
    ├── artifacts/                # OPUS parent/candidate 源文件
    ├── manifest.jsonl            # 完整性索引（含 SHA256）
    └── collector_meta.json
```

## 快速使用

```bash
# 从轨迹目录提取（达到上限自动停止）
python -m sft_collector offline \
  --trajectory-dir /path/to/trajectories \
  --run-dir /path/to/output \
  --skip-verify

# 查看采集状态
cat /path/to/output/sft-collect-status.txt

# 自定义上限
python -m sft_collector offline \
  --trajectory-dir /path/to/trajectories \
  --run-dir /path/to/output \
  --cap-analysis 100 --cap-opus 200 --cap-concept 200

# 补验 pending 样本（需要 OPUS 编译器 + GPU，可以在别的服务器上跑）
python -m sft_collector verify \
  --run-dir /path/to/output \
  --workspace-dir /path/to/opus/workspace

# 合并多个 run
python -m sft_collector etl \
  --run-dirs run1/ run2/ run3/ \
  --output-dir /path/to/final_dataset
```

## 代码结构

```
sft_collector/
├── core/           # 共享组件（schema, context, golden path, verify, quota, writer）
├── offline/        # 离线管线（trajectory parser, state reconstructor, offline agent）
├── online/         # 在线管线（event interceptor, buffer-only middleware）
├── etl/            # 后处理（merge, dedup, split, balance）
└── cli.py          # CLI 入口
```
