# Kernel Analysis Agent — SFT Data Construction Design

**Date**: 2026-09-18
**Goal**: SFT Phase 1 for Qwen3-30B-A3B (MoE, 3B active params) to replace Claude Opus 5 as kernel architect agent
**Scope**: GPU kernel performance analysis + GPU kernel coding on AMD CDNA4/gfx950 (MI355X)
**Next phase**: Agentic RL for tool-use strategy optimization
**Training data**: Kernel analysis trajectories (~2.3-3.2K) + GPU kernel coding data in three languages (Triton, HIP, OPUS)

---

## 1. Design Principles

### 1.1 Core Insight: Teach "How to Research", Not "What the Answer Is"

A 3B active parameter model cannot reliably memorize precise bit encodings, ISA field positions, or compiler constant values. Hallucinating `aux=3` when the correct value is `aux=17` is worse than not knowing — it produces silent kernel corruption.

**SFT teaches:**
- Concept frameworks (what cache policy bits exist, what they control)
- Efficient research paths (which docs to check, in what order)
- When to trust memory vs. when to verify via tools
- Analysis output structure and quality bar

**SFT does NOT teach:**
- Precise numerical values (bit positions, encoding constants)
- Task-specific judgments (leave for RL)
- Tool selection strategy optimization (leave for RL)

### 1.2 Format: Agent Trajectory, Not QA

All SFT data uses the same multi-turn agent trajectory format the model will use in production. This ensures:
- No format gap between SFT and deployment
- SFT → RL transition is seamless (RL refines the same format)
- Model learns tool use as part of the workflow, not as an afterthought

### 1.3 Golden Path Extraction

From raw trajectories (which contain dead ends, retries, truncated fetches), extract the **productive path only** — the minimal sequence of steps that reaches the correct conclusion.

Dead ends are not discarded — they become **RL negative examples** in phase 2.

---

## 2. Data Source: Raw Trajectory Structure

Based on analysis of `session.v3.jsonl` (representative sample):

```
801 lines JSONL, 4.9MB
Content breakdown:
  Tool results:     1,184K chars  (78.6%)  ← most volume, most compressible
  Reasoning:          170K chars  (11.3%)  ← high value, keep most
  Tool args:           84K chars  (5.6%)   ← keep tool call structure
  Agent inbox:         53K chars  (3.5%)   ← team messages, keep selectively
  Text output:         15K chars  (1.0%)   ← sparse but important

Tool call distribution:
  read:        50 calls  (ISA docs, source files, design docs)
  web_fetch:   22 calls  (LLVM/Triton source from GitHub)
  pwsh:        18 calls  (file search, wiki queries)
  grep:        15 calls  (code search)
  send_message: 7 calls  (team communication)
  present:      5 calls  (deliverables)

Deliverables produced: 5 analysis documents
  - cdna4-cache-policy.md
  - E0/E1 analysis.md (gap attribution, touch-count modeling)
  - parameterization-design.md
  - tuning-filter.md
```

---

## 3. SFT Dataset Construction

### 3.1 Dataset 1: Concept Grounding (40% of samples)

**Purpose**: Ensure the model understands domain vocabulary and conceptual frameworks without memorizing precise values.

**Format**: Short agent trajectory where the model demonstrates concept understanding and knows what to verify.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a GPU kernel performance analyst for AMD CDNA4/gfx950..."
    },
    {
      "role": "user",
      "content": "Explain how cache policy control works for gfx950 buffer loads."
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "text": "gfx950 MUBUF instructions have a cache policy (cpol) operand with three independent control bits: SC0 controls L1 scope, SC1 controls L2 scope, and NT is the non-temporal hint. The exact bit positions in the cpol field are defined in LLVM's SIDefines.h under the AMDGPU::CPol namespace. I should state the conceptual framework and note where to verify specifics."
        },
        {
          "type": "text",
          "text": "On gfx950 (CDNA4), buffer load instructions (MUBUF/MTBUF) have a cache policy operand (`cpol`/`aux`) with three independent control dimensions:\n\n1. **SC0** — L1 cache scope control (maps to GLC in LLVM)\n2. **NT** — Non-temporal hint, bypasses cache for streaming access (maps to SLC in LLVM)\n3. **SC1** — L2/system-level scope control (maps to SCC in LLVM)\n\nThese combine to form cache modifier classes:\n- `.ca` (cache all): default, no bits set\n- `.cs`/`.cg` (cache streaming/global): SC0 + NT\n- `.cv` (cache volatile): SC0 + SC1\n\nThe exact bit positions and numeric values should be verified against `LLVM/SIDefines.h` and the ISA manual Chapter 7 encoding tables for the specific compiler version in use."
        }
      ]
    }
  ]
}
```

**Extraction rules:**
- From reasoning blocks: extract conceptual understanding statements
- From deliverables: extract framework descriptions (strip precise constants)
- Augment: rephrase questions from different angles
- Key: response always ends with "verify against [source]" for any precise values

**Coverage topics:**
- Cache hierarchy and policy (L1/L2/MALL, cpol bits, coherence)
- MFMA instruction characteristics (data types, tile sizes, throughput)
- Memory subsystem (LDS banks, VGPR/SGPR budgets, occupancy)
- Data type conversion paths (fp8↔bf16↔fp32, which are exact)
- Roofline model components (peak BW per level, peak FLOPS, arithmetic intensity)
- Kernel parameterization concepts (tiling, blocking, pipelining depth)

### 3.2 Dataset 2: Efficient Research Paths (40% of samples)

**Purpose**: Teach the model the golden path for common research tasks — which tools to use, in what order, and how many sources to cross-validate.

**Format**: Multi-turn agent trajectory with tool calls, stripped of dead ends.

**Extraction pipeline:**

```
Raw trajectory (Opus, 30+ steps):
  ├── read ISA page-084  → partial info
  ├── read ISA page-092  → partial info
  ├── read ISA page-093  → partial info
  ├── read ISA page-602  → nothing useful        ← dead end
  ├── read ISA page-603  → found encoding table   ← keep
  ├── grep LLVM "CPol"   → glob syntax error      ← dead end
  ├── grep LLVM "CPol"   → found SIDefines.h      ← keep
  ├── web_fetch Triton   → truncated at 100KB     ← dead end
  ├── web_fetch Triton   → found cache control    ← keep
  ├── read aiter kernel  → confirmed usage        ← keep
  └── synthesize → deliverable

Golden path (SFT target, 4-5 steps):
  ├── thinking: "Need MUBUF cpol encoding. Check: 1) ISA encoding chapter 2) LLVM source 3) existing kernel code"
  ├── read ISA page-603  → encoding table
  ├── thinking: "Got hardware encoding. Cross-check with compiler mapping."
  ├── grep LLVM SIDefines.h → CPol constants
  ├── thinking: "Hardware and compiler match. Verify against actual kernel usage."
  ├── read aiter/opus_kernel.hpp → confirmed
  └── thinking: "3 sources consistent. Synthesizing."
  └── output: structured analysis
```

**Key design decisions:**
- Always show 2-3 source cross-validation (teach verification culture)
- Thinking blocks explain WHY each tool call is made (not just what)
- Tool results are compressed to the relevant lines only (~500-2000 chars per result, not 32K)
- Include the "I need to verify this" pattern even when the model's training data might contain the answer

**Common research patterns to cover:**
- ISA encoding lookup (which chapter, which pages)
- Compiler lowering path tracing (LLVM → backend → instruction selection)
- Existing kernel code analysis (find patterns, extract design decisions)
- Profiling data interpretation (counter meanings, derived metrics)
- Cross-source validation workflow

### 3.3 Dataset 3: Analysis Report Generation (20% of samples)

**Purpose**: Teach the model to produce structured, high-quality analysis deliverables after research is complete.

**Format**: Agent trajectory where research results are provided as tool outputs, and the model produces a deliverable.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a GPU kernel performance analyst..."
    },
    {
      "role": "user",
      "content": "Analyze the performance gap for kernel E0 against the incumbent. Here is the profiling data: ..."
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "text": "The profiling shows 4x more L1 cache line requests than expected. The incumbent uses direct-to-LDS staging which avoids the L1 entirely. This is a data movement architecture gap, not a compute gap..."
        }
      ],
      "tool_calls": [{"name": "read", "arguments": {"file_path": "hardware-specs.md"}}]
    },
    {
      "role": "tool",
      "content": "<hardware specs: L1 line size=128B, LDS BW=...>"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "text": "With 128B cache lines and 16B element accesses, each tile load touches 8 unique lines but only uses 2 — giving exactly the 4x amplification we see..."
        },
        {
          "type": "text",
          "text": "## E0 Gap Attribution\n\n### 1. Roofline Position\nCompute: 12% of peak MFMA throughput → memory-bound\nBandwidth: 23% of peak L2→L1 BW → bottleneck is L1 access pattern\n\n### 2. Root Cause\n**4x cache-line-request amplification.** Each 16B element access pulls a 128B L1 cache line, but the access pattern across the tile only uses 32B of each line...\n\n### 3. Incumbent Comparison\nThe incumbent avoids this entirely via direct-to-LDS staging...\n\n### 4. Recommendation\nRewrite load path as direct-to-LDS per-M-tile staging..."
        }
      ]
    }
  ]
}
```

**Report types to cover (from trajectory deliverables):**
- Gap attribution (roofline → bottleneck → root cause → action)
- Touch-count modeling (cache geometry → predicted vs actual)
- Parameterization design (axes, invariants, capacity constraints)
- Tuning filter design (pruning rules, ranking model, bucket map)
- Cache policy analysis (bit layout, modifier mapping, verification)

---

## 4. 在线 SFT 数据采集 — Harness Middleware

### 4.1 设计思路：Harness 自动采集

数据采集作为 **DeepSeek Harness 的 middleware** 运行，agent 完全无感知。Harness 在关键事件（`present()`、OPUS 文件写入等）发生时自动触发采集逻辑，无需 agent 主动调用任何 tool。

核心优势：
- **零遗漏** — 不依赖 agent 记住调用，基础设施层保证每个事件都被捕获
- **Parent/candidate 状态精确** — Harness 追踪文件的 shadow copy，修改前后的内容天然可得
- **Golden path 自动推断** — 启发式规则基于 tool result → reasoning 引用链判断，不需要 agent 标注
- **对 agent 零干扰** — 不占用 agent 的 tool call 配额、token 预算或注意力

### 4.1.1 五类 Task Type 的采集模式

Middleware 的 `task_type` 不是事后推断——它由 **Orchestrator 在 run 配置中预先声明**，Middleware 读取后严格执行对应的采集约束。这与 `GEAK-SFT-Model.md` §2 的要求一致：每类 task type 的可见输入必须在 Engineer 执行前冻结。

```yaml
# run 配置中的 collection_mode（Orchestrator 设置）
collection_mode:
  task_type: "cold_start | profile_guided | direction_conditioned | error_recovery | regression_balance"
```

**五类模式的采集行为差异：**

```
cold_start:
  ├── Harness 不注入 profile、direction
  ├── Agent 只能看到：contract + parent_source + baseline
  ├── Middleware 验证：direction_context 和 profile_context 必须为空
  │   若 agent 通过 tool call 自行获取了 profile → 该样本不能标记为 cold_start
  └── 不能从 direction_conditioned 轨迹删除字段后伪造

profile_guided:
  ├── Harness 注入 profile，不注入 direction
  ├── Agent 可见：contract + parent_source + baseline + profile
  ├── Profile 必须来自 parent_source 在同一环境的真实测量
  ├── Middleware 验证：direction_context 必须为空，profile_context 必须非空
  └── 不能从 direction_conditioned 轨迹删除 direction 后伪造

direction_conditioned:
  ├── Harness 注入 profile + TechLead direction
  ├── Agent 可见：contract + parent_source + baseline + profile + direction
  ├── Direction 必须在 Engineer 执行前持久化（见 §4.12）
  └── Middleware 验证：direction_context 非空，且 direction_created_at < agent_first_action

error_recovery:
  ├── 输入是失败后的源码 + 精确错误信息
  ├── Parent_source = 应用了失败 patch 后的版本（不是原始版本）
  ├── Middleware 使用 error-aware shadow copy（见 §4.6.1）
  └── Middleware 验证：error_context 非空，parent_source_hash ≠ baseline_source_hash

regression_balance:
  ├── 输入包含 per_case_benchmark + regression_constraints
  ├── 前提：aggregate speedup > 1 但至少一个 case 超出 regression budget
  ├── Middleware 验证：per_case_benchmark 非空，至少一个 case 有退化标记
  └── 修复后的 patch 必须满足所有 regression constraints
```

**配比目标（与 GEAK-SFT-Model.md §3 对齐）：**

```
每 1,000 条 Kernel SFT 正样本：
  Cold-start:             150  (15%)
  Profile-guided:         150  (15%)
  Direction-conditioned:  450  (45%)
  Error-recovery:         150  (15%)
  Regression-balance:     100  (10%)
```

Orchestrator 必须分别发起五种 `collection_mode` 的 run，不能仅靠 direction_conditioned 运行后事后重分类。

### 4.2 Middleware 架构

```
DeepSeek Harness
  │
  ├── Agent Loop（正常运行，无感知）
  │     ├── tool call: read / web_fetch / grep / pwsh / ...
  │     ├── reasoning blocks
  │     ├── present() deliverables
  │     └── file writes (*.opus, *.hpp, ...)
  │
  └── SFT Collector Middleware（拦截层）
        │
        ├── Event Interceptor
        │   ├── on_present(deliverable)     → 触发 analysis_trajectory 采集
        │   ├── on_file_write(*.opus)       → 记录 candidate snapshot
        │   ├── on_file_read(*.opus, first) → 记录 parent snapshot (shadow copy)
        │   ├── on_tool_result(*)           → 追加到当前 segment buffer
        │   └── on_session_end()            → flush 未完成 segment
        │
        ├── Context Accumulator（持续追踪，见 §4.10）
        │   ├── architecture_context  — ISA specs, cache hierarchy, compute caps
        │   ├── profile_context       — profiling data, counter values, roofline
        │   ├── contract_context      — task contract, operator, dims, dtype
        │   ├── direction_context     — optimization direction, TechLead feedback
        │   ├── baseline_context      — baseline performance numbers
        │   └── error_context         — compile/correctness errors if error-recovery
        │
        ├── Trajectory Buffer（内存）
        │   ├── 当前 segment 的所有事件（tool call + result + reasoning）
        │   ├── file shadow copies（OPUS 文件的 read-before-write 快照）
        │   └── segment 边界标记（由 present() 或 session_end 触发）
        │
        ├── Golden Path Engine（自动推断）
        │   └── 分析 tool result → reasoning 引用链，自动分类 golden / dead end
        │
        ├── Formatter（格式化）
        │   ├── analysis_trajectory → 多轮 messages 格式
        │   ├── opus_kernel → GEAK samples.jsonl schema（含完整 input context）
        │   └── concept_snapshot → 短格式 messages
        │
        ├── Independent Verifier（OPUS 专用，§4.7）
        │   └── fresh workspace compile + correctness
        │
        └── Writer（写入 run 目录）
            └── <run_dir>/sft_collected/
```

### 4.3 Event Interceptor 详细规则

Middleware 拦截以下 Harness 事件：

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 事件                      │ Middleware 动作                                │
├──────────────────────────┼────────────────────────────────────────────────┤
│ tool_call(read, path)    │ 若 path 匹配 *.opus 且是该文件首次 read：       │
│                          │   保存文件内容为 parent shadow copy              │
│                          │ 若 path 匹配架构/ISA/hardware spec 文件：       │
│                          │   → Context Accumulator 更新 architecture      │
│                          │ 追加到当前 segment buffer                       │
├──────────────────────────┼────────────────────────────────────────────────┤
│ tool_result(*)           │ 追加到当前 segment buffer                       │
│                          │ ★ Context Accumulator 扫描并提取（见 §4.10）：  │
│                          │   - profile/rocprof 输出 → profile_context     │
│                          │   - baseline 数据 → baseline_context           │
│                          │   - 编译/运行错误 → error_context              │
├──────────────────────────┼────────────────────────────────────────────────┤
│ reasoning_block(*)       │ 追加到当前 segment buffer                       │
│                          │ ★ Context Accumulator 扫描：                   │
│                          │   - 优化方向/策略描述 → direction_context       │
│                          │   - contract 理解 → contract_context           │
├──────────────────────────┼────────────────────────────────────────────────┤
│ agent/inbox/spliced(*)   │ ★ Context Accumulator 扫描：                   │
│                          │   - TechLead direction → direction_context     │
│                          │   - task assignment → contract_context         │
├──────────────────────────┼────────────────────────────────────────────────┤
│ tool_call(pwsh/bash,     │ 若 cmd 写入 *.opus 文件：                       │
│   cmd writes *.opus)     │   执行后保存文件新内容为 candidate snapshot      │
│                          │   标记 OPUS 修改事件                            │
├──────────────────────────┼────────────────────────────────────────────────┤
│ present(deliverable)     │ ★ 触发 analysis_trajectory 采集：              │
│                          │   1. 冻结当前 segment buffer + context snapshot│
│                          │   2. 运行 Golden Path Engine                   │
│                          │   3. 格式化 + 质量门禁 + 写入                   │
│                          │   4. 开启新 segment（context 保留，不清空）      │
├──────────────────────────┼────────────────────────────────────────────────┤
│ opus_modify_complete     │ ★ 触发 opus_kernel 采集：                      │
│ (file, verify_pass)      │   1. 取出 parent shadow + candidate snapshot   │
│                          │   2. 生成 unified diff                         │
│                          │   3. 从 Context Accumulator 取完整 input context│
│                          │   4. Independent verify（§4.7）                │
│                          │   5. 打包 GEAK schema（含 input 全字段）+ 写入  │
├──────────────────────────┼────────────────────────────────────────────────┤
│ session_end()            │ Flush 当前 segment（如果有 deliverable 级内容） │
│                          │ 保存 Context Accumulator 最终状态到 run 目录    │
└──────────────────────────┴────────────────────────────────────────────────┘
```

### 4.4 Golden Path Engine — 自动推断

不依赖 agent 标注，用启发式规则自动判定每个 tool call 是 golden path 还是 dead end：

```
对 segment 中的每个 tool_call[i]：

  规则 1 — 引用检测（主规则）：
    扫描 tool_call[i] 之后的所有 reasoning blocks
    若 reasoning 中引用了 tool_result[i] 的关键内容（子串匹配 / 关键词重叠）
    → golden path
    否则 → dead end 候选

  规则 2 — 重试检测：
    若 tool_call[i] 和 tool_call[i+1] 指向同一 target 且 tool_call[i] 返回错误
    → tool_call[i] 标记为 dead end（重试失败）
    → tool_call[i+1] 进入规则 1 评估

  规则 3 — 空结果检测：
    若 tool_result 为空、报错、或明显无关（如 "page not found"）
    → dead end

  规则 4 — deliverable 引用检测：
    若 tool_result[i] 的内容出现在最终 deliverable 文本中
    → golden path（即使 reasoning 中未显式引用）

  规则 5 — 最低保留：
    segment 中至少保留 2 个 golden path tool call
    若规则 1-4 只选出 0-1 个，按 reasoning 引用强度排序保留前 2 个

分离后：
  golden_steps → 进入 SFT 样本
  dead_ends    → 保存为 RL negatives（保留完整 tool call + result + 相关 reasoning）
```

### 4.5 Concept Snapshot 自动提取

除了 `present()` 和 OPUS 修改触发的采集，middleware 还自动扫描 reasoning blocks 提取概念快照：

```
对每个 reasoning block：
  1. 检测是否包含概念框架总结模式：
     - 多个并列定义（"X 是 ..., Y 是 ..., Z 是 ..."）
     - 分类/对比结构（"分为三类：..."）
     - 原理解释（"这是因为 ... 所以 ..."）
  2. 若匹配，提取为 concept_snapshot：
     - 将 reasoning 中的概念部分作为 assistant response
     - 从 segment context 推断对应的 user question
     - 扫描精确数值：若有未标注 "verify against [source]" 的常数 → reject
  3. 输出为短格式 messages（无 tool call）
```

### 4.6 File Shadow Copy 机制（OPUS 专用）

为准确获取 OPUS 文件的 parent → candidate diff，middleware 维护 shadow copy：

```
shadow_copies: Dict[filepath, List[Snapshot]]

on_file_read(path):
  if path matches *.opus and path not in shadow_copies:
    shadow_copies[path] = [Snapshot(content=read(path), timestamp=now, type="parent")]

on_file_write(path):
  if path matches *.opus:
    shadow_copies[path].append(Snapshot(content=read(path), timestamp=now, type="candidate"))

on_opus_collect(path):
  parent = shadow_copies[path][0]           # 第一次 read 的内容
  candidate = shadow_copies[path][-1]       # 最后一次 write 的内容
  diff = unified_diff(parent.content, candidate.content)
  # 清理该文件的 shadow copies
```

#### 4.6.1 Error-Recovery 场景的 Shadow Copy 特殊处理

在 `collection_mode=error_recovery` 时，parent_source 必须是"失败后的源码"，不是原始版本。Shadow copy 逻辑调整：

```
error_recovery 模式下：

on_compile_fail(path) 或 on_correctness_fail(path):
  # 当前文件内容就是"失败后的源码"，将其标记为新 parent
  shadow_copies[path].reset_parent(
    Snapshot(content=read(path), timestamp=now, type="error_parent")
  )
  # 记录失败信息
  error_snapshots[path] = {
    "failed_source_hash": sha256(read(path)),
    "compile_error": last_compile_stderr,     # 如果是编译失败
    "correctness_error": last_test_failure,    # 如果是正确性失败
    "exit_code": last_exit_code
  }

on_opus_collect(path):  # error_recovery 模式
  parent = shadow_copies[path].get_error_parent()   # 失败后版本
  candidate = shadow_copies[path][-1]                # 修复后版本
  diff = unified_diff(parent.content, candidate.content)
  # 此时 parent_source_hash ≠ baseline_source_hash，符合 error_recovery 约束
```

这确保 error_recovery 样本的 `parent_source` 指向失败版本，`error_feedback` 包含精确的 compile/correctness 错误，与 `GEAK-SFT-Model.md` §2.4 的要求一致。

### 4.7 OPUS Independent Verify（硬门禁）

OPUS 样本与 Triton/HIP 遵循同一原则：**只有 independent verify 通过的 patch 才能成为正向 SFT target**。Middleware 在采集 OPUS diff 后，必须在隔离环境中独立验证，不能仅依赖 agent 轨迹中的执行证据。

```
OPUS Independent Verify 流程：

on_opus_collect(path, parent, candidate, diff):
  │
  ├── 1. 创建 fresh workspace（与 agent workspace 隔离）
  │     mkdir <run_dir>/opus_verify/<sample_id>/
  │     复制 parent workspace 的完整上下文（config、依赖、build 文件）
  │
  ├── 2. Apply patch
  │     将 unified diff apply 到 fresh workspace 中的 parent source
  │     ✗ apply 失败 → reject(patch_does_not_apply)
  │
  ├── 3. Compile
  │     使用 OPUS compiler 编译 candidate
  │     记录: compile command, exit code, stdout, stderr
  │     ✗ 编译失败 → reject(compile_failed)
  │
  ├── 4. Correctness
  │     运行 correctness test cases（与 agent 运行时使用的相同 cases）
  │     记录: test cases, tolerance, results
  │     ✗ correctness 失败 → reject(correctness_failed)
  │
  ├── 5. Performance（可选，若 benchmark 可用）
  │     运行 performance benchmark
  │     记录: baseline/candidate latency, speedup
  │     ✗ 性能退化超过阈值 → reject(regression_failed)
  │     ✗ benchmark 不可用 → 标记 benchmark_valid=null，不 reject
  │
  ├── 6. 保存 verify receipt
  │     <run_dir>/opus_verify/<sample_id>/receipt.json:
  │     {
  │       "patch_applies": true,
  │       "compile_pass": true,
  │       "compile_command": "...",
  │       "compile_cwd": "...",
  │       "compile_exit_code": 0,
  │       "compile_stdout": "...",
  │       "compile_stderr": "...",
  │       "correctness_pass": true,
  │       "correctness_cases": [...],
  │       "correctness_seeds": [...],
  │       "correctness_tolerance": 1e-3,
  │       "benchmark_valid": true | null,
  │       "warmup_iterations": 5,
  │       "measurement_iterations": 100,
  │       "baseline_repeats": 3,
  │       "candidate_repeats": 3,
  │       "timer_type": "device",
  │       "raw_latencies_baseline": [142.1, 142.5, 142.3],
  │       "raw_latencies_candidate": [123.5, 123.8, 123.2],
  │       "aggregation_method": "median",
  │       "verified_speedup": 1.15 | null,
  │       "variance_note": "...",
  │       "timing_reliable": true,
  │       "kernel_launch_verified": true,
  │       "regression_per_case": [{case_id, pass, budget, actual}],
  │       "verify_workspace": "opus_verify/<sample_id>/",
  │       "verify_timestamp": "...",
  │       "gpu_identity": "...",
  │       "cheating_checks": {
  │         "harness_modified": false,
  │         "reference_delegation": false,
  │         "fixed_output": false,
  │         "cached_result": false,
  │         "input_mutation": false
  │       }
  │     }
  │
  └── 7. 写入正样本（所有硬门禁通过后）
        labels 使用 verify receipt 中的真实值，不使用 agent self-report
        verify_source = "independent_verify"
```

**若当前环境没有 OPUS compiler：**

这不是降级条件——是硬门禁。没有 compiler 就无法验证，无法验证就不能产出正样本。

```
无 OPUS compiler 时的行为：
  ├── Middleware 仍然采集 diff 和 context（写入 pending/）
  ├── 标记 verify_status = "pending_verify"
  ├── 不写入 samples/（不是正样本）
  └── 当 compiler 可用后，批量运行 pending 样本的 verify
      通过的升级为正样本；失败的移入 rejected/
```

### 4.8 质量门禁

所有自动采集的样本必须通过以下门禁才能写入 `samples/`（正样本）：

```
通用门禁：
  ├── reasoning 总长 ≥ 100 chars
  └── final output / deliverable ≥ 200 chars

analysis_trajectory 门禁：
  ├── golden path 包含 ≥ 2 个 tool call（保证交叉验证）
  ├── 最终 deliverable 存在且非空
  └── segment 时间跨度 ≥ 30 seconds（排除意外触发）

opus_kernel 门禁（硬门禁，缺一不可）：
  ├── parent 和 candidate 内容不同（有实际修改）
  ├── unified diff 可以 cleanly apply 到 parent
  ├── independent verify: compile pass          ← 必须
  ├── independent verify: correctness pass      ← 必须
  ├── independent verify: performance 不退化    ← 若 benchmark 可用
  ├── verify receipt 完整                        ← 必须
  ├── diff 不为空且 ≤ 50KB（排除整文件替换）
  ├── input.contract 必须含 operator + language + architecture   ← 必须
  ├── input.parent_source 非空                                   ← 必须
  └── input.architecture_info.target_gpu 非空                    ← 必须

concept_snapshot 门禁：
  ├── 包含 ≥ 2 个独立概念定义
  ├── 不包含未标注来源的精确数值（hex、bit position 等）
  └── 内容与已采集 snapshot 的文本相似度 < 0.8（去重）

reject 的样本写入 <run_dir>/sft_collected/rejected/ 并记录 reason
未能 verify 的 OPUS 样本写入 <run_dir>/sft_collected/pending/ 等待补验
```

### 4.8 写入格式与 Run 目录结构

```
<run_dir>/sft_collected/
├── samples/
│   ├── <sample_id>.json              # 格式化后的 SFT 样本
│   └── ...
├── artifacts/                         # OPUS parent/candidate 文件
│   └── <sample_id>/
│       ├── parent/kernel.opus
│       └── candidate/kernel.opus
├── dead_ends/
│   └── <sample_id>.json              # RL negatives（完整 dead end chain）
├── rejected/
│   └── <sample_id>.json              # 未通过门禁的样本 + rejection reason
├── manifest.jsonl                     # append-only 索引，每条带 SHA256
└── collector_meta.json               # middleware 版本、配置、统计
```

写入保障：
- 先写 temp file → fsync → atomic rename
- manifest.jsonl 每条带 sample SHA256，离线可校验
- 写入失败记录 warning，不阻塞 agent 主任务
- session 中断后通过 manifest 可精确恢复已采集的样本

### 4.9 离线后处理（轻量 ETL）

Middleware 在线采集后，离线 ETL 只做合并和全局操作，不需要重新解析轨迹：

```
Input: 多个 <run_dir>/sft_collected/ 目录

Step 1: 合并
  ├── 收集所有 run 的 manifest.jsonl
  ├── 校验每条 sample 的 SHA256
  ├── 分配全局 sample_id
  └── 合并 artifacts

Step 2: 去重
  ├── 按 source_hash + patch_hash 精确去重
  ├── 按 normalized text hash 模糊去重
  └── 跨语言 implementation family 去重

Step 3: Split 分配
  ├── 按 source lineage 分组
  ├── 分配 train / dev / held_out
  └── 检查无跨 split 泄漏

Step 4: 平衡与增强
  ├── 检查 concept_grounding:research_path:analysis_report ≈ 40:40:20
  ├── 不足时对 concept_grounding 做 question rephrasing 增强
  └── 检查三语言分布（Triton:HIP:OPUS）

Step 5: 输出
  ├── analysis_trajectories.jsonl   (分析轨迹数据)
  ├── kernel_coding_samples.jsonl   (三语言 GEAK schema)
  ├── rl_negatives.jsonl            (dead ends)
  └── manifest.json + checksums
```

### 4.10 Context Accumulator — 全量上下文采集

Middleware 持续追踪 agent 在 session 中获取的所有与 kernel coding 相关的上下文信息。当 OPUS 采集触发时，Context Accumulator 的当前快照作为该样本的 `input` 字段写入，确保训练时模型看到的输入与 agent 当时看到的一致。

**追踪的上下文类型：**

```
Context Accumulator 状态：

architecture_context:
  ├── target_gpu:          "gfx950" | "gfx942"
  ├── gpu_sku:             "MI355X" | "MI308" | ...
  ├── isa_version:         从 ISA doc reads 提取
  ├── cache_hierarchy:     L1 line size, L2 size, MALL capacity
  ├── compute_caps:        MFMA tile sizes, throughput, data types
  ├── memory_subsystem:    LDS banks/size, VGPR/SGPR budget, occupancy table
  ├── special_features:    MXFP support, async copy, ...
  └── source_refs:         [{"file": "ISA-manual-p603", "content_hash": "..."}]

contract_context:
  ├── operator:            "gemm" | "flash_attention" | ...
  ├── entry_point:         function name
  ├── dimensions:          {M, N, K, batch, ...}
  ├── input_dtype:         "fp16" | "bf16" | "fp8" | ...
  ├── output_dtype:        ...
  ├── layout:              "A[M,K] @ B[K,N]"  # 数学布局表示，对齐 GEAK-SFT-Model.md
  ├── quantization:        {format, packing, scale_granularity, block_size}
  ├── language:            "opus" | "triton" | "hip"
  ├── language_version:    "..."              # ★ 编译器/runtime 版本
  ├── backend_version:     "..."              # ★ backend 版本
  ├── modifiable_files:    ["kernel.opus"]    # ★ 允许修改的文件列表
  ├── correctness_tol:     tolerance spec
  ├── optimization_target: "latency" | "throughput" | ...
  ├── regression_constraints: [...]
  ├── shape_regime:        "decode" | "prefill" | "small_batch"
  └── commandment_hash:    "..."              # ★ COMMANDMENT.md 内容 hash

baseline_context:
  ├── baseline_latency:    {mean, std, samples}
  ├── baseline_throughput: ...
  ├── baseline_source_hash: "..."             # ★ 基线版本的源码 hash
  ├── roofline_position:   {compute_util, bandwidth_util, arithmetic_intensity}
  └── bottleneck:          "memory_bound" | "compute_bound" | "latency_bound"

profile_context:                              # null if cold_start
  ├── raw_counters:        [{counter_name, value}, ...]
  ├── derived_metrics:     {L1_hit_rate, LDS_bank_conflicts, occupancy, ...}
  ├── hotspot:             "load_tile" | "mfma_loop" | ...
  ├── profiler_tool:       "rocprof" | "rocprof-compute" | ...
  ├── profiler_version:    "..."
  └── source_refs:         [{"tool_call_id": N, "content_hash": "..."}]

direction_context:                            # null if cold_start / profile_guided
  ├── direction_source:    "tech_lead" | "fallback" | "skill" | "user"  # ★
  ├── direction_id:        "r1_d0"                                       # ★
  ├── direction_prompt:    原始 TechLead 消息文本
  ├── direction_prompt_hash: "..."                                       # ★
  ├── direction_created_at: "2026-09-18T10:30:00Z"                      # ★ 必须 < agent 首次动作时间
  ├── fallback_reason:     null | "..."                                  # ★
  ├── skill_refs:          [{"path": "...", "content_hash": "..."}]      # ★
  ├── skill_corpus_hash:   "..."                                         # ★
  ├── optimization_strategy: "直接 LDS staging 避免 L1 放大" | ...
  ├── key_constraints:     ["不能增加 VGPR 用量", "保持 correctness", ...]
  └── prior_attempts:      [{strategy, outcome}, ...]

per_case_benchmark_context:                   # ★ regression_balance 专用
  ├── cases:               [{case_id, shape, baseline_latency, candidate_latency,
  │                          speedup, regression_budget, budget_exceeded}]
  ├── aggregate_speedup:   1.15
  ├── regressed_cases:     [{case_id, regression_amount}]
  └── regression_feedback: "整体变快但关键 shape M=1 N=2048 退化 15%"

error_context:                                # error_recovery 专用
  ├── failed_source_hash:  失败版本的源码 hash（≠ baseline_source_hash）
  ├── compile_error:       {exit_code, stderr, error_category}
  ├── correctness_error:   {failed_cases, first_failure, tolerance, actual_vs_expected}
  └── error_analysis:      agent reasoning 中的错误归因

skill_retrieval_context:                      # ★ Skill/KB 检索信息
  ├── retrieval_query:     "..."
  ├── candidates:          [{path, rank, score, content_hash}]
  ├── selected_entries:    [{path, content_hash, frontmatter_constraints}]
  ├── corpus_hash:         "..."
  └── corpus_git_sha:      "..."

environment_context:                          # ★ GPU/软件栈/容器环境快照
  ├── rocm_version:        "6.x.x"
  ├── driver_version:      "..."
  ├── hipcc_version:       "..."
  ├── pytorch_version:     "..."
  ├── triton_version:      "..."   # if applicable
  ├── container_image:     "..."
  ├── container_digest:    "sha256:..."
  ├── geak_git_sha:        "..."
  ├── lumen_git_sha:       "..."
  ├── gpu_clocks:          {core, memory}
  ├── gpu_power_mode:      "..."
  └── gpu_temperature:     "..."

model_context:                                # ★ 模型/解码参数
  ├── model_checkpoint:    "Qwen/Qwen3-Coder-Next-FP8"
  ├── tokenizer_revision:  "..."
  ├── adapter_version:     null | "..."
  ├── chat_template:       "..."
  ├── temperature:         0.7
  ├── top_p:               0.9
  ├── max_output_tokens:   32000
  └── stop_tokens:         [...]

orchestrator_context:                         # ★ Orchestrator 预算参数
  ├── budget_total:        N
  ├── budget_used:         M
  ├── round_limit:         3
  ├── candidate_floor:     1.0
  ├── min_improve:         0.02
  ├── max_no_improve:      2
  ├── stopped_by:          null | "budget" | "deadline" | ...
  ├── warm_start_mode:     false
  └── gpu_pool_pin_mode:   "..."

session_identity:                             # ★ §8.1 身份字段
  ├── schema_version:      "geak_kernel_sft_v1"
  ├── dataset_version:     "phase1-v1"
  ├── run_id:              "..."
  ├── eval_dir:            "..."
  ├── kernel_name:         "..."
  ├── round:               1
  ├── engineer_id:         "..."
  ├── collection_mode:     "direction_conditioned"   # = task_type
  ├── creation_timestamp:  "..."
  └── workspace_head:      "..."              # ★ workspace Git commit at round start
```

**提取规则 — 从 session 事件流中自动识别并归类：**

```
session_start:
  → 从 Harness 配置读取 collection_mode, orchestrator_context
  → 从环境读取 environment_context（rocm-smi, hipcc --version, 容器标签等）
  → 从 Harness 读取 model_context（model, tokenizer, 解码参数）
  → 初始化 session_identity（run_id, eval_dir, schema_version 等）
  → 对 workspace 内所有 *.opus 文件做初始 snapshot

tool_call(read, "ISA-manual-*" | "hardware-specs*")
  → 解析 tool_result 提取架构参数
  → 更新 architecture_context

tool_result 匹配 rocprof/profiler 输出格式
  → 解析 counter name-value pairs
  → 更新 profile_context

tool_result 包含 baseline latency/throughput 数据
  → 解析并更新 baseline_context

tool_result 包含 per-case benchmark 数据
  → 逐 case 解析 latency, speedup, regression status
  → 更新 per_case_benchmark_context                    # ★ regression_balance

agent/inbox 消息来自 TechLead
  → ★ 立即冻结 direction_snapshot（含 created_at 时间戳）
  → 填充 direction_context 的完整结构（source, id, prompt_hash 等）
  → direction_created_at 必须 < agent 后续首次 tool call 时间

agent/inbox 或 tool_result 包含 Skill/KB 检索结果
  → 更新 skill_retrieval_context                        # ★

user/message 包含 task assignment
  → 解析 operator、dims、dtype、layout、language、language_version 等
  → 更新 contract_context（含 modifiable_files、commandment_hash）

tool_result 包含 compile error 或 test failure
  → 更新 error_context
  → ★ 若 collection_mode=error_recovery，触发 shadow copy parent 重置（§4.6.1）

reasoning_block 包含策略描述
  → 注意：仅当 direction_context.direction_prompt 为空时才从 reasoning 提取
  → 若 direction 已冻结，不再从 reasoning 覆盖（保证因果性）
```

**关键设计原则：**

1. **只存模型可见信息** — Context Accumulator 只记录 agent 实际看到的内容（tool results、inbox messages），不加入 agent 不可见的 hidden test cases 或 oracle 信息
2. **累积不覆盖** — 新信息追加到 context，不删除旧信息（除非是同一字段的更准确值）。这样采集时的 context 反映 agent 当时的完整知识状态
3. **跨 segment 保留** — Context 在 `present()` segment 边界后不清空，因为架构知识和 contract 在整个 session 中有效
4. **源可追溯** — 每条 context 记录其来源 tool_call_id 和 content_hash，可回溯到 trajectory 原文
5. **Direction 因果性** — direction 在 TechLead 消息到达时立即冻结，后续 reasoning 不能覆盖。`direction_created_at` 必须早于 agent 首次动作，防止事后反推
6. **环境一次性快照** — `environment_context` 和 `model_context` 在 session 开始时采集一次，不在 session 中变化。若环境发生变化（如 GPU 温度漂移），记录变化但不覆盖初始值

### 4.11 OPUS 样本完整 Input 字段

触发 OPUS 采集时，Context Accumulator 的当前快照填入 GEAK-compatible 的 `input` 字段：

```json
{
  "sample_id": "opus-<run_id>-<sequence>",
  "split": "train",
  "sample_domain": "kernel",
  "task_type": "direction_conditioned",
  "input": {
    "contract": {
      "operator": "flash_attention_fwd",
      "language": "opus",
      "architecture": "gfx950",
      "dimensions": {"M": 2048, "N": 2048, "K": 128, "num_heads": 32},
      "input_dtype": "bf16",
      "output_dtype": "bf16",
      "layout": "row_major",
      "optimization_target": "latency",
      "correctness_tolerance": 1e-3,
      "regression_constraints": ["throughput >= 0.98 * baseline"]
    },
    "parent_source": "<parent kernel.opus 完整源码>",
    "architecture_info": {
      "target_gpu": "gfx950",
      "gpu_sku": "MI355X",
      "cache_hierarchy": {
        "L1_line_size": "128B",
        "L2_size": "256MB",
        "LDS_size": "128KB per CU"
      },
      "compute_caps": {
        "mfma_bf16": "512 FLOPS/cycle per CU",
        "mfma_tile": "32x32x16"
      },
      "memory_subsystem": {
        "vgpr_per_cu": 512,
        "max_occupancy_waves": 8
      },
      "source_refs": ["ISA-manual-ch7-p603", "LLVM/SIDefines.h"]
    },
    "baseline": {
      "latency_us": {"mean": 142.3, "std": 2.1, "samples": 10},
      "roofline": {"compute_util": 0.12, "bw_util": 0.23}
    },
    "profile": {
      "counters": {"SQ_WAVES": 1024, "TCC_HIT_sum": 8192, "TCC_MISS_sum": 32768},
      "derived": {"L2_hit_rate": 0.20, "LDS_bank_conflict_rate": 0.0},
      "hotspot": "global_load_tile",
      "profiler": "rocprof-compute"
    },
    "direction": {
      "strategy": "用 direct-to-LDS staging 替代 L1 cache path，消除 4x cache line 放大",
      "techlead_direction": "Focus on data movement, not compute...",
      "constraints": ["VGPR 用量不能超过 128", "保持 wave occupancy >= 4"]
    },
    "error_feedback": null,
    "per_case_benchmark": null
  },
  "output": {
    "patch": "<unified diff>"
  },
  "labels": {
    "patch_applies": true,
    "compile_pass": true,
    "correctness_pass": true,
    "benchmark_valid": true,
    "verified_speedup": 1.15
  },
  "provenance": {
    "schema_version": "geak_kernel_sft_v1",
    "dataset_version": "phase1-v1",
    "run_id": "...",
    "eval_dir": "...",
    "round": 1,
    "engineer_id": "...",
    "candidate_id": "...",
    "kernel_name": "flash_attention_fwd",
    "source_hash": "...",
    "patch_hash": "...",
    "baseline_source_hash": "...",
    "workspace_head": "...",
    "extraction_method": "harness_middleware",
    "context_snapshot_hash": "...",
    "verify_receipt": "opus_verify/<sample_id>/receipt.json",
    "implementation_family_id": "...",
    "source_lineage_id": "...",
    "verify_source": "independent_verify",
    "gpu": "gfx950",
    "gpu_sku": "MI355X",
    "rocm_version": "...",
    "compiler_version": "...",
    "container_digest": "...",
    "lumen_git_sha": "...",
    "geak_git_sha": "...",
    "model_checkpoint": "...",
    "creation_timestamp": "...",
    "commandment_hash": "...",
    "direction_source": "tech_lead",
    "direction_id": "r1_d0",
    "direction_created_at": "...",
    "skill_corpus_hash": "...",
    "orchestrator_budget_total": 10,
    "orchestrator_budget_used": 3,
    "cheating_checks": {
      "harness_modified": false,
      "reference_delegation": false,
      "fixed_output": false,
      "cached_result": false,
      "input_mutation": false
    }
  }
}
```

**与 GEAK Triton/HIP schema 的对齐（含 `GEAK-SFT-Model.md` §8 provenance 要求）：**

| GEAK 标准字段 | OPUS 来源 | 对应 Model.md 章节 |
|--------------|----------|-------------------|
| `input.contract` | Context Accumulator → contract_context | §1, §8.2 |
| `input.contract.language_version` | Context Accumulator → contract_context.language_version | §1 |
| `input.contract.backend_version` | Context Accumulator → contract_context.backend_version | §1 |
| `input.contract.modifiable_files` | Context Accumulator → contract_context.modifiable_files | §6, §8.2 |
| `input.parent_source` | File Shadow Copy → parent snapshot | §1, §8.3 |
| `input.baseline` | Context Accumulator → baseline_context | §1, §2.1 |
| `input.profile` | Context Accumulator → profile_context | §2.2 |
| `input.direction` | Context Accumulator → direction_context（冻结版） | §2.3, §3 |
| `input.error_feedback` | Context Accumulator → error_context | §2.4 |
| `input.per_case_benchmark` | Context Accumulator → per_case_benchmark_context | §2.5 |
| `input.regression_constraints` | Context Accumulator → contract_context.regression_constraints | §2.5 |
| `input.architecture_info` | Context Accumulator → architecture_context | **OPUS 新增** |
| `output.patch` | File Shadow Copy → unified_diff(parent, candidate) | §1 |
| `labels.*` | Independent Verifier receipt（§4.7） | §8.8 |
| `provenance.direction_*` | Context Accumulator → direction_context（冻结版） | §3 |
| `provenance.skill_*` | Context Accumulator → skill_retrieval_context | §7, §8.4 |
| `provenance.environment_*` | Context Accumulator → environment_context | §8.6 |
| `provenance.model_*` | Context Accumulator → model_context | §8.4 |
| `provenance.orchestrator_*` | Context Accumulator → orchestrator_context | §8.5 |
| `provenance.cheating_checks` | Independent Verifier → 作弊检测结果 | §6 |
| `provenance.session_identity.*` | Context Accumulator → session_identity | §8.1 |

**字段完整性门禁：**

```
必填（缺失则 reject）：
  input.contract.operator
  input.contract.language
  input.contract.architecture
  input.contract.language_version
  input.contract.backend_version
  input.parent_source
  input.architecture_info.target_gpu
  provenance.schema_version
  provenance.run_id
  provenance.source_hash
  provenance.patch_hash
  provenance.gpu + gpu_sku
  provenance.rocm_version + compiler_version
  provenance.verify_source = "independent_verify"
  provenance.creation_timestamp

按 task_type 条件必填：
  cold_start:             profile=null, direction=null
  profile_guided:         profile≠null, direction=null
  direction_conditioned:  direction≠null, direction_created_at < agent_first_action
  error_recovery:         error_feedback≠null, failed_source_hash ≠ baseline_source_hash
  regression_balance:     per_case_benchmark≠null, 至少一个 case budget_exceeded=true

可选（缺失写 null，不 reject）：
  input.architecture_info 的详细子字段（cache, compute_caps 等）
  provenance.skill_corpus_hash（若未使用 Skill）
  labels.benchmark_valid, labels.verified_speedup（若 benchmark 不可用）
```

---

## 5. Expected Scale

Per trajectory:
- ~3-5 deliverable segments → ~3-5 analysis_report samples
- ~10-15 research chains → ~10-15 research_path samples
- ~8-12 concept explanations (from reasoning blocks) → ~8-12 concept_grounding samples

From 100+ trajectories:
```
concept_grounding:    ~1000-1200 samples
research_path:        ~1000-1500 samples
analysis_report:       ~300-500 samples
────────────────────────────────────
Total:                ~2300-3200 SFT samples
RL negatives:         ~500-800 (dead end paths, for reward shaping)
```

---

## 6. Validation Criteria

Before training, validate the dataset by checking:

1. **No precise constants in concept_grounding responses** — grep for hex values, bit positions, specific register numbers. Any precise value should be accompanied by "verify against [source]"
2. **All research_path samples have ≥2 tool calls** — single-call samples don't teach cross-validation
3. **All analysis_report samples follow the standard structure** — section headers, quantitative claims, actionable recommendations
4. **Tool results are compressed** — no sample has a single tool result >3000 chars
5. **Golden path is actually golden** — spot-check 20 samples to confirm dead ends were correctly removed

---

## 7. SFT → RL Handoff

What SFT provides as the RL starting point:
- Model knows domain concepts and vocabulary
- Model produces well-structured outputs
- Model uses tools in a reasonable order
- Model attempts cross-validation

What RL then optimizes:
- Tool selection efficiency (minimize steps to answer)
- Dead-end avoidance (learn from negative examples)
- Knowing when to stop researching (confidence calibration)
- Handling novel hardware/scenarios (generalization)
- Multi-step reasoning depth (where 3B active params are the bottleneck)

---

## 8. Co-Training with GPU Kernel Coding Data

### 8.1 Three Code Languages

The kernel coding data spans three languages with different collection pipelines:

| 语言 | 格式 | 采集方式 | Schema |
|------|------|---------|--------|
| **Triton** (Python) | contract/source/feedback → unified diff | Lumen-RL MultiTune 管线（已有） | GEAK `samples.jsonl`（见 `runbook-GEAK-SFT-Dataset.md`） |
| **HIP** (C++) | contract/source/feedback → unified diff | Lumen-RL MultiTune 管线（已有） | GEAK `samples.jsonl`（同 schema） |
| **OPUS** (custom DSL) | contract/source/feedback → unified diff | **Harness Middleware 自动采集**（见 §4, §8.5） | 对齐 GEAK schema |

Triton 和 HIP 已经在生产中：通过完整的 Lumen-RL MultiTune 管线采集，有 compile/correctness/performance 验证和 independent verify。格式定义在 `runbook-GEAK-SFT-Dataset.md` §10.2。

OPUS 是 agent 使用的自定义 kernel DSL。由于没有独立的 MultiTune 管线，OPUS 代码通过 Harness SFT Collector Middleware 在 agent 运行时自动采集（§4 定义）。Agent 无需做任何额外操作。

同样，kernel 分析轨迹数据（concept grounding、research paths、analysis reports）也通过同一 middleware 自动采集——在 `present()` 事件触发时自动执行，而非离线解析轨迹文件。

### 8.2 Compatibility Assessment

The kernel coding data (instruction → code, all three languages) is **directly compatible** with the kernel analysis data:
- **Positive transfer**: Kernel analysis requires reading/understanding source code (LLVM, Triton, existing kernels). Better coding ability directly improves tool result comprehension.
- **Shared vocabulary**: Both datasets use the same domain terms (MFMA, LDS, cache lines, tiling, occupancy).
- **Complementary skills**: Analysis = read code → reason about performance. Coding = understand requirements → write code.
- **Cross-language transfer**: Understanding one GPU kernel language (e.g., Triton) reinforces concepts used in others (e.g., OPUS tiling, HIP shared memory).

### 8.3 Format Differentiation via System Prompt

The two data types use different formats, resolved by system prompt conditioning:

```
Kernel Analysis (agent trajectory format):
  system: "You are a GPU kernel performance analyst. Research thoroughly
           using available tools before drawing conclusions..."
  → multi-turn with tool calls, thinking blocks, structured deliverables

Kernel Coding (instruction → code format):
  system: "You are a GPU kernel developer specializing in AMD CDNA4.
           Write the requested kernel code directly..."
  → single-turn or few-turn, direct code output (unified diff), no tool calls needed
```

The model learns: **different system prompt → different behavior mode**. This is standard SFT practice and well within 30B capacity. All three coding languages share the same coding-mode system prompt — language differentiation comes from the contract and source code context.

### 8.4 Mixing Strategy

Data volumes:
```
Kernel analysis:     ~2,300-3,200 samples (agent trajectory format)
Kernel coding total: ~1,000-10,000 samples (instruction → unified diff)
  ├── Triton:        from GEAK Phase 1 (750-1,000 per arch combo)
  ├── HIP:           from GEAK Phase 1 (750-1,000 per arch combo)
  └── OPUS:          extracted from trajectories (volume TBD, see §8.5)
Ratio (analysis:coding): ~1:1 to 1:3
```

**Recommendation: Simple shuffle.** At this ratio, no special weighting or staging is needed. Both datasets will get adequate exposure per epoch. The three coding languages are shuffled together — no per-language staging.

If coding data is at the upper end (10K), consider:
- Cap kernel coding at ~5K samples (downsample to keep ratio ≤ 1:2)
- Or upsample kernel analysis data via question rephrasing augmentation
- Monitor val loss per-task and per-language to detect if one dominates

### 8.5 OPUS 代码采集 — 通过 `collect_sft` Skill 在线采集

OPUS 代码采集不走 Lumen-RL MultiTune 管线，而是通过 §4 定义的 Harness SFT Collector Middleware 自动采集。Agent 无需任何操作。

**自动采集流程：**

```
Agent 读取 OPUS 文件（首次 read）
  → Middleware 自动保存 parent shadow copy

Agent 修改 OPUS 文件（pwsh/bash write）
  → Middleware 自动保存 candidate snapshot

Agent 编译/运行通过（tool call 返回成功）
  → Middleware 自动触发 opus_kernel 采集：
      ├── 取出 parent shadow + candidate snapshot
      ├── 计算 SHA256，生成 unified diff
      ├── 从 Context Accumulator 取完整 input context（§4.10）：
      │   architecture_info, contract, baseline, profile, direction, error_feedback
      ├── 根据上下文推断 task_type
      ├── Independent verify: fresh workspace compile + correctness（§4.7）
      ├── 打包为 GEAK samples.jsonl 兼容 schema（含完整 input，见 §4.11）
      ├── 保存 parent/ 和 candidate/ artifact 到 run 目录
      └── 写入 <run_dir>/sft_collected/samples/<sample_id>.json
```

**输出 GEAK schema：**

```json
{
  "sample_id": "opus-<run_id>-<sequence>",
  "split": "train",
  "sample_domain": "kernel",
  "task_type": "direction_conditioned",
  "parent_sources": {
    "kernel.opus": {"path": "artifacts/<sample_id>/parent/kernel.opus", "sha256": "..."}
  },
  "candidate_sources": {
    "kernel.opus": {"path": "artifacts/<sample_id>/candidate/kernel.opus", "sha256": "..."}
  },
  "artifact_paths": {
    "parent_workspace": "artifacts/<sample_id>/parent",
    "candidate_workspace": "artifacts/<sample_id>/candidate"
  },
  "output": {"patch": "<unified diff>"},
  "labels": {
    "patch_applies": true,
    "compile_pass": true,
    "correctness_pass": true,
    "benchmark_valid": true,
    "verified_speedup": 1.15
  },
  "provenance": {
    "run_id": "...",
    "candidate_id": "...",
    "source_hash": "...",
    "patch_hash": "...",
    "extraction_method": "harness_middleware",
    "verify_receipt": "opus_verify/<sample_id>/receipt.json",
    "implementation_family_id": "...",
    "verify_source": "independent_verify"
  }
}
```

**与 Triton/HIP 的对比：**

| 维度 | Triton/HIP | OPUS |
|------|-----------|------|
| 采集方式 | Lumen-RL MultiTune 自动管线 | Harness Middleware 自动采集 |
| Verify | MultiTune independent verify（compile + correctness + performance） | Middleware independent verify（compile + correctness + performance 若可用）（§4.7） |
| `labels` 完整性 | 所有字段填充 | compile_pass + correctness_pass 必须为 true；benchmark_valid/verified_speedup 视环境 |
| `verify_source` | `verify_engineer` | `independent_verify` |
| `task_type` | MultiTune 显式设置 | Middleware 从 segment context 自动推断 |

**OPUS 质量保障（与 Triton/HIP 同等标准）：**

OPUS 样本遵循与 Triton/HIP 相同的验证原则——只有 independent verify 通过的才是正样本：

1. **Independent verify 硬门禁**：middleware 在隔离 workspace 中独立运行 compile + correctness（§4.7），不依赖 agent self-report
2. **Patch apply 验证**：unified diff 必须在 fresh workspace 中 cleanly apply
3. **Verify receipt**：每条正样本附带完整 receipt（command、exit code、stdout/stderr、GPU identity）
4. **无 compiler 时不降级**：没有 OPUS compiler 就不产出正样本，样本进入 `pending/` 等待补验
5. **覆盖报告**：OPUS 样本单独追踪统计，与 Triton/HIP 分开报告 verify pass rate

### 8.6 Quality Requirements for All Coding Data

Before mixing, validate coding samples:
1. **Patch must apply** — unified diff must cleanly apply to parent source
2. **GPU kernel specific** — remove any general Python/C++ that isn't GPU-related
3. **Correct hardware targeting** — ensure code targets gfx950/CDNA4 where applicable
4. **Consistent provenance** — every sample must have `source_hash`, `patch_hash`, and traceable origin
5. **Per-language requirements:**
   - Triton/HIP: full GEAK verification chain (compile + correctness + performance + independent verify)
   - OPUS: middleware independent verify (compile + correctness in fresh workspace) + verify receipt（§4.7）

---

## 9. Full Training Pipeline (Updated)

```
Phase 1: SFT (Mixed Co-Training)
  ├── Data A: Kernel Analysis Trajectories (~2.3-3.2K, agent format)
  │   ├── Concept grounding (40%)
  │   ├── Efficient research paths (40%)
  │   └── Analysis report generation (20%)
  │
  ├── Data B: GPU Kernel Coding (three languages, instruction → unified diff)
  │   ├── Triton: ~750-2,000 samples (GEAK-verified, gfx942+gfx950)
  │   ├── HIP:    ~750-2,000 samples (GEAK-verified, gfx942+gfx950)
  │   ├── OPUS:   volume TBD (trajectory-extracted, verify_source=trajectory_only)
  │   └── All share coding-mode system prompt
  │
  ├── Mixing: simple shuffle across all data types and languages
  ├── Total: ~3.3K-8.2K+ samples
  └── Output: model with domain knowledge + tri-language coding + basic tool use

Phase 2: Agentic RL
  ├── Starting point: SFT checkpoint
  ├── Environment: kernel analysis tasks + tool sandbox
  ├── Reward signal: deliverable quality + research efficiency
  ├── Negative signal: dead end paths from raw trajectories (for reward shaping)
  └── Optimizes: tool strategy, dead-end avoidance, reasoning depth
```

### Data Schema Alignment

All kernel coding samples (Triton, HIP, OPUS) use the same GEAK `samples.jsonl` schema defined in `runbook-GEAK-SFT-Dataset.md` §1/§10.2. This ensures:
- Unified loader: one `samples.jsonl` reader handles all three languages
- Content-addressed artifacts: `parent_sources` and `candidate_sources` with SHA256
- Consistent provenance: every sample traces back to run/candidate/environment
- Interoperable split management: lineage-based split assignment works across languages

Kernel analysis trajectory data (Data A) uses a different schema (multi-turn agent messages format) and is stored separately. The two data types merge only at the training dataloader level via simple shuffle.

### SFT → RL Transition

SFT provides the RL starting point — the model already knows domain concepts, can use tools in a reasonable order, and produces structured outputs. RL then optimizes the policy for efficiency and correctness through environment interaction.

The dead end paths extracted during golden path processing (§4 Step 3) serve as negative reward signals during RL — the model learns to avoid tool call patterns that lead to wasted steps.

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| 3B model memorizes values from tool results in training data despite design intent | Silent hallucination in production | Validation criterion #1; also test at inference time by checking if model cites sources |
| Golden path extraction removes necessary context | Model can't understand why a step was taken | Keep reasoning blocks that explain motivation, even if the corresponding dead-end tool call is removed |
| Tool result compression loses critical information | Model learns incomplete analysis patterns | Compression validates against reasoning block references; keep all lines that are cited |
| Dataset too small for 30B model | Underfitting, no generalization | Augment concept_grounding with rephrased questions; consider synthetic trajectory generation from deliverables |
| Format mismatch with actual deployment harness | Training doesn't transfer | Align message format exactly with the DeepSeek Harness agent protocol used in production |
| Coding data overwhelms analysis data | Model favors direct code output over research-first analysis | Cap coding data at 2x analysis data; monitor per-task val loss |
| System prompt conditioning fails | Model mixes analysis and coding behaviors | Include diverse system prompt variants in training; test mode-switching at eval |
| OPUS compiler 不可用导致无法 verify | 无法产出 OPUS 正样本 | 样本进入 pending/ 队列；compiler 就绪后批量补验；不降级为 self-report |
| Golden Path Engine 启发式误判 | 有效 step 被标为 dead end，或 dead end 被留在 golden path | 规则 5 保证最低 2 个 golden step；离线 ETL 抽检 20 条对比人工标注；持续调优规则阈值 |
| Middleware 写入失败或数据损坏 | SFT 样本丢失 | 原子写入（temp → fsync → rename）；manifest 带 SHA256 可校验；写入失败不阻塞 agent |
| Concept snapshot 过度提取 | 产生大量低质量/重复的概念样本 | 文本相似度 ≥ 0.8 自动去重；最低 2 个独立概念定义门禁；离线 ETL 再次去重 |
| File shadow copy 不完整 | Agent 通过非标准方式修改 OPUS 文件（如直接在 reasoning 中生成后一次性写入），middleware 未捕获 parent | Middleware 在 session 开始时对 workspace 内所有 *.opus 文件做初始 snapshot；on_file_write 兜底 |
| One language dominates training | Model specializes in Triton but underperforms on HIP/OPUS | Monitor per-language val loss; enforce rough balance in sampling if needed |
| RL cold-start from SFT checkpoint | RL exploration may be inefficient if SFT policy is too weak | Ensure SFT golden paths are high quality; use dead-end negatives for reward shaping |
