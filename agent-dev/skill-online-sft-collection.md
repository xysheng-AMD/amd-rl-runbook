# Skill: Online SFT Data Collection (Buffer-Only Middleware)

## 目标

在 kernel agent loop 运行时以 **middleware** 形式挂载，实时录制事件到内存 buffer。
Agent 完全无感知——不占用 tool call 配额、token 预算或注意力。

**关键原则：online 只做"录像"，不做"剪辑"。**

所有重处理（golden path、concept extraction、quality gate、independent verify、磁盘写入）
全部推迟到 agent loop 结束后执行。

## 你是谁

你是 Harness 基础设施层的 middleware 配置 agent。你的任务是将 SFT Collector Middleware
正确集成到 agent 运行环境中，确保：
1. agent loop 运行期间零干扰
2. session 结束后事件被完整保存
3. 保存的事件可以被 offline 管线处理

## 架构

```
Agent Loop（正常运行，无感知）
  │
  ├── tool call: read / web_fetch / grep / bash / ...
  ├── reasoning blocks
  ├── present() deliverables
  └── file writes (*.opus, *.hpp, ...)
  │
  │  [事件流]
  ▼
SFT Collector Middleware（buffer-only）
  │
  ├── Event Interceptor → 路由事件（try/except 全包裹）
  └── 内存 buffer → list.append() 只做这一件事
        │
        │  [session 结束]
        ▼
  online_buffer_<timestamp>.jsonl  ← best-effort dump
        │
        │  [agent loop 结束后，显式调用]
        ▼
  Offline Pipeline（golden path / quality gate / verify / 写盘）
```

## 安全保证

| 保证 | 实现方式 |
|------|---------|
| 不阻塞 agent loop | 所有 callback 只做 `list.append`，微秒级 |
| 异常不传播 | interceptor + middleware 每个 handler 独立 `try/except` |
| 内存有上限 | 单条内容 ≤ 50 KB（截断）；总事件 ≤ 100,000 条（丢弃） |
| 不做数据变换 | 不解析 JSON、不解析时间、不调用任何 core 模块 |
| flush 不影响后续 | session_end 时 flush 是 best-effort，失败静默忽略 |
| 不调用 verify | 编译/测试/benchmark 全部推迟到 process_buffer() |

## 集成步骤

### Step 1: 在运行环境中挂载 Middleware

```python
from sft_collector.online.middleware import SFTCollectorMiddleware

config = {
    "run_id": "online-run-001",
    "collection_mode": "direction_conditioned",
    "workspace_dir": "/path/to/workspace",  # process_buffer 时用
    "contract": {
        "operator": "flash_attention_fwd",
        "language": "opus",
        "language_version": "0.1.0",
        "backend_version": "rocm-6.2",
    },
    "architecture": {
        "target_gpu": "gfx950",
        "gpu_sku": "MI355X",
    },
}

middleware = SFTCollectorMiddleware(
    run_dir="/path/to/run",
    config=config,
)
middleware.attach(harness)  # 只注册回调，不做任何处理

# ========================================
# Agent loop 正常运行 — middleware 只在后台 buffer
# ========================================
harness.run()

# Agent loop 结束后，才做重处理
result = middleware.process_buffer(skip_verify=False)
print(result)
# {"analysis": 5, "opus": 3, "concept": 2, "dead_end": 1, "rejected": 0}
```

### Step 2: Harness 事件注册

Middleware 通过 `EventInterceptor` 注册到 Harness 事件系统。
Harness 需要支持以下接口之一：

```python
# 方式 A: 统一事件接口
harness.on_event(callback)  # callback(event_type: str, data: dict)

# 方式 B: 分类 hook 接口
harness.register_hook("tool_call", callback)
harness.register_hook("tool_result", callback)
# ...

# 方式 C: middleware 接口
harness.add_middleware(middleware_instance)
```

### Step 3: 事件录制规则

| 事件 | buffer 记录 |
|------|------------|
| `tool_call(*)` | `{"type": "tool_call", "data": {"name": ..., "arguments": ...}}` |
| `tool_result(*)` | `{"type": "tool_result", "data": {"content": ...}}` (≤50KB) |
| `reasoning(*)` | `{"type": "reasoning", "data": {"text": ...}}` (≤50KB) |
| `inbox(*)` | `{"type": "inbox", "data": {"sender": ..., "text": ...}}` |
| `file_read(*.opus)` | `{"type": "file_read", "data": {"path": ..., "content": ...}}` (≤50KB) |
| `file_write(*.opus)` | `{"type": "file_write", "data": {"path": ..., "content": ...}}` (≤50KB) |
| `present(*)` | `{"type": "present", "data": {"deliverable": ...}}` |
| `session_start` | `{"type": "session_start", "data": {...config...}}` |
| `session_end` | `{"type": "session_end", "data": {}}` |

所有事件附带 `"timestamp"` 字段（`time.time()` 浮点数）。

### Step 4: 验证集成

```bash
# agent loop 结束后检查 buffer 文件
ls <run_dir>/sft_collected/online_buffer_*.jsonl

# 查看事件数
wc -l <run_dir>/sft_collected/online_buffer_*.jsonl

# 用 offline 管线处理（或者代码里直接调 process_buffer()）
python -m sft_collector offline \
  --trajectory-dir <run_dir>/sft_collected/ \
  --run-dir <run_dir>
```

## 与 Offline 管线的关系

Online middleware 的输出是一个 JSONL 文件，格式和 offline 管线的输入兼容。
两者的关系是**生产者-消费者**：

```
Online Middleware  →  online_buffer_*.jsonl  →  Offline Pipeline  →  samples/
```

| 维度 | Online Middleware | Offline Pipeline |
|------|-------------------|------------------|
| 何时运行 | agent loop 期间 | agent loop 结束后 |
| 做什么 | 纯 buffer（list.append） | 全部重处理（golden path, verify, ...） |
| 资源消耗 | 微秒级 CPU + 有限内存 | 可以用满 CPU/GPU |
| 失败影响 | 零 — 异常被吞掉 | 只影响数据采集，不影响 agent |
| 输出 | `online_buffer_*.jsonl` | `samples/`, `dead_ends/`, `rejected/` |

## 代码位置

```
/home/danyzhan/amd-rl-runbook/agent-dev/sft_collector/
├── online/
│   ├── event_interceptor.py   # 事件拦截路由（全 try/except）
│   └── middleware.py           # Buffer-only middleware
├── core/                       # 共享组件（offline 时才调用）
├── offline/                    # 重处理管线
├── etl/                        # 后处理
└── cli.py
```
