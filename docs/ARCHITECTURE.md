# FinAgent · 架构设计

> 这是 v1.0 的**目标蓝图**。实现是渐进的：每课只落地当前版本需要的部分（见第 5 节）。
> 每课结束时，回来把新长出来的部分标成"已实现"。

---

## 1. 分层架构（目标态）

```
+----------------------------------- Interface ------------------------------------+
|   CLI (typer)  [L01]          FastAPI + SSE  [L15]          Gradio demo  [L16]   |
+----------------------------------------+-----------------------------------------+
                                         |
+------------------------------- Orchestration -----------------------------------+
|   answer pipeline [L02] -> tool loop [L04] -> ReAct agent [L07] -> router [L15] |
|   guardrails [L16]                                                               |
+-----------+------------------------------+-------------------------------+-------+
            |                              |                               |
+-----------v----------+   +---------------v------------+   +--------------v-------+
|  LLM Layer           |   |  Tools                     |   |  Retrieval (RAG)     |
|  LLMClient interface |   |  registry        [L04]     |   |  chunk/embed  [L05]  |
|   - OpenAICompat[L01]|   |  calculator      [L04]     |   |  vector index [L05]  |
|     DeepSeek / vLLM  |   |  search_filings  [L07]     |   |  BM25/hybrid  [L06]  |
|   - LocalHF    [L08] |   |  get_metric      [L07]     |   |  reranker     [L06]  |
+-----------^----------+   +----------------------------+   +--------------+-------+
            | loads LoRA adapters / merged weights                         |
+-----------+---------------- Training (offline) -----------+   +----------v-------+
|  build_sft [L10] -> SFT+LoRA [L11] -> QLoRA [L12]          |   |  Data            |
|  -> DPO [L13] -> GRPO [L14] -> merge/quantize [L15]        |<--|  10-K corpus     |
+-----------------------------^------------------------------+   |  FinQA / FinBench|
                              | trajectories (JSONL)             |  eval sets       |
+-----------------------------+------------------------------+   +------------------+
|  Cross-cutting:  Eval harness [L03]  |  Tracing & cost log [L01]  |  Config [L01] |
+-----------------------------------------------------------------------------------+
```

**读图要点**
- **上下分层**：上层依赖下层的**接口**，不依赖具体实现。Orchestration 不知道背后是 DeepSeek 还是 Qwen。
- **横切关注点**（最下面一栏）贯穿所有层：每次 LLM 调用都被 tracing 记录，每个系统版本都被 eval 度量。
- **Training 是离线的**，它的输入是 Orchestration 产出的 agent 轨迹（trajectories），输出是 LLM Layer 加载的权重。这是本项目最有特色的闭环：**线上系统产生数据 → 离线训练 → 更新线上模型**，也就是工业界说的 data flywheel。

## 2. 一个请求的生命周期（以 v0.7 的多跳问题为例）

```
Q: "Compare NVDA and AMD gross margin change from FY2023 to FY2024"
 |
 |-- guardrail: 是投资建议吗? 否 -> 放行                               [L16]
 |-- agent loop step 1: LLM 决定调用 search_filings(NVDA, FY2024, "gross margin")
 |      '-- retriever: BM25 + dense -> RRF -> rerank -> top-5 chunks     [L05-06]
 |-- step 2..4: 同理检索 NVDA FY2023、AMD FY2024、AMD FY2023
 |-- step 5: LLM 调用 calculator("(72.7-56.9)") ...                     [L04]
 |-- step 6: LLM 输出结构化答案 {value, unit, reasoning, sources[]}      [L02]
 |-- 校验: sources 里的 chunk id 真实存在吗? 数值能和工具结果对上吗?       [L16]
 '-- 记录: tracing（tokens/cost/latency）+ 轨迹（完整 messages/tool calls）写入 JSONL  [L01, L07]
```

（上面的数字只是示意，真实值以财报为准。）

## 3. 关键设计决策（ADR 精简版）

每条决策都写清**背景 → 决定 → 代价**。面试时被问"为什么这么设计"，照这个结构回答即可。

### ADR-1 · 统一的 `LLMClient` 接口 + 内部采用 OpenAI chat 格式
- **背景**：本项目最大的变化轴是模型后端：DeepSeek API → 本地 HF（进程内加载）→ 本地推理服务（vLLM / Ollama）。
- **决定**：
  - 上层只调用 `client.chat(messages, tools)`，不关心后端是谁。
  - 内部消息格式**直接采用 OpenAI chat 格式**（`role` / `content` / `tool_calls`），不自己发明。这是事实标准：DeepSeek、vLLM、Ollama、llama.cpp 的服务端都用这种格式，HF 的 `apply_chat_template` 也直接接受它。
  - 于是只需要两个实现：`OpenAICompatClient`（L01 连 DeepSeek，L15 换个 base_url 连本地服务）和 `LocalHFClient`（L08，进程内加载权重）。
- **代价**：绑定了一家公司定义的 schema；各厂商的扩展字段（如 DeepSeek 的 `thinking` 参数、`reasoning_content`）要在 adapter 里单独处理，抽象会有一点"漏"。换来的是几乎零转换成本，值得。

### ADR-2 · Eval 是一等公民，从 v0.3 开始
- **背景**：没有度量，所有"优化"都是猜测。
- **决定**：一个"系统版本" = 一个 YAML config。`finagent eval --config X` 产出不可变的 run 目录（results、summary、config 快照、git commit）。
- **代价**：前期多花约一天建 harness，但后面每一课都省时间。

### ADR-3 · 配置驱动的组件开关
- **决定**：模型、prompt 版本、检索策略（dense/hybrid/rerank）、top-k、工具集合全部写在 `configs/systems/*.yaml`，用 pydantic 校验。
- **好处**：做消融实验只需要改 config，不用改代码；SCOREBOARD 每一行都对应一个 config 文件。

### ADR-4 · Prompt 是文件，不是代码里的字符串
- **决定**：`prompts/*.yaml`（YAML 能表达 system、few-shot 示例、user 模板这样的结构），文件名带版本号；模板变量写成 `{{name}}`（不和 JSON 的花括号、金额的 `$` 冲突）；eval 记录用的是哪个 prompt。
- **好处**：可以 diff、可以回滚，改 prompt 不用碰代码。

### ADR-5 · 一切可追踪（trace everything）
- **决定**：每次 LLM 调用和每个 agent step 都写 JSONL。
- **好处**：同一份数据同时服务**可观测性**（成本、延迟、调试）和**训练**（L10 蒸馏数据）。

### ADR-6 · 不用 LangChain / LlamaIndex
- **背景**：这是学习项目，面试时要能讲清每一行；框架的抽象变化快，而且会把关键机制藏起来。
- **决定**：只用基础库（openai SDK、transformers、PEFT、TRL、FAISS）。
- **代价**：多写一些胶水代码。收获是做完后你能看懂这些框架在做什么，工作中需要时一天就能上手。

### ADR-7 · 先手写，再换库
- numpy 检索 → FAISS；手写 LoRA → PEFT；手写 loss mask → TRL。手写版本保留在 `walkthroughs/` 或测试里，作为理解和对照的基线。

### ADR-8 · 大文件不进 git
- 原始数据、向量索引、checkpoint、eval 运行结果放在 `data/`、`runs/`（已 gitignore）。进 git 的只有代码、config、prompt 和**小体量的 eval 集**（`evalsets/`，它们是项目的"规格"，必须版本化）。

## 4. 目录结构（目标态，标注出现的课次）

```
finagent/
|-- README.md  ROADMAP.md  SCOREBOARD.md  pyproject.toml  .env.example   [L00]
|-- docs/            PRD.md  ARCHITECTURE.md                              [L00]
|-- lessons/         Lxx_*.md 讲义                                        [L00+]
|-- scripts/         check_env.py [L00]  l01_compare.py [L01]（每课的实验脚本）
|-- configs/
|   |-- systems/     v0.1.yaml ... v1.0.yaml（每个系统版本一份）            [L01+]
|   '-- train/       sft_*.yaml  dpo_*.yaml                               [L11+]
|-- prompts/         answer_v1.yaml / v2 / v3（每个版本一个文件）          [L02]
|-- evalsets/        seed_closed_book.jsonl [L01]  finqa / financebench / multihop [L03+]
|-- src/finagent/
|   |-- cli.py                                                            [L01]
|   |-- config.py    tracing.py                                           [L01]
|   |-- llm/         pricing.py（按 token 计费）                          [L01]
|   |-- llm/         base.py  openai_compat.py                            [L01]
|   |                local_hf.py                                          [L08]
|   |-- schemas.py   prompts.py（答案 schema、prompt 模板）                [L02]
|   |-- pipeline/    answer.py                                            [L02]
|   |-- eval/        datasets.py  metrics.py  judge.py  run.py            [L03]
|   |-- tools/       registry.py  calculator.py                           [L04]
|   |                search_filings.py  get_metric.py                     [L07]
|   |-- agent/       loop.py [L04]   react.py [L07]
|   |-- data/        edgar.py  ingest.py                                  [L05]
|   |-- rag/         chunker.py  embedder.py  index.py  retriever.py      [L05]
|   |                bm25.py  hybrid.py  reranker.py  query_rewrite.py    [L06]
|   |-- training/    data/build_sft.py [L10]  sft.py [L11]
|   |                dpo.py [L13]  grpo.py [L14]
|   |-- serve/       api.py  router.py                                    [L15]
|   '-- guard/       guardrails.py                                        [L16]
|-- walkthroughs/    Lxx_*.py 机制演示脚本（每课一个）                      [L04+]
|-- tests/                                                                [L01+]
|-- data/            (gitignored) raw/ processed/ index/ sft/
'-- runs/            (gitignored) eval 运行结果、trajectories/、checkpoints
```

## 5. 演进表：每个版本点亮哪些模块

| 版本 | 新增 | 系统能做什么 |
|---|---|---|
| v0.0 | 骨架、环境 | 什么都做不了，但知道要去哪 |
| v0.1 ✅ | `llm/`（base / openai_compat / pricing）、`cli.py`、`tracing.py`、`config.py` | 闭卷问答，记录成本 |
| v0.2 ✅ | `prompts/`、`schemas.py`、`prompts.py`、`pipeline/answer.py` | 结构化答案，可自动判分 |
| v0.3 | `eval/`、`evalsets/`、SCOREBOARD | **可度量** |
| v0.4 | `tools/`、`agent/loop.py` | 算对数 |
| v0.5 | `data/`、`rag/` 基础 | 有依据地回答，带引用 |
| v0.6 | `rag/` 进阶 | 检索更准 |
| v0.7 | `agent/react.py`、trajectories | 多跳问题；**老师系统完成** |
| v0.8 | `llm/local_hf.py` | 可切换到本地学生 |
| v0.9 | （nanolm 侧实验） | 理解 LoRA |
| v0.10 | `training/data/` | 有了训练数据 |
| v0.11 | `training/sft.py` | 学生变强 |
| v0.12 | QLoRA 配置 | 学生变大（4B） |
| v0.13 | `training/dpo.py` | 学生少编造 |
| v0.14 | `training/grpo.py` | 学生算得更准（进阶） |
| v0.15 | `serve/`（`openai_compat` 改指本地推理服务） | 可服务、可路由 |
| v1.0 | `guard/`、UI、CI | 可交付 |

## 6. 技术栈与选型理由

| 需求 | 选择 | 备选 | 理由 |
|---|---|---|---|
| 老师 LLM | DeepSeek：`deepseek-flash` / `deepseek-v4-pro` | Claude、GPT | 价格远低于同级闭源模型，学生预算友好；支持 tool calls、thinking 模式、JSON mode；模型权重以 MIT 协议开源。价格见 PRD 第 5 节 |
| API SDK | openai（OpenAI 兼容格式） | 各家原生 SDK | 一个 SDK 同时连 DeepSeek 和本地推理服务（vLLM / Ollama） |
| 学生 LLM | Qwen3 0.6B / 1.7B / 4B | Llama 3.2 3B、Gemma | 原生支持 tool calling 的 chat template、Apache-2.0 许可、尺寸梯度覆盖 8GB 可训范围 |
| Embedding | bge-small-en-v1.5 → bge-base | e5、gte | 小而快，8GB 友好 |
| Reranker | bge-reranker-base | MiniLM cross-encoder | 质量与速度平衡 |
| 向量索引 | numpy → FAISS | Chroma、Qdrant | 先理解再用库；FAISS 不需要额外服务 |
| 稀疏检索 | rank_bm25 | Elasticsearch | 零运维 |
| 训练 | transformers + PEFT + TRL + bitsandbytes | Unsloth、Axolotl | 主流，面试通用；Unsloth 在 Windows 上麻烦 |
| 推理服务 | llama.cpp / Ollama；vLLM（WSL2，选做） | SGLang、TGI | Windows 兼容性 |
| API | FastAPI + SSE | Flask | 原生 async、自动文档 |
| 配置 | YAML + pydantic | Hydra | 够用、好懂 |
| 实验记录 | JSONL + TensorBoard（W&B 可选） | MLflow | 轻量 |
| CLI | typer | argparse、click | 类型注解即参数定义 |
