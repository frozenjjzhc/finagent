# FinAgent 学习路线图

> 一句话：做一个能读懂美股财报（10-K）、**算对数**、**给出处**的 AI 财报分析助手。
> 先用 DeepSeek API 把它做到"能用"（老师），再把能力**蒸馏**进一个跑在你 RTX 4070（8GB）上的 1.7B～4B 小模型（学生）。
>
> 同一个项目从 v0.1 演进到 v1.0，每一课只引入**一个**新知识点，而且都有一个用得上它的理由。
> 做完之后，你会覆盖 LLM 应用 + 后训练 + 部署三大块的主要知识点。

---

## 0. 设计原则（为什么这样排课）

1. **一条主线，一个项目。** 不做 17 个玩具 demo，而是一个项目的 17 次演进。每课结束项目都**能跑**，打一个 git tag（v0.1, v0.2 …）。
2. **痛点驱动。** 每一课都由上一版暴露出的问题引出：会编数字 → RAG；算错 → Tools；太贵/数据外发 → 本地小模型；小模型不会 → SFT；显存不够 → QLoRA；还会编 → DPO。
   这也是面试时"你为什么用 X"的标准答案：**技术是从需求里推出来的，不是拿来炫的。**
3. **同一把尺子。** L03 建好 eval 之后，每一课都在同一套评测上报数，写进 `SCOREBOARD.md`。项目做完时，这张表就是你简历上的数字。
4. **先手写，再用库。** numpy 暴力检索 → FAISS；手写 tool loop（不用任何 agent 框架）；在 nanolm 上手写 LoRA → PEFT；手写 loss mask → TRL。不用 LangChain / LlamaIndex，这样面试时每一行你都讲得清。
5. **约束是真的。** 8GB 显存、Windows、学生预算，这些正好是 MLE 每天要面对的工程权衡，不当作借口。
6. **延续 nanolm 的学法。** 每课 = `lessons/Lxx_*.md` 讲义 + 一个 `walkthroughs/Lxx_*.py` 演示脚本（小例子逐步打印，让你看清机制，同 nanolm 里 `*_trace.py` 的学法）+ 真正的项目代码 + 一组实验 + 自测题。

---

## 1. 演进总览

```
 Phase 1  应用层：用好一个强模型（DeepSeek = Teacher）                          Phase 2  模型层
 +------+  +------+  +------+  +------+  +------+  +------+  +------+         +------+
 | v0.1 |->| v0.2 |->| v0.3 |->| v0.4 |->| v0.5 |->| v0.6 |->| v0.7 |-------->| v0.8 |
 | API  |  |Prompt|  | Eval |  |Tools |  | RAG  |  | RAG+ |  |Agent |         |Local |
 +------+  +------+  +------+  +------+  +------+  +------+  +--+---+         +--+---+
                                                                |  trajectories  |
                                                                v  (训练数据)    v
 Phase 4  工程层                Phase 3  训练层：让小模型变强（Student = Qwen 1.7B / 4B）
 +------+  +------+  +------+  +------+  +------+  +------+  +------+  +------+
 | v1.0 |<-| v0.15|<-| v0.14|<-| v0.13|<-| v0.12|<-| v0.11|<-| v0.10|<-| v0.9 |
 | Prod |  |Serve |  | GRPO |  | DPO  |  |QLoRA |  | SFT  |  | Data |  | LoRA |
 +------+  +------+  +------+  +------+  +------+  +------+  +------+  +------+
```

**两个关键转折点：**
- **v0.7 → v0.8**：老师系统做完后，第一次把后端换成本地小模型，分数会掉很多。后面整个 Phase 3 的目标就是把这个差距补回来。
- **v0.7 的轨迹（trajectories）**：L07 让 agent 记录每一步轨迹（JSONL）。这些轨迹就是 L10 蒸馏用的训练数据，前后两个阶段在这里接上。

---

## 2. 课程表

图例：⏱ 预计时长（含实验）· 🔗 与已有知识的连接 · 🎤 学完能答的面试题

### Phase 0 · 立项

#### L00 · 从需求到架构 → v0.0（骨架）⏱ 3h
- **知识点**：需求分析（用户故事、功能需求/非功能需求、非目标）· 需求到技术的映射 · 分层架构与"变化轴" · 演进式架构 · 工程骨架（src layout、依赖、密钥管理）
- **动手**：读懂并改写 `docs/PRD.md`，理解 `docs/ARCHITECTURE.md`，搭环境，跑通 `scripts/check_env.py`
- 🎤 "Design an LLM system for X" 的开场框架 · "RAG vs fine-tuning，怎么选？"

### Phase 1 · 应用层：用好一个强模型

#### L01 · LLM API 基础 → v0.1 ⏱ 4h
- **痛点**：项目还不存在，先让它能说话。
- **知识点**：Chat Completions API（OpenAI 兼容格式：system / user / assistant / tool 四种角色。这是事实标准，DeepSeek、vLLM、Ollama 都用它）· token 与计费（缓存命中和未命中分开计价；高峰/非高峰两档价格）· `finish_reason`（stop / length / tool_calls / content_filter）· streaming · 错误分类与重试（429/5xx 可重试，400/401 不可重试）· 模型选型的能力/价格/延迟三角 · **thinking 模式**（`reasoning_content`、`reasoning_effort`）
- **注意**：DeepSeek **默认开启** thinking 模式，而 thinking 模式下 `temperature` **不报错、但静默无效**。做采样实验要先关掉 thinking（`extra_body={"thinking": {"type": "disabled"}}`）。这种"参数被悄悄忽略"的坑，就是为什么要先读文档、再用实验验证。L08 会在本地模型上亲手实现采样，接上你 nanolm 的代码。
- **动手**：`llm/base.py`（统一接口）+ `llm/openai_compat.py`（用 openai SDK 连 DeepSeek；L15 同一个类换个 base_url 就能连本地推理服务） · `finagent ask "..."` CLI · 每次调用写一行 JSONL（model / tokens / 缓存命中 / cost / latency）
- **实验**：closed-book 问 10 个财报问题（如 "NVIDIA FY2025 数据中心收入？"），手工核对。你会看到模型**很自信地编数字**，还会碰到**知识截止**问题。再用同一批题对比 `deepseek-flash` 和 `deepseek-v4-pro`、thinking 开和关的效果与成本，定下老师模型。
- 🎤 What is a token? · 怎么处理 rate limit？ · 怎么在模型之间做选择？

#### L02 · Prompt Engineering + 结构化输出 → v0.2 ⏱ 4h
- **痛点**：v0.1 输出是自由文本，格式飘忽、单位混乱（million/billion），程序没法判分。
- **知识点**：system prompt 的职责（角色、规则、格式、拒答边界）· zero-shot / few-shot · 先推理后回答（CoT）与 thinking · **结构化输出**（DeepSeek 的 JSON mode `response_format={"type": "json_object"}` 只保证输出是合法 JSON，**不保证符合你的 schema**，偶尔还会返回空内容；所以 pydantic 校验 + 失败重试要自己做）· prompt 模板化与版本化（prompt 是文件，不是代码里的字符串）· 常见反模式
- **动手**：`prompts/answer_v1.md …` · 输出 schema `Answer{value, unit, reasoning, sources, confidence}` · 解析失败的处理
- **实验**：3 个 prompt 版本在 10 道题上**手工**打分。你会发现手工打分慢、不可复现、样本太少分不出高下，这就是下一课要解决的问题。
- 🎤 怎么让 LLM 输出稳定可解析？ · few-shot 什么时候反而有害？ · CoT 为什么有效？

#### L03 · Eval 驱动开发 → v0.3 ⏱ 6h ★ 全课程最重要的一课
- **痛点**："我改了 prompt，到底有没有变好？"
- **知识点**：golden set 构建（覆盖面、难度分层、来源：FinanceBench 开源子集 + 自写题；FinQA 需要给定上下文才能作答，挪到 L04）· **dev / test 划分**（防止把 prompt 调到 test 上过拟合）· 指标：数值题用带容差的 exact match（相对误差 < 1%），文本题用 LLM-as-judge + rubric · judge 的偏差（位置、长度、自我偏好；被测系统和 judge 都是 DeepSeek 时要特别注意自我偏好）与人工校准 · **样本量与置信区间**（100 道题上差 3% 往往不显著，用 bootstrap 看）· 可复现（run 目录记录 config + prompt 版本 + commit）
- **动手**：`evalsets/*.jsonl` · `eval/metrics.py` · `eval/judge.py` · `eval/run.py` → `runs/<时间戳>/{results.jsonl, summary.json}` · `finagent eval --set financebench --split dev` · `finagent compare`（配对 bootstrap + McNemar）
- **实验**：定量重比 L02 的 3 个 prompt，SCOREBOARD 写入第一行 baseline。
- 🔗 相当于 nanolm 里的 val loss，只是这里要自己定义"好"是什么。
- 🎤 How would you evaluate an LLM system? · LLM-as-judge 可靠吗，怎么校准？ · 怎么判断一个提升是真的？

#### L04 · Tool Use（函数调用）→ v0.4 ⏱ 5h
- **痛点**：eval 显示，即使把财报原文直接给模型（oracle context），增长率、比率、CAGR 这类多步算术仍然会错。（L04 实测修正：flash 的四则运算其实很准，弱点是开方类运算；FinQA 上计算器的提升主要来自"把计算做完整"）
- **知识点**：LLM 为什么算不好数（tokenization 把数字切得很乱，L08 你会亲眼看到）· function calling 的本质：模型只输出"调用意图"（JSON），执行在你这边 · tool schema 设计（名字和描述的写法直接影响调用准确率）· **手写 tool loop**：请求 → `finish_reason=tool_calls` → 执行 → 用 `role: tool` 消息回传结果 → 再请求 · 并行调用 · 错误回传（把异常信息当作工具结果返回，让模型自己纠正）· thinking 模式下带工具时，之前各轮的 `reasoning_content` 要原样传回（DeepSeek 文档的要求）· 安全：计算器用 AST 白名单，**绝不用 `eval()`**
- **动手**：`tools/registry.py`（装饰器从函数签名 + docstring 自动生成 schema）· `tools/calculator.py` · `agent/loop.py` · `walkthroughs/L04_tool_loop.py`
- **实验**：先构建 FinQA 评测子集（每题附带财报片段，也就是给定上下文），再比较 no-tool vs calculator。控制变量：上下文完全相同，只变工具，就是你 CSCI 566 用的那套方法。另加纯算术探针（`scripts/l04_arith_probe.py`）和两个消融（不给使用说明、开 JSON mode）；主对比重复跑两次。
- 🎤 function calling 底层怎么工作？ · 怎么设计好的 tool schema？ · 模型传了坏参数怎么办？

#### L05 · RAG ① 基础管线 → v0.5 ⏱ 7h
- **痛点**：closed-book 时模型不知道新财报，还会编；一份 10-K 有 5～10 万 tokens，整份塞进上下文又贵又慢，还有 lost-in-the-middle 问题。
- **知识点**：ingest → chunk → embed → index → retrieve → generate 全流程 · 数据获取与清洗（SEC EDGAR HTML、财报 PDF；可移植你 ai-equity-research 里的 `edgar.py`）· chunking：定长 + overlap vs 按章节（Item 7 MD&A / Item 8 财务报表）· **表格怎么处理**（财报 RAG 的核心难点）· embedding 原理（bi-encoder、对比学习、cosine）· **先用 numpy 手写暴力检索，再换 FAISS**，理解 ANN（IVF / HNSW 的直觉）· 带引用的生成
- **动手**：`data/ingest.py` · `rag/{chunker, embedder, index, retriever}.py` · `walkthroughs/L05_retrieval.py`（5 个玩具 chunk，把相似度矩阵打印出来）
- **实验**：closed-book vs RAG(top-5) 的端到端准确率 + 检索 Recall@k（FinanceBench 带 evidence 页码，可以自动判断检索是否命中）
- **实际做法（L05 记录）**：语料改用 FinanceBench 自带的 PDF（HTML 没有页码，没法和证据页对照），`edgar.py` 的移植推迟到需要新财报时；表格用"按坐标重建行"处理；另加 oracle（直接给证据页）上限实验和 small-to-big（用块检索、给整页）消融。
- 🎤 讲一遍 RAG pipeline · chunk size 怎么选？ · RAG vs 长上下文 vs 微调？

#### L06 · RAG ② 检索质量工程 → v0.6 ⏱ 6h
- **痛点**：错误分析发现大部分错误出在"没检索到"。比如问 FY2023 却检索到 FY2022 的段落；"Item 7A"、具体数字这类精确词，dense 检索不擅长。
- **知识点**：**错误分析方法**（先把错误分成检索错和生成错两桶，再决定优化哪边）· BM25 与稀疏检索 · hybrid + RRF 融合 · cross-encoder reranker（比 bi-encoder 准但慢，所以用两阶段）· metadata 过滤（公司/年份）· query 改写 / HyDE / multi-query · 检索指标 Recall@k / MRR / nDCG · 消融实验
- **动手**：`rag/{bm25, hybrid, reranker, query_rewrite}.py`，所有组件由 config 开关
- **实验**：消融表 dense / +BM25 / +rerank / +filter / +rewrite，同时报检索指标和端到端准确率，再扫一遍 chunk size
- ⭐ **选修 S1 · Embedding 微调**：用 (query, 正确 chunk) 对，以 InfoNCE loss 对比微调 bge-small。这是你在本项目里的第一次"微调"，1 小时左右能跑完。
- 🎤 RAG 答错了你怎么 debug？ · 为什么要 rerank？ · dense vs sparse？

#### L07 · Agent：ReAct + Agentic RAG → v0.7 ⏱ 6h（Phase 1 收官）
- **痛点**："对比 NVDA 和 AMD 近两年毛利率变化"这种问题，一次检索加一次生成做不到，需要拆解 → 多次检索 → 计算 → 汇总。
- **知识点**：workflow vs agent（什么时候**不该**用 agent）· ReAct 循环（Thought → Action → Observation）· 把检索本身做成工具（agentic RAG）· 任务拆解 · 终止条件、max_steps、死循环检测、成本上限 · 上下文管理（工具结果太长怎么办）· **prompt injection**：检索到的文档里写着"忽略之前的指令"怎么办 · **轨迹日志**：messages / tool calls / results 全部存成 JSONL，这就是 Phase 3 的训练数据
- **动手**：`agent/react.py` · 工具 `search_filings` / `calculator` / `get_financial_metric`（可选：接 XBRL 结构化数据）· `runs/trajectories/*.jsonl`
- **实验**：新增约 30 道 multi-hop 题，对比 single-shot RAG vs agent 的准确率、平均步数、每题成本
- 🏁 **到这里你已经有一个完整可用的老师系统。** 可以先写进简历（RAG + Agent + Eval，有真实数字），刚好赶上申请季。
- ⭐ **选修 S2 · MCP**：把你的财报工具包装成 MCP server，让 Claude Code / Claude Desktop 直接调用。
- 🎤 Design an agent for X · 怎么防止死循环和成本失控？ · RAG 里的 prompt injection 怎么防？

### Phase 2 · 模型层：把模型搬回本地

#### L08 · 本地推理与模型解剖 → v0.8 ⏱ 6h
- **痛点**：老师系统依赖外部 API：数据要发给第三方、每次延迟好几秒、模型和价格都不由你控制。先看看本地小模型直接上场有多差。
- **知识点**：transformers 加载 Qwen3-1.7B（或届时同尺寸最新小模型）· **tokenizer**：BPE vs 你 nanolm 的 char-level，看数字怎么被切（呼应 L04）· **chat template**：messages 如何变成带特殊 token 的原始字符串（SFT 的基础，必须亲眼看一遍）· 对照你的 `gpt.py` 看结构差异（RMSNorm、RoPE、GQA、SwiGLU）· 生成：greedy / temperature / top-k / top-p 亲手调 · **KV cache** · 显存算术（参数量 × 字节数；bf16 / int8 / 4bit）· bitsandbytes 4-bit 加载 · 本地模型的 tool calling（Qwen 模板里的工具格式）
- **动手**：`llm/local_hf.py`，实现**同一个** `LLMClient` 接口（内部消息本来就是 OpenAI 格式，可以直接交给 `tokenizer.apply_chat_template`），改一行 config 就能用本地模型跑完整 eval · `walkthroughs/L08_chat_template.py`
- **实验**：SCOREBOARD 新增 Qwen 裸跑（同样的 RAG + tools）。预期分数远低于老师：工具格式错、不会停、算错。这个差距就是 Phase 3 要补的。
- 🔗 nanolm 的待办（top-k 采样、KV cache、BPE）在这一课一次补齐。
- 🎤 什么是 chat template？ · 部署一个 7B 模型要多少显存？ · 讲讲 KV cache · 为什么用 GQA？

### Phase 3 · 训练层：让小模型变强

#### L09 · LoRA 原理：在你的 nanolm 上手写 → v0.9 ⏱ 4h
- **痛点**：1.7B 全参微调在 8GB 上根本放不下。算一笔账：权重 + 梯度 + Adam 状态 ≈ 16 bytes/参数，1.7B 就是约 27GB。
- **知识点**：全参微调的显存账 · 低秩假设（intrinsic dimension）· LoRA：`W' = W + (alpha/r)·B·A`，A 随机初始化、**B 零初始化**（为什么？）· rank / alpha / target modules 的作用 · merge 后推理零额外开销 · 与 adapter / prefix tuning / prompt tuning 的对比
- **动手**：在 nanolm 的 `gpt.py` 上写 `LoRALinear`，冻结底座只训 A/B，把莎士比亚模型微调到另一种文风（字符表对齐即可）· 代码放在 `nanolm/lora_trace.py` + `nanolm/lora.py`
- **实验**：full FT vs LoRA(r=1/4/16)，对比可训练参数量、显存、val loss；手写 merge 并验证 merge 前后输出一致
- 🔗 你熟悉 10M 模型的每一行，这里只多一个新概念，而且几分钟就能训完。
- 🎤 讲讲 LoRA，B 为什么初始化为 0？ · r 和 alpha 怎么选？ · LoRA vs 全参微调？

#### L10 · SFT 数据工程 → v0.10 ⏱ 6h
- **痛点**：想让小模型学会老师的行为，先要有高质量训练数据。数据质量决定了效果上限。
- **知识点**：SFT 数据格式（messages + tool calls，对齐 Qwen chat template）· 三个来源：
  ① **程序化合成**：FinQA 自带 gold program（如 `subtract(5829, 5735), divide(#0, 5735)`），可以直接转成 calculator 工具调用轨迹，零 API 成本
  ② **知识蒸馏**：让 L07 的老师 agent 跑训练集问题，做 **rejection sampling**（只保留最终答案正确的轨迹）；thinking 模式返回的 `reasoning_content` 还能拿来做**思维链蒸馏**（DeepSeek-R1-Distill 系列就是这么做的）
  ③ 少量人工精修
  · 质量控制：去重、长度分布、格式校验、多样性 · **数据泄漏检查**（训练集和 eval 集做 n-gram 去重，否则分数是假的）· 省钱：批量任务放在非高峰时段（半价）+ 上下文缓存（固定前缀放在前面，命中部分按缓存价计费）· **数据许可**：用谁的输出训练模型，先查清条款（DeepSeek 模型权重以 MIT 协议开源，API 条款在本课开始前再核实）· data card
- **动手**：`training/data/{build_sft, filters}.py` → `data/sft/{train,val}.jsonl`
- 🎤 怎么构建微调数据集？ · 什么是蒸馏？ · 怎么避免 contamination？

#### L11 · SFT with LoRA → v0.11 ⏱ 7h
- **痛点**：数据有了，开始训练。
- **知识点**：SFT 的 loss 仍然是 next-token CE，**和 nanolm 完全一样**，区别是**只在 assistant token 上算 loss**（labels 设为 -100 做 mask），要亲手打印一条样本的 input_ids / labels 对照看 · packing vs padding · PEFT `LoraConfig` + TRL `SFTTrainer` · 超参：lr（LoRA 常用 1e-4～2e-4，比全参大很多）、epoch、batch 与梯度累积、warmup + cosine · 训练动态：train/val 曲线、过拟合（你在 nanolm 3000 步后见过）、灾难性遗忘 · 先用 0.6B 快速迭代，再上 1.7B
- **动手**：`training/sft.py` · `configs/train/sft_*.yaml` · `walkthroughs/L11_loss_mask.py` · TensorBoard 日志
- **实验**：base vs SFT；消融 r ∈ {8, 32}、数据量 {25%, 50%, 100%}（data scaling 曲线）、只用程序化数据 vs 加蒸馏数据
- 🎤 为什么要 mask prompt tokens？ · LoRA 的学习率为什么大？ · SFT 过拟合怎么发现？

#### L12 · QLoRA 与显存工程 → v0.12 ⏱ 5h
- **痛点**：1.7B 的提升到顶了，想上 4B，可 4B 的 bf16 权重光存下来就要 8GB。
- **知识点**：量化基础（对称/非对称、per-channel/group）· NF4 + double quantization · QLoRA = 4-bit 冻结底座 + bf16 LoRA · gradient checkpointing（用计算换显存）· paged / 8-bit optimizer · **实测显存分解**：weights / grads / optimizer / activations · 序列长度对显存的影响
- **实验**：1.7B-LoRA vs 4B-QLoRA，比较准确率、训练时间、推理速度、峰值显存，整理成 trade-off 表
- 🎤 讲讲 QLoRA · 训练时显存都花在哪？ · 什么是 gradient checkpointing？

#### L13 · 偏好对齐：DPO → v0.13 ⏱ 7h
- **痛点**：SFT 后做错误分析，模型学会了格式，但不确定时还是会硬编数字、引用不存在的 chunk、回答啰嗦。SFT 只教了"该怎么做"，没教"什么是错的"。
- **知识点**：RLHF 全景（SFT → Reward Model → PPO）及其痛点 · **DPO 推导**：KL 约束下的 reward 最大化 → 最优策略闭式解 → 隐式 reward → Bradley-Terry → DPO loss（纯文本公式一步步推）· beta 的含义 · reference model · **偏好数据构建**：让 SFT 模型对同一题采样 N 次，用可验证信号（数值对错、引用是否存在）+ LLM judge 打分，组成 (chosen, rejected) 对 · on-policy vs off-policy · 常见失败：长度偏差、likelihood displacement
- **动手**：`training/dpo.py` · `walkthroughs/L13_dpo_loss.py`（玩具数字手算一遍 DPO loss）
- **实验**：SFT vs SFT+DPO，额外统计编造引用率和合理拒答率
- 🎤 推导 DPO · DPO vs PPO？ · 偏好数据怎么来？

#### L14 · （进阶）可验证奖励的强化学习：GRPO → v0.14 ⏱ 8h
- **痛点**：DPO 依赖离线偏好对，而数值题的对错**可以自动验证**，那就能直接用 RL 在线优化。
- **知识点**：policy gradient 直觉 · PPO 的 clip 与 value model · **GRPO**：组内相对优势，省掉 value model（DeepSeek-R1 的路线）· reward 设计（正确性 + 格式 + 工具使用）· reward hacking 实例 · KL penalty · 小模型做 RL 的难点
- **实验**：0.6B / 1.7B + LoRA 在 FinQA 训练集上跑，观察 reward 曲线和 eval 分数
- ⚠️ 8GB 上可行但很紧张。时间不够可以跳过，面试时能把原理讲清楚也有价值。
- 🎤 GRPO vs PPO · 什么是 RLVR？ · 举一个 reward hacking 的例子

### Phase 4 · 工程层：上线

#### L15 · 推理服务与优化 → v0.15 ⏱ 6h
- **痛点**：模型训好了，但 HF `generate` 慢、没有 API、不能并发。
- **知识点**：merge LoRA → 完整权重 · 推理引擎：vLLM（PagedAttention、continuous batching；Windows 需走 WSL2）vs llama.cpp / Ollama（GGUF 量化，Windows 友好）· OpenAI-compatible server · FastAPI + SSE 流式封装 agent · 性能指标：TTFT、tokens/s、吞吐、P50/P95 延迟 · 压测 · **模型路由/级联**：简单题走本地学生，难题或低置信度升级到老师，画出成本 vs 准确率曲线
- **动手**：`serve/api.py` · `llm/openai_compat.py` 把 base_url 指向本地服务（L01 写的类原样复用，这就是内部采用 OpenAI 格式的回报）· `bench/load_test.py`
- 🎤 vLLM 为什么吞吐高？ · TTFT 和吞吐怎么权衡？ · 设计一个省钱的 LLM serving 系统

#### L16 · 生产化与作品集 → v1.0 ⏱ 8h
- **知识点**：guardrails（输入端：prompt injection 检测、越界请求如"该买哪只股票"要拒答投资建议；输出端：数值校验、引用校验）· 可观测性（每个请求的 span：检索/工具/生成各自的耗时与 token）· **回归测试**：CI 里跑 smoke eval（20 题），分数下降就失败 · 缓存（语义缓存的坑）· Demo UI（Gradio）· Docker
- **交付**：README 叙事 + 最终 SCOREBOARD + 架构图 · 用真实数字写的简历 bullet · 2 分钟 demo 视频 · STAR 讲述稿
- 🎤 System design: "Design a RAG-based QA system for financial documents"（这个项目就是你的答案）

### 选修（任意时间插入）
| 编号 | 内容 | 建议插在 |
|---|---|---|
| S1 | Embedding 对比微调（InfoNCE） | L06 后 |
| S2 | MCP server：把工具暴露给 Claude Desktop / Claude Code | L07 后 |
| S3 | Multi-agent（研究员 + 审稿人） | L07 后 |
| S4 | 上下文缓存省钱实战：调整 prompt 结构，提高缓存命中率 | L10 时 |
| S5 | 继续预训练（domain-adaptive pretraining），在 nanolm 上做 | L11 后 |

---

## 3. 知识覆盖矩阵（学完能覆盖的范围）

| 知识领域 | 具体知识点 | 课次 |
|---|---|---|
| LLM 基础 | tokens、Chat Completions API、finish_reason、streaming、thinking 模式、采样、模型选型 | L01, L08 |
| Prompt | system prompt、few-shot、CoT/thinking、结构化输出、prompt 版本化 | L02 |
| 评测 | golden set、dev/test、数值 EM、LLM-as-judge 与校准、bootstrap CI、回归测试 | L03, L16 |
| Tool / Agent | function calling、schema 设计、tool loop、ReAct、终止与成本控制、MCP | L04, L07, S2 |
| RAG | chunking、embedding、ANN/FAISS、BM25、hybrid/RRF、rerank、query 改写、检索指标 | L05, L06 |
| 模型内部 | BPE、chat template、RoPE/GQA/RMSNorm、KV cache、显存算术 | L08（+ nanolm） |
| 参数高效微调 | LoRA 数学与实现、PEFT、QLoRA/NF4、gradient checkpointing | L09, L12 |
| SFT 与数据 | 数据合成、蒸馏、rejection sampling、去重、contamination、loss mask、packing | L10, L11 |
| 对齐与 RL | RLHF 全景、DPO 推导与实现、GRPO、reward hacking | L13, L14 |
| 推理与部署 | 量化 GGUF、vLLM/llama.cpp、TTFT/吞吐、FastAPI/SSE、模型级联 | L15 |
| 工程与安全 | 配置驱动、tracing、成本核算、prompt injection、guardrails、CI、Docker | 全程, L16 |

**刻意没覆盖的**：从头预训练（nanolm 已覆盖原理）、多模态、分布式训练（8GB 单卡做不了，面试时能讲 DDP/FSDP/ZeRO 的概念即可，可以另开专题）。

---

## 4. 预期成果（最终 SCOREBOARD 的样子）

| 版本 | 系统 | 模型 | FinQA acc | FinanceBench acc | Multi-hop acc | $/题 | P50 延迟 |
|---|---|---|---|---|---|---|---|
| v0.3 | closed-book | DeepSeek | 低 | 很低 | 低 | $ | 快 |
| v0.7 | Agent + hybrid RAG + tools | DeepSeek | **最高** | **最高** | **最高** | $$$ | 慢 |
| v0.8 | 同上 | Qwen 1.7B 裸跑 | 很低 | 很低 | 很低 | ≈0 | 中 |
| v0.11 | + SFT(LoRA) | Qwen 1.7B | ↑↑ | ↑ | ↑ | ≈0 | 中 |
| v0.13 | + DPO | Qwen 1.7B / 4B | ↑ | ↑ | ↑ | ≈0 | 中 |
| v0.15 | 路由：学生为主，难题给老师 | 混合 | 接近老师 | 接近老师 | 接近老师 | $ | 快 |

**你的简历故事**（数字做完再填）：
> Built a financial-filings QA agent (hybrid RAG + tool use + eval harness), then distilled it from DeepSeek-V4 into a local Qwen-1.7B via LoRA-SFT and DPO, recovering **X%** of teacher accuracy at **Y×** lower cost per query; hybrid retrieval + reranking raised Recall@5 from **A** to **B**.

---

## 5. 节奏建议

- 总量约 **100 小时**（17 课 × 平均 5～6h）。
- 现在是秋季学期加申请季。按每周 12～15h 算，大约 **8 周**做完。
- **优先冲到 L07**（约 40h、3～4 周）：到这里就有一个带真实数字的 RAG + Agent + Eval 项目，可以写进简历投递。Phase 3 在申请期间继续做，面试时能讲"正在做蒸馏，目前 SFT 已提升 X%"，这比"做完了一个 RAG demo"强很多。
- API：用 DeepSeek，全课程预计 $10–50（见 `docs/PRD.md` 第 5 节）。老师用 flash 还是 v4-pro，L01 实测后再定。

## 6. 每课的学习协议

```
课前   读讲义的"痛点 + 原理"部分（Claude 写，含详细原理和库用法）
课中   先跑 walkthroughs/ 演示脚本看清机制 → 读/写正式代码 → 跑实验
课后   更新 SCOREBOARD → 做自测题 → git commit + tag v0.x → 勾掉下面的进度
```

## 7. 进度

- [ ] L00 需求到架构（环境、git 已完成；剩：改写 PRD、写 ADR-9）
- [x] L01 LLM API 基础（2026-09-26，v0.1）
- [x] L02 Prompt + 结构化输出（2026-09-27，v0.2；作业为选做）
- [x] L03 Eval 驱动开发 ★（2026-09-27，v0.3；test 基线 37.0%）
- [x] L04 Tool Use（2026-09-27，v0.4；FinQA dev：计算器 +5.4 个百分点，两次平均）
- [x] L05 RAG 基础（2026-09-28，v0.5；dev：R@5 48%，答错率 44% → 13%，oracle 88%）
- [ ] L06 RAG 检索工程
- [ ] L07 Agent 🏁（可写进简历）
- [ ] L08 本地推理
- [ ] L09 手写 LoRA（nanolm）
- [ ] L10 SFT 数据工程
- [ ] L11 SFT with LoRA
- [ ] L12 QLoRA
- [ ] L13 DPO
- [ ] L14 GRPO（进阶）
- [ ] L15 推理服务
- [ ] L16 生产化 + 作品集 🎯
