# FinAgent

读懂美股 10-K 财报、**算对数**、**给出处**的 AI 财报分析助手。

这是一个学习型项目：从 Prompt / Eval / Tools / RAG / Agent 起步，做出一个基于 DeepSeek API 的"老师系统"；再用 LoRA-SFT + DPO 把它的能力蒸馏进一个跑在 8GB 显卡上的本地 Qwen 小模型（"学生"）。

| 文档 | 内容 |
|---|---|
| [ROADMAP.md](ROADMAP.md) | 17 课演进路线与进度 |
| [docs/PRD.md](docs/PRD.md) | 需求：做什么、做到什么程度、不做什么 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构、设计决策（ADR）、目录规划 |
| [SCOREBOARD.md](SCOREBOARD.md) | 每个版本的评测结果 |
| [lessons/](lessons/) | 每课讲义 |

## 当前状态

**v0.2**：闭卷；prompt 模板 + JSON 结构化输出 + schema 校验（L02）。`finagent ask` 默认输出结构化答案，`--config configs/systems/v0.1.yaml` 可回到裸调模式

## 快速开始（Windows / PowerShell）

```powershell
conda create -n finagent python=3.12 -y
conda activate finagent
pip install -e ".[dev]"
Copy-Item .env.example .env   # 然后填入 DEEPSEEK_API_KEY
python scripts/check_env.py
```
