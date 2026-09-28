"""系统配置：一个"系统版本" = configs/systems/ 下的一个 YAML 文件（ADR-3）。

用 pydantic 校验，原则是"宁可报错，不要静默"：
  - 字段名拼错（比如 temprature）直接报错，而不是被悄悄忽略
  - DeepSeek 会静默忽略的参数组合（thinking 模式 + temperature），在这里变成显式错误
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

# src/finagent/config.py -> parents[2] 就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMConfig(BaseModel):
    """一个 LLM 后端的全部设置。"""

    model_config = ConfigDict(extra="forbid")  # 出现未定义的字段就报错

    backend: Literal["openai_compat"] = "openai_compat"  # L08 会加上 "local_hf"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"  # 只写环境变量的名字；key 本身永远不进 config，也不进 git
    thinking: bool = False
    reasoning_effort: Literal["low", "high", "max"] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    timeout_s: float = Field(default=300, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _reject_silently_ignored(self) -> LLMConfig:
        if self.thinking and self.temperature is not None:
            raise ValueError(
                "thinking 模式下 DeepSeek 会静默忽略 temperature。要调温度，请设 thinking: false"
            )
        if not self.thinking and self.reasoning_effort is not None:
            raise ValueError("reasoning_effort 只在 thinking: true 时有效")
        return self


class PromptConfig(BaseModel):
    """用哪个 prompt 模板（L02）。"""

    model_config = ConfigDict(extra="forbid")

    name: str  # 对应 prompts/<name>.yaml
    max_attempts: int = Field(default=2, ge=1)  # 输出校验失败时最多尝试几次（含第一次）


class ToolsConfig(BaseModel):
    """模型可以调用哪些工具（L04）。"""

    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=list)  # 工具名，见 tools/__init__.py 的 AVAILABLE_TOOLS
    max_rounds: int = Field(default=6, ge=1)  # 每次尝试最多几轮工具调用，防止模型一直调用下去
    # 工具循环里要不要开 JSON mode。默认关：L04 实测开着 JSON mode 时模型几乎不调用工具
    # （FinQA dev 8 道多步题：开 2/8，关 6/8）。关掉后 JSON 格式靠 prompt + pydantic 校验 + 重试把关
    json_mode: bool = False


class RetrievalConfig(BaseModel):
    """检索（L05）：回答前先从语料里找出相关的片段，作为上下文交给模型。"""

    model_config = ConfigDict(extra="forbid")

    index: str | None = None  # 索引名：configs/index/<index>.yaml 描述，建好后在 data/index/<index>/
    top_k: int = Field(default=5, ge=1)  # 取相似度最高的几个文本块
    # 交给模型的是什么：chunk = 检索到的文本块本身；page = 文本块所在的整页（small-to-big，见 L05 讲义）
    context: Literal["chunk", "page"] = "page"
    # oracle = 不检索，直接用标准答案标注的证据页（"检索永远正确"的上限实验）
    oracle: bool = False

    @model_validator(mode="after")
    def _index_unless_oracle(self) -> RetrievalConfig:
        if not self.oracle and not self.index:
            raise ValueError("不是 oracle 模式时，必须指定 index")
        return self


class SystemConfig(BaseModel):
    """一个完整的系统版本。L01 只有 llm；L02 加上 prompt；L04 加上 tools；L05 加上 retrieval。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt: PromptConfig | None = None  # None = 裸调模式（v0.1）：没有模板，也不要求结构化输出
    tools: ToolsConfig | None = None  # None = 不带工具（v0.3 及以前）
    retrieval: RetrievalConfig | None = None  # None = 不检索（闭卷，或者题目自带上下文）


class EmbedderConfig(BaseModel):
    """向量模型（L05）。"""

    model_config = ConfigDict(extra="forbid")

    model: str = "BAAI/bge-small-en-v1.5"
    revision: str | None = None  # 固定到 HuggingFace 上的某个 commit，和评测集固定版本是一个道理
    query_prefix: str = ""  # 问题前面要加的话（bge 系列要求加，文本块不加）
    device: str = "auto"  # auto = 有 GPU 用 GPU
    batch_size: int = Field(default=64, ge=1)


class ChunkConfig(BaseModel):
    """怎么切块（L05）。token 数按向量模型的分词器来数。"""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(default=256, ge=16)
    overlap_tokens: int = Field(default=32, ge=0)
    title: bool = True  # 每块开头加文档标题（如 "3M 10-K 2018"）

    @model_validator(mode="after")
    def _overlap_smaller(self) -> ChunkConfig:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens 必须小于 max_tokens")
        return self


class IndexConfig(BaseModel):
    """一个索引 = 语料 + 切块方式 + 向量模型 + 索引类型（L05）。configs/index/<name>.yaml。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    corpus_layout: Literal["rows", "text"] = "rows"  # 用哪种方式抽取的 PDF 文字（data/ingest.py）
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    backend: Literal["numpy", "faiss"] = "numpy"  # numpy = 手写的暴力检索；faiss = FAISS 库
    faiss_factory: str = "Flat"  # FAISS 的索引类型：Flat（精确）、HNSW32、IVF256,Flat（近似）


class JudgeConfig(BaseModel):
    """LLM 裁判的设置（L03）。"""

    model_config = ConfigDict(extra="forbid")

    llm: LLMConfig
    prompt: str = "judge_v1"  # prompts/<prompt>.yaml
    max_attempts: int = Field(default=2, ge=1)


class EvalConfig(BaseModel):
    """评测怎么进行（L03）。它描述的是"怎么考试"，所以和被评测的系统配置分开存放。"""

    model_config = ConfigDict(extra="forbid")

    judge: JudgeConfig
    relative_tolerance: float = Field(default=0.01, gt=0)  # 数值题默认允许 1% 的相对误差
    bootstrap_resamples: int = Field(default=2000, ge=100)
    bootstrap_seed: int = 0


def load_eval_config(path: str | Path = "configs/eval.yaml") -> EvalConfig:
    load_dotenv(PROJECT_ROOT / ".env")
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    return EvalConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))


def load_index_config(name_or_path: str | Path) -> IndexConfig:
    """按名字（configs/index/<name>.yaml）或路径读取索引配置。"""
    p = Path(name_or_path)
    if p.suffix != ".yaml":
        p = PROJECT_ROOT / "configs" / "index" / f"{name_or_path}.yaml"
    elif not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    return IndexConfig.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))


def load_config(path: str | Path) -> SystemConfig:
    """读取系统配置，顺带加载项目根目录的 .env（密钥只从环境变量读取）。"""
    load_dotenv(PROJECT_ROOT / ".env")
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p  # 允许在任何目录下使用"相对项目根目录"的路径
    # safe_load 只构造基本类型（dict/list/str/数字），不会执行 YAML 里夹带的任意 Python 对象
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return SystemConfig.model_validate(data)


def with_overrides(cfg: LLMConfig, **overrides: Any) -> LLMConfig:
    """临时覆盖部分字段（值为 None 的忽略），并**重新校验**。

    为什么不用 cfg.model_copy(update=...)：model_copy 不会重新运行校验器，
    "thinking + temperature" 这种非法组合会被悄悄放过。
    """
    updates = {k: v for k, v in overrides.items() if v is not None}
    return LLMConfig.model_validate({**cfg.model_dump(), **updates})
