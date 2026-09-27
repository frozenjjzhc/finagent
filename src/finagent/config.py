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


class SystemConfig(BaseModel):
    """一个完整的系统版本。L01 只有 llm；L02 加上 prompt；后面还会加 retrieval、tools。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt: PromptConfig | None = None  # None = 裸调模式（v0.1）：没有模板，也不要求结构化输出


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
