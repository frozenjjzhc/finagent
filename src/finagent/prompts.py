"""Prompt 模板：从 prompts/*.yaml 读取、填入变量、组装成 messages（ADR-4：prompt 是文件，不是代码里的字符串）。

模板变量写成 {{变量名}}，由 render() 替换。为什么不用 Python 自带的两种方式：
  - str.format 用 {变量名}：prompt 里的 JSON 示例本身全是花括号，format 会把它们当成变量，直接报错
  - string.Template 用 $变量名：财报 prompt 里到处是 "$4.2 billion" 这样的金额，同样会冲突
所以自己写一个只认 {{...}} 的替换，简单，而且不会和 JSON、美元符号打架。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from finagent.config import PROJECT_ROOT
from finagent.llm.base import Message

PROMPTS_DIR = PROJECT_ROOT / "prompts"
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render(template: str, **variables: object) -> str:
    """把模板里的 {{name}} 换成 variables["name"]。缺了哪个变量就报错，绝不留下没填的空位。"""

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"模板变量 {{{{{key}}}}} 没有提供值")
        return str(variables[key])

    return _PLACEHOLDER.sub(substitute, template)


class FewShotExample(BaseModel):
    """一组示例：用户这样问 → 助手这样答。它会作为一轮真实的对话放进 messages。"""

    model_config = ConfigDict(extra="forbid")
    user: str
    assistant: str


class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    system: str
    examples: list[FewShotExample] = Field(default_factory=list)
    user_template: str = "{{question}}"

    @property
    def variables(self) -> set[str]:
        """模板里用到的变量名，例如 {"today", "question"}。

        render() 只会在"缺了变量"时报错，"多给了变量"时会静默忽略。所以调用方要自己检查：
        比如评测集带了上下文，prompt 却没有 {{context}}，上下文就会被悄悄丢掉（L04）。
        """
        texts = [self.system, self.user_template]
        return {m.group(1) for text in texts for m in _PLACEHOLDER.finditer(text)}

    def build_messages(self, question: str, **variables: object) -> list[Message]:
        """组装顺序：system → 示例（user/assistant 成对）→ 真正的问题。

        system 和示例每次都一样，放在前面，能命中前缀缓存；
        每次都变的内容（日期、问题）放在最后一条 user 消息里。
        """
        messages: list[Message] = [{"role": "system", "content": render(self.system, **variables)}]
        for ex in self.examples:
            messages.append({"role": "user", "content": ex.user})
            messages.append({"role": "assistant", "content": ex.assistant})
        user_text = render(self.user_template, question=question, **variables)
        messages.append({"role": "user", "content": user_text})
        return messages


def load_prompt(name: str) -> PromptTemplate:
    """按名字读取 prompts/<name>.yaml。"""
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"找不到 prompt 文件 {path.name}，现有的：{available}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptTemplate.model_validate(data)


def list_prompts() -> list[Path]:
    return sorted(PROMPTS_DIR.glob("*.yaml"))
