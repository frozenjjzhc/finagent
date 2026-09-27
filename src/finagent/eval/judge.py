"""LLM 裁判：给文字类答案判分（参考答案制，二分类）。

裁判本身也是一个大模型，会有偏差。常见的几种：
  - 位置偏差：两个答案放在一起比较时，偏向排在前面（或后面）的那个。我们只判一个答案，没有这个问题
  - 长度偏差：偏向写得长的答案。对策：只给裁判看最终回答，不给推理过程
  - 自我偏好：偏向和自己风格相近（同一个模型写的）的答案。对策：裁判用和被测系统不同的模型
所以裁判的判定要和人工标注对一对（校准），见 scripts/judge_agreement.py。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finagent.eval.datasets import EvalItem
from finagent.eval.metrics import Grade
from finagent.llm.base import LLMClient
from finagent.llm.structured import call_structured
from finagent.prompts import PromptTemplate
from finagent.schemas import Answer, format_value


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(max_length=1000)
    verdict: Literal["correct", "incorrect"]


def candidate_text(answer: Answer) -> str:
    """给裁判看的候选答案：只有最终回答和数值，故意不包含推理过程（减少长度偏差）。"""
    text = answer.answer_text
    if answer.value is not None:
        text += f" (value: {format_value(answer)})"
    return text


@dataclass
class Judge:
    """一个带统计的裁判：记录调用次数和花费，供评测报告使用。"""

    client: LLMClient
    prompt: PromptTemplate
    max_attempts: int = 2
    calls: int = 0
    cost_usd: float = 0.0
    failures: list[str] = field(default_factory=list)
    # 评测时多个线程会同时调用裁判；"+=" 不是原子操作，不加锁可能少算
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __call__(self, item: EvalItem, answer: Answer) -> Grade:
        messages = self.prompt.build_messages(
            item.question, reference=item.gold_text, candidate=candidate_text(answer)
        )
        result = call_structured(self.client, messages, Verdict, self.max_attempts)
        with self._lock:
            self.calls += result.attempts
            self.cost_usd += result.cost_usd or 0.0
            if result.value is None:
                self.failures.append(item.id)
        if result.value is None:
            return Grade(
                "error", "judge", f"裁判输出不合格：{'；'.join(result.errors)}", "judge_failed"
            )
        v = result.value
        label = "correct" if v.verdict == "correct" else "incorrect"
        return Grade(label, "judge", v.reasoning, "" if label == "correct" else "judge_incorrect")
