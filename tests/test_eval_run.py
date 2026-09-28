"""裁判与评测运行器的端到端测试：用假客户端代替真实 API，检查记录目录里的每一样东西。"""

import json

import pytest

from finagent.config import EvalConfig, JudgeConfig, LLMConfig, load_config
from finagent.eval.datasets import EvalItem
from finagent.eval.judge import Judge, candidate_text
from finagent.eval.run import compare_runs, run_eval
from finagent.llm.base import LLMResponse, ToolCall, Usage
from finagent.prompts import load_prompt
from finagent.schemas import Answer

EVAL_CFG = EvalConfig(judge=JudgeConfig(llm=LLMConfig(model="fake-judge")), bootstrap_resamples=200)


class FakeClient:
    """系统用：永远回答"不知道"。裁判用：永远判"correct"。按模型名区分。"""

    def __init__(self, cfg, tag: str = "") -> None:
        self.is_judge = cfg.model == "fake-judge"

    def chat(self, messages, on_delta=None, json_mode=False) -> LLMResponse:
        if self.is_judge:
            content = json.dumps({"reasoning": "matches", "verdict": "correct"})
        else:
            content = json.dumps(
                {"reasoning": "r", "status": "unknown", "value": None, "unit": None,
                 "answer_text": "I do not know.", "sources": [], "confidence": "low"}
            )  # fmt: skip
        return LLMResponse(
            content=content,
            finish_reason="stop",
            model="fake",
            usage=Usage(),
            latency_s=0.01,
            cost_usd=0.001,
        )


def test_candidate_text_hides_reasoning() -> None:
    a = Answer(reasoning="LONG REASONING", status="answered", value=1577, unit="USD_millions",
               answer_text="Capex was $1,577 million.", confidence="high")  # fmt: skip
    text = candidate_text(a)
    assert "LONG REASONING" not in text and "$1,577 million" in text  # 裁判看不到推理，减少长度偏差


def test_judge_counts_calls_and_cost() -> None:
    judge = Judge(FakeClient(LLMConfig(model="fake-judge")), load_prompt("judge_v1"))
    it = EvalItem(
        id="x",
        source="financebench",
        split="dev",
        question="Q?",
        answer_type="text",
        gold_text="Yes",
    )
    a = Answer(reasoning="r", status="answered", answer_text="Yes.", confidence="high")
    assert judge(it, a).label == "correct"
    assert judge.calls == 1 and judge.cost_usd == 0.001


def test_run_eval_writes_a_complete_record(tmp_path) -> None:
    system = load_config("configs/systems/v0.2.yaml")
    run_dir = run_eval(
        system,
        "seed",
        "dev",
        EVAL_CFG,
        today="2026-09-27",
        out_root=tmp_path,
        make_client=FakeClient,
    )
    for name in (
        "results.jsonl",
        "summary.json",
        "report.md",
        "config.yaml",
        "prompt.yaml",
        "eval.yaml",
        "meta.json",
    ):
        assert (run_dir / name).exists(), name

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["n"] == 10
    assert summary["metrics"]["abstained"] == 10  # 假系统全都说不知道
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["today"] == "2026-09-27" and len(meta["dataset_sha256"]) == 64

    # 同一个评测跑两次再比较：结果完全相同，差异应当"不显著"
    run_b = run_eval(
        system,
        "seed",
        "dev",
        EVAL_CFG,
        today="2026-09-27",
        out_root=tmp_path / "b",
        make_client=FakeClient,
    )
    assert "不显著" in compare_runs(run_dir, run_b, resamples=200)


class ToolFakeClient:
    """带工具的假系统（L04）：第一轮请求计算器，拿到结果后把它当作答案。裁判不会被用到。"""

    def __init__(self, cfg, tag: str = "") -> None:
        pass

    def chat(self, messages, on_delta=None, json_mode=False, tools=None) -> LLMResponse:
        last = messages[-1]
        if tools and last["role"] != "tool":
            call = ToolCall("call_0", "calculate", json.dumps({"expression": "(110 - 100) / 100"}))
            return LLMResponse("", "tool_calls", "fake", Usage(), 0.01, 0.001, tool_calls=[call])
        content = json.dumps(
            {"reasoning": "r", "status": "answered", "value": float(last["content"]),
             "unit": "other", "answer_text": "t", "sources": [], "confidence": "high"}
        )  # fmt: skip
        return LLMResponse(content, "stop", "fake", Usage(), 0.01, 0.001)


def test_run_eval_with_tools_and_context(tmp_path) -> None:
    system = load_config("configs/systems/v0.4.yaml")
    run_dir = run_eval(
        system,
        "finqa",
        "dev",
        EVAL_CFG,
        limit=3,
        out_root=tmp_path,
        make_client=ToolFakeClient,
    )
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text("utf-8").splitlines()]
    assert all(r["llm_calls"] == 2 and r["attempts"] == 1 for r in rows)
    assert rows[0]["tool_calls"][0]["result"] == "0.1"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["tools"]["used_rate"] == 1.0 and summary["tools"]["mean_llm_calls"] == 2.0
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["tools"] == ["calculate"]
    assert "工具（calculate）" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_prompt_must_fit_the_evalset(tmp_path) -> None:
    # FinQA 带上下文，闭卷 prompt 没有 {{context}}：上下文会被悄悄丢掉，所以在花钱之前就报错
    system = load_config("configs/systems/v0.3.yaml")
    with pytest.raises(ValueError, match="context"):
        run_eval(system, "finqa", "dev", EVAL_CFG, limit=1, out_root=tmp_path)


# ---------------------------------------------------------------- 检索（L05）


class FakeContextBuilder:
    """假的检索：每道题都返回两段材料，第 1 段是证据页，第 2 段不是。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def retrieve(self, questions):
        from finagent.rag.retriever import Source

        return [
            [Source(1, "DOC", 7, "Doc 10-K", "evidence text", 0.9), Source(2, "OTHER", 1, "Other", "noise", 0.5)]
            for _ in questions
        ]  # fmt: skip


class CitingClient(FakeClient):
    """系统回答时引用 [1]，并确认上下文确实进了 prompt。"""

    def chat(self, messages, on_delta=None, json_mode=False, tools=None) -> LLMResponse:
        if self.is_judge:
            return super().chat(messages, on_delta, json_mode)
        assert "[1] Doc 10-K, page 8" in messages[-1]["content"]  # 检索结果以编号的形式进了 prompt
        content = json.dumps(
            {"reasoning": "r", "status": "unknown", "value": None, "unit": None,
             "answer_text": "I do not know.", "sources": ["[1]"], "confidence": "low"}
        )  # fmt: skip
        return LLMResponse(content, "stop", "fake", Usage(), 0.01, 0.001)


def test_run_eval_with_retrieval(tmp_path, monkeypatch) -> None:

    items = [
        EvalItem(id=f"q{i}", source="financebench", split="dev", question=f"Q{i}?",
                 answer_type="text", gold_text="x", evidence=[{"doc_name": "DOC", "page": 7}])
        for i in range(3)
    ]  # fmt: skip
    monkeypatch.setattr("finagent.eval.run.load_evalset", lambda name, split: items)
    system = load_config("configs/systems/v0.5.yaml")
    run_dir = run_eval(
        system, "financebench", "dev", EVAL_CFG, out_root=tmp_path,
        make_client=CitingClient, make_context=FakeContextBuilder,
    )  # fmt: skip
    rows = [json.loads(x) for x in (run_dir / "results.jsonl").read_text("utf-8").splitlines()]
    assert all(r["retrieval_hit"] and r["doc_hit"] for r in rows)
    assert all(r["cited"] == [1] and r["cited_evidence"] for r in rows)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["retrieval"]["hit_rate"] == 1.0 and summary["retrieval"]["mean_sources"] == 2.0
    assert "## 检索" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_retrieval_needs_a_context_prompt(tmp_path) -> None:
    system = load_config("configs/systems/v0.5.yaml")
    with pytest.raises(ValueError, match="context"):
        run_eval(system, "financebench", "dev", EVAL_CFG, prompt_name="answer_v3", limit=1,
                 out_root=tmp_path, make_context=FakeContextBuilder)  # fmt: skip
