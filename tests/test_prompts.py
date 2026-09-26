"""Prompt 模板的测试：变量替换、三个版本的文件都能正常加载，以及 few-shot 示例本身的质量。"""

import json

import pytest

from finagent.config import PROJECT_ROOT
from finagent.pipeline.answer import parse_answer
from finagent.prompts import list_prompts, load_prompt, render

# ---------------------------------------------------------------- render


def test_render_replaces_placeholders() -> None:
    assert render("Today is {{today}}.", today="2026-09-26") == "Today is 2026-09-26."


def test_render_missing_variable_raises() -> None:
    with pytest.raises(KeyError):
        render("Question: {{question}}")


def test_render_leaves_json_braces_alone() -> None:
    template = 'Reply like {"value": 1}. Today: {{today}}'
    assert render(template, today="X") == 'Reply like {"value": 1}. Today: X'
    # 对比：同样的模板交给 str.format 会直接报错，这就是我们不用它的原因
    with pytest.raises((KeyError, ValueError)):
        template.format(today="X")


# ---------------------------------------------------------------- 三个版本的 prompt 文件

PROMPT_NAMES = [p.stem for p in list_prompts()]


def test_prompt_files_exist() -> None:
    assert {"answer_v1", "answer_v2", "answer_v3"} <= set(PROMPT_NAMES)


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_builds_valid_messages(name: str) -> None:
    messages = load_prompt(name).build_messages("What is X?", today="2026-01-01")
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "What is X?" in messages[-1]["content"]
    assert "2026-01-01" in messages[-1]["content"]
    assert all("{{" not in str(m["content"]) for m in messages)  # 没有漏填的变量
    # DeepSeek 的 JSON mode 要求 prompt 里出现 "json" 这个词
    assert "json" in messages[0]["content"].lower()


# ---------------------------------------------------------------- few-shot 示例的质量


def all_examples() -> list:
    return [
        pytest.param(ex, id=f"{name}-example{i + 1}")
        for name in PROMPT_NAMES
        for i, ex in enumerate(load_prompt(name).examples)
    ]


@pytest.mark.parametrize("example", all_examples())
def test_example_answers_match_the_schema(example) -> None:
    # 示例里的回答本身必须合格：示例是模型模仿的对象，示例错了，模型就会学着错
    parse_answer(example.assistant)


def test_examples_do_not_leak_eval_questions() -> None:
    # 示例不能和评测题重合，否则就是拿答案教模型，分数是假的（数据泄漏）
    seed = PROJECT_ROOT / "evalsets" / "seed_closed_book.jsonl"
    eval_questions = [
        json.loads(line)["question"]
        for line in seed.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for name in PROMPT_NAMES:
        for ex in load_prompt(name).examples:
            assert not any(q in ex.user for q in eval_questions), f"{name} 的示例包含评测题"
