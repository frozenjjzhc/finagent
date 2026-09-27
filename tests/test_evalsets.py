"""评测集文件本身的测试：格式合法、划分正确、没有重复、和 MANIFEST 一致。"""

import hashlib
import json

import pytest

from finagent.eval.datasets import EVALSETS_DIR, load_evalset

SETS = ["financebench", "seed", "finqa"]


@pytest.mark.parametrize("name", SETS)
def test_items_are_valid_and_ids_unique(name) -> None:
    items = load_evalset(name)  # 每一行都会被 EvalItem 校验
    ids = [it.id for it in items]
    assert len(ids) == len(set(ids))


def test_financebench_split() -> None:
    dev, test = load_evalset("financebench", "dev"), load_evalset("financebench", "test")
    assert (len(dev), len(test)) == (50, 100)
    assert not {it.id for it in dev} & {it.id for it in test}  # dev 和 test 不能有重叠
    # 分层划分：每种题型在 dev 和 test 里都要有
    for cat in ("metrics-generated", "domain-relevant", "novel-generated"):
        assert any(it.category == cat for it in dev) and any(it.category == cat for it in test)


def test_all_metric_questions_have_numeric_gold() -> None:
    metrics = [it for it in load_evalset("financebench") if it.category == "metrics-generated"]
    assert len(metrics) == 50 and all(it.answer_type == "number" for it in metrics)


def test_seed_is_dev_only() -> None:
    # 种子题在 L01/L02 里被用来挑 prompt，已经"用过了"，不能出现在 test 里
    assert all(it.split == "dev" for it in load_evalset("seed"))


def test_files_match_manifest() -> None:
    manifest = json.loads((EVALSETS_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        # 按字节算哈希：Windows 上 git 可能把换行转成 CRLF，这里先统一成 LF
        data = (EVALSETS_DIR / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == info["sha256"], (
            f"{name} 和 MANIFEST 不一致：被手工改过？"
        )


def test_finqa_split_and_context() -> None:
    dev, test = load_evalset("finqa", "dev"), load_evalset("finqa", "test")
    assert (len(dev), len(test)) == (120, 150)
    for split, n in ((dev, 40), (test, 50)):  # 按计算步数分层，每层一样多
        for stratum in ("1-step", "2-step", "3+-step"):
            assert sum(it.category == stratum for it in split) == n
    for it in dev + test:
        assert it.context and "Table:" in it.context  # 开卷题：每题都带财报片段
        assert it.answer_type == "number" and it.gold_unit in ("percent", "other")


def test_finqa_dev_and_test_come_from_different_sources() -> None:
    # dev 只来自 FinQA 的 dev.json，test 只来自 test.json：同一页财报不会同时出现在两边
    dev_docs = {it.evidence[0]["doc_name"] for it in load_evalset("finqa", "dev")}
    test_docs = {it.evidence[0]["doc_name"] for it in load_evalset("finqa", "test")}
    assert not dev_docs & test_docs


def test_old_evalsets_do_not_gain_new_fields() -> None:
    # L04 给 EvalItem 加了 context、sign_agnostic；取默认值时不写进文件，
    # 这样 L03 的评测集文件逐字节不变，旧记录的哈希仍然对得上
    for name in ("financebench", "seed"):
        first = json.loads((EVALSETS_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert "context" not in first and "sign_agnostic" not in first
