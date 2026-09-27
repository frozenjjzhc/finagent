"""构建评测集（L03 起）：下载原始数据 → 换算标准答案 → 划分 dev/test → 写入 evalsets/。

用法（在项目根目录）：
    python scripts/build_evalsets.py

生成的文件（提交到 git，评测集是项目的"规格"）：
    evalsets/financebench.jsonl   FinanceBench 开源子集 150 题（dev 50 / test 100）       [L03]
    evalsets/seed.jsonl           L01 手工核实的 10 道种子题（全部是 dev）                [L03]
    evalsets/finqa.jsonl          FinQA 子集 270 题（dev 120 / test 150），每题带财报片段  [L04]
    evalsets/MANIFEST.json        数据来源、固定的版本、文件哈希、各划分的题数

可复现的三个保证：
    1. 下载地址固定到原始仓库的某个 commit，而不是会变的 main 分支
    2. 下载后核对 sha256，文件被改过就拒绝继续
    3. 划分用固定随机种子；同样的输入，永远得到同样的 dev/test
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from finagent.config import PROJECT_ROOT
from finagent.eval.datasets import (
    EVALSETS_DIR,
    EvalItem,
    dump_items,
    parse_finqa_gold,
    parse_gold_number,
)

FB_COMMIT = "6e138b2f29670bc416456cd16258f9a48b93b76e"  # 2024-09-05 "update gold label"
FB_FILES = {
    "financebench_open_source.jsonl": "a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877",
    "financebench_document_information.jsonl": "1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575",
}
FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"  # 2022-06-06，仓库最后一次更新
FINQA_FILES = {
    "dev.json": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
    "test.json": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
}
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_SEED = 42
DEV_PER_CATEGORY = {"metrics-generated": 17, "domain-relevant": 17, "novel-generated": 16}  # 共 50
# FinQA 按计算步数分层，每层抽同样多的题。原始数据里 1 步的题占六成，
# 按比例抽的话多步题太少，看不出"计算器对多步计算帮助更大"这类差别
FINQA_PER_STEPS = {"dev": 40, "test": 50}  # 每层几道：dev 3×40=120，test 3×50=150
# 题目自己说明了方向（"下降了多少"），答 42 和 -42 都算对（见 metrics.grade_number）
DIRECTION_WORDS = re.compile(
    r"\b(decreas|declin|reduc|drop|fell|fall|lower|less|loss)", re.IGNORECASE
)


def fetch(url: str, path: Path, sha256: str) -> Path:
    """下载（已有就跳过）并核对哈希。"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载 {url}")
        urllib.request.urlretrieve(url, path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != sha256:
        sys.exit(f"{path.name} 的哈希对不上（{actual}），文件可能被改过，停止构建")
    return path


def fetch_fb(name: str) -> Path:
    url = f"https://raw.githubusercontent.com/patronus-ai/financebench/{FB_COMMIT}/data/{name}"
    return fetch(url, RAW_DIR / name, FB_FILES[name])


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_financebench() -> list[EvalItem]:
    questions = read_jsonl(fetch_fb("financebench_open_source.jsonl"))
    docs = {d["doc_name"]: d for d in read_jsonl(fetch_fb("financebench_document_information.jsonl"))}

    items, unparsed = [], []
    for q in questions:
        doc = docs[q["doc_name"]]
        parsed = (
            parse_gold_number(q["answer"], q["question"])
            if q["question_type"] == "metrics-generated"
            else None
        )
        if q["question_type"] == "metrics-generated" and parsed is None:
            unparsed.append(q["financebench_id"])
        value, unit, decimals = parsed if parsed else (None, None, None)
        items.append(
            EvalItem(
                id="fb" + q["financebench_id"].rsplit("_", 1)[-1],
                source="financebench",
                split="test",  # 先全部标 test，下面再挑出 dev
                question=q["question"],
                answer_type="number" if parsed else "text",
                gold_text=q["answer"],
                gold_value=value,
                gold_unit=unit,
                gold_decimals=decimals,
                category=q["question_type"],
                company=q["company"],
                period=str(doc["doc_period"]),
                doc_type=doc["doc_type"],
                justification=q.get("justification") or "",
                evidence=[
                    {
                        "doc_name": e.get("evidence_doc_name") or q["doc_name"],
                        "page": e.get("evidence_page_num"),
                    }
                    for e in q.get("evidence") or []
                ],
            )
        )
    if unparsed:
        print(f"⚠️ 这些数值题的标准答案没能换算，按文字题处理（交给裁判）：{unparsed}")

    # 分层划分：每种题型各挑固定数量进 dev。先按 id 排序再打乱，保证结果和文件里的行序无关
    rng = random.Random(SPLIT_SEED)
    for cat, n_dev in DEV_PER_CATEGORY.items():
        group = sorted((it for it in items if it.category == cat), key=lambda it: it.id)
        for it in rng.sample(group, n_dev):
            it.split = "dev"
    return sorted(items, key=lambda it: it.id)


def build_seed() -> list[EvalItem]:
    """L01 手工核实过的 10 道题。它们在 L01/L02 里被用来挑 prompt，所以只能放进 dev。"""
    units = {"USD millions": "USD_millions", "%": "percent"}
    items = []
    for q in read_jsonl(EVALSETS_DIR / "seed_closed_book.jsonl"):
        trap = q["type"] == "trap"
        items.append(
            EvalItem(
                id=f"seed_{q['id']}",
                source="seed",
                split="dev",
                question=q["question"],
                answer_type="invalid_premise" if trap else "number",
                gold_text=str(q["answer"]),
                gold_value=None if trap else float(q["answer"]),
                gold_unit=None if trap else units[q["unit"]],
                gold_decimals=None if trap else (1 if q["unit"] == "%" else 0),
                # q06：10-K 里把 18.51% 四舍五入写作 19%，两个都算对
                tolerance_abs=0.5 if q["id"] == "q06" else None,
                category=q["type"],
                company=q["company"],
                period=q["fiscal_year"],
                doc_type="10k",
                justification=q["source"]["location"],
            )
        )
    return items


def finqa_context(doc: dict) -> str:
    """把 FinQA 的一页财报（表格前的文字 + 表格 + 表格后的文字）拼成一段纯文本上下文。

    用 FinQA 清洗过的 table（去掉了 HTML 标签，数字没有千分位逗号），而不是原始的 table_ori。
    """
    ticker, year, page = doc["filename"].removesuffix(".pdf").split("/")
    header = f"Excerpt from the annual report of {ticker} ({year}), {page.replace('_', ' ')}."

    def sentences(parts: list[str]) -> str:
        return "\n".join(p.strip() for p in parts if len(p.strip()) > 1)  # 去掉只有一个 "." 的行

    table = "\n".join(" | ".join(cell.strip() for cell in row) for row in doc["table"])
    return (
        f"{header}\n\n{sentences(doc['pre_text'])}\n\nTable:\n{table}\n\n"
        f"{sentences(doc['post_text'])}"
    ).strip()


def build_finqa() -> tuple[list[EvalItem], Counter]:
    """FinQA：每题附带一页财报（文字 + 表格），答案要从中取数并计算。

    我们的 dev 只从 FinQA 的 dev.json 抽，test 只从 test.json 抽；FinQA 的 train.json 留给 L10
    做训练数据。三者互不重叠，训练时就不会碰到评测题（避免数据泄漏）。
    """
    items, dropped = [], Counter()
    for split in ("dev", "test"):
        url = f"https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_COMMIT}/dataset/{split}.json"
        path = fetch(url, RAW_DIR / "finqa" / f"{split}.json", FINQA_FILES[f"{split}.json"])
        raw = json.loads(path.read_text(encoding="utf-8"))

        pool: dict[str, list[EvalItem]] = {"1-step": [], "2-step": [], "3+-step": []}
        seen = set()
        for doc in raw:
            qa = doc["qa"]
            parsed = parse_finqa_gold(qa["answer"], qa["exe_ans"])
            if parsed is None:
                numeric = isinstance(qa["exe_ans"], float) and re.search(r"\d", qa["answer"])
                dropped["答案文字与程序结果对不上" if numeric else "不是数值题（yes/no、空答案）"] += 1
                continue
            key = (doc["filename"], " ".join(qa["question"].lower().split()))
            if key in seen:  # 同一页财报上一字不差的重复问题
                dropped["重复题"] += 1
                continue
            seen.add(key)
            value, unit, decimals = parsed
            n_steps = len(qa["steps"])
            stratum = "1-step" if n_steps == 1 else "2-step" if n_steps == 2 else "3+-step"
            ticker, year, _ = doc["filename"].split("/")
            pool[stratum].append(
                EvalItem(
                    id="fq_" + re.sub(r"[^A-Za-z0-9]+", "_", doc["id"].replace(".pdf", "")),
                    source="finqa",
                    split=split,
                    question=qa["question"].strip(),
                    answer_type="number",
                    gold_text=qa["answer"].strip(),
                    gold_value=value,
                    gold_unit=unit,
                    gold_decimals=decimals,
                    category=stratum,
                    company=ticker,
                    period=year,
                    doc_type="10k",
                    justification=f"FinQA program: {qa['program']}",
                    evidence=[{"doc_name": doc["filename"], "gold_inds": sorted(qa["gold_inds"])}],
                    context=finqa_context(doc),
                    sign_agnostic=bool(DIRECTION_WORDS.search(qa["question"])),
                )
            )
        rng = random.Random(SPLIT_SEED)
        for group in pool.values():
            group.sort(key=lambda it: it.id)
            items += rng.sample(group, FINQA_PER_STEPS[split])
    return sorted(items, key=lambda it: it.id), dropped


def main() -> None:
    fb, seed = build_financebench(), build_seed()
    finqa, finqa_dropped = build_finqa()
    outputs = {"financebench.jsonl": fb, "seed.jsonl": seed, "finqa.jsonl": finqa}
    manifest = {
        "financebench": {
            "source": "https://github.com/patronus-ai/financebench",
            "commit": FB_COMMIT,
            "license": "CC BY-NC 4.0",
            "raw_sha256": FB_FILES,
        },
        "finqa": {
            "source": "https://github.com/czyssrs/FinQA",
            "commit": FINQA_COMMIT,
            "license": "MIT",
            "raw_sha256": FINQA_FILES,
            "dropped": dict(finqa_dropped),
        },
        "split_seed": SPLIT_SEED,
        "files": {},
    }
    for name, items in outputs.items():
        text = dump_items(items)
        # newline="\n"：Windows 上默认会把换行写成 \r\n，导致同一份评测集在不同系统上哈希不同
        (EVALSETS_DIR / name).write_text(text, encoding="utf-8", newline="\n")
        manifest["files"][name] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "splits": dict(Counter(it.split for it in items)),
            "answer_types": dict(Counter(it.answer_type for it in items)),
            "categories": dict(Counter(it.category for it in items)),
        }
    (EVALSETS_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for name, info in manifest["files"].items():
        print(
            f"{name}: {info['splits']}  题型 {info['categories']}  答案类型 {info['answer_types']}"
        )

    print("\n数值题的标准答案换算（请人工抽查几条）：")
    for it in [x for x in fb if x.answer_type == "number"][:8]:
        print(
            f"  {it.id}  原文 {it.gold_text!r:<12} → {it.gold_value:g} {it.gold_unit}（小数位 {it.gold_decimals}）"
        )

    print(f"\nFinQA 丢弃的题（两个原始划分合计）：{dict(finqa_dropped)}")
    print("FinQA 标准答案（请人工抽查几条）：")
    for it in finqa[:6]:
        print(
            f"  {it.id}  原文 {it.gold_text!r:<10} → {it.gold_value:g} {it.gold_unit}  "
            f"{it.justification}"
        )
    lengths = sorted(len(it.context or "") for it in finqa)
    print(f"FinQA 上下文长度（字符）：中位数 {lengths[len(lengths) // 2]}，最长 {lengths[-1]}")


if __name__ == "__main__":
    main()
