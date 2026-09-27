"""裁判校准（L03）：把某次评测里裁判的判定导出成表格，你人工标注后，计算你和裁判的一致程度。

用法（在项目根目录）：
    python scripts/judge_agreement.py export runs/eval/<某次评测>   # 生成 evalsets/judge_calibration/<评测名>.csv
    （用 Excel 或任意编辑器打开，在 human_label 列填 correct 或 incorrect）
    python scripts/judge_agreement.py score runs/eval/<某次评测>    # 计算一致率和 Cohen's kappa

为什么要做：裁判也是大模型，会有偏差。只有和人工标注对过，才知道裁判给的分数能信几分。
标注文件放在 evalsets/ 下并提交到 git：人工标注花了时间，是有价值的数据；runs/ 不进 git。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from finagent.eval.datasets import EVALSETS_DIR

LABELS = ("correct", "incorrect")
CALIBRATION_DIR = EVALSETS_DIR / "judge_calibration"


def labels_path(run_dir: Path) -> Path:
    return CALIBRATION_DIR / f"{run_dir.resolve().name}.csv"


def export(run_dir: Path) -> None:
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    judged = [r for r in rows if r["grader"] == "judge" and r["grade"] in LABELS]
    out = labels_path(run_dir)
    if out.exists():
        raise SystemExit(f"{out} 已存在，里面可能有你的标注，不覆盖")
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    # utf-8-sig：带 BOM，Excel 打开时才能正确识别 UTF-8
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "question",
                "reference",
                "candidate",
                "judge_verdict",
                "judge_reasoning",
                "human_label",
            ]
        )
        for r in judged:
            a = r["answer"] or {}
            w.writerow(
                [
                    r["id"],
                    r["question"],
                    r["gold_text"],
                    a.get("answer_text", ""),
                    r["grade"],
                    r["detail"],
                    "",
                ]
            )
    print(f"导出了 {len(judged)} 条裁判判定 → {out}")
    print(
        "在 human_label 列填 correct 或 incorrect（只看参考答案判断，不要先看裁判的结论），然后运行 score。"
    )


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa：扣除"碰巧一致"之后的一致程度。1 = 完全一致，0 = 和瞎猜一样。

    只看一致率会被误导：如果 90% 的答案都是错的，裁判无脑判"错"也有 90% 的一致率。
    kappa = (实际一致率 - 碰巧一致率) / (1 - 碰巧一致率)
    """
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    judge_counts, human_counts = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(judge_counts[c] * human_counts[c] for c in LABELS) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def score(run_dir: Path) -> None:
    with labels_path(run_dir).open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human_label"].strip()]
    bad = [r["id"] for r in rows if r["human_label"].strip() not in LABELS]
    if bad:
        raise SystemExit(f"human_label 只能填 correct 或 incorrect，这些行不对：{bad}")
    pairs = [(r["judge_verdict"], r["human_label"].strip()) for r in rows]
    if not pairs:
        raise SystemExit("还没有任何人工标注")
    agree = sum(a == b for a, b in pairs)
    print(
        f"已标注 {len(pairs)} 条：一致 {agree} 条（{100 * agree / len(pairs):.0f}%），Cohen's kappa = {cohen_kappa(pairs):.2f}"
    )
    print("\n            人工=correct  人工=incorrect")
    for j in LABELS:
        print(
            f"裁判={j:<9} {sum(1 for a, b in pairs if a == j and b == 'correct'):>8}  {sum(1 for a, b in pairs if a == j and b == 'incorrect'):>12}"
        )
    print("\n不一致的条目：")
    for r in rows:
        if r["judge_verdict"] != r["human_label"].strip():
            print(
                f"  {r['id']}：裁判 {r['judge_verdict']}，你 {r['human_label'].strip()}。裁判的理由：{r['judge_reasoning'][:150]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="裁判校准")
    parser.add_argument("action", choices=["export", "score"])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    export(args.run_dir) if args.action == "export" else score(args.run_dir)


if __name__ == "__main__":
    main()
