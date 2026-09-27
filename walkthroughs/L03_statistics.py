"""L03 演示：为什么 10 道题的评测不可靠，以及 bootstrap 怎么量化这种不可靠。

运行（在项目根目录）：
    python walkthroughs/L03_statistics.py
完全离线，不调用任何 API。

    第 1 步  模拟：一个真实准确率 70% 的系统，用 10 / 50 / 150 / 1000 道题去测，测出来的分数会怎样波动
    第 2 步  bootstrap：只有一次评测结果时，怎么估计这种波动（置信区间）
    第 3 步  重新分析 L02：v2 答对 8 题、v3 答对 7 题，到底能不能说 v2 更好
    第 4 步  想分辨 5 个百分点的差距，大概需要多少道题
"""

from __future__ import annotations

import math
import random
import statistics

from finagent.eval.metrics import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff


def banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def step1_simulation() -> None:
    banner("第 1 步 · 模拟：真实准确率 70% 的系统，用不同数量的题去测")
    rng = random.Random(0)
    true_acc = 0.7
    print(f"{'题数':>6} | {'测出的分数（中间 95% 的范围）':<28} | 标准差")
    for n in (10, 50, 150, 1000):
        # 重复 2000 次"出一套 n 道题的卷子、测一次"
        scores = sorted(sum(rng.random() < true_acc for _ in range(n)) / n for _ in range(2000))
        lo, hi = scores[50], scores[1949]
        print(
            f"{n:>6} | {100 * lo:5.1f}% ~ {100 * hi:5.1f}%{'':<14} | {100 * statistics.stdev(scores):4.1f} 个百分点"
        )
    print("\n  同一个系统，只测 10 道题，分数可能在 40% 到 100% 之间乱跳；")
    print("  题数翻 4 倍，波动大约减半（波动 ∝ 1/√n）。")


def step2_bootstrap() -> None:
    banner("第 2 步 · bootstrap：只有一次结果时，怎么估计波动")
    results = [1.0] * 7 + [0.0] * 3  # 10 道题答对 7 道
    print(f"一次评测的结果：{[int(x) for x in results]}（答对 7/10）")
    rng = random.Random(1)
    print("有放回地重抽 10 道题，重新算准确率，示范 5 次：")
    for _ in range(5):
        sample = [results[rng.randrange(10)] for _ in range(10)]
        print(f"  {[int(x) for x in sample]} → {100 * sum(sample) / 10:.0f}%")
    mean, lo, hi = bootstrap_ci(results)
    print(
        f"重复 2000 次，取中间 95%：准确率 {100 * mean:.0f}%，95% 置信区间 {100 * lo:.0f}% ~ {100 * hi:.0f}%"
    )
    _, lo2, hi2 = bootstrap_ci([1.0] * 105 + [0.0] * 45)
    print(f"同样 70% 的准确率，如果是 150 道题：95% 置信区间 {100 * lo2:.1f}% ~ {100 * hi2:.1f}%")


def step3_revisit_l02() -> None:
    banner("第 3 步 · 重新分析 L02：v2 答对 8 题，v3 答对 7 题")
    #        q01 q02 q03 q04 q05 q06 q07 q08 q09 q10（1 = 答对，来自 SCOREBOARD 的 L02 记录）
    v2 = [1.0, 0, 1, 1, 0, 1, 1, 1, 1, 1]
    v3 = [1.0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
    for name, r in (("v2", v2), ("v3", v3)):
        m, lo, hi = bootstrap_ci(r)
        print(f"  {name}：{100 * m:.0f}%，95% 置信区间 {100 * lo:.0f}% ~ {100 * hi:.0f}%")
    diff, lo, hi = paired_bootstrap_diff(v2, v3)
    a_only = sum(1 for x, y in zip(v2, v3, strict=True) if x > y)
    b_only = sum(1 for x, y in zip(v2, v3, strict=True) if y > x)
    print(
        f"  配对差 v2 − v3：{100 * diff:+.0f} 个百分点，95% 置信区间 {100 * lo:+.0f} ~ {100 * hi:+.0f}"
    )
    print(
        f"  McNemar：v2 对 v3 错 {a_only} 道，v2 错 v3 对 {b_only} 道 → p = {mcnemar_exact(a_only, b_only):.2f}"
    )
    print("\n  两个区间大面积重叠；配对差的区间包含 0；p = 1.00。")
    print('  结论：L02 的数据不足以说明 v2 比 v3 好。这正是那时我们说"差 1 题不可靠"的定量依据。')


def step4_how_many_questions() -> None:
    banner("第 4 步 · 想分辨 5 个百分点的差距，大概需要多少道题")
    p, half_width = 0.7, 0.05
    n_independent = math.ceil((1.96 / half_width) ** 2 * p * (1 - p))
    print(
        f"准确率约 {100 * p:.0f}% 时，要让单个系统的 95% 置信区间缩到 ±{100 * half_width:.0f} 个百分点："
    )
    print(f"  n ≈ (1.96 / {half_width})² × {p} × {1 - p:.1f} ≈ {n_independent} 道题")
    print("比较两个系统时，用配对的方法（同一批题）会省很多：两个系统在大多数题上结果相同，")
    print('只有结果不同的那些题提供信息。所以配对比较时，关键是"结果不同的题"够不够多。')
    print(
        "\n  我们的 FinanceBench dev 只有 50 道：它能发现大的差距（十几个百分点），发现不了小的。"
    )
    print("  这就是为什么要同时报告置信区间：它诚实地告诉你，这个评测集能分辨到什么程度。")


if __name__ == "__main__":
    step1_simulation()
    step2_bootstrap()
    step3_revisit_l02()
    step4_how_many_questions()
