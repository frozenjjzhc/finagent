"""调用记录（tracing）：每次 LLM 调用往 runs/calls/<本地日期>.jsonl 追加一行。

为什么用 JSONL（一行一个 JSON 对象）：
  - 只追加、不改写：程序中途崩溃，也不会损坏之前的记录
  - 分析方便：逐行 json.loads，或者 pandas.read_json(path, lines=True)
这些记录现在用来看成本、延迟和排查问题；L07 之后它们也是训练数据的来源（ADR-5）。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from finagent.config import PROJECT_ROOT

CALLS_DIR = PROJECT_ROOT / "runs" / "calls"

# 多线程并发调用时（比如对比实验），加锁保证每一行完整写入、不会和别的线程穿插
_lock = threading.Lock()


def _local_today() -> str:
    # 文件按"你所在地的日期"命名，方便查找；先取带时区的 UTC 时间，再转成本地时区
    return datetime.now(timezone.utc).astimezone().date().isoformat()


def _path_for(day: str | None) -> Path:
    return CALLS_DIR / f"{day or _local_today()}.jsonl"


def log_call(record: dict) -> Path:
    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    path = _path_for(None)
    line = json.dumps(record, ensure_ascii=False)  # ensure_ascii=False：中文原样保存，方便直接看
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def read_calls(day: str | None = None) -> list[dict]:
    """读取某一天（YYYY-MM-DD，默认今天）的全部调用记录。"""
    path = _path_for(day)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
