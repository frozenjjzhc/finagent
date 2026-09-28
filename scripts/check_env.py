"""L00 · 环境自检脚本

用法:
    python scripts/check_env.py              # 检查到 L01 为止需要的东西
    python scripts/check_env.py --ping       # 额外用 key 调一次 DeepSeek 的模型列表接口（不计费）
    python scripts/check_env.py --lesson 5   # 上 L05 之前再跑一次

它检查四类东西:
    1. Python 版本和当前环境
    2. 各阶段依赖: 当前课次需要但没装 -> [MISSING]；以后才需要 -> [LATER]（正常）
    3. 密钥与配置: .env 是否存在、是否被 gitignore、API key 是否设置（--ping 时再验证 key 真的能用）
    4. GPU 与磁盘

几个设计点（也是写工具脚本的好习惯）:
    - 用 importlib.metadata 查"已安装的版本号"，而不是真的 import 包。
      import torch 要好几秒，有的包 import 时还有副作用；查元数据只需几毫秒。
    - API key 只报告"有没有"，绝不打印内容（哪怕部分内容）。日志和截图都可能外泄。
    - 有 [MISSING] 时退出码为 1。以后放进 CI（L16）可以直接当检查步骤用。
    - 只输出 ASCII 标记和中文，不用 emoji：Windows 控制台重定向到文件时用 GBK 编码，emoji 会报错。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OK = "[OK]     "
MISSING = "[MISSING]"
LATER = "[LATER]  "
WARN = "[WARN]   "

# (从第几课开始需要, 分组说明, [pip 发行包名])
REQUIREMENTS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "核心: LLM API / 配置 / CLI",
        ["openai", "pydantic", "python-dotenv", "pyyaml", "typer", "rich"],
    ),
    (1, "开发工具", ["pytest", "ruff"]),
    (
        5,
        "RAG: 数据与检索（embedding 模型需要 torch）",
        ["torch", "numpy", "pymupdf", "sentence-transformers", "faiss-cpu"],
    ),
    (6, "RAG 进阶：关键词检索", ["rank-bm25"]),
    (
        8,
        "本地模型与训练",
        ["transformers", "accelerate", "peft", "trl", "datasets", "bitsandbytes"],
    ),
    (15, "推理服务", ["fastapi", "uvicorn"]),
]


def pkg_version(name: str) -> str | None:
    """返回已安装包的版本号；没装返回 None。"""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def env_name() -> str:
    """推断当前运行的是哪个 Python 环境。"""
    prefix = Path(sys.prefix)
    if sys.prefix != sys.base_prefix:  # venv / virtualenv
        return f"venv:{prefix.name}"
    if prefix.parent.name == "envs":  # conda 命名环境: .../anaconda3/envs/<name>
        return prefix.name
    return "base"


def check_python() -> int:
    v = sys.version_info
    print(f"[Python] {v.major}.{v.minor}.{v.micro}  ({sys.executable})")
    problems = 0
    if v < (3, 10):
        print(f"  {MISSING} 需要 Python >= 3.10（推荐 3.12）")
        problems += 1
    else:
        print(f"  {OK} 版本满足 >= 3.10")

    name = env_name()
    if name == "base":
        print(f"  {WARN} 当前在 base 环境。建议: conda activate finagent（见讲义 7.1）")
    else:
        print(f"  {OK} 当前环境: {name}")
    return problems


def check_packages(lesson: int) -> int:
    problems = 0
    for start, group, names in REQUIREMENTS:
        needed_now = start <= lesson
        header = "当前需要" if needed_now else f"L{start:02d} 起需要"
        print(f"\n[{header}] {group}")
        for name in names:
            ver = pkg_version(name)
            if ver:
                print(f"  {OK} {name:<24}{ver}")
            elif needed_now:
                print(f"  {MISSING} {name:<24}未安装")
                problems += 1
            else:
                print(f"  {LATER} {name:<24}未安装（L{start:02d} 再装）")
    return problems


def load_dotenv_file() -> str:
    """尝试加载项目根目录的 .env，返回状态说明。"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return "absent"
    try:
        from dotenv import load_dotenv
    except ImportError:
        return "no-dotenv"
    load_dotenv(env_file)
    return "loaded"


def check_secrets(lesson: int, dotenv_status: str) -> int:
    print("\n[密钥与配置]")
    problems = 0

    if dotenv_status == "absent":
        print(f"  {MISSING} .env 不存在: Copy-Item .env.example .env，再填写")
        problems += 1
    elif dotenv_status == "no-dotenv":
        print(f"  {WARN} .env 存在，但 python-dotenv 未安装，读不到里面的变量")
    else:
        print(f"  {OK} .env 已加载")

    gitignore = ROOT / ".gitignore"
    ignored = gitignore.exists() and ".env" in gitignore.read_text(encoding="utf-8").split()
    if ignored:
        print(f"  {OK} .env 已被 .gitignore 忽略")
    else:
        print(f"  {MISSING} .gitignore 里没有 .env，密钥有被提交的风险！")
        problems += 1

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        print(f"  {MISSING} DEEPSEEK_API_KEY 未设置（填到 .env 里）")
        problems += 1
    elif not key.startswith("sk-"):
        print(f"  {WARN} DEEPSEEK_API_KEY 已设置，但不是以 sk- 开头，检查是否复制完整")
    else:
        print(f"  {OK} DEEPSEEK_API_KEY 已设置（不显示内容）")

    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    ua_ok = "@" in ua and "example.com" not in ua
    if ua_ok:
        print(f"  {OK} SEC_USER_AGENT 已设置")
    elif lesson >= 5:
        print(f"  {MISSING} SEC_USER_AGENT 需要填你的真实邮箱（EDGAR 要求）")
        problems += 1
    else:
        print(f"  {LATER} SEC_USER_AGENT 还是占位符（L05 再填）")
    return problems


def ping_api(base_url: str) -> int:
    """调一次 GET /models：不消耗 token、不计费，但能同时验证 key、网络和 base_url。"""
    print(f"\n[API 连通性] {base_url}")
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        print(f"  {MISSING} 没有 DEEPSEEK_API_KEY，跳过")
        return 1
    try:
        import openai
    except ImportError:
        print(f"  {MISSING} openai 未安装，跳过")
        return 1

    # max_retries=0：自检要的是快速暴露问题，而不是默默重试
    client = openai.OpenAI(api_key=key, base_url=base_url, timeout=15, max_retries=0)
    try:
        models = sorted(m.id for m in client.models.list())
    # 从具体到宽泛地捕获：AuthenticationError 是 APIStatusError 的子类，必须写在前面
    except openai.AuthenticationError:
        print(f"  {MISSING} key 无效（401）：检查 .env 里的值是否复制完整、是否已在平台上删除")
        return 1
    except openai.APIStatusError as e:
        print(f"  {MISSING} 服务器返回 HTTP {e.status_code}")
        return 1
    except openai.APIConnectionError as e:
        print(f"  {MISSING} 连不上服务器（{type(e).__name__}），检查网络或代理")
        return 1
    print(f"  {OK} key 有效。当前可用模型: {', '.join(models)}")
    return 0


def check_gpu(lesson: int) -> int:
    print("\n[GPU]")
    need = lesson >= 8
    torch_ver = pkg_version("torch")
    if not torch_ver:
        tag = MISSING if need else LATER
        print(f"  {tag} torch 未安装，跳过 GPU 检查（L05 前按讲义 7.3 安装 CUDA 版）")
        return 1 if need else 0

    if "+cu" not in torch_ver:
        print(
            f"  {WARN} torch {torch_ver} 看起来是 CPU 版。重装: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )

    import torch  # 只有真正需要时才 import（慢）

    if not torch.cuda.is_available():
        tag = MISSING if need else WARN
        print(f"  {tag} torch {torch_ver} 已安装，但 CUDA 不可用")
        return 1 if need else 0

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    bf16 = "支持" if torch.cuda.is_bf16_supported() else "不支持"
    print(f"  {OK} {props.name} | 显存 {vram_gb:.1f} GB | CUDA {torch.version.cuda} | bf16 {bf16}")
    return 0


def check_disk() -> None:
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    probe = hf_home if hf_home.exists() else Path(hf_home.anchor)
    free_gb = shutil.disk_usage(probe).free / 1024**3
    tag = OK if free_gb >= 30 else WARN
    print("\n[磁盘]")
    print(f"  {tag} HF 模型缓存: {hf_home}（所在盘剩余 {free_gb:.0f} GB；全课程约需 20-30 GB）")


def main() -> int:
    parser = argparse.ArgumentParser(description="FinAgent 环境自检")
    parser.add_argument(
        "--lesson", type=int, default=1, help="你接下来要上的课次，决定哪些依赖算'必须'（默认 1）"
    )
    parser.add_argument(
        "--ping", action="store_true", help="用 key 调一次模型列表接口，验证 key 真的可用（不计费）"
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="OpenAI 兼容服务的地址（默认 DeepSeek）",
    )
    args = parser.parse_args()

    # 输出被重定向到 GBK 文件时，遇到编码不了的字符用 ? 代替，而不是崩溃
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    print(f"FinAgent 环境自检 | 目标课次 L{args.lesson:02d} | 项目目录 {ROOT}")
    print("=" * 72)

    dotenv_status = load_dotenv_file()
    problems = (
        check_python()
        + check_packages(args.lesson)
        + check_secrets(args.lesson, dotenv_status)
        + check_gpu(args.lesson)
    )
    if args.ping:
        problems += ping_api(args.base_url)
    check_disk()

    print("\n" + "=" * 72)
    if problems:
        print(f"有 {problems} 项 [MISSING] 需要处理。")
    else:
        print(f"全部通过，可以开始 L{args.lesson:02d}。")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
