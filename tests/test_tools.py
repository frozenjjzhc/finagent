"""工具的测试（L04）：计算器的 AST 白名单，以及注册表如何生成 schema、校验参数、回传错误。全程不联网。"""

import json

import pytest

from finagent.llm.base import ToolCall
from finagent.tools import AVAILABLE_TOOLS, build_registry, tool
from finagent.tools.calculator import evaluate, format_number
from finagent.tools.registry import ToolError

# ---------------------------------------------------------------- 计算器：能算的


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(5829 - 5735) / 5735 * 100", 1.6390584132519617),  # 增长率
        ("(113300 / 60100) ** (1 / 5) - 1", (113300 / 60100) ** 0.2 - 1),  # CAGR：分数次幂
        ("-3 + +2", -1.0),  # 正负号
        ("max(1, 2.5) - abs(-3) + min(4, 5)", 3.5),  # 白名单里的函数
        ("1e3 / 4", 250.0),  # 科学计数法
    ],
)
def test_evaluate(expression: str, expected: float) -> None:
    assert evaluate(expression) == pytest.approx(expected)


def test_format_number_hides_float_noise() -> None:
    assert format_number(0.1 + 0.2) == "0.3"  # 不是 0.30000000000000004
    assert format_number(1 / 3) == "0.333333333333"
    assert format_number(-0.0) == "0"


# ---------------------------------------------------------------- 计算器：必须拒绝的


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("__import__('os').system('dir')", "only these functions"),  # 经典攻击：eval 会照做
        ("open('secret.txt').read()", "only these functions"),
        ("(1).__class__", "Attribute"),  # 属性访问
        ("x + 1", "Name"),  # 变量名
        ("[1, 2][0]", "Subscript"),  # 下标
        ("'1' + '2'", "only numbers"),  # 字符串
        ("True + 1", "only numbers"),  # bool 是 int 的子类，也要挡住
        ("lambda: 1", "Lambda"),
        ("1 if 2 else 3", "IfExp"),
        ("5,829 - 5,735", "thousands separators"),  # 千分位逗号：在 Python 里是元组
        ("15% * 3", "'%' is not supported"),  # % 在 Python 里是取余，不是百分号
        ("20 % 3", "'%' is not supported"),
        ("import os", "not a valid arithmetic expression"),  # 语句，不是表达式
        ("1 / 0", "division by zero"),
        ("10 ** 10 ** 10", "too large"),  # 整数的话会卡死程序；float 立刻溢出
        ("(-8) ** 0.5", "not a real number"),  # 会算出复数
        ("1" * 600, "too long"),
        ("(" * 240 + "1" + ")" * 240, "too many nested parentheses"),  # 解析器自己的限制
    ],
)
def test_evaluate_rejects(expression: str, reason: str) -> None:
    with pytest.raises(ToolError, match=reason):
        evaluate(expression)


# ---------------------------------------------------------------- 注册表：schema


def test_schema_is_generated_from_signature_and_docstring() -> None:
    schema = AVAILABLE_TOOLS["calculate"].schema()
    fn = schema["function"]
    assert schema["type"] == "function" and fn["name"] == "calculate"
    assert fn["description"].startswith("Evaluate an arithmetic expression")
    params = fn["parameters"]
    assert params["required"] == ["expression"]
    assert params["additionalProperties"] is False  # 多传参数也算错
    assert params["properties"]["expression"]["type"] == "string"
    assert "5829" in params["properties"]["expression"]["description"]  # 来自 docstring 的 Args 段
    assert "title" not in json.dumps(schema)  # pydantic 的 title 已去掉，省 token


def test_tool_decorator_handles_defaults_and_multiline_arg_docs() -> None:
    @tool
    def lookup(company: str, year: int = 2024) -> dict:
        """Look up something.

        Args:
            company: The ticker,
                for example NVDA.
            year: Fiscal year.
        """
        return {"company": company, "year": year}

    params = lookup.schema()["function"]["parameters"]
    assert params["required"] == ["company"]  # 有默认值的参数不是必填
    assert params["properties"]["year"]["default"] == 2024
    assert params["properties"]["company"]["description"] == "The ticker, for example NVDA."
    assert lookup.run('{"company": "NVDA"}') == ('{"company": "NVDA", "year": 2024}', True)


def test_tool_without_docstring_is_rejected() -> None:
    def no_doc(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="docstring"):
        tool(no_doc)


# ---------------------------------------------------------------- 注册表：执行与错误回传


@pytest.mark.parametrize(
    ("arguments", "ok", "text"),
    [
        ('{"expression": "(5829 - 5735) / 5735 * 100"}', True, "1.63905841325"),
        ('{"expression": "5,829 - 5,735"}', False, "thousands separators"),
        ("not json", False, "not valid JSON"),  # 模型生成的参数字符串不是合法 JSON
        ('["1 + 1"]', False, "must be a JSON object"),
        ("{}", False, "expression: Field required"),  # 漏了参数
        ('{"expression": 5}', False, "valid string"),  # 类型不对
        ('{"expression": "1+1", "precision": 2}', False, "Extra inputs"),  # 多了参数
    ],
)
def test_execute_returns_errors_as_text(arguments: str, ok: bool, text: str) -> None:
    step = build_registry(["calculate"]).execute(ToolCall("call_0", "calculate", arguments))
    assert step.ok is ok
    assert text in step.result
    assert ok or step.result.startswith("Error:")  # 出错时不抛异常，而是回传一条错误信息


def test_unknown_tool_name() -> None:
    step = build_registry(["calculate"]).execute(ToolCall("call_0", "search_web", "{}"))
    assert not step.ok and "unknown tool" in step.result and "calculate" in step.result


def test_unexpected_exception_inside_tool_is_reported() -> None:
    @tool
    def broken(x: int) -> int:
        """Always fails.

        Args:
            x: Anything.
        """
        return x // 0

    result, ok = broken.run('{"x": 1}')
    assert not ok and result.startswith("Error: ZeroDivisionError")


def test_build_registry_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="未知的工具"):
        build_registry(["calculate", "calculater"])  # 配置里拼错了名字：报错，而不是悄悄少一个工具


# ---------------------------------------------------------------- 第二个工具：单位换算（补充讲义的示例）


def test_convert_units_schema_uses_enum() -> None:
    params = AVAILABLE_TOOLS["convert_units"].schema()["function"]["parameters"]
    assert params["required"] == ["value", "from_scale", "to_scale"]
    assert params["properties"]["value"]["type"] == "number"
    # Literal 类型 → enum：模型只能从这几个值里选
    scales = ["units", "thousands", "millions", "billions"]
    assert params["properties"]["from_scale"]["enum"] == scales


@pytest.mark.parametrize(
    ("arguments", "ok", "text"),
    [
        ('{"value": 5829, "from_scale": "thousands", "to_scale": "millions"}', True, "5.829"),
        ('{"value": 8.7, "from_scale": "billions", "to_scale": "millions"}', True, "8700"),
        ('{"value": 3, "from_scale": "millions", "to_scale": "millions"}', True, "3"),
        # pydantic 默认是宽松模式（L02）：字符串 "5829" 会被转成数字 5829
        ('{"value": "5829", "from_scale": "thousands", "to_scale": "millions"}', True, "5.829"),
        # 不在 enum 里的值：被拦下，错误信息里列出了允许的取值
        ('{"value": 1, "from_scale": "thousand", "to_scale": "millions"}', False, "'thousands'"),
        # Python 的 json.loads 接受 NaN，工具自己要检查
        ('{"value": NaN, "from_scale": "units", "to_scale": "units"}', False, "finite"),
    ],
)
def test_convert_units(arguments: str, ok: bool, text: str) -> None:
    step = build_registry(["convert_units"]).execute(ToolCall("c", "convert_units", arguments))
    assert step.ok is ok and text in step.result


def test_registry_with_two_tools() -> None:
    registry = build_registry(["calculate", "convert_units"])
    assert [s["function"]["name"] for s in registry.schemas()] == ["calculate", "convert_units"]
    step = registry.execute(ToolCall("c", "calculate", '{"expression": "5829 - 5735"}'))
    assert step.result == "94"  # 按名字分派到正确的函数
