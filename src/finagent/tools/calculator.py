"""计算器工具（L04）：安全地计算模型写出的算术表达式。

为什么不直接用 eval()：
    eval 会执行**任意 Python 代码**。表达式是模型写的，而模型的输入里可能混进了不可信的内容
    （L05 起是检索到的网页和文件，L07 会专门讲 prompt injection）。只要有人让模型写出
        __import__('os').system('del /q *')
    eval 就会照做。所以绝不能把模型的输出交给 eval。

做法：AST 白名单。
    1. ast.parse 只做"语法分析"，把字符串变成一棵语法树，**不执行任何东西**
    2. 自己遍历这棵树，只认识白名单里的几种节点：数字、+ - * / **、正负号、括号、abs/min/max
    3. 其余一律拒绝：变量名、属性访问（x.y）、下标、字符串、import、lambda……
    这是"默认拒绝"：不是列出哪些危险，而是只列出哪些允许。没想到的东西自动被挡在外面。

    例如 "(5829 - 5735) / 5735 * 100" 的语法树（ast.dump 的简化版）：
        BinOp(
          left=BinOp(
            left=BinOp(left=Constant(5829), op=Sub(), right=Constant(5735)),
            op=Div(), right=Constant(5735)),
          op=Mult(), right=Constant(100))
    遍历时从最里层算起：5829-5735=94 → 94/5735 → ×100。

白名单也管"语义"，不只管安全：
    % 在 Python 里是取余，但模型写 "15%" 时想的是百分之十五。与其让 20 % 3 悄悄算出 2，
    不如直接拒绝 %，并告诉模型该怎么写。数字里的千分位逗号同理："5,829" 在 Python 里是元组 (5, 829)。
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from finagent.tools.registry import ToolError, tool

MAX_EXPRESSION_LENGTH = 500

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {"abs": abs, "min": min, "max": max}


def evaluate(expression: str) -> float:
    """计算算术表达式。不合法或不安全时抛出 ToolError，错误信息是写给模型看的（英文）。"""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"expression is too long (max {MAX_EXPRESSION_LENGTH} characters)")
    if "%" in expression:
        raise ToolError(
            "'%' is not supported. Write percentages as numbers: use 0.15 for 15%, "
            "and multiply by 100 yourself to get a percentage"
        )
    try:
        tree = ast.parse(
            expression.strip(), mode="eval"
        )  # mode="eval"：只接受单个表达式，不接受语句
    except SyntaxError as e:
        raise ToolError(f"not a valid arithmetic expression ({e.msg})") from None
    except (RecursionError, MemoryError):  # 括号嵌套几百层时，解析器自己就会出错
        raise ToolError("expression is nested too deeply") from None
    try:
        result = _eval(tree.body)
    except ZeroDivisionError:
        raise ToolError("division by zero") from None
    except OverflowError:
        raise ToolError("the result is too large") from None
    if not math.isfinite(result):  # inf / nan：比如 1e308 * 10
        raise ToolError("the result is not a finite number")
    return result


def _eval(node: ast.expr) -> float:
    """递归计算一个节点。不在白名单里的节点，一律拒绝。"""
    if isinstance(node, ast.Constant):
        # bool 是 int 的子类（True + True == 2），要先排除
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"only numbers are allowed, got {node.value!r}")
        # 全部转成 float：整数的 ** 可以算出几百万位的大整数（9**9**9 会卡死程序），
        # float 超出范围会立刻抛 OverflowError。float 有约 15 位有效数字，财务计算足够了
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        result = _BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(result, complex):  # (-8) ** 0.5 在 Python 里会得到复数
            raise ToolError(
                "the result is not a real number (fractional power of a negative number)"
            )
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolError(f"only these functions are allowed: {', '.join(_FUNCTIONS)}")
        if node.keywords or not node.args:
            raise ToolError(f"{node.func.id}() takes one or more numbers as arguments")
        return _FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))
    if isinstance(node, ast.Tuple):
        raise ToolError("write numbers without thousands separators: 5829, not 5,829")
    raise ToolError(
        f"'{type(node).__name__}' is not allowed; use only numbers, + - * / **, parentheses "
        f"and {', '.join(_FUNCTIONS)}()"
    )


def format_number(x: float) -> str:
    """12 位有效数字：足够精确，又不会出现 0.30000000000000004 这样的浮点噪音。"""
    text = f"{x:.12g}"
    return "0" if text == "-0" else text


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly and return the numeric result. Use this tool for every calculation (differences, sums, ratios, percentage changes, averages, growth rates) instead of doing arithmetic in your head. Independent calculations can be requested in parallel.

    Args:
        expression: One arithmetic expression using plain numbers, + - * / ** and parentheses, for example "(5829 - 5735) / 5735 * 100". Do not use thousands separators, currency signs, percent signs or variable names.
    """
    return format_number(evaluate(expression))
