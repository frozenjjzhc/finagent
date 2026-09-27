# L02 补充：JSON 详解

> 所有示例的输出都是在本机实际运行得到的（Python 3.12 + pydantic 2.13）。

---

## 1. JSON 是什么，为什么到处都是它

JSON 是一种**纯文本**的数据格式，用来表示结构化数据。它和编程语言无关，人能读、机器好解析，所以成了数据交换的事实标准。

在 FinAgent 里，它几乎无处不在：

| 位置 | 形式 | 课次 |
|---|---|---|
| API 的请求体、响应体 | JSON | L01 |
| 流式响应里每一行 `data: ...` | JSON | L01 |
| 调用记录 `runs/calls/*.jsonl` | JSONL | L01 |
| 评测集 `evalsets/*.jsonl` | JSONL | L01、L03 |
| 模型的结构化答案 | JSON（JSON mode） | L02 |
| prompt 里的格式说明、few-shot 示例的回答 | 写在文本里的 JSON | L02 |
| 工具定义（JSON Schema）和工具调用参数 | JSON、JSON 字符串 | L04 |
| SFT 训练数据 | JSONL | L10 |

---

## 2. 语法：只有 6 种值，规则少但很严

```
object   {"key": 值, "key2": 值}      键必须是双引号字符串
array    [值, 值, 值]
string   "只能用双引号"               转义：\"  \\  \n  \t  \uXXXX
number   123   -4.5   1e6            不能有前导 0，不能写 +1，没有 NaN / Infinity
true / false                          必须小写
null                                  必须小写
```

另外几条硬规定：**不能有结尾多余的逗号、不能写注释、只能用双引号**；空格和换行随意；文本编码用 UTF-8。

### 看起来像 JSON，其实不是

下面这些写法，大模型在**不开 JSON mode** 时经常输出（它见过大量 Python 代码，所以常写出 Python 风格）。`json.loads` 的真实报错：

| 写法 | 例子 | `json.loads` 的报错 |
|---|---|---|
| 单引号 | `{'value': 1}` | Expecting property name enclosed in double quotes |
| Python 的 True / None | `{"ok": True, "v": None}` | Expecting value |
| 结尾多逗号 | `{"value": 1,}` | Expecting property name enclosed in double quotes |
| 注释 | `{"value": 1 // 单位是百万}` | Expecting ',' delimiter |
| 数字带千分位逗号 | `{"value": 130,497}` | Expecting property name enclosed in double quotes |
| markdown 代码块 | ` ```json {...} ``` ` | Expecting value: line 1 column 1 (char 0) |
| 前面夹着说明文字 | `Here is the JSON: {...}` | Expecting value: line 1 column 1 (char 0) |
| 被截断（max_tokens 不够） | `{"reasoning": "Growth = (107,556 - 90,7` | Unterminated string starting at ... |
| 键没加引号 | `{value: 1}` | Expecting property name enclosed in double quotes |

**最危险的是不报错的情况**：`[130,497]` 是合法的 JSON，意思是一个包含 130 和 497 **两个数**的数组，而不是 130,497。报错的写法你至少知道出了问题，不报错却改了意思的，才是真正的坑。

---

## 3. JSON 和 Python 的对应关系，以及 json 模块

| JSON | Python |
|---|---|
| object | dict |
| array | list（tuple 也会被转成 array，但读回来是 list） |
| string | str |
| number | int / float |
| true / false | True / False |
| null | None |

**转换是不对称的**，实测：

```python
json.dumps({"a": (1, 2), "b": None, "c": True, 1: "x"})
→ '{"a": [1, 2], "b": null, "c": true, "1": "x"}'   # tuple 变 list，整数键 1 变成字符串 "1"
json.loads(上面的结果)
→ {'a': [1, 2], '1': 'x'}                          # 读回来已经不是原来的样子了

json.dumps(0.1 + 0.2) → '0.30000000000000004'     # 浮点数本身的误差，JSON 原样保留
```

浮点误差提醒我们：**真正的财务系统计算金额，用 `Decimal` 或者以"分"为单位的整数**，不用 float。我们的项目只是比较答案，用 float 加容差就够了。

### 四个函数，几个常用参数

| 函数 | 作用 |
|---|---|
| `json.dumps(obj)` / `json.loads(s)` | Python 对象 ↔ 字符串 |
| `json.dump(obj, f)` / `json.load(f)` | Python 对象 ↔ 文件 |

| 参数 | 作用 | 实测 |
|---|---|---|
| `ensure_ascii=False` | 中文原样输出，不转成 `\uXXXX` | 默认：`{"q": "营收"}`；设为 False：`{"q": "营收"}` |
| `indent=2` | 缩进、换行，方便人看 | 输出会带换行，**不能当 JSONL 的一行** |
| `sort_keys=True` | 按键名排序 | `{"reasoning": "x", "value": 1}` |
| `default=str` | 遇到不认识的类型怎么处理 | `{"d": "2026-09-27"}` |
| `allow_nan=False` | 遇到 NaN 就报错，而不是输出非法 JSON | 见下文 |

### json 模块的五个坑（都已实测）

1. **默认会输出非法 JSON**：`json.dumps(float("nan"))` → `'NaN'`。JSON 规范里没有 NaN，别的语言读到会报错。要严格就加 `allow_nan=False`，它会直接报 `ValueError`。
2. **能读非标准的内容**：`json.loads("NaN")` → `nan`。Python 能读，不代表别的系统能读。
3. **重复的键不报错，悄悄保留最后一个**：`json.loads('{"value": 1, "value": 2}')` → `{'value': 2}`。如果模型输出了两个 `value`，你拿到的是后一个，而且毫无察觉。需要检测的话，可以用 `object_pairs_hook` 参数拿到全部键值对。
4. **不认识的类型直接报错**：`json.dumps({"d": date(...)})` → `TypeError: Object of type date is not JSON serializable`。用 `default=str` 或 pydantic 的 `model_dump_json` 解决。
5. **大整数**：Python 的整数没有大小限制，`2**53 + 1` 能原样读写；但 JavaScript 超过 2^53 就会丢精度。所以很多 API 把很长的 ID 写成字符串。

### 报错里的位置信息

```python
json.loads('{"a": 1,\n "b": }')
→ JSONDecodeError: msg='Expecting value' lineno=2 colno=7 pos=15
```

`parse_answer` 把这条信息原样放进重试消息里反馈给模型（"不是合法的 JSON：Expecting value: line 1 column 1 (char 0)"），模型就知道错在哪个位置。

---

## 4. 键的顺序

- **JSON 规范说 object 是无序的**，不应该依赖键的顺序。
- 但 Python 3.7 以后 dict 保留插入顺序，`json.loads` 按文档里的顺序建 dict。实测：`list(json.loads('{"value": 1, "reasoning": "x"}'))` → `['value', 'reasoning']`。上一次我们检查"模型输出的第一个键是不是 reasoning"，靠的就是这一点。
- `sort_keys=True` 会重新排序。

两个实际影响：
1. **"先推理后回答"只是软约束**：顺序对生成有意义，对解析没有意义，解析器也不检查它（上一次讲过）。
2. **前缀缓存要求字节完全一致**：同样的内容，键的顺序不同，就是不同的字节，缓存命中不了。所以构造 messages、工具列表时要保持固定的顺序，不能每次随机。

---

## 5. JSONL：一行一个 JSON

```
{"id": "q01", "question": "...", "answer": 130497}
{"id": "q02", "question": "...", "answer": 193737}
```

为什么日志、数据集都用它：
- **只追加**：新记录写在文件末尾，不用读取、修改整个文件；
- **可以流式处理**：逐行读，不用一次性把几 GB 的文件读进内存；
- **坏一行只影响一行**：程序中途崩溃留下的半行，不会让整个文件读不出来；
- **方便切分和统计**：行数就是记录数，`head`、`wc -l` 就能用。

**唯一的硬规定：每条记录必须在一行之内。**
- `json.dumps` 默认不换行，字符串里的换行会被转义成 `\n`。实测：`'{"s": "line1\\nsay \\"hi\\""}'`；
- `indent=2` 会引入真正的换行，JSONL 就坏了。

读取的标准写法（项目里用的就是它）：

```python
with path.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]   # 跳过空行
```

---

## 6. JSON 里套 JSON（双重编码）

L04 的工具调用会遇到这种情况：API 返回的工具参数 `arguments` 是一个**JSON 字符串**，外面再包一层 JSON。实测：

```python
args = {"expression": "(107556 - 90757) / 90757"}
tool_call = {"name": "calculator", "arguments": json.dumps(args)}
json.dumps(tool_call)
→ '{"name": "calculator", "arguments": "{\"expression\": \"(107556 - 90757) / 90757\"}"}'
```

第一次 `json.loads` 之后，`arguments` 的类型还是 **`str`**，要再 `json.loads` 一次才能拿到 dict。而且这个字符串是模型生成的，可能不合法，所以要像 L02 一样校验。

---

## 7. 大模型输出 JSON：失败类型与防线

| 失败类型 | 例子 | 谁来挡住 |
|---|---|---|
| 代码块、说明文字 | ` ```json ... ``` ` | JSON mode |
| Python 风格 | `True`、`None`、单引号 | JSON mode |
| 输出被截断 | Unterminated string | 检查 `finish_reason=length`，调大 `max_tokens` |
| 类型不对 | `"value": "130,497"` | pydantic |
| 取值不在范围内 | `"unit": "USD billions"` | pydantic 的 `Literal` |
| 缺字段、多字段 | | pydantic 的必填字段、`extra="forbid"` |
| 自相矛盾 | 说 unknown，却给了 value | pydantic 的 `model_validator` |
| **数字本身是错的** | `"value": 130000` | **以上都挡不住** → 评测（L03）、检索（L05） |

最后一行最重要：**结构合格 ≠ 内容正确。** 校验保证的是"程序能用"，不是"答案对"。

### 为什么不用"JSON 修复库"

有些库（比如 `json_repair`）能自动修复单引号、结尾逗号等问题，让解析"总能成功"。代价是**悄悄改动内容可能改变意思**，比如 `[130,497]` 这种情况。我们选择**严格解析 + 带反馈重试**：有问题就明确暴露出来。修复库可以作为最后的兜底，但要把"修过"这件事记录下来。

---

## 8. JSON Schema、pydantic，以及 JSON mode 背后的原理

### 8.1 JSON Schema：描述 JSON 长什么样的 JSON

JSON Schema 是一个标准，用一段 JSON 描述"合格的 JSON 应该长什么样"：有哪些字段、什么类型、哪些必填、可选值有哪些。pydantic 能从 `Answer` 自动生成它（`Answer.model_json_schema()`，实测节选）：

```
字段顺序: ['reasoning', 'status', 'value', 'unit', 'answer_text', 'sources', 'confidence']
必填字段: ['reasoning', 'status', 'answer_text', 'confidence']
status:   {"enum": ["answered", "unknown", "invalid_premise"], "type": "string"}
value:    {"anyOf": [{"type": "number"}, {"type": "null"}], "default": null}
additionalProperties: False
```

注意：`value`、`unit`、`sources` 有默认值，所以不在必填列表里。模型可以不写它们，不写就取默认值。

**它在哪里用**：L04 定义工具时，工具的 `parameters` 字段**就是一个 JSON Schema**，我们会从 Python 函数的签名自动生成它。OpenAI 的"严格结构化输出"也直接接收 JSON Schema。DeepSeek 的 JSON mode 不接收 schema，所以我们把格式写在 prompt 里，再用 pydantic 校验。

### 8.2 pydantic 与 JSON

| 方法 | 作用 |
|---|---|
| `Answer.model_validate(dict)` | 校验一个 dict |
| `Answer.model_validate_json(str)` | 解析加校验一步完成（内部用 Rust 实现，更快） |
| `a.model_dump()` / `a.model_dump_json()` | 转回 dict / JSON 字符串，按字段定义的顺序输出 |

**`parse_answer` 为什么分成两步**（先 `json.loads`，再 `model_validate`），而不用 `model_validate_json` 一步完成：为了**区分"不是 JSON"和"不符合 schema"两种失败**，给模型更准确的反馈。

**宽松模式与严格模式**（实测）：

| | `{"value": "130497"}` | `{"value": 130497}` |
|---|---|---|
| 默认（宽松） | ✅ 自动转成 130497.0 | ✅ |
| `strict=True` | ❌ Input should be a valid number | ✅ |

我们用默认的宽松模式：无害的类型差异（数字写成了字符串）自动容忍；但 `"130,497"` 这种带逗号的，宽松模式也转不了，照样校验失败。

### 8.3 JSON mode 为什么能"保证"合法：约束解码

回忆 nanolm 的生成：每一步算出下一个 token 的概率分布，然后采样。JSON mode 在采样前多做一件事：

```
每一步：
  1. 模型算出所有 token 的 logits
  2. 根据"目前已经写了什么"，判断接下来哪些 token 在语法上合法
     （例如写到 {"value": 之后，只能接数字、负号、引号、[、{、true、false、null）
  3. 把不合法 token 的 logits 设为负无穷
  4. 再 softmax、采样
```

这叫**约束解码**（constrained decoding）：从源头上不可能生成语法错误的 JSON。但它只管语法，所以管不了内容。

**一个副作用**：如果模型本来没打算写 JSON，被强行约束后，它可能一直输出空格，直到撞上 `max_tokens`。OpenAI 的文档明确提醒过这一点。DeepSeek 要求 prompt 里必须出现 "json" 一词，很可能出于同样的原因。

L08 以后用本地模型时，可以用约束解码库按**完整的 schema** 来约束，包括字段顺序、枚举值，比 DeepSeek 的 JSON mode 更强。

---

## 9. 安全

- **`json.loads` 是安全的**：它只产生数据，不执行任何代码。
- **绝不要用 `eval()` 处理模型输出**或任何"长得像 JSON"的文本，那等于执行任意代码。同理，`pickle`、`yaml.load` 也不安全，我们用的是 `yaml.safe_load`。
- **模型输出一律当作不可信的外部输入**：先校验，再使用。L04 的计算器工具会用 AST 白名单来解析表达式，也是这个原因。

---

## 10. 自测题

1. 下面哪些是合法的 JSON？`{'a': 1}`、`{"a": 1,}`、`{"a": NaN}`、`[130,497]`、`"hello"`、`null`
2. `json.dumps({1: (1, 2)})` 的结果是什么？
3. `parse_answer` 为什么先 `json.loads` 再 `model_validate`，而不是一步 `model_validate_json`？
4. JSON mode 能保证什么？它是怎么做到的？为什么 prompt 里必须写 "json"？
5. 写 JSONL 时为什么不能用 `indent`？
6. L04 的工具参数为什么要 `json.loads` 两次？

**参考答案**

1. 只有后三个合法。`[130,497]` 合法，但它是两个数；`"hello"` 和 `null` 作为顶层值也合法。前三个分别错在单引号、结尾逗号、NaN。
2. `'{"1": [1, 2]}'`：整数键变成字符串，tuple 变成 array。
3. 为了区分"不是 JSON"和"不符合 schema"两种失败，反馈给模型的错误信息更准确。
4. 保证输出是语法合法的 JSON，不保证内容。做法是约束解码：每一步只允许语法上合法的 token。不告诉模型要写 JSON，它在约束下可能一直输出空白直到 `max_tokens`，所以要求 prompt 里出现 "json"。
5. `indent` 会在一条记录内部插入换行，而 JSONL 要求一条记录恰好占一行。
6. API 返回的 `arguments` 是一个 JSON 字符串，外面还包着一层 JSON。第一次解析外层，拿到的是字符串；第二次才解析出参数本身。
