"""配置校验：验证"宁可报错，不要静默"真的生效。"""

import pytest
from pydantic import ValidationError

from finagent.config import PROJECT_ROOT, LLMConfig, load_config, with_overrides


def test_v01_config_loads() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "systems" / "v0.1.yaml")
    assert cfg.name == "v0.1"
    assert cfg.llm.model == "deepseek-flash"
    assert cfg.llm.thinking is False


def test_typo_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMConfig.model_validate({"temprature": 0.5})  # 拼错的字段名


def test_thinking_with_temperature_is_rejected() -> None:
    with pytest.raises(ValidationError, match="静默忽略"):
        LLMConfig(thinking=True, temperature=0.7)


def test_effort_without_thinking_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(thinking=False, reasoning_effort="high")


def test_overrides_are_revalidated() -> None:
    base = LLMConfig(temperature=0.7)
    with pytest.raises(ValidationError):
        with_overrides(base, thinking=True)  # 合法 + 覆盖 = 非法，必须被发现
    assert with_overrides(base, model="deepseek-v4-pro", thinking=None).model == "deepseek-v4-pro"
