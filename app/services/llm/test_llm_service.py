"""Unit tests for the high level LLM service facade.

These tests focus on verifying that :mod:`app.services.llm.unified_service`
invokes the correct provider hooks and handles formatted responses without
requiring access to real third-party APIs.  The previous version of this test
suite defined ``async def`` tests directly, which Pytest could not collect
without an async plugin and therefore failed immediately.  The rewritten
tests below use synchronous wrappers together with lightweight dummy
providers, making the behaviour deterministic and plugin-free.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from app.services.llm.base import TextModelProvider
from app.services.llm.config_validator import LLMConfigValidator
from app.services.llm.manager import LLMServiceManager
from app.services.llm.unified_service import UnifiedLLMService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyTextProvider(TextModelProvider):
    """Deterministic text provider used for unit testing."""

    _narration_payload: Dict[str, Any] = {
        "title": "人工智能解说示例",
        "content": "这是一段用于测试的示例文案。",
        "duration": 42,
        "items": [
            {
                "_id": 1,
                "timestamp": "00:00:00,000-00:00:05,000",
                "picture": "主角在森林中开始建造木屋的镜头",
                "narration": "在清晨的森林里，他选定了一块平坦的空地准备搭建木屋。",
                "OST": 2,
            }
        ],
    }

    _subtitle_analysis: str = (
        "剧情分析：影片讲述了一位角色在频道中分享人工智能知识的故事，"
        "内容结构层层递进，突出角色的学习动机与成长轨迹。"
    )

    @property
    def provider_name(self) -> str:  # pragma: no cover - trivial
        return "dummy"

    @property
    def supported_models(self) -> List[str]:  # pragma: no cover - trivial
        return ["dummy-model"]

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        **_: Any,
    ) -> str:
        """Return deterministic responses tailored to the test scenario."""

        if response_format == "json":
            return json.dumps(self._narration_payload, ensure_ascii=False)

        if "00:00:01,000" in prompt:
            return self._subtitle_analysis

        return "人工智能是一门研究机器如何模拟人类智能行为的科学。"

    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - unused
        return {"echo": payload}


@pytest.fixture
def dummy_text_provider(monkeypatch: pytest.MonkeyPatch) -> DummyTextProvider:
    """Provide a dummy text provider and patch the manager to return it."""

    provider = DummyTextProvider(api_key="fake", model_name="dummy-model")

    monkeypatch.setattr(
        LLMServiceManager,
        "get_text_provider",
        lambda provider_name=None: provider,
    )

    return provider


def _run_async(coro):
    """Execute an async coroutine inside the tests without pytest plugins."""

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests covering UnifiedLLMService async helpers
# ---------------------------------------------------------------------------


def test_text_generation(dummy_text_provider: DummyTextProvider):
    """``generate_text`` should delegate to the configured provider."""

    result = _run_async(
        UnifiedLLMService.generate_text(
            prompt="请用一句话介绍人工智能。",
            system_prompt="你是一个专业的AI助手。",
            temperature=0.7,
        )
    )

    assert "人工智能" in result


def test_json_generation(dummy_text_provider: DummyTextProvider):
    """JSON formatted requests should return structured payloads."""

    raw = _run_async(
        UnifiedLLMService.generate_text(
            prompt="请生成一个简单的解说文案示例，包含标题与内容。",
            system_prompt="你是一个专业的文案撰写专家。",
            temperature=0.7,
            response_format="json",
        )
    )

    parsed = json.loads(raw)

    assert parsed["title"] == "人工智能解说示例"
    assert parsed["duration"] == 42
    assert parsed["items"] and parsed["items"][0]["_id"] == 1


def test_narration_script_generation(dummy_text_provider: DummyTextProvider):
    """The narration helper should parse the provider JSON into segments."""

    segments = _run_async(
        UnifiedLLMService.generate_narration_script(
            prompt="根据以下视频描述生成解说文案。",
            temperature=0.8,
            validate_output=True,
        )
    )

    assert isinstance(segments, list)
    assert segments[0]["timestamp"] == "00:00:00,000-00:00:05,000"
    assert "木屋" in segments[0]["picture"]


def test_subtitle_analysis(dummy_text_provider: DummyTextProvider):
    """Subtitle analysis should pass validation and return the summary."""

    subtitle_content = """
1
00:00:01,000 --> 00:00:05,000
大家好，欢迎来到我的频道。

2
00:00:05,000 --> 00:00:10,000
今天我们要学习如何使用人工智能。

3
00:00:10,000 --> 00:00:15,000
人工智能是一项非常有趣的技术。
"""

    analysis = _run_async(
        UnifiedLLMService.analyze_subtitle(
            subtitle_content=subtitle_content,
            temperature=0.7,
            validate_output=True,
        )
    )

    assert "剧情分析" in analysis


# ---------------------------------------------------------------------------
# Tests that keep their real implementations but assert structured output
# ---------------------------------------------------------------------------


def test_config_validation():
    """The configuration validator should always return a structured report."""

    results = LLMConfigValidator.validate_all_configs()
    summary = results["summary"]

    assert summary["total_vision_providers"] == len(results["vision_providers"])
    assert summary["total_text_providers"] == len(results["text_providers"])
    assert isinstance(summary["errors"], list)


def test_provider_info_structure():
    """Provider metadata should expose separate vision/text dictionaries."""

    info = UnifiedLLMService.get_provider_info()

    assert "vision_providers" in info
    assert "text_providers" in info
    assert isinstance(info["vision_providers"], dict)
    assert isinstance(info["text_providers"], dict)
