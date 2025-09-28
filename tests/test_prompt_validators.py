import pytest

from app.services.prompts.validators import (
    PromptOutputValidator,
    PromptValidationError,
)


def test_validate_narration_script_normalizes_timestamps():
    payload = {
        "items": [
            {
                "_id": 1,
                "timestamp": "0:00-5",
                "picture": "场景",
                "narration": "解说内容",
            }
        ]
    }

    result = PromptOutputValidator.validate_narration_script(payload)

    assert result["items"][0]["timestamp"] == "00:00:00,000-00:00:05,000"


def test_validate_narration_script_supports_decimal_seconds():
    payload = {
        "items": [
            {
                "_id": 1,
                "timestamp": "00:00:03.5 - 00:00:06",
                "picture": "场景",
                "narration": "解说内容",
            }
        ]
    }

    result = PromptOutputValidator.validate_narration_script(payload)

    assert result["items"][0]["timestamp"] == "00:00:03,500-00:00:06,000"


def test_validate_narration_script_rejects_invalid_timestamp():
    payload = {
        "items": [
            {
                "_id": 1,
                "timestamp": "invalid",
                "picture": "场景",
                "narration": "解说内容",
            }
        ]
    }

    with pytest.raises(PromptValidationError):
        PromptOutputValidator.validate_narration_script(payload)

