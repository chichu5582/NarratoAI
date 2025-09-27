import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.script_generator import ScriptProcessor


@pytest.mark.parametrize(
    "time_range, expected",
    [
        ("00:00:00,000-00:00:10,000", 25),  # 10 秒 -> floor(10/0.4) = 25
        ("00:00:00,000-00:00:01,500", 10),  # 1.5 秒 -> floor(1.5/0.4)=3, 下限为 10
        ("00:00:00,000-00:40:00,000", 500),  # 40 分钟 -> 上限为 500
    ],
)
def test_calculate_duration_and_word_count(time_range, expected):
    processor = ScriptProcessor.__new__(ScriptProcessor)

    assert processor.calculate_duration_and_word_count(time_range) == expected
