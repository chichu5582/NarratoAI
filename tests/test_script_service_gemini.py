import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import config
from app.services.script_service import ScriptGenerator


class ProcessWithGeminiTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_variant_uses_openai_analyzer(self):
        generator = ScriptGenerator()
        fake_frames = [
            "/tmp/frame_000001_00:00:00,000.jpg",
            "/tmp/frame_000002_00:00:05,000.jpg",
            "/tmp/frame_000003_00:00:10,000.jpg",
        ]

        class DummyAnalyzer:
            instantiated = False

            def __init__(self, model_name, api_key, base_url):
                DummyAnalyzer.instantiated = True
                self.model_name = model_name
                self.api_key = api_key
                self.base_url = base_url

            async def analyze_images(self, images, prompt, batch_size):
                self.images = images
                self.prompt = prompt
                self.batch_size = batch_size
                return [
                    {"batch_index": 0, "response": "analysis"},
                ]

        class DummyProcessor:
            def __init__(self, *args, **kwargs):
                pass

            def process_frames(self, frame_content_list):
                return frame_content_list

        progress_events = []

        def progress_callback(progress, message):
            progress_events.append((progress, message))

        with patch.dict(
            config.app,
            {
                "vision_gemini_api_key": "test-key",
                "vision_gemini_model_name": "test-model",
                "vision_gemini_base_url": "https://vision.example.com",
                "vision_analysis_prompt": "describe",
                "text_llm_provider": "gemini",
                "text_gemini_api_key": "text-key",
                "text_gemini_model_name": "gemini-pro",
                "text_gemini_base_url": "https://text.example.com",
            },
            clear=False,
        ), patch(
            "app.services.script_service.gemini_analyzer.VisionAnalyzer",
            side_effect=AssertionError("Should not instantiate VisionAnalyzer"),
        ), patch(
            "app.utils.gemini_openai_analyzer.GeminiOpenAIAnalyzer",
            DummyAnalyzer,
        ), patch(
            "app.services.script_service.ScriptProcessor",
            DummyProcessor,
        ):
            result = await generator._process_with_gemini(
                keyframe_files=fake_frames,
                video_theme="Adventure",
                custom_prompt="",
                vision_batch_size=2,
                progress_callback=progress_callback,
                vision_provider="gemini(openai)",
            )

        self.assertTrue(DummyAnalyzer.instantiated)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["picture"], "analysis")
        self.assertGreaterEqual(len(progress_events), 3)


class ExtractKeyframesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = Path(self.temp_dir.name) / "video.mp4"
        self.video_path.write_bytes(b"fake video content")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_extract_keyframes_respects_interval_skip_and_threshold(self):
        generator_temp_dir = Path(self.temp_dir.name) / "generator"
        generator_temp_dir.mkdir()

        processed_calls = []

        class DummyVideoProcessor:
            def __init__(self, video_path: str):
                self.video_path = video_path

            def process_video_pipeline(self, output_dir: str, interval_seconds: float = 5.0, use_hw_accel: bool = True):
                processed_calls.append(
                    {
                        "output_dir": output_dir,
                        "interval_seconds": interval_seconds,
                        "use_hw_accel": use_hw_accel,
                    }
                )

                os.makedirs(output_dir, exist_ok=True)

                def create_frame(frame_number: int, seconds: float):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    milliseconds = int(round((seconds % 1) * 1000))
                    time_str = f"{hours:02d}{minutes:02d}{secs:02d}{milliseconds:03d}"
                    path = Path(output_dir) / f"keyframe_{frame_number:06d}_{time_str}.jpg"
                    path.write_bytes(b"frame")

                create_frame(0, 0)
                create_frame(90, 3)
                create_frame(150, 5)
                create_frame(170, 5.6)
                create_frame(210, 7)

        with patch("app.services.script_service.utils.temp_dir", return_value=str(generator_temp_dir)), patch(
            "app.services.script_service.video_processor.VideoProcessor",
            DummyVideoProcessor,
        ):
            generator = ScriptGenerator()

            filtered = await generator._extract_keyframes(
                video_path=str(self.video_path),
                frame_interval_input=2,
                skip_seconds=4,
                threshold=30,
            )

            self.assertEqual(len(processed_calls), 1)
            self.assertEqual(processed_calls[0]["interval_seconds"], 2.0)
            self.assertTrue(processed_calls[0]["use_hw_accel"])

            filtered_names = [Path(path).name for path in filtered]
            self.assertEqual(filtered_names, [
                "keyframe_000150_000005000.jpg",
                "keyframe_000210_000007000.jpg",
            ])

            cached = await generator._extract_keyframes(
                video_path=str(self.video_path),
                frame_interval_input=2,
                skip_seconds=0,
                threshold=10,
            )

            self.assertEqual(len(processed_calls), 1, "should use cached keyframes on subsequent calls")
            cached_names = [Path(path).name for path in cached]
            self.assertEqual(cached_names, [
                "keyframe_000000_000000000.jpg",
                "keyframe_000090_000003000.jpg",
                "keyframe_000150_000005000.jpg",
                "keyframe_000170_000005600.jpg",
                "keyframe_000210_000007000.jpg",
            ])


if __name__ == "__main__":
    unittest.main()
