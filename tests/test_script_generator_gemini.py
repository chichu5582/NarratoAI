import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from app.config import config
from app.services.script_generator import ScriptGenerator


class ProcessWithGeminiTests(unittest.IsolatedAsyncioTestCase):
    def test_normalize_provider_handles_aliases(self):
        generator = ScriptGenerator()

        self.assertEqual(
            generator._normalize_provider("Silicon Flow"),
            ("siliconflow", "siliconflow"),
        )
        self.assertEqual(
            generator._normalize_provider("Qwen-VL"),
            ("qwenvl", "qwenvl"),
        )

    async def test_extract_keyframes_passes_supported_video_processor_kwargs(self):
        """Ensure only supported kwargs are passed to the video processor."""

        async def run_extraction(generator: ScriptGenerator, video_path: Path):
            captured_kwargs = {}

            with patch(
                "app.services.script_generator.video_processor.VideoProcessor"
            ) as MockProcessor:
                instance = MockProcessor.return_value

                def fake_process_video_pipeline(*args, **kwargs):
                    captured_kwargs.clear()
                    captured_kwargs.update(kwargs)
                    os.makedirs(kwargs["output_dir"], exist_ok=True)
                    frame_path = Path(kwargs["output_dir"]) / "frame_000001.jpg"
                    frame_path.write_bytes(b"")

                instance.process_video_pipeline.side_effect = fake_process_video_pipeline

                keyframes = await generator._extract_keyframes(
                    str(video_path),
                    skip_seconds=0,
                    threshold=30,
                    frame_interval=2,
                    use_hw_accel=False,
                )

            self.assertTrue(keyframes)
            return captured_kwargs

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            video_path = temp_path / "video.mp4"
            video_path.write_bytes(b"fake video content")

            with patch(
                "app.services.script_generator.utils.temp_dir",
                return_value=tmp_dir,
            ):
                generator = ScriptGenerator()

            kwargs = await run_extraction(generator, video_path)

        expected_keys = {"output_dir", "interval_seconds", "use_hw_accel"}
        self.assertTrue(kwargs)
        self.assertEqual(set(kwargs), expected_keys)
        self.assertEqual(kwargs["interval_seconds"], 2)
        self.assertFalse(kwargs["use_hw_accel"])

    async def test_openai_variant_uses_openai_analyzer(self):
        generator = ScriptGenerator()
        fake_frames = [
            "/tmp/frame_000001_00:00:00,000.jpg",
            "/tmp/frame_000002_00:00:05,000.jpg",
            "/tmp/frame_000003_00:00:10,000.jpg",
        ]

        class DummyAnalyzer:
            instantiated = 0

            def __init__(self, model_name, api_key, base_url):
                DummyAnalyzer.instantiated += 1
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

        provider_variants = ["gemini(openai)", "Gemini (OpenAI)"]

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
            "app.services.script_generator.gemini_analyzer.VisionAnalyzer",
            side_effect=AssertionError("Should not instantiate VisionAnalyzer"),
        ), patch(
            "app.utils.gemini_openai_analyzer.GeminiOpenAIAnalyzer",
            DummyAnalyzer,
        ), patch(
            "app.services.script_generator.ScriptProcessor",
            DummyProcessor,
        ):
            for provider in provider_variants:
                progress_events = []

                def progress_callback(progress, message):
                    progress_events.append((progress, message))

                result = await generator._process_with_gemini(
                    keyframe_files=fake_frames,
                    video_theme="Adventure",
                    custom_prompt="",
                    vision_batch_size=2,
                    progress_callback=progress_callback,
                    vision_provider=provider,
                )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["picture"], "analysis")
        self.assertGreaterEqual(len(progress_events), 3)

        self.assertEqual(DummyAnalyzer.instantiated, len(provider_variants))

    async def test_generate_script_routes_to_registered_provider(self):
        generator = ScriptGenerator()
        fake_frames = [
            "/tmp/frame_000001_00:00:00,000.jpg",
            "/tmp/frame_000002_00:00:05,000.jpg",
        ]

        async def dummy_extract(*args, **kwargs):
            return fake_frames

        processed_payload = [{"timestamp": "00:00:00,000-00:00:05,000"}]

        with patch.object(generator, "_extract_keyframes", new=AsyncMock(side_effect=dummy_extract)), \
            patch.object(
                generator,
                "_process_with_registered_provider",
                new=AsyncMock(return_value=processed_payload),
            ) as mock_process, \
            patch(
                "app.services.script_generator.LLMServiceManager.list_vision_providers",
                return_value=["gemini", "siliconflow"],
            ), \
            patch.dict(
                config.app,
                {
                    "vision_analysis_prompt": "describe",
                    "vision_siliconflow_api_key": "vision-key",
                    "vision_siliconflow_model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
                    "vision_siliconflow_base_url": "https://api.siliconflow.cn/v1",
                    "text_llm_provider": "siliconflow",
                    "text_siliconflow_api_key": "text-key",
                    "text_siliconflow_model_name": "deepseek-ai/DeepSeek-R1",
                    "text_siliconflow_base_url": "https://api.siliconflow.cn/v1",
                },
                clear=False,
            ):
            script = await generator.generate_script(
                video_path="/tmp/video.mp4",
                vision_llm_provider="siliconflow",
            )

        self.assertEqual(script, processed_payload)
        mock_process.assert_awaited_once()
        args, _ = mock_process.await_args
        self.assertEqual(args[0], fake_frames)
        self.assertEqual(args[5], "siliconflow")
        self.assertEqual(args[6], "siliconflow")
        self.assertIsInstance(args[7], dict)

    async def test_registered_provider_normalizes_results(self):
        generator = ScriptGenerator()
        keyframes = [
            "/tmp/frame_000001_00:00:00,000.jpg",
            "/tmp/frame_000002_00:00:05,000.jpg",
            "/tmp/frame_000003_00:00:10,000.jpg",
            "/tmp/frame_000004_00:00:15,000.jpg",
        ]

        class DummyVisionProvider:
            model_name = "siliconflow-model"

            async def analyze_images(self, images, prompt, batch_size):
                self.images = images
                self.prompt = prompt
                self.batch_size = batch_size
                return ["batch-one", {"response": "batch-two"}]

        dummy_provider = DummyVisionProvider()

        with patch(
            "app.services.script_generator.LLMServiceManager.get_vision_provider",
            return_value=dummy_provider,
        ), patch.object(
            ScriptGenerator,
            "_generate_script_from_results",
            return_value="processed",
        ) as mock_generate, patch.dict(
            config.app,
            {"vision_analysis_prompt": "describe"},
            clear=False,
        ):
            result = await generator._process_with_registered_provider(
                keyframe_files=keyframes,
                video_theme="Adventure",
                custom_prompt="",
                vision_batch_size=2,
                progress_callback=lambda *_: None,
                vision_provider_key="siliconflow",
                normalized_text_provider="siliconflow",
                text_settings={
                    "api_key": "text-key",
                    "model_name": "deepseek-ai/DeepSeek-R1",
                    "base_url": "https://api.siliconflow.cn/v1",
                },
                vision_model_name="siliconflow-model",
            )

        self.assertEqual(result, "processed")
        mock_generate.assert_called_once()
        normalized_results = mock_generate.call_args.args[0]
        self.assertEqual(len(normalized_results), 2)
        self.assertEqual(normalized_results[0]["response"], "batch-one")
        self.assertNotIn("error", normalized_results[0])
        self.assertEqual(normalized_results[0]["model_used"], "siliconflow-model")
        self.assertEqual(normalized_results[1]["response"], "batch-two")


if __name__ == "__main__":
    unittest.main()
