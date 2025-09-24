import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from app.utils.gemini_openai_analyzer import GeminiOpenAIAnalyzer


class GeminiOpenAIAnalyzerTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_images_returns_structured_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for idx in range(2):
                path = os.path.join(tmpdir, f"frame_{idx:06d}_00-00-0{idx}.jpg")
                Image.new("RGB", (32, 32), color=(idx * 10, 0, 0)).save(path, format="JPEG")
                image_paths.append(path)

            with patch.object(GeminiOpenAIAnalyzer, "_configure_client", lambda self: setattr(self, "client", object())):
                analyzer = GeminiOpenAIAnalyzer(
                    model_name="gemini-vision",
                    api_key="test-key",
                    base_url="https://example.com",
                )

            async def fake_generate(prompt, batch):
                self.assertEqual(prompt, "Describe scene")
                return SimpleNamespace(text=f"analysis-{len(batch)}")

            with patch.object(analyzer, "_generate_content_with_retry", AsyncMock(side_effect=fake_generate)), \
                patch("app.utils.gemini_openai_analyzer.asyncio.sleep", AsyncMock()):
                results = await analyzer.analyze_images(image_paths, "Describe scene", batch_size=1)

        self.assertEqual(len(results), 2)
        for batch_index, result in enumerate(results):
            self.assertIsInstance(result, dict)
            self.assertEqual(result["batch_index"], batch_index)
            self.assertEqual(result["images_processed"], 1)
            self.assertEqual(result["model_used"], "gemini-vision")
            self.assertIn("analysis", result.get("response", ""))
            self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()
