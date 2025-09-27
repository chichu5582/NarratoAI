import unittest
from unittest.mock import MagicMock, patch

from app.utils import script_generator


class GeminiGeneratorTests(unittest.TestCase):
    def test_native_generator_posts_to_gemini_endpoint(self):
        generator = script_generator.GeminiGenerator(
            model_name="gemini-pro",
            api_key="fake-key",
            prompt="base",
            base_url="https://gemini.example.com",
        )

        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Generated narration"},
                        ]
                    }
                }
            ]
        }

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = response_payload

        with patch("app.utils.script_generator.requests.post", return_value=fake_response) as mock_post:
            script = generator.generate_script("Scene description", 120)

        self.assertEqual(script, "Generated narration")
        mock_post.assert_called_once()
        requested_url = mock_post.call_args[0][0]
        self.assertEqual(requested_url, "https://gemini.example.com/models/gemini-pro:generateContent")

    def test_openai_compatible_generator_invokes_openai_client(self):
        calls = []

        class DummyResponse:
            def __init__(self, content: str):
                message = type("Message", (), {"content": content})()
                choice = type("Choice", (), {"message": message})()
                self.choices = [choice]

        class DummyCompletions:
            def __init__(self, recorder):
                self._recorder = recorder

            def create(self, **kwargs):
                self._recorder.append(kwargs)
                return DummyResponse("Generated via OpenAI proxy")

        class DummyChat:
            def __init__(self, recorder):
                self.completions = DummyCompletions(recorder)

        class DummyOpenAI:
            def __init__(self, api_key, base_url):
                self.api_key = api_key
                self.base_url = base_url
                self.chat = DummyChat(calls)

        with patch("openai.OpenAI", DummyOpenAI):
            generator = script_generator.GeminiOpenAIGenerator(
                model_name="gemini-pro",
                api_key="proxy-key",
                prompt="base",
                base_url="https://proxy.example.com/v1",
            )
            script = generator.generate_script("Scene description", 80)

        self.assertEqual(script, "Generated via OpenAI proxy")
        self.assertEqual(len(calls), 1)
        recorded_call = calls[0]
        self.assertEqual(recorded_call["model"], "gemini-pro")
        self.assertIn("messages", recorded_call)
        self.assertEqual(recorded_call["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
