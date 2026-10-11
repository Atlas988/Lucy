import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.audio import VoiceOutput
from core.ai_engine import Lucy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LucyTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.memory_path = self.root / "memory.json"
        self.knowledge_path = self.root / "knowledge.json"
        self.knowledge_path.write_text(
            json.dumps(
                {
                    "Python": {
                        "summary": "Python is a high-level programming language."
                    },
                    "Empty topic": {"summary": None},
                }
            ),
            encoding="utf-8",
        )
        self.lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
        )

    def test_simple_conversation_and_help(self):
        self.assertIn("Hello", self.lucy.respond("hello"))
        self.assertIn("/remember", self.lucy.respond("/help"))
        self.assertIn("knowledge-only mode", self.lucy.respond("/status"))

    def test_remembered_facts_persist_between_instances(self):
        self.assertEqual("I'll remember that.", self.lucy.respond("/remember I like tea"))
        restarted_lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
        )
        response = restarted_lucy.respond("/recall tea")
        self.assertIn("I like tea", response)

    def test_local_knowledge_retrieval(self):
        response = self.lucy.respond("What is Python?")
        self.assertIn("Python is a high-level programming language.", response)
        self.assertIn("LUCY_MODEL_PATH", response)

    def test_ai_alias_matches_artificial_intelligence_entry(self):
        self.knowledge_path.write_text(
            json.dumps(
                {
                    "Artificial Intelligence": {
                        "summary": "AI is the study of intelligent machines."
                    }
                }
            ),
            encoding="utf-8",
        )
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
        )
        self.assertIn("study of intelligent machines", lucy.respond("What is AI?"))

    def test_calculator_and_rejects_code(self):
        self.assertEqual("14", self.lucy.respond("=2 + 3 * 4"))
        self.assertIn("couldn't calculate", self.lucy.respond("=__import__('os')"))

    def test_search_uses_wikipedia_summary(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"extract": "A short article summary."}'

        with patch("core.ai_engine.urllib.request.urlopen", return_value=FakeResponse()):
            self.assertEqual("A short article summary.", self.lucy.respond("/search Python"))

    def test_optional_voice_output_uses_platform_speech_engine(self):
        engine = Mock()
        speech_module = Mock(init=Mock(return_value=engine))
        with patch.dict(sys.modules, {"pyttsx3": speech_module}):
            voice = VoiceOutput()
            voice.speak("Hello")
        engine.setProperty.assert_called_once_with("rate", 175)
        engine.say.assert_called_once_with("Hello")
        engine.runAndWait.assert_called_once_with()

    def test_configured_missing_model_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            Lucy(
                model_path=self.root / "missing.gguf",
                memory_path=self.memory_path,
                knowledge_path=self.knowledge_path,
            )

    def test_cli_runs_from_another_working_directory(self):
        environment = os.environ.copy()
        environment.pop("LUCY_MODEL_PATH", None)
        environment.pop("LUCY_VOICE", None)
        environment["LUCY_MEMORY_PATH"] = str(self.root / "cli-memory.json")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "main.py")],
            input="/status\n/quit\n",
            capture_output=True,
            text=True,
            cwd=self.root,
            env=environment,
            timeout=10,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Lucy is running", result.stdout)
        self.assertIn("Goodbye.", result.stdout)


if __name__ == "__main__":
    unittest.main()
