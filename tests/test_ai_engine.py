import json
import io
import os
import subprocess
import sys
import tempfile
import types
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.audio import VoiceOutput
from core.coding_workspace import CodingWorkspace
from core.computer_access import ComputerAccess
from core.ai_engine import Lucy
import main as lucy_main


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
        self.assertEqual(
            "I'm ready to help. What would you like to work on?",
            self.lucy.respond("how are you?"),
        )
        self.assertEqual(
            "Good morning! What would you like to talk about?",
            self.lucy.respond("good morning"),
        )
        self.assertIn("/remember", self.lucy.respond("/help"))
        self.assertIn("knowledge-only mode", self.lucy.respond("/status"))

    def test_supplied_emotion_taxonomy_is_available_as_local_reference(self):
        answer = self.lucy.respond("What does lonely mean?")

        self.assertIn("Lonely", answer)
        self.assertIn("isolated or disconnected", answer)
        self.assertIn("not a clinical assessment", answer)

    def test_supplied_research_notes_are_searchable_without_participant_records(self):
        image_data = self.lucy.respond(
            "What is the supplied image-triplet research data?"
        )
        checkpoint = self.lucy.respond("What is the supplied fMRI checkpoint?")

        self.assertIn("visual similarity/grouping data", image_data)
        self.assertIn("not emotion", image_data)
        self.assertIn("not a conversational language model", checkpoint)
        self.assertFalse(any("participant_" in summary for _, summary in self.lucy.knowledge))

    def test_supplied_emotion_reference_is_included_in_model_context(self):
        lucy = self.make_api_lucy()
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return self.fake_chat_response(
                {"role": "assistant", "content": "The supplied taxonomy groups loneliness under sadness."}
            )

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
                answer = lucy.respond("What does lonely mean?")

        self.assertIn("sadness", answer)
        self.assertIn("isolated or disconnected", requests[0]["messages"][0]["content"])

    def test_common_identity_questions_describe_configured_ai_mode(self):
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="http://localhost:11434/v1",
            api_model="qwen3:0.6b",
        )

        self.assertIn("qwen3", lucy.respond("what are you").casefold())
        self.assertIn("Ollama", lucy.respond("who are you?"))

    def test_user_identity_is_recalled_from_the_current_conversation(self):
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="https://api.example.test/v1",
            api_model="chat-model",
        )

        introduction = lucy.respond("Im aiden")
        answer = lucy.respond("who am i")

        self.assertIn("Aiden", introduction)
        self.assertEqual("You told me your name is Aiden.", answer)
        self.assertNotIn("Aiden", lucy.facts)

    def test_identity_correction_does_not_repeat_an_incorrect_name(self):
        self.lucy.respond("I'm Aiden")
        self.assertEqual(
            "Sorry, I misunderstood. What should I call you?",
            self.lucy.respond("no"),
        )
        self.assertEqual(
            "I don't know your name yet. What would you like me to call you?",
            self.lucy.respond("who am i"),
        )

    def test_affirmations_are_handled_locally_instead_of_calling_offline_api(self):
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="http://localhost:11434/v1",
            api_model="qwen3:0.6b",
        )
        lucy.respond("Im aiden")
        lucy.respond("who am i")

        with patch(
            "core.ai_engine.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ) as urlopen:
            first = lucy.respond("yes")
            second = lucy.respond("YEs")

        self.assertIn("Got it, Aiden", first)
        self.assertIn("Got it, Aiden", second)
        self.assertIn("who am i", lucy.history[-6]["content"].casefold())
        self.assertEqual("Aiden", lucy.user_name)
        urlopen.assert_not_called()

    def test_no_after_name_confirmation_does_not_drop_name(self):
        self.lucy.respond("I'm Aiden")
        self.lucy.respond("who am i")

        response = self.lucy.respond("no")

        self.assertEqual("Okay, thanks for letting me know.", response)
        self.assertEqual("Aiden", self.lucy.user_name)

    def test_name_detection_does_not_mistake_small_talk_for_a_name(self):
        self.assertIsNone(Lucy._extract_user_name("I'm learning Python."))
        self.assertIsNone(Lucy._extract_user_name("I am happy today."))
        self.assertEqual("Ada Lovelace", Lucy._extract_user_name("My name is Ada Lovelace."))

    def test_clearing_history_also_clears_unsaved_conversation_identity(self):
        self.lucy.respond("I'm Aiden")
        self.lucy.respond("/clear")

        self.assertIn("don't know your name yet", self.lucy.respond("who am i"))

    def test_common_date_questions_return_the_current_local_date(self):
        with patch("core.ai_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value.astimezone.return_value.strftime.return_value = (
                "Today is Sunday, October 11, 2026."
            )

            responses = [
                self.lucy.respond("tell me the date"),
                self.lucy.respond("what is today's date"),
                self.lucy.respond("tell me the date?"),
                self.lucy.respond("what day is it?"),
            ]

        self.assertEqual(
            ["Today is Sunday, October 11, 2026."] * 4,
            responses,
        )
        mock_datetime.now.assert_called()

    def test_environment_auto_detects_installed_ollama_qwen3(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        environment = {
            "LUCY_MEMORY_PATH": str(self.memory_path),
            "LUCY_MODEL_PATH": "",
            "LUCY_API_BASE_URL": "",
            "LUCY_API_KEY": "",
            "LUCY_API_MODEL": "",
            "LUCY_CODING_API_MODEL": "",
        }
        models = [
            {"name": "qwen3:0.6b", "size": 522_653_767},
            {"name": "qwen2.5-coder:0.5b", "size": 397_000_000},
            {"name": "qwen3:1.7b", "size": 1_359_293_444},
        ]
        response.read.return_value = json.dumps({"models": models}).encode("utf-8")

        with patch.dict(os.environ, environment), patch(
            "core.ai_engine.urllib.request.urlopen", return_value=response
        ) as urlopen:
            lucy = Lucy.from_environment()

        self.assertEqual("http://localhost:11434/v1", lucy.api_base_url)
        self.assertEqual("qwen3:0.6b", lucy.api_model)
        self.assertEqual("qwen2.5-coder:0.5b", lucy.coding_api_model)
        self.assertEqual(1, urlopen.call_count)

    def test_environment_rejects_local_models_over_nine_hundred_mb(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(
            {"models": [{"name": "qwen3:1.7b", "size": 1_359_293_444}]}
        ).encode("utf-8")
        environment = {
            "LUCY_MEMORY_PATH": str(self.memory_path),
            "LUCY_API_BASE_URL": "http://localhost:11434/v1",
            "LUCY_API_MODEL": "qwen3:1.7b",
            "LUCY_API_KEY": "",
            "LUCY_CODING_API_MODEL": "",
            "LUCY_MODEL_PATH": "",
        }

        with patch.dict(os.environ, environment), patch(
            "core.ai_engine.urllib.request.urlopen", return_value=response
        ), self.assertRaisesRegex(RuntimeError, "exceeds Lucy's 900 MB model limit"):
            Lucy.from_environment()

    def test_environment_rejects_oversized_local_coding_model(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(
            {"models": [{"name": "qwen2.5-coder:1.5b", "size": 986_000_000}]}
        ).encode("utf-8")
        environment = {
            "LUCY_MEMORY_PATH": str(self.memory_path),
            "LUCY_API_BASE_URL": "http://localhost:11434/v1",
            "LUCY_API_MODEL": "qwen3:0.6b",
            "LUCY_API_KEY": "",
            "LUCY_CODING_API_MODEL": "qwen2.5-coder:1.5b",
            "LUCY_MODEL_PATH": "",
        }

        with patch.dict(os.environ, environment), patch(
            "core.ai_engine.urllib.request.urlopen", return_value=response
        ), self.assertRaisesRegex(RuntimeError, "coding model .* exceeds Lucy's 900 MB model limit"):
            Lucy.from_environment()

    def test_environment_accepts_local_model_at_nine_hundred_mb(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(
            {"models": [{"name": "qwen3:1.5b", "size": 900_000_000}]}
        ).encode("utf-8")
        environment = {
            "LUCY_MEMORY_PATH": str(self.memory_path),
            "LUCY_API_BASE_URL": "http://localhost:11434/v1",
            "LUCY_API_MODEL": "qwen3:1.5b",
            "LUCY_API_KEY": "",
            "LUCY_CODING_API_MODEL": "",
            "LUCY_MODEL_PATH": "",
        }

        with patch.dict(os.environ, environment), patch(
            "core.ai_engine.urllib.request.urlopen", return_value=response
        ):
            lucy = Lucy.from_environment()

        self.assertEqual("qwen3:1.5b", lucy.api_model)

    def test_environment_keeps_knowledge_only_when_ollama_is_unavailable(self):
        environment = {
            "LUCY_MEMORY_PATH": str(self.memory_path),
            "LUCY_MODEL_PATH": "",
            "LUCY_API_BASE_URL": "",
            "LUCY_API_KEY": "",
            "LUCY_API_MODEL": "",
        }

        with patch.dict(os.environ, environment), patch(
            "core.ai_engine.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Ollama is offline"),
        ):
            lucy = Lucy.from_environment()

        self.assertFalse(lucy.api_base_url)
        self.assertIsNone(lucy.model)

    def test_invalid_api_base_url_fails_during_startup(self):
        for url in (
            "file:///tmp/lucy",
            "https://user:password@example.test/v1",
            "https://api.example.test/v1?key=secret",
            "https://api.example.test/v1#fragment",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(
                ValueError, "LUCY_API_BASE_URL"
            ):
                Lucy(
                    memory_path=self.memory_path,
                    knowledge_path=self.knowledge_path,
                    api_base_url=url,
                )

    def test_api_timeout_must_be_within_supported_range(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 600"):
            Lucy(
                memory_path=self.memory_path,
                knowledge_path=self.knowledge_path,
                api_timeout=0,
            )

        with patch.dict(os.environ, {"LUCY_API_TIMEOUT": "not-a-number"}):
            with self.assertRaisesRegex(ValueError, "LUCY_API_TIMEOUT must be a number"):
                Lucy._api_timeout_from_environment()

    def test_local_model_context_length_is_validated(self):
        with patch.dict(os.environ, {"LUCY_MODEL_CONTEXT": "not-a-number"}):
            with self.assertRaisesRegex(ValueError, "LUCY_MODEL_CONTEXT must be an integer"):
                Lucy._model_context_length()

        with patch.dict(os.environ, {"LUCY_MODEL_CONTEXT": "4096"}):
            self.assertEqual(4096, Lucy._model_context_length())

    def test_llama_cpp_backend_loads_local_gguf_without_probing_ollama(self):
        model_path = self.root / "local.gguf"
        model_path.write_bytes(b"small test model")
        generated = []

        class FakeLlama:
            def __init__(self, **kwargs):
                self.options = kwargs

            def create_chat_completion(self, **kwargs):
                generated.append(kwargs)
                return {
                    "choices": [
                        {"message": {"content": "The local model is working."}}
                    ]
                }

        llama_module = types.ModuleType("llama_cpp")
        llama_module.Llama = FakeLlama
        environment = {
            "LUCY_MODEL_PATH": str(model_path),
            "LUCY_MODEL_BACKEND": "llama-cpp",
            "LUCY_API_BASE_URL": "",
            "LUCY_API_KEY": "",
            "LUCY_API_MODEL": "",
            "LUCY_CODING_API_MODEL": "",
            "LUCY_MEMORY_PATH": str(self.memory_path),
        }

        with patch.dict(os.environ, environment), patch.dict(
            sys.modules, {"llama_cpp": llama_module}
        ), patch.object(
            Lucy,
            "_available_local_backends",
            return_value={"llama-cpp": True, "gpt4all": False},
        ), patch("core.ai_engine.urllib.request.urlopen") as urlopen:
            lucy = Lucy.from_environment()
            response = lucy.respond("Tell me a short story.")

        self.assertEqual("llama-cpp", lucy.model_backend)
        self.assertIn("llama.cpp local model loaded", lucy.respond("/status"))
        self.assertEqual("The local model is working.", response)
        self.assertEqual(str(model_path), lucy.model.options["model_path"])
        self.assertEqual(4096, lucy.model.options["n_ctx"])
        self.assertEqual([], urlopen.call_args_list)
        self.assertEqual("user", generated[0]["messages"][-1]["role"])
        self.assertEqual("Tell me a short story.", generated[0]["messages"][-1]["content"])

    def test_llama_cpp_backend_reuses_recent_dialogue_for_followups(self):
        model_path = self.root / "local.gguf"
        model_path.write_bytes(b"small test model")
        requests = []

        class FakeLlama:
            def __init__(self, **kwargs):
                pass

            def create_chat_completion(self, **kwargs):
                requests.append(kwargs)
                content = "A fox visited a garden." if len(requests) == 1 else "I told you about a fox."
                return {"choices": [{"message": {"content": content}}]}

        llama_module = types.ModuleType("llama_cpp")
        llama_module.Llama = FakeLlama
        with patch.dict(sys.modules, {"llama_cpp": llama_module}), patch.object(
            Lucy,
            "_available_local_backends",
            return_value={"llama-cpp": True, "gpt4all": False},
        ):
            lucy = Lucy(
                model_path=model_path,
                model_backend="llama-cpp",
                memory_path=self.memory_path,
                knowledge_path=self.knowledge_path,
            )
            first = lucy.respond("Tell me a story about a fox.")
            second = lucy.respond("What did you tell me about?")

        self.assertIn("fox", first)
        self.assertIn("fox", second)
        self.assertEqual("Tell me a story about a fox.", requests[1]["messages"][-3]["content"])
        self.assertEqual("A fox visited a garden.", requests[1]["messages"][-2]["content"])

    def test_llama_cpp_chat_model_can_use_lucys_restricted_calculator_tool(self):
        model_path = self.root / "local.gguf"
        model_path.write_bytes(b"small test model")
        requests = []

        class FakeLlama:
            def __init__(self, **kwargs):
                pass

            def create_chat_completion(self, **kwargs):
                requests.append(kwargs)
                if len(requests) == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        LucyTests._calculate_tool_call(
                                            "local-calc", "6 * 7"
                                        )
                                    ],
                                }
                            }
                        ]
                    }
                return {
                    "choices": [
                        {"message": {"content": "6 times 7 is 42."}}
                    ]
                }

        llama_module = types.ModuleType("llama_cpp")
        llama_module.Llama = FakeLlama
        with patch.dict(sys.modules, {"llama_cpp": llama_module}), patch.object(
            Lucy,
            "_available_local_backends",
            return_value={"llama-cpp": True, "gpt4all": False},
        ), patch("core.ai_engine.urllib.request.urlopen") as urlopen:
            lucy = Lucy(
                model_path=model_path,
                model_backend="llama-cpp",
                memory_path=self.memory_path,
                knowledge_path=self.knowledge_path,
            )
            answer = lucy.respond("What is 6 times 7?")

        self.assertEqual("6 times 7 is 42.", answer)
        self.assertIn("tools", requests[0])
        self.assertEqual("tool", requests[1]["messages"][-1]["role"])
        self.assertEqual("42", requests[1]["messages"][-1]["content"])
        urlopen.assert_not_called()

    def test_macos_prefers_llama_cpp_when_both_local_runtimes_are_available(self):
        model_path = self.root / "local.gguf"
        model_path.write_bytes(b"small test model")

        class FakeLlama:
            def __init__(self, **kwargs):
                self.options = kwargs

        llama_module = types.ModuleType("llama_cpp")
        llama_module.Llama = FakeLlama
        with patch.dict(sys.modules, {"llama_cpp": llama_module}), patch.object(
            Lucy,
            "_available_local_backends",
            return_value={"llama-cpp": True, "gpt4all": True},
        ), patch("core.ai_engine.os.sys.platform", "darwin"):
            lucy = Lucy(
                model_path=model_path,
                memory_path=self.memory_path,
                knowledge_path=self.knowledge_path,
            )

        self.assertEqual("llama-cpp", lucy.model_backend)

    def test_gpt4all_backend_remains_available_for_local_gguf_models(self):
        model_path = self.root / "local.gguf"
        model_path.write_bytes(b"small test model")

        class FakeGPT4All:
            def __init__(self, **kwargs):
                self.options = kwargs

            def generate(self, prompt, max_tokens, temp):
                return "The GPT4All local model is working."

        gpt4all_module = types.ModuleType("gpt4all")
        gpt4all_module.GPT4All = FakeGPT4All
        with patch.dict(sys.modules, {"gpt4all": gpt4all_module}), patch.object(
            Lucy,
            "_available_local_backends",
            return_value={"llama-cpp": False, "gpt4all": True},
        ):
            lucy = Lucy(
                model_path=model_path,
                model_backend="gpt4all",
                memory_path=self.memory_path,
                knowledge_path=self.knowledge_path,
            )
            answer = lucy.respond("Tell me something interesting.")

        self.assertEqual("gpt4all", lucy.model_backend)
        self.assertEqual("The GPT4All local model is working.", answer)
        self.assertIn("GPT4All local model loaded", lucy.respond("/status"))

    def test_explicit_model_backend_requires_model_path(self):
        with self.assertRaisesRegex(ValueError, "requires LUCY_MODEL_PATH"):
            Lucy(
                model_backend="llama-cpp",
                memory_path=self.memory_path,
                knowledge_path=self.knowledge_path,
            )

    def test_api_timeout_environment_setting_is_applied(self):
        environment = {
            "LUCY_MEMORY_PATH": str(self.memory_path),
            "LUCY_API_BASE_URL": "https://api.example.test/v1",
            "LUCY_API_KEY": "",
            "LUCY_API_MODEL": "small-chat-model",
            "LUCY_CODING_API_MODEL": "",
            "LUCY_MODEL_PATH": "",
            "LUCY_API_TIMEOUT": "9.5",
        }
        with patch.dict(os.environ, environment):
            lucy = Lucy.from_environment()

        self.assertEqual(9.5, lucy.api_timeout)
        self.assertIn("9.5s timeout", lucy.respond("/status"))

    def test_api_uses_configured_timeout_and_retries_transient_read_only_failure(self):
        lucy = self.make_api_lucy(api_timeout=7)
        response = self.fake_chat_response(
            {"role": "assistant", "content": "It recovered."}
        )
        timeouts = []

        def fail_then_recover(request, timeout):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise urllib.error.URLError("temporary network failure")
            return response

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fail_then_recover), patch(
            "core.ai_engine.time.sleep"
        ) as sleep:
            message = lucy._api_request(
                [{"role": "user", "content": "hello"}],
                include_tools=False,
            )

        self.assertEqual("It recovered.", message["content"])
        self.assertEqual([7.0, 7.0], timeouts)
        sleep.assert_called_once()

    def test_api_does_not_retry_requests_that_include_tools(self):
        lucy = self.make_api_lucy()
        with patch(
            "core.ai_engine.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ) as urlopen, patch("core.ai_engine.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "Could not reach"):
                lucy._api_request(
                    [{"role": "user", "content": "hello"}],
                    include_tools=True,
                )

        urlopen.assert_called_once()
        sleep.assert_not_called()

    def test_api_rejects_oversized_response(self):
        lucy = self.make_api_lucy()
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b"x" * (4 * 1024 * 1024 + 1)

        with patch("core.ai_engine.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "exceeded Lucy's 4 MB limit"):
                lucy._api_request(
                    [{"role": "user", "content": "hello"}],
                    include_tools=False,
                )

    def test_api_rejects_unsupported_response_structure(self):
        lucy = self.make_api_lucy()
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"choices":[]}'

        with patch("core.ai_engine.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "unsupported response"):
                lucy._api_request(
                    [{"role": "user", "content": "hello"}],
                    include_tools=False,
                )

    def test_corrupt_memory_is_reported_without_overwriting_source(self):
        self.memory_path.write_text("{broken", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "invalid JSON and was left unchanged"):
            Lucy(
                memory_path=self.memory_path,
                knowledge_path=self.knowledge_path,
            )

        self.assertEqual("{broken", self.memory_path.read_text(encoding="utf-8"))

    def test_memory_save_failure_is_reported_and_rolls_back_session_state(self):
        self.lucy.respond("hello")
        previous_history = self.lucy.history.copy()
        self.lucy._save_memory = Mock(side_effect=PermissionError("read-only"))

        response = self.lucy.respond("how are you?")

        self.assertIn("could not save this conversation", response)
        self.assertIn("read-only", response)
        self.assertEqual(previous_history, self.lucy.history)

    def test_failed_clear_keeps_previous_history(self):
        self.lucy.respond("hello")
        previous_history = self.lucy.history.copy()
        self.lucy._save_memory = Mock(side_effect=PermissionError("read-only"))

        response = self.lucy.respond("/clear")

        self.assertIn("couldn't clear the saved conversation", response)
        self.assertEqual(previous_history, self.lucy.history)

    def test_cli_stays_running_after_a_conversation_backend_error(self):
        lucy = Mock()
        lucy.model = None
        lucy.api_base_url = "http://localhost:11434/v1"
        lucy.respond.side_effect = [RuntimeError("temporary outage"), "I recovered."]
        output = io.StringIO()
        errors = io.StringIO()

        with patch("main.Lucy.from_environment", return_value=lucy), patch(
            "builtins.input",
            side_effect=["first message", "try again", "/quit"],
        ), patch.dict(os.environ, {"LUCY_VOICE": ""}), patch(
            "sys.stdout", output
        ), patch("sys.stderr", errors):
            result = lucy_main.main()

        self.assertEqual(0, result)
        self.assertIn("Lucy: I recovered.", output.getvalue())
        self.assertIn("temporary outage", output.getvalue())
        self.assertEqual("", errors.getvalue())
        self.assertEqual(2, lucy.respond.call_count)

    def test_remembered_facts_persist_between_instances(self):
        self.assertEqual("I'll remember that.", self.lucy.respond("/remember I like tea"))
        restarted_lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
        )
        response = restarted_lucy.respond("/recall tea")
        self.assertIn("I like tea", response)

    def test_natural_language_memory_request_is_saved(self):
        self.assertEqual(
            "I'll remember that.",
            self.lucy.respond("Please remember that my favorite color is blue."),
        )
        self.assertEqual(["my favorite color is blue"], self.lucy.facts)

    def test_feedback_persists_as_a_user_preference(self):
        approvals = []
        self.lucy.approval_callback = lambda action: approvals.append(action) or True
        self.assertIn("remember it for future", self.lucy.respond("/feedback keep answers concise"))
        self.assertIn("save this correction", approvals[0])
        restarted_lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
        )
        self.assertIn("User preference: keep answers concise", restarted_lucy.facts)
        self.assertEqual([], restarted_lucy.history)

    def test_feedback_without_approval_is_session_only_and_not_saved_to_history(self):
        lucy = self.make_api_lucy(approval_callback=lambda action: False)

        response = lucy.respond("/feedback correct me gently")

        self.assertIn("this conversation only", response)
        self.assertIn("User preference: correct me gently", lucy.session_lessons)
        self.assertEqual([], lucy.facts)
        self.assertEqual([], lucy.history)
        self.assertNotIn("correct me gently", self.memory_path.read_text(encoding="utf-8"))

    def test_model_receives_session_corrections_and_is_told_to_self_correct(self):
        lucy = self.make_api_lucy(approval_callback=lambda action: False)
        lucy.respond("/feedback explain mistakes briefly")
        request_messages = []

        def fake_urlopen(request, timeout):
            request_messages.extend(json.loads(request.data.decode("utf-8"))["messages"])
            return self.fake_chat_response({"role": "assistant", "content": "Understood."})

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            lucy.respond("What should you do if I correct you?")

        system_message = request_messages[0]["content"]
        self.assertIn("Current-conversation user corrections and preferences", system_message)
        self.assertIn("explain mistakes briefly", system_message)
        self.assertIn("reconsider your answer", system_message)

    def test_clear_only_removes_conversation_history(self):
        self.lucy.respond("/remember I like tea")
        self.lucy.respond("hello")
        self.assertTrue(self.lucy.history)

        response = self.lucy.respond("/clear")

        self.assertIn("cleared", response)
        self.assertEqual([], self.lucy.history)
        self.assertEqual([], self.lucy.session_lessons)
        self.assertEqual(["I like tea"], self.lucy.facts)

    def test_local_knowledge_retrieval(self):
        response = self.lucy.respond("What is Python?")
        self.assertIn("Python is a high-level programming language.", response)
        self.assertIn("LUCY_MODEL_PATH", response)

    def test_coding_reference_is_available_in_knowledge_mode(self):
        topics = self.lucy.respond("/coding")
        self.assertIn("Python programming", topics)
        self.assertIn("Rust programming", topics)
        self.assertIn("Application security basics", topics)

        python_reference = self.lucy.respond("/coding Python")
        self.assertIn("Python programming", python_reference)
        self.assertIn("Never use eval on untrusted input", python_reference)

    def test_coding_questions_prioritize_programming_reference(self):
        response = self.lucy.respond("How should I write Python code?")
        self.assertIn("Python programming", response)
        self.assertIn("project-specific coding help", response)

    def test_configured_model_receives_relevant_coding_reference_and_guidance(self):
        lucy = self.make_api_lucy()
        request_messages = []

        def fake_urlopen(request, timeout):
            request_messages.extend(json.loads(request.data.decode("utf-8"))["messages"])
            return self.fake_chat_response(
                {"role": "assistant", "content": "Use small, tested functions."}
            )

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            lucy.respond("How should I write Python code?")

        system_message = request_messages[0]["content"]
        self.assertIn("Python programming", system_message)
        self.assertIn("smallest complete change", system_message)

    def test_coding_questions_use_sub_600_mb_specialist_without_tools(self):
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="http://localhost:11434/v1",
            api_model="qwen3:0.6b",
            coding_api_model="qwen2.5-coder:0.5b",
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return self.fake_chat_response(
                {"role": "assistant", "content": "A function is defined with def."}
            )

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            answer = lucy.respond("Explain how to write a Python function.")

        self.assertIn("function", answer)
        self.assertEqual("qwen2.5-coder:0.5b", requests[0]["model"])
        self.assertNotIn("tools", requests[0])
        self.assertEqual(0.3, requests[0]["temperature"])

    def test_project_coding_keeps_tool_capable_qwen3_model_and_thinks(self):
        project = self.root / "project"
        project.mkdir()
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="http://localhost:11434/v1",
            api_model="qwen3:0.6b",
            coding_api_model="qwen2.5-coder:0.5b",
            code_directory=project,
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return self.fake_chat_response(
                {"role": "assistant", "content": "I can help review your Python code."}
            )

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            lucy.respond("Explain how to write a Python function.")

        self.assertEqual("qwen3:0.6b", requests[0]["model"])
        self.assertIn("tools", requests[0])
        self.assertIs(requests[0]["think"], True)

    def test_file_analysis_without_workspace_explains_required_setup(self):
        lucy = self.make_api_lucy()

        with patch.object(lucy, "_api_response") as api_response:
            answer = lucy.respond("analayze your main file")

        self.assertIn("coding workspace is disabled", answer)
        self.assertIn("LUCY_CODE_DIR", answer)
        self.assertNotIn("<search_learned_files", answer)
        api_response.assert_not_called()

    def test_file_analysis_reads_only_requested_file_and_uses_model_without_tools(self):
        project = self.root / "project"
        project.mkdir()
        (project / "reasoning.py").write_text(
            "def answer(question):\n    return question.strip()\n",
            encoding="utf-8",
        )
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="http://localhost:11434/v1",
            api_model="qwen3:0.6b",
            coding_api_model="qwen2.5-coder:0.5b",
            code_directory=project,
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return self.fake_chat_response(
                {"role": "assistant", "content": "The function strips whitespace from its input."}
            )

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            answer = lucy.respond("analayze reasoning.py")

        self.assertIn("strips whitespace", answer)
        self.assertEqual("qwen3:0.6b", requests[0]["model"])
        self.assertNotIn("tools", requests[0])
        self.assertEqual(0.1, requests[0]["temperature"])
        self.assertIs(requests[0]["think"], False)
        self.assertIn(
            "def answer(question)",
            requests[0]["messages"][-1]["content"],
        )
        self.assertNotIn("BEGIN FILE", requests[0]["messages"][0]["content"])

    def test_python_file_analysis_reports_verified_structure_after_generic_model_reply(self):
        project = self.root / "project"
        project.mkdir()
        (project / "main.py").write_text(
            "import os\n"
            "import sys\n"
            "from core.ai_engine import Lucy\n\n"
            "def confirm_computer_action(action):\n"
            "    return bool(action)\n\n"
            "def main():\n"
            "    return Lucy()\n\n"
            "if __name__ == \"__main__\":\n"
            "    sys.exit(main())\n",
            encoding="utf-8",
        )
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="http://localhost:11434/v1",
            api_model="qwen3:0.6b",
            code_directory=project,
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return self.fake_chat_response(
                {
                    "role": "assistant",
                    "content": "The file initializes the application and handles user interaction.",
                }
            )

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            answer = lucy.respond("analyze main.py")

        self.assertIn("Top-level definitions: `confirm_computer_action`", answer)
        self.assertIn("`main`", answer)
        self.assertIn("calls `main()` when run directly", answer)
        self.assertIn("Verified Python structure", requests[0]["messages"][-1]["content"])
        self.assertNotIn("tools", requests[0])

    def test_python_file_analysis_keeps_static_outline_if_model_returns_no_text(self):
        project = self.root / "project"
        project.mkdir()
        (project / "main.py").write_text(
            "def main():\n    pass\n\n"
            "if __name__ == \"__main__\":\n    main()\n",
            encoding="utf-8",
        )
        lucy = self.make_api_lucy(code_directory=project)

        with patch.object(
            lucy,
            "_api_response",
            side_effect=RuntimeError("The configured chat API returned an empty response."),
        ):
            answer = lucy.respond("analyze main.py")

        self.assertIn("The model returned no analysis text", answer)
        self.assertIn("Top-level definitions: `main`", answer)
        self.assertIn("calls `main()` when run directly", answer)

    def test_analysis_followup_filename_completes_pending_request(self):
        project = self.root / "project"
        project.mkdir()
        (project / "reasoning.py").write_text(
            "def answer(question):\n    return question.strip()\n",
            encoding="utf-8",
        )
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="http://localhost:11434/v1",
            api_model="qwen3:0.6b",
            coding_api_model="qwen2.5-coder:0.5b",
            code_directory=project,
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return self.fake_chat_response(
                {"role": "assistant", "content": "The file defines a small text-normalizing function."}
            )

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            prompt = lucy.respond("analyze your main file")
            answer = lucy.respond("reasoning.py")

        self.assertIn("relative file path", prompt)
        self.assertIn("text-normalizing", answer)
        self.assertEqual(1, len(requests))
        self.assertNotIn("tools", requests[0])

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

    @staticmethod
    def fake_chat_response(message):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(
            {"choices": [{"message": message}]}
        ).encode("utf-8")
        return response

    def make_api_lucy(self, **kwargs):
        return Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="http://localhost:11434/v1",
            api_model="local-model",
            **kwargs,
        )

    def test_openai_compatible_api_runs_safe_tool_and_returns_answer(self):
        lucy = self.make_api_lucy(api_key="local-test-key")
        tool_call = {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": json.dumps({"expression": "2 + 3 * 4"}),
            },
        }
        responses = [
            self.fake_chat_response({"role": "assistant", "tool_calls": [tool_call]}),
            self.fake_chat_response({"role": "assistant", "content": "The answer is 14."}),
        ]
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            self.assertEqual(120, timeout)
            return responses.pop(0)

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            answer = lucy.respond("What is 2 + 3 * 4?")

        self.assertEqual("The answer is 14.", answer)
        self.assertEqual(
            "Bearer local-test-key",
            dict((key.lower(), value) for key, value in requests[0].header_items())[
                "authorization"
            ],
        )
        follow_up = json.loads(requests[1].data.decode("utf-8"))
        self.assertEqual("tool", follow_up["messages"][-1]["role"])
        self.assertEqual("14", follow_up["messages"][-1]["content"])

    def test_local_ollama_qwen3_disables_thinking(self):
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="http://localhost:11434/v1",
            api_model="qwen3:0.6b",
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return self.fake_chat_response(
                {"role": "assistant", "content": "Hello! How can I help?"}
            )

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
                lucy.respond("Tell me a short joke.")

        self.assertIs(requests[0]["think"], False)

    def test_non_ollama_api_does_not_receive_ollama_thinking_option(self):
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            api_base_url="https://api.example.test/v1",
            api_model="qwen3:0.6b",
        )
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return self.fake_chat_response(
                {"role": "assistant", "content": "Hello! How can I help?"}
            )

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
                lucy.respond("Tell me a short joke.")

        self.assertNotIn("think", requests[0])

    def test_model_cannot_save_memory_without_explicit_request(self):
        lucy = self.make_api_lucy()
        tool_call = {
            "id": "call-memory",
            "type": "function",
            "function": {
                "name": "remember_fact",
                "arguments": json.dumps({"fact": "Lucy should retain this"}),
            },
        }
        responses = [
            self.fake_chat_response({"role": "assistant", "tool_calls": [tool_call]}),
            self.fake_chat_response({"role": "assistant", "content": "I answered your question."}),
        ]
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            return responses.pop(0)

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            lucy.respond("What is 2 + 2?")

        self.assertEqual([], lucy.facts)
        tool_result = json.loads(requests[1].data.decode("utf-8"))["messages"][-1]
        self.assertIn("did not explicitly ask", tool_result["content"])

    def test_api_conversation_includes_previous_turns(self):
        lucy = self.make_api_lucy()
        responses = [
            self.fake_chat_response({"role": "assistant", "content": "Python is a language."}),
            self.fake_chat_response({"role": "assistant", "content": "It is dynamically typed."}),
        ]
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            return responses.pop(0)

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            lucy.respond("Explain Python.")
            lucy.respond("How is it typed?")

        second_turn = json.loads(requests[1].data.decode("utf-8"))["messages"]
        self.assertIn("conversational assistant", second_turn[0]["content"])
        self.assertIn("use prior turns to understand follow-up questions", second_turn[0]["content"])
        self.assertEqual("Explain Python.", second_turn[-3]["content"])
        self.assertEqual("Python is a language.", second_turn[-2]["content"])

    @staticmethod
    def _calculate_tool_call(call_id, expression):
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": json.dumps({"expression": expression}),
            },
        }

    def test_repeated_identical_tool_request_is_stopped_before_second_execution(self):
        lucy = self.make_api_lucy()
        repeated_call = self._calculate_tool_call("call-1", "2 + 2")
        responses = [
            self.fake_chat_response({"role": "assistant", "tool_calls": [repeated_call]}),
            self.fake_chat_response(
                {
                    "role": "assistant",
                    "tool_calls": [self._calculate_tool_call("call-2", "2 + 2")],
                }
            ),
        ]

        with patch(
            "core.ai_engine.urllib.request.urlopen",
            side_effect=lambda request, timeout: responses.pop(0),
        ), patch.object(lucy, "_run_tool", return_value="4") as run_tool:
            with self.assertRaisesRegex(RuntimeError, "repeated the 'calculate' tool request"):
                lucy._api_response("What is 2 + 2?")

        run_tool.assert_called_once_with("calculate", {"expression": "2 + 2"})

    def test_excessive_tool_batch_is_rejected_before_any_tool_runs(self):
        lucy = self.make_api_lucy()
        calls = [
            self._calculate_tool_call(f"call-{index}", str(index))
            for index in range(7)
        ]
        response = self.fake_chat_response({"role": "assistant", "tool_calls": calls})

        with patch("core.ai_engine.urllib.request.urlopen", return_value=response), patch.object(
            lucy, "_run_tool"
        ) as run_tool:
            with self.assertRaisesRegex(RuntimeError, "per-turn limit of 6"):
                lucy._api_response("Calculate several values.")

        run_tool.assert_not_called()

    def test_model_can_use_bounded_tool_rounds_and_return_final_response(self):
        lucy = self.make_api_lucy()
        responses = [
            self.fake_chat_response(
                {
                    "role": "assistant",
                    "tool_calls": [
                        self._calculate_tool_call(f"call-{index}", str(index))
                    ],
                }
            )
            for index in range(5)
        ]
        responses.append(
            self.fake_chat_response(
                {"role": "assistant", "content": "I calculated all five values."}
            )
        )

        with patch(
            "core.ai_engine.urllib.request.urlopen",
            side_effect=lambda request, timeout: responses.pop(0),
        ) as urlopen:
            answer = lucy._api_response("Calculate five values.")

        self.assertEqual("I calculated all five values.", answer)
        self.assertEqual(6, urlopen.call_count)

    def test_tool_loop_stops_before_executing_a_sixth_tool_round(self):
        lucy = self.make_api_lucy()
        responses = [
            self.fake_chat_response(
                {
                    "role": "assistant",
                    "tool_calls": [
                        self._calculate_tool_call(f"call-{index}", str(index))
                    ],
                }
            )
            for index in range(6)
        ]

        with patch(
            "core.ai_engine.urllib.request.urlopen",
            side_effect=lambda request, timeout: responses.pop(0),
        ), patch.object(lucy, "_run_tool", return_value="calculated") as run_tool:
            with self.assertRaisesRegex(RuntimeError, "within Lucy's 5-round tool limit"):
                lucy._api_response("Calculate values.")

        self.assertEqual(5, run_tool.call_count)

    def test_reflection_requires_language_model_and_conversation(self):
        self.assertIn("no conversation", self.lucy.respond("/reflect"))
        self.lucy.respond("hello")
        self.assertIn("needs a configured language model", self.lucy.respond("/reflect"))

    def test_reflection_proposes_lessons_and_saves_only_approved_items(self):
        approvals = []
        lucy = self.make_api_lucy(
            approval_callback=lambda action: approvals.append(action) or "concise" in action
        )
        conversation = [
            {"role": "user", "content": "Please keep answers concise."},
            {"role": "assistant", "content": "Understood."},
        ]
        lucy.history = list(conversation)
        responses = [
            self.fake_chat_response(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "lessons": [
                                "Keep answers concise.",
                                "Use examples when explaining code.",
                            ]
                        }
                    ),
                }
            )
        ]
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return responses.pop(0)

        with patch("core.ai_engine.urllib.request.urlopen", side_effect=fake_urlopen):
            result = lucy.respond("/reflect")

        self.assertIn("Saved approved lessons", result)
        self.assertIn("Not saved", result)
        self.assertEqual(["User preference: Keep answers concise."], lucy.facts)
        self.assertEqual(conversation, lucy.history)
        self.assertEqual(2, len(approvals))
        self.assertNotIn("tools", requests[0])
        self.assertNotIn("tool_choice", requests[0])

    def test_reflection_does_not_save_when_approval_is_unavailable(self):
        lucy = self.make_api_lucy()
        lucy.history = [{"role": "user", "content": "I like short answers."}]
        response = self.fake_chat_response(
            {
                "role": "assistant",
                "content": json.dumps({"lessons": ["Prefer short answers."]}),
            }
        )

        with patch("core.ai_engine.urllib.request.urlopen", return_value=response):
            result = lucy.respond("/reflect")

        self.assertIn("approval was declined or unavailable", result)
        self.assertEqual([], lucy.facts)

    def test_reflection_rejects_invalid_model_output(self):
        lucy = self.make_api_lucy()
        lucy.history = [{"role": "user", "content": "Please explain gently."}]
        response = self.fake_chat_response({"role": "assistant", "content": "not json"})

        with patch("core.ai_engine.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "invalid reflection JSON"):
                lucy.respond("/reflect")

    def test_explicit_natural_memory_request_allows_memory_tool(self):
        self.assertTrue(Lucy._user_requested_memory("Please remember my name is Ada."))
        self.assertTrue(Lucy._user_requested_memory("Can you remember that I like tea?"))
        self.assertFalse(Lucy._user_requested_memory("What do you remember about me?"))

    def test_learning_is_limited_to_selected_nonhidden_text_files(self):
        source = self.root / "learning"
        source.mkdir()
        (source / "notes.md").write_text("Lucy can learn this note.", encoding="utf-8")
        (source / ".hidden.txt").write_text("Do not show.", encoding="utf-8")
        (source / "credentials.txt").write_text("Do not read.", encoding="utf-8")
        (source / "image.png").write_bytes(b"not text")
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            learn_directory=source,
            learned_path=self.root / "learned.json",
        )

        self.assertEqual(["notes.md"], lucy.computer.list_files())
        self.assertIn("learned notes.md", lucy.respond("/learn notes.md"))
        self.assertIn("learn this note", lucy.respond("/learned notes"))

    def test_learning_rejects_traversal_and_symlinks(self):
        source = self.root / "learning"
        source.mkdir()
        outside = self.root / "private.txt"
        outside.write_text("outside", encoding="utf-8")
        access = ComputerAccess(source, self.root / "learned.json")

        with self.assertRaises(ValueError):
            access.learn_file("../private.txt")
        with self.assertRaises(ValueError):
            access.learn_file(str(outside))
        link = source / "linked.txt"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            access.learn_file("linked.txt")

    def test_failed_learned_library_save_rolls_back_in_memory_changes(self):
        source = self.root / "learning"
        source.mkdir()
        (source / "notes.md").write_text("A durable note.", encoding="utf-8")
        access = ComputerAccess(
            root_directory=source,
            library_path=self.root / "learned.json",
        )

        with patch.object(access, "_save_library", side_effect=OSError("read-only")):
            with self.assertRaisesRegex(OSError, "read-only"):
                access.learn_file("notes.md")
        self.assertNotIn("notes.md", access.documents)

        access.documents["notes.md"] = {"text": "saved note", "digest": "digest"}
        with patch.object(access, "_save_library", side_effect=OSError("read-only")):
            with self.assertRaisesRegex(OSError, "read-only"):
                access.forget_file("notes.md")
        self.assertIn("notes.md", access.documents)

    def test_model_file_learning_requires_an_explicit_user_request(self):
        source = self.root / "learning"
        source.mkdir()
        (source / "notes.txt").write_text("approved notes", encoding="utf-8")
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            learn_directory=source,
            learned_path=self.root / "learned.json",
        )

        denied = lucy._run_tool("learn_user_file", {"path": "notes.txt"})
        self.assertIn("explicitly asks", denied)
        lucy._current_user_request = "Please learn notes.txt."
        allowed = lucy._run_tool("learn_user_file", {"path": "notes.txt"})
        self.assertIn("learned notes.txt", allowed)

    def test_file_learning_requires_configured_folder(self):
        with self.assertRaises(RuntimeError):
            self.lucy.computer.list_files()
        self.assertIn("disabled", self.lucy._list_user_files())

    def test_opening_an_app_requires_approval_and_allowlist(self):
        with patch("core.computer_access.platform.system", return_value="Darwin"), patch(
            "core.computer_access.shutil.which", return_value="/usr/bin/open"
        ), patch("core.computer_access.subprocess.run") as run:
            denied = ComputerAccess.open_app("Notes", lambda action: False)
            run.assert_not_called()
            allowed = ComputerAccess.open_app("Notes", lambda action: True)

        self.assertIn("not open", denied)
        self.assertEqual("Opened Notes.", allowed)
        run.assert_called_once_with(
            ["/usr/bin/open", "-a", "Notes"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn(
            "approved app list",
            ComputerAccess.open_app("Terminal", lambda action: True),
        )

    def test_opening_url_requires_approval_and_rejects_private_addresses(self):
        with patch("core.computer_access.platform.system", return_value="Darwin"), patch(
            "core.computer_access.shutil.which", return_value="/usr/bin/open"
        ), patch("core.computer_access.subprocess.run") as run:
            denied = ComputerAccess.open_url("https://example.com", lambda action: False)
            run.assert_not_called()
            opened = ComputerAccess.open_url("https://example.com", lambda action: True)

        self.assertIn("not open", denied)
        self.assertEqual("Opened https://example.com in your browser.", opened)
        with self.assertRaises(ValueError):
            ComputerAccess.open_url("http://127.0.0.1", lambda action: True)

    def test_web_search_extracts_public_result_links(self):
        html_page = (
            '<a class="result__a" href="https://example.com/article">Example Article</a>'
            '<a class="result__a" href="javascript:alert(1)">Bad result</a>'
        ).encode("utf-8")
        with patch.object(
            ComputerAccess,
            "_fetch_public_url",
            return_value=("https://html.duckduckgo.com/html/", "text/html", html_page),
        ):
            results = ComputerAccess.search_web("Lucy test")

        self.assertEqual(
            [{"title": "Example Article", "url": "https://example.com/article"}],
            results,
        )

    def test_web_page_reader_strips_active_content(self):
        page = (
            "<html><script>secret()</script><h1>Useful</h1><p>Public page text.</p></html>"
        ).encode("utf-8")
        with patch.object(
            ComputerAccess,
            "_fetch_public_url",
            return_value=("https://example.com", "text/html", page),
        ):
            result = ComputerAccess.read_web_page("https://example.com")
        self.assertIn("Useful", result)
        self.assertIn("Public page text.", result)
        self.assertNotIn("secret()", result)

    def test_public_web_access_blocks_local_hosts_and_nonstandard_ports(self):
        for url in (
            "http://127.0.0.1/",
            "http://localhost/",
            "http://192.168.1.1/",
            "https://example.com:8443/",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ComputerAccess._validate_public_url(url, resolve_host=False)
        with patch(
            "core.computer_access.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("10.0.0.5", 0))],
        ), self.assertRaises(ValueError):
            ComputerAccess._validate_public_url("https://example.com/")

    def test_coding_workspace_supports_multiple_languages_and_requires_approval(self):
        project = self.root / "project"
        project.mkdir()
        (project / "src").mkdir()
        (project / "src" / "app.py").write_text("print('old')\n", encoding="utf-8")
        workspace = CodingWorkspace(project)
        previews = []

        def approve(preview):
            previews.append(preview)
            return True

        result = workspace.preview_and_write(
            "src/app.py",
            "print('new')\n",
            approve,
        )
        self.assertIn("Applied your approved change", result)
        self.assertIn("-print('old')", previews[0])
        self.assertEqual("print('new')\n", workspace.read_file("src/app.py"))

        result = workspace.preview_and_write(
            "src/App.swift",
            "print(\"hello\")\n",
            approve,
        )
        self.assertIn("Applied your approved change", result)
        self.assertEqual('print("hello")\n', workspace.read_file("src/App.swift"))

    def test_coding_workspace_denies_writes_without_approval(self):
        project = self.root / "project"
        project.mkdir()
        workspace = CodingWorkspace(project)

        result = workspace.preview_and_write("new.py", "print('no')\n")

        self.assertIn("was not applied", result)
        self.assertFalse((project / "new.py").exists())

    def test_coding_workspace_rejects_secrets_traversal_and_symlinks(self):
        project = self.root / "project"
        project.mkdir()
        (project / "app.py").write_text("pass\n", encoding="utf-8")
        outside = self.root / "outside.py"
        outside.write_text("secret\n", encoding="utf-8")
        (project / "linked.py").symlink_to(outside)
        workspace = CodingWorkspace(project)

        for path in ("../outside.py", ".env", "api_key.py", "linked.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                workspace.read_file(path)

    def test_code_removal_is_approved_and_recoverable(self):
        project = self.root / "project"
        project.mkdir()
        original = project / "old.py"
        original.write_text("print('keep me')\n", encoding="utf-8")
        workspace = CodingWorkspace(project)
        previews = []

        denied = workspace.delete_file("old.py", lambda preview: False)
        self.assertIn("not approved", denied)
        self.assertTrue(original.exists())

        removed = workspace.delete_file(
            "old.py",
            lambda preview: previews.append(preview) or True,
        )
        self.assertIn("recoverable trash", removed)
        self.assertFalse(original.exists())
        self.assertIn("-print('keep me')", previews[0])

        restored = workspace.restore_file("old.py", lambda preview: True)
        self.assertIn("Restored", restored)
        self.assertEqual("print('keep me')\n", original.read_text(encoding="utf-8"))

    def test_python_check_does_not_execute_code_and_reports_syntax(self):
        project = self.root / "project"
        project.mkdir()
        marker = self.root / "executed"
        (project / "app.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
            encoding="utf-8",
        )
        workspace = CodingWorkspace(project)
        self.assertIn("syntax is valid", workspace.check_python_syntax("app.py"))
        self.assertFalse(marker.exists())
        (project / "broken.py").write_text("if:\n", encoding="utf-8")
        self.assertIn("syntax error", workspace.check_python_syntax("broken.py"))

    def test_model_project_change_requires_explicit_request_and_approval(self):
        project = self.root / "project"
        project.mkdir()
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            code_directory=project,
        )
        denied = lucy._run_tool(
            "write_project_file",
            {"path": "app.js", "content": "console.log('hello');\n"},
        )
        self.assertIn("explicitly asks", denied)

        lucy._current_user_request = "Add a JavaScript file."
        changed = lucy._run_tool(
            "write_project_file",
            {"path": "app.js", "content": "console.log('hello');\n"},
        )
        self.assertIn("was not applied", changed)
        self.assertFalse((project / "app.js").exists())

        lucy.approval_callback = lambda preview: True
        changed = lucy._run_tool(
            "write_project_file",
            {"path": "app.js", "content": "console.log('hello');\n"},
        )
        self.assertIn("Applied your approved change", changed)
        self.assertTrue((project / "app.js").exists())

    def test_project_file_reads_require_user_request(self):
        project = self.root / "project"
        project.mkdir()
        (project / "app.py").write_text("print('hello')\n", encoding="utf-8")
        lucy = Lucy(
            memory_path=self.memory_path,
            knowledge_path=self.knowledge_path,
            code_directory=project,
        )

        denied = lucy._run_tool("read_project_file", {"path": "app.py"})
        self.assertIn("explicitly asks", denied)
        lucy._current_user_request = "Please review app.py."
        self.assertIn("print('hello')", lucy._run_tool("read_project_file", {"path": "app.py"}))

    def test_explicit_coding_and_removal_requests_are_recognized(self):
        self.assertTrue(Lucy._user_requested_coding("Add a Swift example."))
        self.assertTrue(Lucy._user_requested_coding("Could you update the JavaScript app?"))
        self.assertTrue(Lucy._user_requested_removal("Please delete the old module."))
        self.assertFalse(Lucy._user_requested_removal("Could you review the old module?"))

    def test_optional_voice_output_uses_platform_speech_engine(self):
        engine = Mock()
        speech_module = Mock(init=Mock(return_value=engine))
        with patch("core.audio.platform.system", return_value="Linux"), patch.dict(
            sys.modules, {"pyttsx3": speech_module}
        ):
            voice = VoiceOutput()
            voice.speak("Hello")
        engine.setProperty.assert_called_once_with("rate", 175)
        engine.say.assert_called_once_with("Hello")
        engine.runAndWait.assert_called_once_with()

    def test_macos_voice_uses_native_say_command(self):
        with patch("core.audio.platform.system", return_value="Darwin"), patch(
            "core.audio.shutil.which", return_value="/usr/bin/say"
        ), patch("core.audio.subprocess.run") as run, patch.dict(
            sys.modules, {"pyttsx3": None}
        ):
            voice = VoiceOutput()
            voice.speak("Hello from Lucy")
        run.assert_called_once_with(
            ["/usr/bin/say", "Hello from Lucy"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_configured_missing_model_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            Lucy(
                model_path=self.root / "missing.gguf",
                memory_path=self.memory_path,
                knowledge_path=self.knowledge_path,
            )

    def test_gpt4all_model_larger_than_nine_hundred_mb_is_rejected(self):
        model_path = self.root / "oversized.gguf"
        with model_path.open("wb") as model_file:
            model_file.truncate(900_000_001)

        with self.assertRaisesRegex(ValueError, "exceeds Lucy's 900 MB model limit"):
            Lucy(
                model_path=model_path,
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
