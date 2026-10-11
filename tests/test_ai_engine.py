import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.audio import VoiceOutput
from core.coding_workspace import CodingWorkspace
from core.computer_access import ComputerAccess
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
        self.assertEqual("Explain Python.", second_turn[-3]["content"])
        self.assertEqual("Python is a language.", second_turn[-2]["content"])

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
