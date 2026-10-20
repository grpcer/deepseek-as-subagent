from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_mcp import agent_loop, trace_log
from deepseek_mcp.config import Config
from deepseek_mcp.provider_response import ProviderResponse


def _response(*, reasoning: str, content: str, tool_call: dict | None = None):
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if tool_call is not None:
        message["tool_calls"] = [{"id": "call-1", "type": "function", "function": tool_call}]
    return ProviderResponse.from_payload({
        "choices": [{
            "finish_reason": "tool_calls" if tool_call else "stop",
            "message": message,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


@unittest.skipIf(os.name == "nt", "private trace logging is POSIX-only")
class TraceLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir(mode=0o700)
        self.workspace = self.root / "project"
        self.workspace.mkdir()
        self.trace_file = self.log_dir / "trace.log"
        for patcher in (
            patch.object(trace_log, "trace_path", return_value=self.trace_file),
            patch.object(trace_log, "_stream", None),
            patch.object(trace_log, "_unavailable", False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._close_stream)
        self.addCleanup(self._tmp.cleanup)

    def _close_stream(self) -> None:
        if trace_log._stream is not None:
            trace_log._stream.close()

    def _config(self, **overrides) -> Config:
        return Config("sk-test", self.workspace, allowed_tools=["Read", "Bash"], **overrides)

    def _run(self, config: Config) -> dict:
        responses = iter([
            _response(
                reasoning="I should inspect the docs first.",
                content="",
                tool_call={"name": "Read", "arguments": json.dumps({"path": "doc.txt", "offset": 3})},
            ),
            _response(reasoning="Enough information now.", content="summary ready"),
        ])
        with (
            patch.object(agent_loop, "_call_with_retry", side_effect=lambda *a, **k: next(responses)),
            patch.object(agent_loop, "execute_in_subprocess", return_value="SECRET FILE BODY"),
        ):
            return agent_loop.run_agent("summarize docs", config)

    def test_trace_is_disabled_by_default(self) -> None:
        config = self._config()
        self.assertFalse(config.trace)
        self._run(config)
        self.assertFalse(self.trace_file.exists())

    def test_enabled_trace_records_reasoning_and_tool_summary(self) -> None:
        result = self._run(self._config(trace=True))

        text = self.trace_file.read_text(encoding="utf-8")
        self.assertEqual(stat.S_IMODE(self.trace_file.stat().st_mode), 0o600)
        self.assertIn("=== run start", text)
        self.assertIn("summarize docs", text)
        self.assertIn("I should inspect the docs first.", text)
        self.assertIn("→ Read path=doc.txt offset=3 [ok, 16 chars]", text)
        self.assertIn("summary ready", text)
        self.assertIn(f"=== run end · done · turns={result['turns_used']}", text)
        # Tool results carry file contents and are reduced to their size.
        self.assertNotIn("SECRET FILE BODY", text)

    def test_tool_summary_excludes_file_contents_and_edit_payloads(self) -> None:
        cases = {
            "Write": {"path": "a.py", "content": "PAYLOAD"},
            "Edit": {"path": "a.py", "old_string": "PAYLOAD", "new_string": "PAYLOAD"},
            "NotebookEdit": {"path": "n.ipynb", "new_source": "PAYLOAD"},
        }
        for tool, args in cases.items():
            with self.subTest(tool=tool):
                summary = trace_log.describe_tool_args(tool, args)
                self.assertIn("path=", summary)
                self.assertNotIn("PAYLOAD", summary)

    def test_bash_and_search_arguments_are_truncated_previews(self) -> None:
        summary = trace_log.describe_tool_args("Bash", {"command": "echo " + "x" * 1000})
        self.assertLess(len(summary), trace_log.MAX_ARG_CHARS + 40)
        self.assertIn("chars]", summary)
        summary = trace_log.describe_tool_args("Grep", {"pattern": "TODO", "glob": "*.py"})
        self.assertEqual(summary, "pattern=TODO glob=*.py")

    def test_unwritable_trace_does_not_fail_the_run(self) -> None:
        os.chmod(self.log_dir, 0o755)  # not private: the stream refuses to open
        result = self._run(self._config(trace=True))
        self.assertEqual(result["final_message"], "summary ready")
        self.assertTrue(trace_log._unavailable)

    def test_trace_setting_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "trace must be true or false"):
            self._config(trace="yes")

    def test_config_file_trace_key_is_loaded(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_WORKSPACE": str(self.workspace)}):
            self.assertTrue(Config._from_data({"trace": True}, "sk-test").trace)
            self.assertFalse(Config._from_data({}, "sk-test").trace)
            with self.assertRaisesRegex(RuntimeError, "trace must be true or false"):
                Config._from_data({"trace": 1}, "sk-test")


if __name__ == "__main__":
    unittest.main()
