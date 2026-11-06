# pyright: reportAttributeAccessIssue=false
"""First-run setup tests — no network, no real downloader, no real terms.

    python -m unittest discover -s tests -v

What is pinned here is the promise, not the download: nothing is fetched without an authorisation
that can be pointed at, the terms are shown before anything happens, and the record says which
channel the acceptance came through.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bootstrap  # noqa: E402
import hermes_voicevox as vv  # noqa: E402
from hermes_voicevox import setup  # noqa: E402

TERMS_SAMPLE = """# VOICEVOX 音声モデル 利用規約
## 許諾内容
1. 商用・非商用問わず利用することができます
## 禁止事項
- 逆コンパイル・リバースエンジニアリング
"""


def make_data(core_dir: str, models: tuple[str, ...] = ("0.vvm",)) -> None:
    for path in ("onnxruntime/lib", "dict/open_jtalk_dic_utf_8-1.11", "models/vvms"):
        os.makedirs(os.path.join(core_dir, path), exist_ok=True)
    for name, text in (("models/TERMS.txt", TERMS_SAMPLE),
                       ("onnxruntime/TERMS.txt", "VOICEVOX ONNX Runtime 利用規約\n")):
        with open(os.path.join(core_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    for name in models:
        open(os.path.join(core_dir, "models", "vvms", name), "wb").close()


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="voicevox-setup-")
        self.core = os.path.join(self.dir, "core")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def runner(self, *, succeed: bool = True, seen: list | None = None):
        """Fake downloader: `n` answers with the terms, `y` fetches (or fails)."""
        def _run(binary, core_dir, answer):
            if seen is not None:
                seen.append(answer)
            if answer == "n":
                return subprocess.CompletedProcess([], 1, TERMS_SAMPLE, "")
            if succeed:
                make_data(core_dir)
                return subprocess.CompletedProcess([], 0, "ok", "")
            return subprocess.CompletedProcess([], 1, "", "disk full")
        return _run


class TestStatus(Base):
    def test_reports_what_is_missing(self):
        state = setup.status(self.core)
        self.assertFalse(state["data_present"])
        self.assertIn("models", state["missing"])
        self.assertIsNone(state["consent"])

    def test_reports_consent_when_present(self):
        make_data(self.core)
        with open(os.path.join(self.core, ".consent.json"), "w", encoding="utf-8") as fh:
            json.dump({"accepted_by": "yuki-js", "terms_text_sha256": "abc"}, fh)
        state = setup.status(self.core)
        self.assertTrue(state["data_present"])
        self.assertEqual(state["consent"]["accepted_by"], "yuki-js")


class TestDownloadAuthorisation(Base):
    def test_refuses_without_any_authorisation(self):
        with self.assertRaises(setup.SetupRefused):
            setup.download(self.core, approved_by="", approval_ref="")

    def test_refuses_half_an_authorisation(self):
        for kwargs in ({"approved_by": "yuki-js", "approval_ref": ""},
                       {"approved_by": "", "approval_ref": "chat"}):
            with self.assertRaises(setup.SetupRefused):
                setup.download(self.core, **kwargs)

    def test_shows_terms_before_accepting(self):
        seen: list[str] = []
        with mock.patch.object(setup, "run_downloader", self.runner(seen=seen)), \
             mock.patch.object(setup, "ensure_downloader", lambda *a, **k: "dl"):
            setup.download(self.core, approved_by="yuki-js",
                           approval_ref="chat 2026-11-06: 'go'")
        self.assertEqual(seen, ["n", "y"], "the terms run must come first")

    def test_records_the_authorisation_and_the_terms_digest(self):
        with mock.patch.object(setup, "run_downloader", self.runner()), \
             mock.patch.object(setup, "ensure_downloader", lambda *a, **k: "dl"):
            result = setup.download(self.core, approved_by="yuki-js",
                                    approval_ref="chat 2026-11-06: 'go'")
        self.assertEqual(result["models"], ["0.vvm"])
        self.assertEqual(len(result["terms_sha256"]), 64)
        with open(os.path.join(self.core, ".consent.json"), encoding="utf-8") as fh:
            record = json.load(fh)
        self.assertEqual(record["accepted_by"], "yuki-js")
        self.assertIn("go", record["approval_ref"])
        self.assertIn("conversation", record["consent_channel"])
        self.assertIn("models_terms", record["assets"])

    def test_failed_download_raises_and_records_nothing(self):
        with mock.patch.object(setup, "run_downloader", self.runner(succeed=False)), \
             mock.patch.object(setup, "ensure_downloader", lambda *a, **k: "dl"):
            with self.assertRaises(RuntimeError):
                setup.download(self.core, approved_by="yuki-js", approval_ref="chat")
        self.assertFalse(os.path.exists(os.path.join(self.core, ".consent.json")))


class TestShowTerms(Base):
    def test_answers_no_and_downloads_nothing(self):
        seen: list[str] = []
        with mock.patch.object(setup, "run_downloader", self.runner(seen=seen)):
            text = setup.show_terms("dl", self.core)
        self.assertEqual(seen, ["n"])
        self.assertIn("利用規約", text)
        self.assertFalse(os.path.exists(os.path.join(self.core, ".consent.json")))


class TestCli(Base):
    def test_no_arguments_reports_missing_data(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(bootstrap.main(["--core", self.core]), 2)
        self.assertIn("--show-terms", buffer.getvalue())

    def test_accept_without_authorisation_exits_3(self):
        with mock.patch.object(bootstrap.setup, "ensure_downloader", lambda *a, **k: "dl"), \
             mock.patch.object(sys.stdin, "isatty", lambda: False), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(bootstrap.main(["--core", self.core, "--accept"]), 3)

    def test_accept_with_authorisation_succeeds(self):
        with mock.patch.object(bootstrap.setup, "run_downloader", self.runner()), \
             mock.patch.object(bootstrap.setup, "ensure_downloader", lambda *a, **k: "dl"), \
             redirect_stdout(io.StringIO()):
            code = bootstrap.main(["--core", self.core, "--accept",
                                   "--approved-by", "yuki-js",
                                   "--approval-ref", "chat 2026-11-06: 'go'"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(self.core, ".consent.json")))


class TestSetupTool(Base):
    """The tool an agent calls — the path that exists when there is no terminal."""

    def call(self, **args):
        return json.loads(vv._handle_setup(args))

    def test_status_is_an_envelope(self):
        result = self.call(action="status", core=self.core)
        self.assertTrue(result["success"])
        self.assertFalse(result["data_present"])

    def test_download_without_approval_is_refused_with_what_is_missing(self):
        result = self.call(action="download", core=self.core)
        self.assertFalse(result["success"])
        self.assertEqual(result["needs"], ["approved_by", "approval_ref"])
        self.assertIn("Ask the operator", result["hint"])

    def test_download_with_approval_fetches_and_records(self):
        with mock.patch.object(setup, "run_downloader", self.runner()), \
             mock.patch.object(setup, "ensure_downloader", lambda *a, **k: "dl"):
            result = self.call(action="download", core=self.core, approved_by="yuki-js",
                               approval_ref="chat 2026-11-06: 'go'")
        self.assertTrue(result["success"])
        self.assertEqual(result["models"], ["0.vvm"])
        self.assertTrue(os.path.exists(os.path.join(self.core, ".consent.json")))

    def test_show_terms_downloads_nothing(self):
        seen: list[str] = []
        with mock.patch.object(setup, "run_downloader", self.runner(seen=seen)), \
             mock.patch.object(setup, "ensure_downloader", lambda *a, **k: "dl"):
            result = self.call(action="show_terms", core=self.core)
        self.assertTrue(result["success"])
        self.assertFalse(result["downloaded"])
        self.assertIn("利用規約", result["terms"])
        self.assertEqual(seen, ["n"])

    def test_unknown_action(self):
        result = self.call(action="delete-everything", core=self.core)
        self.assertFalse(result["success"])
        self.assertIn("unknown action", result["error"])


class TestRegistration(Base):
    def test_register_exposes_tool_and_provider(self):
        try:
            import agent.tts_registry as registry
        except Exception:
            self.skipTest("Hermes Agent is not importable here")

        calls = {}

        class Ctx:
            def register_tts_provider(self, provider):
                calls["provider"] = provider
                registry.register_provider(provider)

            def register_tool(self, **kwargs):
                calls["tool"] = kwargs

        registry._reset_for_tests()
        try:
            vv.register(Ctx())
            self.assertEqual(calls["tool"]["name"], "voicevox_setup")
            self.assertIn("download", calls["tool"]["schema"]["parameters"]["properties"]["action"]["enum"])
            self.assertIsNotNone(registry.get_provider("voicevox"))
        finally:
            registry._reset_for_tests()


if __name__ == "__main__":
    unittest.main()
