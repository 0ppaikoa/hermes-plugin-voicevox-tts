# pyright: reportAttributeAccessIssue=false
"""Bootstrap tests — no network, no real downloader, no real terms.

    python -m unittest discover -s tests -v

The point of these tests is the promise, not the download: the script must refuse to accept
anything without a terminal, without the exact phrase, and it must record what it accepted.
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
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bootstrap  # noqa: E402

TERMS_SAMPLE = """# VOICEVOX 音声モデル 利用規約
## 許諾内容
1. 商用・非商用問わず利用することができます
## 禁止事項
- 逆コンパイル・リバースエンジニアリング
"""


def make_data(core_dir: str, models: tuple[str, ...] = ("0.vvm",)) -> None:
    os.makedirs(os.path.join(core_dir, "onnxruntime", "lib"), exist_ok=True)
    os.makedirs(os.path.join(core_dir, "dict", "open_jtalk_dic_utf_8-1.11"), exist_ok=True)
    os.makedirs(os.path.join(core_dir, "models", "vvms"), exist_ok=True)
    with open(os.path.join(core_dir, "models", "TERMS.txt"), "w", encoding="utf-8") as fh:
        fh.write(TERMS_SAMPLE)
    with open(os.path.join(core_dir, "onnxruntime", "TERMS.txt"), "w", encoding="utf-8") as fh:
        fh.write("VOICEVOX ONNX Runtime 利用規約\n")
    for name in models:
        open(os.path.join(core_dir, "models", "vvms", name), "wb").close()


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="voicevox-bootstrap-")
        self.core = os.path.join(self.dir, "core")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class TestVerify(Base):
    def test_missing_data_exits_2_and_offers_both_commands(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = bootstrap.verify(self.core)
        self.assertEqual(code, 2)
        out = buffer.getvalue()
        self.assertIn("--show-terms", out)
        self.assertIn("--accept", out)

    def test_existing_data_reports_the_consent_record(self):
        make_data(self.core)
        with open(os.path.join(self.core, ".consent.json"), "w", encoding="utf-8") as fh:
            json.dump({"accepted_at": "2026-11-06T00:00:00Z", "accepted_by": "yuki",
                       "terms_text_sha256": "abc"}, fh)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = bootstrap.verify(self.core)
        self.assertEqual(code, 0)
        self.assertIn("同意の記録あり", buffer.getvalue())

    def test_data_without_record_is_flagged_not_trusted(self):
        make_data(self.core)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(bootstrap.verify(self.core), 0)
        self.assertIn("同意の記録はありません", buffer.getvalue())


class TestAccept(Base):
    def test_refuses_without_a_terminal(self):
        seen: list[str] = []

        def fake_runner(binary, core_dir, answer):
            seen.append(answer)
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

        with mock.patch.object(bootstrap, "run_downloader", fake_runner), \
             mock.patch.object(sys.stdin, "isatty", lambda: False):
            code = bootstrap.accept("dl", self.core)
        self.assertEqual(code, 3)
        self.assertEqual(seen, [], "nothing may be run without a TTY")

    def test_refuses_with_only_half_the_authorisation(self):
        seen: list[str] = []

        def fake_runner(binary, core_dir, answer):
            seen.append(answer)
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

        for kwargs in ({"approved_by": "yuki-js"}, {"approval_ref": "chat"}):
            with mock.patch.object(bootstrap, "run_downloader", fake_runner), \
                 mock.patch.object(sys.stdin, "isatty", lambda: False), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(bootstrap.accept("dl", self.core, **kwargs), 3, kwargs)
        self.assertEqual(seen, [], "half an authorisation is no authorisation")

    def test_agent_channel_works_with_a_recorded_authorisation(self):
        """No terminal (an agent), but the operator said yes in the conversation."""
        seen: list[str] = []

        def fake_runner(binary, core_dir, answer):
            seen.append(answer)
            if answer == "n":
                return subprocess.CompletedProcess(args=[], returncode=1,
                                                   stdout=TERMS_SAMPLE, stderr="")
            make_data(core_dir)
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

        with mock.patch.object(bootstrap, "run_downloader", fake_runner), \
             mock.patch.object(sys.stdin, "isatty", lambda: False), \
             redirect_stdout(io.StringIO()):
            code = bootstrap.accept("dl", self.core, approved_by="yuki-js",
                                    approval_ref="chat 2026-11-06T07:55Z: 'go ahead'")

        self.assertEqual(code, 0)
        self.assertEqual(seen, ["n", "y"], "terms still get shown before anything is fetched")
        with open(os.path.join(self.core, ".consent.json"), encoding="utf-8") as fh:
            record = json.load(fh)
        self.assertEqual(record["accepted_by"], "yuki-js")
        self.assertIn("delegated", record["consent_channel"])
        self.assertIn("go ahead", record["approval_ref"])

    def test_interactive_record_says_so(self):
        def fake_runner(binary, core_dir, answer):
            if answer == "n":
                return subprocess.CompletedProcess(args=[], returncode=1,
                                                   stdout=TERMS_SAMPLE, stderr="")
            make_data(core_dir)
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

        with mock.patch.object(bootstrap, "run_downloader", fake_runner), \
             mock.patch.object(sys.stdin, "isatty", lambda: True), \
             mock.patch("builtins.input", lambda *_: bootstrap.PHASE), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(bootstrap.accept("dl", self.core), 0)
        with open(os.path.join(self.core, ".consent.json"), encoding="utf-8") as fh:
            self.assertIn("interactive", json.load(fh)["consent_channel"])

    def test_refuses_the_wrong_phrase(self):
        seen: list[str] = []

        def fake_runner(binary, core_dir, answer):
            seen.append(answer)
            return subprocess.CompletedProcess(args=[], returncode=1, stdout=TERMS_SAMPLE, stderr="")

        with mock.patch.object(bootstrap, "run_downloader", fake_runner), \
             mock.patch.object(sys.stdin, "isatty", lambda: True), \
             mock.patch("builtins.input", lambda *_: "yes"), \
             redirect_stdout(io.StringIO()):
            code = bootstrap.accept("dl", self.core)
        self.assertEqual(code, 3)
        self.assertEqual(seen, ["n"], "only the terms-only run may happen before consent")
        self.assertFalse(os.path.exists(os.path.join(self.core, ".consent.json")))

    def test_accepts_with_the_phrase_and_records_what_was_accepted(self):
        seen: list[str] = []

        def fake_runner(binary, core_dir, answer):
            seen.append(answer)
            if answer == "n":
                return subprocess.CompletedProcess(args=[], returncode=1,
                                                   stdout=TERMS_SAMPLE, stderr="")
            make_data(core_dir, ("0.vvm", "1.vvm"))
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

        with mock.patch.object(bootstrap, "run_downloader", fake_runner), \
             mock.patch.object(sys.stdin, "isatty", lambda: True), \
             mock.patch("builtins.input", lambda *_: bootstrap.PHASE), \
             redirect_stdout(io.StringIO()):
            code = bootstrap.accept("dl", self.core)

        self.assertEqual(code, 0)
        self.assertEqual(seen, ["n", "y"], "terms first, acceptance second")
        record_path = os.path.join(self.core, ".consent.json")
        self.assertTrue(os.path.exists(record_path))
        with open(record_path, encoding="utf-8") as fh:
            record = json.load(fh)
        self.assertEqual(record["phrase"], bootstrap.PHASE)
        self.assertEqual(record["models"], ["0.vvm", "1.vvm"])
        self.assertIn("models_terms", record["assets"])
        self.assertIn("onnxruntime_terms", record["assets"])
        self.assertEqual(len(record["terms_text_sha256"]), 64)


class TestShowTerms(Base):
    def test_answers_no_and_returns_the_official_text(self):
        seen: list[str] = []

        def fake_runner(binary, core_dir, answer):
            seen.append(answer)
            return subprocess.CompletedProcess(args=[], returncode=1,
                                               stdout=TERMS_SAMPLE, stderr="")

        buffer = io.StringIO()
        with mock.patch.object(bootstrap, "run_downloader", fake_runner), redirect_stdout(buffer):
            text = bootstrap.show_terms("dl", self.core)
        self.assertEqual(seen, ["n"])
        self.assertIn("利用規約", text)
        self.assertIn("利用規約", buffer.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.core, ".consent.json")))


class TestMain(Base):
    def test_default_run_does_not_touch_the_network(self):
        with mock.patch.object(bootstrap, "ensure_downloader",
                               side_effect=AssertionError("must not download")), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(bootstrap.main(["--core", self.core]), 2)


if __name__ == "__main__":
    unittest.main()
