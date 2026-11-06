# pyright: reportMissingImports=false
"""Live tests against a real VOICEVOX CORE install. Skipped unless asked for.

    HERMES_VOICEVOX_LIVE=1 HERMES_VOICEVOX_CORE_DIR=/path/to/core \\
        python -m unittest tests.test_live -v

Requires ``voicevox_core`` importable in the running interpreter plus a downloaded core
directory (onnxruntime/, dict/, models/). The point of this file is the memory claim in
docs/plan.md §12: the plugin must stay flat across a long reply because it never hands CORE
more than one sentence at a time.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hermes_voicevox as vv  # noqa: E402

LIVE = os.environ.get("HERMES_VOICEVOX_LIVE") == "1"


def rss_mb() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    return -1.0


@unittest.skipUnless(LIVE, "set HERMES_VOICEVOX_LIVE=1 to run against real CORE")
class TestLiveCore(unittest.TestCase):
    def setUp(self) -> None:
        core_dir = os.environ.get("HERMES_VOICEVOX_CORE_DIR")
        if not core_dir or not os.path.isdir(core_dir):
            self.skipTest("HERMES_VOICEVOX_CORE_DIR is not set to a real core directory")
        try:
            import voicevox_core  # noqa: F401
        except Exception as exc:
            self.skipTest(f"voicevox_core is not importable here: {exc}")
        self.dir = tempfile.mkdtemp(prefix="voicevox-live-")
        self.provider = vv.VoicevoxProvider()

    def tearDown(self) -> None:
        self.provider.release()
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_lists_real_styles(self):
        voices = {v["id"]: v["display"] for v in self.provider.list_voices()}
        self.assertGreater(len(voices), 5)
        self.assertEqual(voices.get(3), "ずんだもん（ノーマル）")

    def test_speaks_a_sentence(self):
        path = self.provider.synthesize(
            "こんにちは、ずんだもんなのだ。", os.path.join(self.dir, "one.mp3"))
        self.assertTrue(path.endswith(".wav"))
        with wave.open(path) as reader:
            self.assertEqual(reader.getframerate(), 24000)
            self.assertEqual(reader.getnchannels(), 1)
            self.assertGreater(reader.getnframes() / reader.getframerate(), 0.5)

    def test_named_voice_selects_the_style(self):
        path = self.provider.synthesize(
            "四国めたんなので。", os.path.join(self.dir, "metan.wav"), voice="四国めたん")
        with wave.open(path) as reader:
            self.assertGreater(reader.getnframes(), 0)

    def test_long_reply_stays_flat_in_memory(self):
        """The claim under test: 1,000 chars must not cost the ~5 GB a single call would."""
        text = "これは長さを測るための日本語の文章です。ずんだもんがゆっくり喋ります。" * 27
        text = text[:1000]
        self.provider.synthesize("ウォームアップなのだ。", os.path.join(self.dir, "warm.wav"))
        before = rss_mb()
        path = self.provider.synthesize(text, os.path.join(self.dir, "long.wav"))
        after = rss_mb()
        with wave.open(path) as reader:
            seconds = reader.getnframes() / reader.getframerate()
        growth = after - before
        self.assertGreater(seconds, 60, "sanity: 1,000 chars should be a long utterance")
        self.assertLess(
            growth, 250.0,
            f"memory grew {growth:.0f} MB (was {before:.0f} MB, now {after:.0f} MB) — "
            "sentence chunking is not working",
        )
        print(f"\n  live: 1,000 chars -> {seconds:.0f}s audio, RSS {before:.0f} -> {after:.0f} MB "
              f"({growth:+.0f} MB)")

    def test_release_returns_memory(self):
        self.provider.synthesize("テストなのだ。", os.path.join(self.dir, "a.wav"))
        before = rss_mb()
        self.provider.release()
        after = rss_mb()
        self.assertLessEqual(after, before)
        print(f"\n  live: release {before:.0f} -> {after:.0f} MB")


if __name__ == "__main__":
    unittest.main()
