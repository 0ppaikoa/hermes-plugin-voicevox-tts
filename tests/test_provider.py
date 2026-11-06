# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""Contract tests — no CORE, no models, no ONNX Runtime, no network.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest
import wave
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_core  # noqa: E402
import hermes_voicevox as vv  # noqa: E402
from hermes_voicevox import core  # noqa: E402


def make_core_dir(root: str, models: tuple[str, ...] = ("0.vvm",)) -> str:
    """Lay out a CORE directory the way the official downloader does."""
    os.makedirs(os.path.join(root, "onnxruntime", "lib"), exist_ok=True)
    open(os.path.join(root, "onnxruntime", "lib", "libvoicevox_onnxruntime.so.1.17.3"), "wb").close()
    os.makedirs(os.path.join(root, "dict", "open_jtalk_dic_utf_8-1.11"), exist_ok=True)
    os.makedirs(os.path.join(root, "models", "vvms"), exist_ok=True)
    for name in models:
        open(os.path.join(root, "models", "vvms", name), "wb").close()
    return root


class Base(unittest.TestCase):
    def setUp(self) -> None:
        fake_core.reset()
        fake_core.FakeCore().install()
        self.dir = tempfile.mkdtemp(prefix="voicevox-unit-")
        self.core_dir = make_core_dir(os.path.join(self.dir, "core"))
        self._env = {k: os.environ.get(k) for k in
                     (core.ENV_CORE_DIR, "HERMES_VOICEVOX_SPEAKER",
                      "HERMES_VOICEVOX_IDLE_UNLOAD", "HERMES_VOICEVOX_CPU_THREADS")}
        os.environ[core.ENV_CORE_DIR] = self.core_dir
        os.environ.pop("HERMES_VOICEVOX_IDLE_UNLOAD", None)
        self.provider = vv.VoicevoxProvider()

    def tearDown(self) -> None:
        fake_core.FakeCore.uninstall()
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.dir, ignore_errors=True)

    def out(self, name: str = "reply.mp3") -> str:
        return os.path.join(self.dir, name)

    def read_wav(self, path: str):
        with wave.open(path) as reader:
            return reader.getnchannels(), reader.getframerate(), reader.getnframes()


# --- splitting ------------------------------------------------------------------

class TestSplitting(unittest.TestCase):
    def test_splits_japanese_sentences(self):
        # With a cap smaller than a sentence, each sentence must travel separately — this is the
        # split Hermes never performs (its splitter needs whitespace after the terminator).
        pieces = vv.split_text("こんにちは。今日はいい天気ですね。明日は雨です。", piece_max=10)
        self.assertEqual(len(pieces), 3)
        self.assertTrue(pieces[0].startswith("こんにちは"))
        for piece in pieces:
            self.assertLessEqual(len(piece), 10)

    def test_never_returns_a_piece_over_the_cap(self):
        text = "あ" * 300
        for piece in vv.split_text(text, piece_max=40):
            self.assertLessEqual(len(piece), 40, piece)

    def test_hard_splits_a_sentence_without_punctuation(self):
        pieces = vv.split_text("あ" * 150, piece_max=50)
        self.assertEqual([len(p) for p in pieces], [50, 50, 50])

    def test_merges_short_sentences(self):
        pieces = vv.split_text("はい。そうですね。", piece_max=40)
        self.assertEqual(len(pieces), 1, "two tiny sentences should travel together")

    def test_never_exceeds_the_cap_across_lines(self):
        pieces = vv.split_text("一行目。\n二行目。", piece_max=10)
        self.assertTrue(pieces)
        for piece in pieces:
            self.assertLessEqual(len(piece), 10)

    def test_empty_text(self):
        self.assertEqual(vv.split_text(""), [])
        self.assertEqual(vv.split_text("   \n  "), [])


# --- style selection ------------------------------------------------------------

class TestStyleMatching(unittest.TestCase):
    VOICES = [
        {"id": 3, "character": "ずんだもん", "style": "ノーマル"},
        {"id": 1, "character": "ずんだもん", "style": "あまあま"},
        {"id": 2, "character": "四国めたん", "style": "ノーマル"},
    ]

    def test_by_id_int_and_str(self):
        self.assertEqual(vv.match_style(self.VOICES, 2), 2)
        self.assertEqual(vv.match_style(self.VOICES, "2"), 2)

    def test_by_character_name_takes_first_style(self):
        self.assertEqual(vv.match_style(self.VOICES, "ずんだもん"), 3)

    def test_by_character_and_style(self):
        self.assertEqual(vv.match_style(self.VOICES, "ずんだもん（あまあま）"), 1)

    def test_unknown_name(self):
        self.assertIsNone(vv.match_style(self.VOICES, "存在しない"))

    def test_none_and_blank(self):
        self.assertIsNone(vv.match_style(self.VOICES, None))
        self.assertIsNone(vv.match_style(self.VOICES, "   "))


# --- provider identity ----------------------------------------------------------

class TestIdentity(Base):
    def test_name(self):
        self.assertEqual(self.provider.name, "voicevox")
        self.assertEqual(self.provider.display_name, "VOICEVOX CORE")

    def test_is_available_never_gates(self):
        # is_available is the tool's check_fn: False would remove text_to_speech entirely.
        os.environ[core.ENV_CORE_DIR] = "/nonexistent"
        self.assertIs(self.provider.is_available(), True)

    def test_not_pcm_streaming_in_v1(self):
        self.assertFalse(self.provider.streams_pcm)

    def test_setup_schema_names_the_core_dir(self):
        schema = self.provider.get_setup_schema()
        self.assertEqual([v["key"] for v in schema["env_vars"]], [core.ENV_CORE_DIR])

    def test_voice_compatible_tracks_ffmpeg(self):
        original = shutil.which
        try:
            shutil.which = lambda name: "/usr/bin/ffmpeg"
            self.assertIs(self.provider.voice_compatible, True)
            shutil.which = lambda name: None
            self.assertIs(self.provider.voice_compatible, False)
        finally:
            shutil.which = original


# --- catalogue ------------------------------------------------------------------

class TestVoices(Base):
    def test_lists_styles_from_vvm_metadata(self):
        voices = {v["id"]: v["display"] for v in self.provider.list_voices()}
        self.assertEqual(voices[3], "ずんだもん（ノーマル）")
        self.assertEqual(voices[2], "四国めたん（ノーマル）")

    def test_listing_does_not_start_onnx(self):
        self.provider.list_voices()
        self.assertEqual(fake_core.Onnxruntime.loaded, [],
                         "listing voices must not load ONNX Runtime")

    def test_listing_is_empty_without_core(self):
        os.environ[core.ENV_CORE_DIR] = "/nonexistent"
        self.assertEqual(self.provider.list_voices(), [])

    def test_default_voice_is_the_configured_speaker(self):
        self.assertEqual(self.provider.default_voice(), "3")


# --- synthesis ------------------------------------------------------------------

class TestSynthesis(Base):
    def test_writes_wav_and_rewrites_the_extension(self):
        path = self.provider.synthesize("こんにちは。", self.out("reply.mp3"))
        self.assertTrue(path.endswith(".wav"))
        self.assertTrue(os.path.exists(path))

    def test_one_call_becomes_several_small_pieces(self):
        text = "これは一文目です。" * 20          # 180 chars, 20 sentences
        self.provider.synthesize(text, self.out())
        synth = fake_core.Synthesizer.instances[-1]
        self.assertGreater(len(synth.queries), 1, "long text must be split before CORE sees it")
        for query in synth.queries:
            self.assertLessEqual(len(query.text), core.PIECE_MAX_CHARS)

    def test_pieces_are_concatenated_into_one_file(self):
        text = "あ" * 120
        self.provider.synthesize(text, self.out())
        synth = fake_core.Synthesizer.instances[-1]
        expected = sum(max(0.1, len(q.text) / 6.0) for q in synth.queries)
        channels, rate, frames = self.read_wav(self.out("reply.wav"))
        self.assertEqual((channels, rate), (1, 24000))
        self.assertAlmostEqual(frames / rate, expected, delta=0.2)

    def test_no_tts_shortcut_is_used(self):
        self.provider.synthesize("テスト。", self.out())
        # the fake raises if tts() is called; reaching here proves audio_query+synthesis
        self.assertTrue(fake_core.Synthesizer.instances[-1].queries)

    def test_speed_is_applied_to_the_query(self):
        self.provider.synthesize("はやく。", self.out(), speed=1.5)
        self.assertAlmostEqual(fake_core.Synthesizer.instances[-1].queries[0].speed_scale, 1.5)

    def test_speed_is_clamped(self):
        self.provider.synthesize("はやすぎ。", self.out(), speed=99)
        self.assertEqual(fake_core.Synthesizer.instances[-1].queries[0].speed_scale, 4.0)

    def test_voice_name_selects_a_style(self):
        self.provider.synthesize("テスト。", self.out(), voice="四国めたん")
        self.assertEqual(fake_core.Synthesizer.instances[-1].queries[0].style_id, 2)

    def test_unknown_voice_names_the_alternatives(self):
        with self.assertRaises(vv.VoicevoxError) as caught:
            self.provider.synthesize("テスト。", self.out(), voice="存在しない")
        self.assertIn("ずんだもん", str(caught.exception))

    def test_empty_text_still_produces_audio(self):
        path = self.provider.synthesize("", self.out("empty.wav"))
        self.assertTrue(os.path.exists(path))

    def test_model_is_loaded_once_across_calls(self):
        self.provider.synthesize("一つ目。", self.out("a.wav"))
        self.provider.synthesize("二つ目。", self.out("b.wav"))
        synth = fake_core.Synthesizer.instances[-1]
        self.assertEqual(len(fake_core.Synthesizer.instances), 1, "one Synthesizer per provider")
        self.assertEqual(list(synth.loaded), ["0.vvm"])

    def test_second_model_replaces_the_first(self):
        os.environ[core.ENV_CORE_DIR] = make_core_dir(
            os.path.join(self.dir, "core2"), ("0.vvm", "1.vvm"))
        provider = vv.VoicevoxProvider()
        provider.synthesize("ずんだもん。", self.out("a.wav"), voice="3")
        provider.synthesize("ひまり。", self.out("b.wav"), voice="14")
        synth = fake_core.Synthesizer.instances[-1]
        self.assertIn("1.vvm", synth.loaded)
        self.assertNotIn("0.vvm", synth.loaded, "only one model should stay resident")
        self.assertTrue(synth.unloaded)


# --- failures -------------------------------------------------------------------

class TestFailures(Base):
    def test_missing_core_dir_is_actionable(self):
        os.environ[core.ENV_CORE_DIR] = "/nonexistent/core"
        with self.assertRaises(vv.VoicevoxError) as caught:
            self.provider.synthesize("テスト。", self.out())
        message = str(caught.exception)
        self.assertIn("tts.voicevox.core", message)
        self.assertIn(core.ENV_CORE_DIR, message)

    def test_missing_models_is_actionable(self):
        empty = make_core_dir(os.path.join(self.dir, "empty"), ())
        os.environ[core.ENV_CORE_DIR] = empty
        with self.assertRaises(vv.VoicevoxError) as caught:
            vv.VoicevoxProvider().synthesize("テスト。", self.out())
        self.assertIn("models/vvms", str(caught.exception))

    def test_warm_never_raises(self):
        os.environ[core.ENV_CORE_DIR] = "/nonexistent/core"
        vv.VoicevoxProvider().warm()   # no assertion needed: it must simply not raise

    def test_release_never_raises_and_is_idempotent(self):
        provider = vv.VoicevoxProvider()
        provider.release()
        provider.synthesize("テスト。", self.out())
        provider.release()
        provider.release()
        self.assertTrue(fake_core.Synthesizer.instances[-1].unloaded)


# --- lifecycle ------------------------------------------------------------------

class TestLifecycle(Base):
    def test_idle_unload_drops_the_model(self):
        os.environ["HERMES_VOICEVOX_IDLE_UNLOAD"] = "0.05"
        provider = vv.VoicevoxProvider()
        provider.synthesize("テスト。", self.out())
        synth = fake_core.Synthesizer.instances[-1]
        self.assertEqual(list(synth.loaded), ["0.vvm"])
        deadline = time.time() + 3.0
        while time.time() < deadline and synth.loaded:
            time.sleep(0.02)
        self.assertEqual(list(synth.loaded), [], "idle timer must release the model")

    def test_idle_unload_zero_disables_the_timer(self):
        os.environ["HERMES_VOICEVOX_IDLE_UNLOAD"] = "0"
        provider = vv.VoicevoxProvider()
        provider.synthesize("テスト。", self.out())
        time.sleep(0.2)
        self.assertTrue(fake_core.Synthesizer.instances[-1].loaded)

    def test_cpu_threads_are_explicit(self):
        provider = vv.VoicevoxProvider()
        provider.synthesize("テスト。", self.out())
        threads = fake_core.Synthesizer.instances[-1].cpu_num_threads
        self.assertGreaterEqual(threads, 1)
        self.assertLessEqual(threads, 4, "CORE's default 0 behaves like a single thread")

    def test_unload_uses_the_model_uuid_not_a_style_id(self):
        provider = vv.VoicevoxProvider()
        provider.synthesize("テスト。", self.out())
        provider.release()
        synth = fake_core.Synthesizer.instances[-1]
        self.assertTrue(all(not isinstance(u, int) for u in synth.unloaded),
                        "unload_voice_model wants a UUID; a style id raises TypeError")


# --- Hermes integration ---------------------------------------------------------

class TestHermesIntegration(Base):
    def setUp(self):
        super().setUp()
        try:
            import agent.tts_registry as registry
        except Exception:
            self.skipTest("Hermes Agent is not importable here")
        registry._reset_for_tests()
        self.registry = registry

    def tearDown(self):
        if hasattr(self, "registry"):
            self.registry._reset_for_tests()
        super().tearDown()

    def test_register_exposes_the_provider(self):
        class Ctx:
            def register_tts_provider(self, provider):
                self.registry_ref.register_provider(provider)

        ctx = Ctx()
        ctx.registry_ref = self.registry
        vv.register(ctx)
        self.assertIsNotNone(self.registry.get_provider("voicevox"))

    def test_dispatch_reaches_this_provider(self):
        try:
            from tools.tts_tool_plugins import _dispatch_to_plugin_provider
        except Exception:
            self.skipTest("tools.tts_tool_plugins is not importable here")
        self.registry.register_provider(vv.VoicevoxProvider())
        out = _dispatch_to_plugin_provider(
            "ディスパッチ経由です。", self.out("d.wav"), "voicevox", {"provider": "voicevox"})
        self.assertTrue(out and out.endswith(".wav"))
        self.assertTrue(os.path.exists(out))

    def test_does_not_shadow_a_builtin(self):
        self.assertNotIn("voicevox", self.registry._BUILTIN_NAMES)


if __name__ == "__main__":
    unittest.main()
