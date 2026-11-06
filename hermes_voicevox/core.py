"""VOICEVOX CORE engine wrapper: lazy init, one model at a time, sentence-sized synthesis.

Why this module exists rather than a straight ``synth.tts(text)`` call: one CORE call costs
roughly 23–34 MB of resident memory per second of generated audio, the memory is never returned
inside the process, and 800 characters in a single call OOM-killed a 7 GiB container. The same
4,000 characters synthesised sentence by sentence stayed flat at 310 MB. So every public entry
point here splits first and concatenates after — see docs/plan.md §12.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import threading
import wave
from io import BytesIO
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

ENV_CORE_DIR = "HERMES_VOICEVOX_CORE_DIR"

#: Ceiling for one piece handed to CORE. CORE costs ~30 MB of resident memory per second of
#: audio, and Japanese runs ~6 chars/second, so 40 chars ≈ 6.8 s ≈ 200 MB per piece. Short
#: enough that a long reply stays flat, long enough that prosody survives (a piece is a
#: sentence, not a fragment).
PIECE_MAX_CHARS = 40
#: Absolute ceiling per call — protects against a pathological reply, not a normal one.
TOTAL_MAX_CHARS = 6000
#: Refuse to synthesise more pieces than this in one call (each piece is a CORE round trip).
MAX_PIECES = 600

DEFAULT_SPEAKER = 3  # ずんだもん（ノーマル）
DEFAULT_IDLE_UNLOAD_SECONDS = 300.0

_SENTENCE_END = re.compile(r"(?<=[。．！？!?])")
_CLAUSE_END = re.compile(r"(?<=[、，,])")


class VoicevoxError(RuntimeError):
    """Anything the user can act on: missing CORE, missing model, unreadable text."""


# --- filesystem layout ------------------------------------------------------------

def _find_onnxruntime(core_dir: str) -> Optional[str]:
    """The ONNX Runtime shared library the downloader laid down.

    Deliberately not ``Onnxruntime.LIB_RECOMMENDED_VERSIONED_FILENAME``: that names 1.23.2 while
    the official downloader installs 1.17.3, and ``load_once`` ignores ``filename`` once the
    process-wide singleton exists, so a wrong guess fails far from here.
    """
    lib_dir = os.path.join(core_dir, "onnxruntime", "lib")
    preferred = os.path.join(lib_dir, "libvoicevox_onnxruntime.so")
    if os.path.exists(preferred):
        return preferred
    candidates = sorted(glob.glob(os.path.join(lib_dir, "libvoicevox_onnxruntime.so*")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(lib_dir, "*onnxruntime*.so*")))
    return candidates[-1] if candidates else None


def _find_dict(core_dir: str) -> Optional[str]:
    base = os.path.join(core_dir, "dict")
    if os.path.isdir(os.path.join(base, "open_jtalk_dic_utf_8-1.11")):
        return os.path.join(base, "open_jtalk_dic_utf_8-1.11")
    found = sorted(glob.glob(os.path.join(base, "open_jtalk_dic*")))
    return found[-1] if found else None


def _find_vvms(core_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(core_dir, "models", "vvms", "*.vvm")))


# --- text splitting ---------------------------------------------------------------

def split_text(text: str, piece_max: int = PIECE_MAX_CHARS) -> list[str]:
    """Split into sentence-sized pieces; hard-split any sentence that is still too long.

    Hermes' own splitter keys on ``(?<=[.!?])\\s+``, which never fires for Japanese prose
    (``。`` is not followed by a space), so this is the only splitting that happens.
    """
    text = (text or "").strip()
    if not text:
        return []

    pieces: list[str] = []
    for block in text.splitlines():
        block = block.strip()
        if not block:
            continue
        for sentence in _SENTENCE_END.split(block):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= piece_max:
                pieces.append(sentence)
                continue
            for clause in _CLAUSE_END.split(sentence):
                clause = clause.strip()
                if not clause:
                    continue
                while len(clause) > piece_max:
                    pieces.append(clause[:piece_max])
                    clause = clause[piece_max:].lstrip()
                if clause:
                    pieces.append(clause)

    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) <= piece_max:
            merged[-1] = merged[-1] + piece
        else:
            merged.append(piece)
    return merged[:MAX_PIECES]


def _normalize_style_name(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("　", "")


def match_style(voices: Iterable[dict[str, Any]], wanted: Any) -> Optional[int]:
    """Resolve a style id from ``36``, ``"36"``, ``"ずんだもん"`` or ``"ずんだもん（ささやき）"``."""
    if wanted is None or isinstance(wanted, bool):
        return None
    if isinstance(wanted, int):
        return wanted
    text = str(wanted).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    target = _normalize_style_name(text)
    best: Optional[tuple[int, int]] = None
    for voice in voices:
        character = _normalize_style_name(str(voice.get("character") or ""))
        style = _normalize_style_name(str(voice.get("style") or ""))
        labels = (
            _normalize_style_name(f"{voice.get('character')}（{voice.get('style')}）"),
            character + style,
            style,
            character,
        )
        if target not in labels:
            continue
        priority = labels.index(target)
        if best is None or priority < best[0]:
            best = (priority, int(voice["id"]))
    return best[1] if best else None


# --- engine -----------------------------------------------------------------------

class CoreEngine:
    """Owns the process-wide CORE objects. One instance per provider; all calls serialised."""

    def __init__(self, config: "ConfigSource") -> None:
        self._config = config
        self._lock = threading.RLock()
        self._synth: Any = None
        self._ort: Any = None
        self._jtalk: Any = None
        self._modules: Any = None
        self._model_ids: dict[int, Any] = {}      # style id -> model UUID
        self._loaded: dict[Any, str] = {}         # model UUID -> vvm path
        self._style_index: Optional[dict[int, dict[str, Any]]] = None
        self._idle_timer: Optional[threading.Timer] = None
        self._idle_generation = 0

    # -- configuration ------------------------------------------------

    def core_dir(self) -> str:
        return self._config.core_dir()

    def _blocking(self):
        if self._modules is not None:
            return self._modules
        try:
            from voicevox_core import blocking  # noqa: PLC0415 - intentionally lazy
        except Exception as exc:  # pragma: no cover - depends on install
            raise VoicevoxError(
                "VOICEVOX CORE is not installed for this Python. Install the official wheel "
                "(not on PyPI) into the interpreter Hermes runs:\n"
                "  pip install https://github.com/VOICEVOX/voicevox_core/releases/download/"
                "0.17.1/voicevox_core-0.17.1-cp310-abi3-manylinux_2_34_x86_64.whl\n"
                f"(import error: {exc})"
            ) from exc
        self._modules = blocking
        return blocking

    # -- voices (no ONNX Runtime needed) ------------------------------

    def voice_index(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Style catalogue read from the ``.vvm`` files' metadata.

        Costs 0.3 ms for two models and needs no ONNX Runtime, so callers may use it freely.
        """
        if self._style_index is not None and not refresh:
            return list(self._style_index.values())

        blocking = self._blocking()
        core_dir = self.core_dir()
        vvms = _find_vvms(core_dir)
        if not vvms:
            raise VoicevoxError(
                f"no voice models found under {core_dir}/models/vvms. Run the VOICEVOX CORE "
                "downloader (--only models) or point tts.voicevox.core at the right directory."
            )

        index: dict[int, dict[str, Any]] = {}
        for path in vvms:
            try:
                model = blocking.VoiceModelFile.open(path)
            except Exception as exc:
                logger.warning("skipping unreadable voice model %s: %s", path, exc)
                continue
            try:
                for character in model.metas:
                    for style in character.styles:
                        index[int(style.id)] = {
                            "id": int(style.id),
                            "display": f"{character.name}（{style.name}）",
                            "character": character.name,
                            "style": style.name,
                            "language": "ja",
                            "model": os.path.basename(path),
                        }
            finally:
                with _suppress():
                    model.close()
        self._style_index = index
        return list(index.values())

    # -- lifecycle ----------------------------------------------------

    def _ensure_engine(self):
        with self._lock:
            if self._synth is not None:
                return self._synth
            blocking = self._blocking()
            core_dir = self.core_dir()
            if not core_dir or not os.path.isdir(core_dir):
                raise VoicevoxError(
                    "VOICEVOX CORE directory is not configured or does not exist "
                    f"({core_dir or '<unset>'}). Set tts.voicevox.core in config.yaml or "
                    f"{ENV_CORE_DIR} to the directory holding onnxruntime/, dict/ and models/."
                )
            onnx = _find_onnxruntime(core_dir)
            if not onnx:
                raise VoicevoxError(
                    f"no ONNX Runtime library under {core_dir}/onnxruntime/lib. Run the "
                    "VOICEVOX CORE downloader (--only onnxruntime)."
                )
            dict_dir = _find_dict(core_dir)
            if not dict_dir:
                raise VoicevoxError(
                    f"no Open JTalk dictionary under {core_dir}/dict. Run the VOICEVOX CORE "
                    "downloader (--only dict)."
                )

            try:
                ort = blocking.Onnxruntime.load_once(filename=onnx)
            except Exception as exc:
                raise VoicevoxError(f"could not load ONNX Runtime from {onnx}: {exc}") from exc
            try:
                jtalk = blocking.OpenJtalk(dict_dir)
            except Exception as exc:
                raise VoicevoxError(
                    f"could not load the Open JTalk dictionary at {dict_dir}: {exc}"
                ) from exc

            try:
                synth = blocking.Synthesizer(
                    ort, jtalk, cpu_num_threads=self._config.cpu_num_threads()
                )
            except Exception as exc:
                raise VoicevoxError(f"could not create the VOICEVOX synthesizer: {exc}") from exc

            self._ort, self._jtalk, self._synth = ort, jtalk, synth
            logger.info(
                "VOICEVOX CORE ready (cpu_num_threads=%s, core=%s)",
                self._config.cpu_num_threads(), core_dir,
            )
            return synth

    def _ensure_model(self, style_id: int):
        """Load the model that owns *style_id*, dropping any other to keep the footprint small."""
        with self._lock:
            synth = self._ensure_engine()
            if style_id in self._model_ids:
                return synth

            voices = self.voice_index()
            entry = next((v for v in voices if v["id"] == style_id), None)
            if entry is None:
                available = ", ".join(str(v["id"]) for v in voices[:20])
                raise VoicevoxError(
                    f"style id {style_id} is not in any model under {self.core_dir()}/models/vvms "
                    f"(available: {available})."
                )
            path = os.path.join(self.core_dir(), "models", "vvms", entry["model"])

            blocking = self._blocking()
            model = blocking.VoiceModelFile.open(path)
            try:
                for loaded_id in list(self._loaded):
                    with _suppress():
                        synth.unload_voice_model(loaded_id)
                self._loaded.clear()
                self._model_ids.clear()
                synth.load_voice_model(model, on_existing="RELOAD")
                self._loaded[model.id] = path
                self._model_ids[style_id] = model.id
                for voice in voices:
                    if voice["model"] == entry["model"]:
                        self._model_ids[voice["id"]] = model.id
            finally:
                with _suppress():
                    model.close()
            return synth

    def unload(self) -> int:
        """Drop every loaded model. Idempotent, never raises — safe under a global lock."""
        with self._lock:
            self._cancel_idle_timer()
            if self._synth is None:
                return 0
            dropped = 0
            for model_id in list(self._loaded):
                try:
                    self._synth.unload_voice_model(model_id)
                    dropped += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("unload_voice_model(%s) failed: %s", model_id, exc)
            self._loaded.clear()
            self._model_ids.clear()
            return dropped

    # -- idle unload ---------------------------------------------------

    def _cancel_idle_timer(self) -> None:
        self._idle_generation += 1
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_unload(self) -> None:
        """Hermes' lease never fires for tool-driven synthesis, so the plugin owns its own timer."""
        seconds = self._config.idle_unload_seconds()
        self._cancel_idle_timer()
        if seconds <= 0:
            return
        generation = self._idle_generation
        timer = threading.Timer(seconds, self._idle_unload, args=(generation,))
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_unload(self, generation: int) -> None:
        if generation != self._idle_generation:
            return
        try:
            dropped = self.unload()
            if dropped:
                logger.info("VOICEVOX CORE released %d model(s) after idling", dropped)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("idle unload failed: %s", exc)

    # -- synthesis -----------------------------------------------------

    def synthesize(self, text: str, style_id: int, *, speed: Optional[float] = None) -> bytes:
        """Return WAV bytes for *text*, synthesising one sentence at a time."""
        blocking = self._blocking()
        pieces = split_text(text)
        if not pieces:
            pieces = [""]
        if sum(len(p) for p in pieces) > TOTAL_MAX_CHARS:
            raise VoicevoxError(
                f"text is longer than {TOTAL_MAX_CHARS} characters; split it before synthesis"
            )

        with self._lock:
            synth = self._ensure_model(style_id)
            frames: list[bytes] = []
            params = None
            try:
                for piece in pieces:
                    query = synth.create_audio_query(piece, style_id)
                    if speed is not None:
                        with _suppress():
                            query.speed_scale = max(0.25, min(4.0, float(speed)))
                    wav = synth.synthesis(query, style_id)
                    with wave.open(BytesIO(wav)) as reader:
                        if params is None:
                            params = reader.getparams()
                        frames.append(reader.readframes(reader.getnframes()))
            except VoicevoxError:
                raise
            except Exception as exc:
                name = type(exc).__name__
                if name == "AnalyzeTextError":
                    raise VoicevoxError(
                        "VOICEVOX could not analyse the text (Open JTalk rejected it). "
                        "Non-Japanese text, or a fragment that is only punctuation, does this."
                    ) from exc
                raise VoicevoxError(f"VOICEVOX synthesis failed: {name}: {exc}") from exc
            finally:
                self._schedule_idle_unload()

        out = BytesIO()
        with wave.open(out, "wb") as writer:
            writer.setparams(params or (1, 2, 24000, 0, "NONE", "not compressed"))
            for chunk in frames:
                writer.writeframes(chunk)
        return out.getvalue()


class _suppress:
    """Tiny contextlib.suppress stand-in so exceptions in cleanup never mask the real one."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return True


class ConfigSource:
    """Reads ``tts.voicevox`` per call, under the caller's profile scope."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def _section(self) -> dict[str, Any]:
        try:
            from hermes_cli.config import load_config_readonly  # noqa: PLC0415

            cfg = load_config_readonly() or {}
            tts = cfg.get("tts")
            section = tts.get("voicevox") if isinstance(tts, dict) else None
            if not isinstance(section, dict):
                section = cfg.get("voicevox")
            return section if isinstance(section, dict) else {}
        except Exception:
            return {}

    def core_dir(self) -> str:
        for env in (ENV_CORE_DIR, "VOICEVOX_CORE_DIR"):
            value = os.environ.get(env)
            if value:
                return os.path.abspath(os.path.expanduser(value))
        value = self._section().get("core")
        return os.path.abspath(os.path.expanduser(str(value))) if value else ""

    def cpu_num_threads(self) -> int:
        raw = self._section().get("cpu_num_threads")
        if raw is None:
            raw = os.environ.get("HERMES_VOICEVOX_CPU_THREADS")
        try:
            if raw is not None:
                threads = int(raw)
                if threads > 0:
                    return min(threads, 16)
        except (TypeError, ValueError):
            pass
        # Never rely on CORE's default: `cpu_num_threads=0` measured like a single thread.
        try:
            cores = len(os.sched_getaffinity(0))
        except Exception:  # pragma: no cover - non-Linux
            cores = os.cpu_count() or 1
        return max(1, min(4, cores))

    def idle_unload_seconds(self) -> float:
        raw = self._section().get("idle_unload_seconds")
        if raw is None:
            raw = os.environ.get("HERMES_VOICEVOX_IDLE_UNLOAD")
        if raw is None:
            return DEFAULT_IDLE_UNLOAD_SECONDS
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return DEFAULT_IDLE_UNLOAD_SECONDS

    def default_speaker(self) -> int:
        raw = self._section().get("speaker")
        if raw is None:
            raw = os.environ.get("HERMES_VOICEVOX_SPEAKER")
        try:
            return int(raw) if raw is not None else DEFAULT_SPEAKER
        except (TypeError, ValueError):
            return DEFAULT_SPEAKER
