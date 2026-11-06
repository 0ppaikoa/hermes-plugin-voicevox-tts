"""VOICEVOX CORE as a Hermes Agent text-to-speech provider.

Registers the provider name ``voicevox``. Hermes renders replies with whatever ``tts.provider``
names, so once this plugin is enabled and selected every spoken reply comes out of a locally
running, in-process VOICEVOX CORE — no engine process, no HTTP, no port.

    tts:
      provider: voicevox
      voicevox:
        core: /opt/voicevox_core        # holds onnxruntime/, dict/, models/
        speaker: 3                      # ずんだもん（ノーマル）; names work too
        cpu_num_threads: 4
        idle_unload_seconds: 300

Setup (the wheel is not on PyPI, and the models need a manual licence click):

    pip install https://github.com/VOICEVOX/voicevox_core/releases/download/0.17.1/\
voicevox_core-0.17.1-cp310-abi3-manylinux_2_34_x86_64.whl
    ./download-linux-x64 --only onnxruntime --only dict --only models -o /opt/voicevox_core
    hermes plugins enable voicevox

Generated audio must credit VOICEVOX (e.g. ``VOICEVOX:ずんだもん``) — that is the voice models'
licence, not this plugin's.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Optional

try:  # Hermes core
    from agent.tts_provider import TTSProvider  # type: ignore[assignment]
except Exception:  # pragma: no cover - lets the test suite import without Hermes

    class TTSProvider:  # type: ignore[no-redef]
        streams_pcm: bool = False
        stream_sample_rate: Optional[int] = None

        def list_voices(self) -> list[dict[str, Any]]:
            return []

        def default_voice(self) -> Optional[str]:
            return None

        def synthesize(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError


from . import setup as _setup  # noqa: E402
from .core import (  # noqa: E402
    ENV_CORE_DIR,
    ConfigSource,
    CoreEngine,
    VoicevoxError,
    match_style,
    split_text,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "voicevox"

__all__ = ["VoicevoxProvider", "register", "VoicevoxError", "split_text", "match_style"]


def _with_wav_extension(path: str) -> str:
    stem, ext = os.path.splitext(path)
    return path if ext.lower() in {".wav", ".wave"} else f"{stem}.wav"


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class VoicevoxProvider(TTSProvider):
    """Speaks through VOICEVOX CORE in the Hermes process."""

    def __init__(self) -> None:
        self._config = ConfigSource()
        self._engine = CoreEngine(self._config)

    # -- identity ------------------------------------------------------

    @property
    def name(self) -> str:
        return PLUGIN_NAME

    @property
    def display_name(self) -> str:
        return "VOICEVOX CORE"

    @property
    def voice_compatible(self) -> bool:
        """WAV in, Opus out — but only if Hermes can run the conversion.

        Hermes converts a ``voice_compatible`` plugin's WAV to Opus/OGG for voice bubbles; when
        ffmpeg is missing the conversion returns nothing and the file is delivered as audio.
        Claiming it unconditionally would produce an empty result on ffmpeg-less machines.
        """
        return _ffmpeg_available()

    def is_available(self) -> bool:
        """Always True.

        ``is_available`` is ``text_to_speech``'s ``check_fn``: returning False removes the tool
        from the schema entirely, and a caller that then invokes it gets an unrelated "install
        Edge TTS" message. Reporting readiness here would therefore trade a precise error for a
        vanished capability, so every real failure is raised from :meth:`synthesize` instead.
        """
        return True

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "VOICEVOX CORE",
            "badge": "local",
            "tag": "In-process — no engine process, no port, nothing leaves the machine",
            "env_vars": [
                {
                    "key": ENV_CORE_DIR,
                    "prompt": "VOICEVOX CORE directory (holds onnxruntime/, dict/, models/)",
                    "url": "https://github.com/VOICEVOX/voicevox_core/releases",
                },
            ],
        }

    # -- catalogue -----------------------------------------------------

    def list_voices(self) -> list[dict[str, Any]]:
        """Styles from the ``.vvm`` metadata; empty (not an exception) when CORE is absent."""
        try:
            return [
                {
                    "id": voice["id"],
                    "display": voice["display"],
                    "language": voice["language"],
                    "gender": "",
                }
                for voice in self._engine.voice_index()
            ]
        except Exception as exc:
            logger.debug("VOICEVOX voice listing unavailable: %s", exc)
            return []

    def default_voice(self) -> Optional[str]:
        return str(self._config.default_speaker())

    # -- lifecycle -----------------------------------------------------

    def warm(self) -> None:
        """Pre-load the configured model. Best effort: Hermes never surfaces this error."""
        try:
            self._engine._ensure_model(self._config.default_speaker())
        except Exception as exc:
            logger.debug("VOICEVOX warm-up skipped: %s", exc)

    def release(self) -> None:
        """Drop the resident model. Cheap, idempotent, never raises.

        Hermes can call this inline while holding its global lease lock (``keep_warm_seconds: 0``),
        so it must not block for long and must never call back into TTS.
        """
        try:
            self._engine.unload()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("VOICEVOX release failed: %s", exc)

    # -- synthesis -----------------------------------------------------

    def _resolve_style(self, voice: Any, model: Any) -> int:
        wanted = voice if voice not in (None, "") else model
        if wanted in (None, ""):
            return self._config.default_speaker()
        style = match_style(self._engine.voice_index(), wanted)
        if style is None:
            voices = self._engine.voice_index()
            preview = ", ".join(f"{v['id']}={v['display']}" for v in voices[:8])
            raise VoicevoxError(
                f"voice {wanted!r} does not exist in the installed models. Available: {preview}"
                + (" ..." if len(voices) > 8 else "")
            )
        return style

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "mp3",
        **extra: Any,
    ) -> str:
        """Render *text* to a WAV file and return its path.

        ``format`` is accepted and ignored: CORE returns WAV, and re-encoding here would add an
        ffmpeg dependency plus a second lossy step for nothing. The path is rewritten so the
        extension matches the bytes.
        """
        style = self._resolve_style(voice, model)
        audio = self._engine.synthesize(text, style, speed=speed)

        target = _with_wav_extension(output_path)
        with open(target, "wb") as handle:
            handle.write(audio)
        logger.info("VOICEVOX spoke %d chars as style %s -> %s", len(text or ""), style, target)
        return target


_SETUP_SCHEMA: dict[str, Any] = {
    "name": "voicevox_setup",
    "description": (
        "Prepare the data VOICEVOX CORE needs (ONNX Runtime, dictionary, voice models) for the "
        "`voicevox` TTS provider. Actions: `status` reports what is installed and what licence "
        "text was accepted; `show_terms` prints the official terms and downloads nothing; "
        "`download` fetches the data. Call `download` ONLY after the operator has explicitly "
        "agreed in the conversation, and pass their exact words as approval_ref — never invent "
        "one. This tool exists because the official downloader's prompt asks a human, and the "
        "human's answer arrives in chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "show_terms", "download"]},
            "approved_by": {"type": "string",
                            "description": "who agreed (required for action=download)"},
            "approval_ref": {"type": "string",
                             "description": "where and when they agreed, quoting them "
                                            "(required for action=download)"},
            "core": {"type": "string",
                     "description": "optional CORE directory override (default: tts.voicevox.core)"},
        },
        "required": ["action"],
    },
}


def _core_dir(override: Any = None) -> str:
    if isinstance(override, str) and override.strip():
        return os.path.abspath(os.path.expanduser(override.strip()))
    configured = ConfigSource().core_dir()
    return configured or os.path.abspath(os.path.expanduser(_setup.DEFAULT_CORE))


def _handle_setup(args: dict[str, Any], **_kw: Any) -> str:
    action = str(args.get("action") or "status").strip().lower()
    core_dir = _core_dir(args.get("core"))

    def envelope(ok: bool, **payload: Any) -> str:
        return json.dumps({"success": ok, **payload}, ensure_ascii=False)

    try:
        if action == "status":
            return envelope(True, **_setup.status(core_dir))
        if action == "show_terms":
            text = _setup.show_terms(_setup.ensure_downloader(), core_dir)
            return envelope(True, core=core_dir, terms=text, downloaded=False)
        if action == "download":
            result = _setup.download(
                core_dir,
                approved_by=str(args.get("approved_by") or ""),
                approval_ref=str(args.get("approval_ref") or ""),
            )
            return envelope(True, **result)
        return envelope(False, error=f"unknown action {action!r}; expected status|show_terms|download")
    except _setup.SetupRefused as exc:
        return envelope(False, error=str(exc),
                        needs=["approved_by", "approval_ref"],
                        hint="Ask the operator in the conversation first, then pass their words.")
    except Exception as exc:
        return envelope(False, error=f"{type(exc).__name__}: {exc}")


def register(ctx) -> None:
    """Plugin entry point."""
    ctx.register_tts_provider(VoicevoxProvider())
    try:
        ctx.register_tool(
            name="voicevox_setup",
            toolset="tts",
            schema=_SETUP_SCHEMA,
            handler=_handle_setup,
            emoji="🗣",
        )
    except Exception as exc:
        # The provider is the point; the setup tool is a convenience. A host that refuses the
        # tool must still get speech, so this never propagates.
        logger.warning("voicevox_setup tool was not registered: %s", exc)
