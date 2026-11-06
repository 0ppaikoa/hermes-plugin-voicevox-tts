"""A stand-in for ``voicevox_core.blocking`` so the suite runs without CORE, models or ONNX.

Mirrors the handful of shapes this plugin touches, and records every call so tests can assert on
*how* synthesis was driven (piece sizes, model loads, unloads) rather than only on output.
"""

from __future__ import annotations

import io
import struct
import sys
import types
import uuid
import wave

RATE = 24000
SAMPLE_BYTES = 2  # int16


def wav_bytes(seconds: float) -> bytes:
    frames = max(1, int(RATE * seconds))
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(SAMPLE_BYTES)
        writer.setframerate(RATE)
        writer.writeframes(b"\x00\x01" * frames)
    return out.getvalue()


class Style:
    def __init__(self, id_: int, name: str) -> None:
        self.id = id_
        self.name = name


class Character:
    def __init__(self, name: str, styles: list[Style]) -> None:
        self.name = name
        self.styles = styles


MODELS = {
    "0.vvm": [
        Character("四国めたん", [Style(2, "ノーマル"), Style(0, "あまあま")]),
        Character("ずんだもん", [Style(3, "ノーマル"), Style(1, "あまあま")]),
    ],
    "1.vvm": [Character("冥鳴ひまり", [Style(14, "ノーマル")])],
}


class Query:
    def __init__(self, text: str, style_id: int) -> None:
        self.text = text
        self.style_id = style_id
        self.speed_scale = 1.0
        self.kana = text


class VoiceModelFile:
    def __init__(self, path: str) -> None:
        self.path = path
        self.name = path.rsplit("/", 1)[-1]
        self.id = uuid.uuid5(uuid.NAMESPACE_DNS, self.name)
        self.metas = MODELS.get(self.name, [])
        self.closed = False

    @staticmethod
    def open(path: str) -> "VoiceModelFile":
        return VoiceModelFile(path)

    def close(self) -> None:
        self.closed = True


class Synthesizer:
    instances: list["Synthesizer"] = []

    def __init__(self, onnxruntime, open_jtalk, *, cpu_num_threads: int = 0, **_kw) -> None:
        self.cpu_num_threads = cpu_num_threads
        self.queries: list[Query] = []
        self.loaded: dict = {}
        self.unloaded: list = []
        self.on_existing: list[str] = []
        self._metas: list[Character] = []
        Synthesizer.instances.append(self)

    def create_audio_query(self, text: str, style_id: int) -> Query:
        query = Query(text, style_id)
        self.queries.append(query)
        return query

    def synthesis(self, query: Query, style_id: int) -> bytes:
        # 6 characters ≈ 1 second, so summed durations are predictable in assertions.
        return wav_bytes(max(0.1, len(query.text) / 6.0))

    def tts(self, text: str, style_id: int, **_kw) -> bytes:  # pragma: no cover - not used
        raise AssertionError("the plugin must not call tts(); it drives audio_query+synthesis")

    def load_voice_model(self, model: VoiceModelFile, *, on_existing: str = "ERROR") -> None:
        if model.name in self.loaded and on_existing == "ERROR":
            raise RuntimeError("ModelAlreadyLoadedError")
        self.loaded[model.name] = model.id
        self.on_existing.append(on_existing)
        self._metas.extend(model.metas)

    def unload_voice_model(self, model_id) -> None:
        if not isinstance(model_id, uuid.UUID):
            raise TypeError("argument 'voice_model_id': Expected a `uuid.UUID` instance.")
        self.unloaded.append(model_id)
        for name, mid in list(self.loaded.items()):
            if mid == model_id:
                del self.loaded[name]
        self._metas = []

    def metas(self) -> list[Character]:
        return list(self._metas)


class Onnxruntime:
    loaded: list[str] = []
    _instance = None

    @staticmethod
    def load_once(*, filename: str = "default") -> "Onnxruntime":
        Onnxruntime.loaded.append(filename)
        if Onnxruntime._instance is None:
            Onnxruntime._instance = Onnxruntime()
        return Onnxruntime._instance


class OpenJtalk:
    opened: list[str] = []

    def __init__(self, dict_dir: str) -> None:
        if not dict_dir:
            raise RuntimeError("NotLoadedOpenjtalkDictError")
        OpenJtalk.opened.append(dict_dir)


def reset() -> None:
    Synthesizer.instances.clear()
    Onnxruntime.loaded.clear()
    Onnxruntime._instance = None
    OpenJtalk.opened.clear()


class FakeCore:
    """Installable module pair mimicking ``voicevox_core`` / ``voicevox_core.blocking``."""

    def install(self) -> None:
        blocking = types.ModuleType("voicevox_core.blocking")
        blocking.Onnxruntime = Onnxruntime
        blocking.OpenJtalk = OpenJtalk
        blocking.Synthesizer = Synthesizer
        blocking.VoiceModelFile = VoiceModelFile
        blocking.UserDict = object

        root = types.ModuleType("voicevox_core")
        root.blocking = blocking
        root.__version__ = "0.17.1-fake"

        sys.modules["voicevox_core"] = root
        sys.modules["voicevox_core.blocking"] = blocking

    @staticmethod
    def uninstall() -> None:
        sys.modules.pop("voicevox_core", None)
        sys.modules.pop("voicevox_core.blocking", None)
