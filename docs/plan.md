# Implementation plan

Status: **draft, nothing implemented.** Every "verified" statement below was read out of source
code or release artifacts while writing this document; anything inferred is labelled as such.

Goal: a Hermes Agent TTS provider that renders replies through **VOICEVOX CORE** — the synthesis
library, loaded in-process — with no engine process, no HTTP, no IPC.

Non-goals for v1: the VOICEVOX editor, the ENGINE's HTTP API, singing synthesis
(`FrameAudioQuery` / `Score`), user dictionaries, GPU acceleration tuning.


## 1. What Hermes requires of a TTS provider

Verified against `/opt/hermes` at `2bd378ed1b` (2026-11-05).

**Registration.** A plugin calls `ctx.register_tts_provider(provider)`; the object must be a
subclass of `agent.tts_provider.TTSProvider`. A directory plugin needs `plugin.yaml` plus an
`__init__.py` exposing `register(ctx)`, and it only loads when its key is in `plugins.enabled`
— plugins are **opt-in** (`hermes_cli/plugins_discovery.py`: "plugins are opt-in").

**Name resolution order** (`tools/tts_tool_plugins.py`, `tools/tts_tool.py`):

1. a built-in name (`edge`, `elevenlabs`, `openai`, `minimax`, `xai`, `mistral`, `gemini`,
   `neutts`, `kittentts`, `piper`, `deepinfra`) always wins;
2. a `tts.providers.<name>: type: command` entry wins over a plugin ("config is more local than
   a plugin install");
3. otherwise a registered plugin provider is dispatched.

`voicevox` is free — it collides with none of the eleven.

**The ABC** (`agent/tts_provider.py`):

```python
class TTSProvider(CatalogProviderBase):
    streams_pcm: bool = False
    stream_sample_rate: int | None = None
    def synthesize(self, text, output_path, *, voice=None, model=None,
                   speed=None, format="mp3", **extra) -> str: ...
    def list_voices(self) -> list[dict]: ...
    def default_voice(self) -> str | None: ...
    def stream(self, text, *, voice=None, model=None, format="opus", **extra): ...
    def warm(self) -> None: ...
    def release(self) -> None: ...
```

Three constraints that shape the design:

- **`is_available()` must not touch the network and must not raise** — the picker calls it on
  every paint (`agent/provider_base.py`). So it reports *importability* (is `voicevox_core`
  importable? is a CORE directory configured?), never *liveness*.
- **`warm()` / `release()` are exactly the hook this plugin needs.** Hermes calls them when a
  surface turns speech output on and when the last lease across surfaces is released, honouring
  `tts.keep_warm_seconds` (default 60). That is the load/unload point that keeps idle RAM near
  zero — the "low RAM" requirement solves itself if the model is loaded lazily and dropped on
  release.
- **`format` is a request, not a promise.** The dispatcher passes `tts.output_format` (default
  `mp3`); the ABC says a provider that cannot honour it should pick the closest equivalent and
  make `output_path` carry the right extension. CORE returns WAV, so the plugin writes `.wav`.

**Delivery.** `tools/tts_tool.py::_finalize_voice_delivery` converts a plugin's output to
Opus/OGG when the provider reports `voice_compatible` and the platform wants a voice bubble; a
provider that already wrote `.ogg` skips the conversion. CORE gives WAV, so the plugin reports
`voice_compatible = ffmpeg present` and lets Hermes do the single WAV→Opus step. Without ffmpeg
the reply arrives as a plain audio attachment instead of a voice bubble — degraded, not broken.

**Text length.** Hermes splits long replies before calling the provider; the cap comes from
`tts.<provider>.max_text_length`, else `FALLBACK_MAX_TEXT_LENGTH`. So `tts.voicevox.max_text_length`
works with no plugin code.


## 2. What VOICEVOX CORE gives us

Verified against release **0.17.1** (2026-10-22).

**Distribution.** CORE ships as (a) a C API shared library (`voicevox_core-linux-x64-0.17.1.zip`
→ `lib/libvoicevox_core.so` + `include/voicevox_core.h`), and (b) **official Python wheels** built
`cp310-abi3` for linux/macos/windows. The Python wheel is the sane route: no `ctypes` bindings to
hand-write, no struct layout to get wrong.

Wheel contents (`voicevox_core-0.17.1-cp310-abi3-manylinux_2_34_x86_64.whl`, 1.86 MB):

```
voicevox_core/__init__.py      enums + exceptions
voicevox_core/blocking.py      Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile, SynthesisStream, UserDict
voicevox_core/asyncio.py       same, async
voicevox_core/_rust.abi3.so    the binding itself
```

There is **no `libvoicevox_core.so` in the wheel** — the binding is self-contained apart from
ONNX Runtime, which it loads by filename at runtime.

**Not on PyPI.** `https://pypi.org/pypi/voicevox_core/json` and `voicevox-core` both return 404.
Installation means a release-asset URL, a vendored wheel, or an explicit documented step. This is
the single ugliest part of the developer experience and section 7 deals with it.

**Blocking API** (from `_rust/blocking.pyi`):

```python
Onnxruntime.load_once(*, filename: str = LIB_RECOMMENDED_VERSIONED_FILENAME) -> Onnxruntime
Onnxruntime.LIB_MIN_REQUIRED_MINOR_VERSION = 17     # 1.x >= 17
Onnxruntime.LIB_MAX_SUPPORTED_MINOR_VERSION = 29
Onnxruntime.LIB_RECOMMENDED_VERSION = "1.23.2"

OpenJtalk(open_jtalk_dict_dir)                       # dict directory
Synthesizer(onnxruntime, open_jtalk, *, acceleration_mode="AUTO", cpu_num_threads=0)
Synthesizer.load_voice_model(model, *, on_existing="ERROR")
Synthesizer.unload_voice_model(voice_model_id)
Synthesizer.is_loaded_voice_model(voice_model_id) -> bool
Synthesizer.metas() -> list[CharacterMeta]

VoiceModelFile.open(path) -> VoiceModelFile          # .vvm; .metas, .id, .close()
Synthesizer.tts(text, style_id, *, enable_interrogative_upspeak=True) -> bytes   # WAV
Synthesizer.tts_from_kana(kana, style_id, *, ...) -> bytes
```

`Onnxruntime` is a **process-wide singleton** ("インスタンスは高々一つ"), and `SynthesisStream`
exists in the blocking API — relevant if we later want `streams_pcm` (section 6).

**Data the plugin needs on disk**, as laid out by the official downloader
(`download-linux-x64`, targets `c-api` / `onnxruntime` / `additional-libraries` / `models` /
`dict`, `--models-pattern` accepts a glob, default device `cpu`):

```
<core-dir>/onnxruntime/lib/libvoicevox_onnxruntime.so
<core-dir>/dict/open_jtalk_dic_utf_8-1.11/…
<core-dir>/models/vvms/0.vvm                # ずんだもん lives in 0.vvm
```

Two frictions found while trying this here: the downloader **prompts interactively** for
agreement to the 「VOICEVOX 音声モデル 利用規約」 and ONNX Runtime terms (an agent cannot accept
those on a user's behalf — the README must make the step manual), and the models target is
version-pinned to `>=0.16, <0.18`.

**Licences.** CORE source and build artifacts are MIT from 0.16 onward. The **voice models are
not**: they are usable commercially and non-commercially, redistributable inside an application,
but generated audio must be credited (`VOICEVOX:ずんだもん`). The plugin must not bury that.


## 3. Prior art to reuse

`yuki-js/readthecard` → `packages/backend/src/services/voicevox-service.ts` already drives CORE
from a non-Rust host (koffi FFI, 354 lines). Facts worth lifting straight into this plugin:

- `ZUNDAMON_STYLE_ID = 3` (ノーマル); ずんだもん's model is inside `0.vvm`.
- dict directory name `open_jtalk_dic_utf_8-1.11`.
- Metadata comes from the model file (`voicevox_voice_model_file_create_metas_json`) — that is
  the voice listing, so `list_voices()` needs no model inference, only the `.vvm` files.
- It degrades by *warning and disabling itself* when the library is missing rather than throwing
  at import. For an agent host that is the right instinct, but see 5: here a missing CORE must
  surface as an actionable error, not silence.
- It loads `0.vvm` eagerly at startup; we will not (section 5).

The C API call order there (`make_default_load_onnxruntime_options` → `onnxruntime_load_once` →
`open_jtalk_rc_new` → `make_default_initialize_options` → `synthesizer_new` →
`voice_model_file_open` → `synthesizer_load_voice_model` → `make_default_tts_options` →
`synthesizer_tts` → `wav_free`) is the same ceremony the Python binding performs internally.


## 4. Module layout

```
hermes_voicevox/
    __init__.py        VoicevoxProvider(TTSProvider) + register(ctx)
    core.py            CoreEngine: singleton ownership, lazy init, model cache, lock
    voices.py          style resolution (id, "ずんだもん", "ずんだもん（あまあま）"), metas parsing
    plugin.yaml        manifest
tests/
    test_provider.py   provider contract, no CORE required
    test_core.py       CoreEngine against a fake voicevox_core module
    test_live.py       real CORE, skipped unless HERMES_VOICEVOX_LIVE=1
```

`core.py` holds one `Synthesizer` behind a lock; `__init__.py` holds no state. Every CORE import
is inside a function so that importing the plugin on a machine without CORE succeeds and
`is_available()` can answer honestly.


## 5. Lifecycle and memory

```
first synthesis  ──► load ONNX Runtime (once per process)
                 ──► OpenJtalk(dict)
                 ──► Synthesizer(…, cpu_num_threads=0)
                 ──► VoiceModelFile.open(vvm) for the requested style only
                 ──► load_voice_model  ──► tts() ──► write WAV
release()        ──► unload_voice_model(...) ──► drop the Synthesizer reference
                     (keep ONNX Runtime: it is a singleton and re-loading it is not supported)
```

Design rules:

- **One model at a time by default.** CORE's `load_voice_model` is per-model; loading every `.vvm`
  in the directory is what makes people give up on CORE as "heavy". The plugin loads the model
  that owns the requested style, and unloads the previous one when the style changes
  (`on_existing="ERROR"` is fine because we track what we loaded).
- **`release()` unloads; `warm()` pre-loads** for the configured style. Hermes already holds a
  lease while speech is on and releases it after `tts.keep_warm_seconds`, so "idle" is a state
  the plugin can reach without inventing a timer.
- **A lock around synthesis.** Hermes may resolve up to three consecutive sentences concurrently
  on the streaming path; CORE's thread-safety is not documented. Serialise, do not hope.
- **Never initialize at import.** A gateway that imports the plugin must not pay model load time
  (or fail) at boot.

Open question: whether one `Synthesizer` can hold two models at once cheaply enough to avoid a
reload when a reply switches character. To be measured in P2, not guessed.


## 6. Streaming (deferred)

CORE exposes `SynthesisStream`, so `streams_pcm = True` with a real `stream()` is plausible:
synthesize to a stream, drop the WAV header, yield int16 PCM at `FRAME_RATE * 100` Hz. That would
let Hermes start speaking on the first chunk instead of the whole file. Not v1 — it doubles the
surface and the failure modes, and `synthesize()` already produces speech. Keep the door open by
never assuming the file path is the only output.

A cheaper intermediate: keep the WAV in `synthesize()`, and let Hermes' existing
`min_len`-based sentence splitting be the granularity. Measure perceived latency first.


## 7. Installation and the PyPI gap

Because `voicevox_core` is not on PyPI, `pip install hermes-plugin-voicevox-tts` cannot pull it
transitively. Options:

| Option | Verdict |
|---|---|
| Document `pip install <release wheel URL>` | honest, brittle (URL pins a platform + version) |
| Extra index / vendored wheels in this repo | works, but ships ~2 MB per platform and needs CI per target |
| Optional dependency: plugin works, `is_available()` False, error tells you what to install | **v1** |

v1 ships **no hard dependency**: `hermes_voicevox` imports fine everywhere, and the first
synthesis that finds no `voicevox_core` raises one message naming the wheel to install and the
downloader command to run. That also keeps `is_available()` cheap and network-free.

Separate question for later: whether to publish wheels of *this* plugin for every platform, or
only sdist + instructions.


## 8. Testing

- **Unit** (no CORE): provider contract, `format`/extension rewriting, style-name resolution,
  `is_available()` never raising, `register()` landing in `agent.tts_registry`, dispatch through
  `tools.tts_tool_plugins._dispatch_to_plugin_provider`, built-in names unshadowable.
- **Fake CORE**: a stub module implementing `Onnxruntime`/`OpenJtalk`/`Synthesizer`/
  `VoiceModelFile` with the same signatures, to test lifecycle (load once, switch model, unload
  on release, lock) without a 100 MB download.
- **Live** (opt-in): real CORE + dict + one `.vvm` in a throwaway venv, synthesize one Japanese
  sentence, assert `RIFF` header, duration within ±20 % of expectation, and record cold-start
  time and RSS. This is the only test that proves the thing works, so it must exist before the
  first release even if it is skipped in CI by default.

A prototype written during planning (HTTP-engine shaped, since discarded) reached 30/30 passing
tests including the real dispatch path; the same harness is reusable here, which is why the
unit-test scope above is not wishful.


## 9. Phases

**P1 — skeleton.** `VoicevoxProvider` + `core.py` + config resolution + error messages; unit and
fake-CORE tests green. *Done when:* `python -m unittest` passes with no CORE installed, and a
machine without CORE gets one actionable message instead of a traceback.

**P2 — real speech.** Live test in a throwaway venv: downloader → dict + `0.vvm` → synthesize →
valid WAV. Measure cold start, steady-state RSS, and whether two styles fit in one process.
*Done when:* the live test passes and the numbers are in the README.

**P3 — voices and lifecycle.** `list_voices()` from `.vvm` metas, style-name resolution,
`warm()`/`release()` wired to Hermes' lease, model switch without leak. *Done when:* toggling
speech off drops RSS back near baseline.

**P4 — polish.** Streaming decision (section 6), Windows/macOS verification, README with the
credit obligation, and only then a published release.

Nothing is published until P2 passes. A repo full of untested CORE bindings would be worse than
no repo.


## 10. Open questions

1. **Core only, or Core with an engine fallback?** One backend is cleaner; two means two config
   stories and two failure modes. My default: Core only.
2. **Where does the plugin find CORE?** `tts.voicevox.core` in config, `HERMES_VOICEVOX_CORE_DIR`
   env, or both (env wins)? The readthecard service uses `VOICEVOX_CORE_DIR`; matching that name
   costs nothing.
3. **Multi-style replies.** Does Aok need to change voice mid-conversation (ずんだもん → めたん),
   or is one style per config enough for v1?
4. **Windows first-class?** Hermes runs there; CORE has a `win_amd64` wheel. Claiming support
   without testing on it would be dishonest, so either test it or say "Linux/macOS in v1".
5. **Python floor.** The wheel is `cp310-abi3`; `manylinux_2_34` implies glibc ≥ 2.34. Hermes'
   own runtime is newer, so this is documentation, not a blocker.
