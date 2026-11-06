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

Only `synthesize()` is abstract (`agent/tts_provider.py`); `list_voices`, `default_voice`,
`stream`, `warm`, `release` all have defaults. `voice_compatible` is a read-only **property** the
delivery code reads, so "ffmpeg present" has to be an override, not an attribute.

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
Opus/OGG whenever the provider reports `voice_compatible` — it does **not** consult `want_opus`
for plugin providers, so this happens on CLI and desktop too, not only where a voice bubble is
wanted. A provider that already wrote `.ogg` skips the conversion. CORE gives WAV, so the plugin
reports `voice_compatible` = ffmpeg present and lets Hermes do the single WAV→Opus step; without
ffmpeg the file is delivered as-is.

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


## 3b. What the experiments settled

Full numbers in [experiments.md](experiments.md). The six that change the design:

1. **`cpu_num_threads` must be explicit.** The documented default `0` behaved like a single
   thread here (real-time factor 0.71–1.03); passing `4` gave 0.28–0.31. Never rely on the
   default — `min(4, cpu_count)` and a config override.
2. **Nothing costs anything until a model loads.** Import 16 ms, ONNX Runtime 12 ms, OpenJtalk and
   `Synthesizer` instant. Model load is 0.66–0.88 s and ~145 MB. So `is_available()` can be a pure
   filesystem check and `warm()` can be the only eager path.
3. **Voice listing needs no ONNX Runtime.** Reading `.vvm` metas works with no runtime loaded:
   11 styles over two files in 0.3 ms. `list_voices()` must never load a model.
4. **`release()` really returns memory** — 354 MB → 150 MB in 95 ms. ONNX Runtime itself stays
   (singleton, unloadable by design), so the honest idle figure is ~150 MB, not ~30 MB.
5. **`unload_voice_model` invalidates the loaded styles.** The next `tts()` raises
   `StyleNotFoundError`, so the provider reloads transparently (0.66 s) rather than trusting cache.
6. **Concurrent `tts()` was byte-identical to serial** on four threads. A lock stays as insurance,
   not because CORE is broken.

Two traps that must be encoded:

- `Onnxruntime.load_once(filename=…)` **ignores `filename` once the singleton exists**, so a bad
  path is silently accepted after a successful first call.
- The default filename (`libvoicevox_onnxruntime.so.1.23.2`) does **not** match what the official
  downloader installs (`…so.1.17.3`). Resolve the real file; do not trust the default.



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
first synthesis  ──► load ONNX Runtime (once per process, 12 ms)
                 ──► OpenJtalk(dict)                       (~0 ms)
                 ──► Synthesizer(…, cpu_num_threads=min(4, cpu_count))
                 ──► VoiceModelFile.open(vvm) for the requested style only
                 ──► load_voice_model                      (0.66–0.88 s, ~+145 MB)
                 ──► tts() ──► write WAV
release()        ──► unload_voice_model(...)               (95 ms, −185 MB)
                     (keep ONNX Runtime: it is a singleton and re-loading it is not supported)
```

Design rules:

- **One model at a time by default.** `load_voice_model` is per-model; loading every `.vvm`
  in the directory is what makes people give up on CORE as "heavy". Load the model that owns the
  requested style, unload the previous one when the style changes. Measured: a second model costs
  +88 MB and 0.69 s, so keeping two characters resident is affordable but is not the default.
- **`release()` unloads; `warm()` pre-loads** for the configured style. Hermes already holds a
  lease while speech is on and releases it after `tts.keep_warm_seconds`, so "idle" is a state the
  plugin can reach without inventing a timer. Measured: 354 MB → 150 MB.
- **Reload after unload.** `unload_voice_model` makes later `tts()` raise `StyleNotFoundError`
  even though the `.vvm` is still on disk; the provider must reconnect the model transparently
  instead of caching "loaded once, forever".
- **A lock around synthesis.** Four concurrent calls were byte-identical to serial here, so this
  is insurance against the streaming path interleaving differently — not a bug fix.
- **Never initialize at import.** A gateway that imports the plugin must not pay model load time
  (or fail) at boot; measured import cost is 16 ms and must stay that way.

Resolved by measurement: two models in one `Synthesizer` coexist fine (11 styles across `0.vvm`
and `1.vvm`, +88 MB), so switching character within a reply is a policy choice about RAM, not a
limitation.



## 6. Streaming (deferred)

Measured: `Synthesizer.streaming_synthesis(audio_query, style_id, segment_length=0.3)` returns a
`SynthesisStream` iterator whose **first element is the 44-byte RIFF header** and whose remainder
is raw PCM. For 4.28 s of audio: 16 chunks, 205 356 bytes total = 44 + 205 312, sizes
`{44, 4608, 14336}`, all even.

So `streams_pcm = True` is a known-shape job, not a research project: consume the query, skip the
44-byte header, yield int16 LE mono at **24 000 Hz** (`FRAME_RATE 93.75 × 256`), and declare
`stream_sample_rate = 24000`. That lets Hermes start speaking on the first segment instead of the
whole file.

Not v1: it doubles the surface and the failure modes, and `synthesize()` already produces speech.
The door stays open by never assuming the file path is the only output.


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

v1 ships **no hard dependency**, but the original version of this section was wrong about how
that surfaces. `check_tts_requirements()` (`tools/tts_tool.py`) is the `text_to_speech` tool's
`check_fn` and it delegates to `_plugin_provider_is_available()` → `provider.is_available()`;
the registry **drops the tool entirely** when that returns False (`tools/registry.py`). So
"is_available() False, but synthesis raises a nice error" is unreachable — there is no tool left
to raise anything.

Decision: `is_available()` returns **True whenever the plugin is loaded**, and every failure
(no `voicevox_core`, no core dir, no `.vvm`, missing dict) is raised from `synthesize()` with the
exact remedy. The tool therefore stays visible and a misconfiguration comes back as one readable
error in the conversation instead of a silently missing capability. The verdict is TTL-cached by
the registry, not recomputed per paint, so this stays cheap.

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

**P2 — real speech.** *Already answered by the experiments:* real CORE, real dict, real `0.vvm`,
4.28 s of audio out of `tts()`, and the timings/RSS in [experiments.md](experiments.md). What
remains for P2 is the part experiments did not prove — the provider inside Hermes: `register()`,
dispatch through `tools.tts_tool`, `warm()`/`release()` against the lease, and a style switch
without a leak. *Done when:* a live test synthesizes through the Hermes dispatcher, not through
the binding directly.


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
   or is one style per config enough for v1? Measured: a second resident model costs +88 MB and
   0.69 s once — so "yes" is affordable; this is a product question, not a technical one.

4. **Windows first-class?** Hermes runs there; CORE has a `win_amd64` wheel. Claiming support
   without testing on it would be dishonest, so either test it or say "Linux/macOS in v1".
5. **Python floor.** The wheel is `cp310-abi3`; `manylinux_2_34` implies glibc ≥ 2.34. Hermes'
   own runtime is newer, so this is documentation, not a blocker.


## 11. Independent review, and what it changed

An independent reviewer checked every claim above against `/opt/hermes` and the real binding.
What it broke, and what we now do instead:

| finding | consequence |
|---|---|
| `is_available()` is the tool's `check_fn`; False **removes** `text_to_speech` | never gate on availability — report in `synthesize()` (§7) |
| `warm()`/`release()` fire only from voice-mode leases (CLI `/voice`, desktop, TUI), **never** from the `text_to_speech` tool | an agent-initiated reply would pin ~150 MB forever. Add a plugin-owned idle unload timer |
| `Synthesizer.tts()` has no `speed` parameter | use `create_audio_query` → set `speed_scale` → `synthesis()` |
| only `tts.voice`/`model`/`speed` reach the provider; `tts.voicevox.*` does not | the plugin reads its own section per call, under the caller's profile scope |
| `release()` can run inline under the global lease lock when `keep_warm_seconds: 0` | `release()` must be cheap, idempotent, must never raise and must never call back into TTS |
| `plugins.isolation: host` runs the plugin in a host process reached over RPC | "no IPC" is only true for `in_process`; say so instead of over-claiming |
| ONNX Runtime is a process-wide singleton and `load_once` ignores `filename` after the first success | the first profile/CORE dir wins for the process; a second profile's `core:` is ignored |
| Hermes splits text on `[.!?;:,]` + whitespace only | Japanese `。、` never triggers a sentence split; a 4400-char reply is hard-cut at the cap. The plugin should split on Japanese punctuation itself and concatenate WAVs |
| `min(4, os.cpu_count())` ignores cgroup quota | size the default from CPU affinity/`cpu.max`, then cap at 4 |
| the previous "30/30 tests" line referred to a discarded HTTP prototype | removed; it proved nothing about this design |

Reviewer's verdict was "not safe as written", with three pre-conditions: fix the availability/error
contract, make unload independent of the lease, and pin the config/speed/isolation story. All three
are now in §§5, 7 and above.

Two of its criticisms were themselves imprecise and were checked rather than adopted: the Opus
conversion happens regardless of `want_opus` (already corrected in §1), and `0.vvm`/`1.vvm` style
ids are **disjoint** (0–8, 10 vs 14), which is what makes "the model that owns the style"
well-defined rather than ambiguous.

## 12. Memory ceilings, measured in an isolated Job

Run in a throwaway `batch/v1 Job` in the `aok` namespace (`python:3.11-slim`, limit 7 Gi, 4 CPU) so
an OOM could not touch the gateway. Same wheel, same dict, same `0.vvm`, `cpu_num_threads=4`.

### One `tts()` call per reply — the shape that fails

| text | audio | peak RSS | delta | outcome |
|---|---|---|---|---|
| 100 chars | 18.0 s | 618 MB | +412 MB | ok |
| 200 chars | 36.0 s | 1,496 MB | +878 MB | ok |
| 400 chars | 71.5 s | 3,908 MB | +2,412 MB | ok |
| 800 chars | ~143 s | — | — | **OOMKilled (exit 137)** |

Memory does not come back: `after == peak` in every row. The cost is roughly **23–34 MB per second
of generated audio**, so a single call's footprint is a function of how long it speaks, not of the
text's byte length. Hermes' own fallback cap is 4,000 chars, which sits well past the OOM point.

### Sentence-at-a-time — the shape that holds

Splitting on `。！？` and synthesizing each piece into the same WAV:

| text | pieces | audio | rtf | peak RSS | delta |
|---|---|---|---|---|---|
| 200 chars | 12 | 34.0 s | 0.29 | 310 MB | +111 MB |
| 400 chars | 23 | 67.6 s | 0.30 | 310 MB | 0 |
| 800 chars | 46 | 134.8 s | 0.29 | 310 MB | 0 |
| 1,600 chars | 92 | 269.5 s | 0.30 | 310 MB | 0 |
| 2,400 chars | 137 | 403.9 s | 0.29 | 310 MB | 0 |
| 4,000 chars | 229 | 673.2 s | 0.29 | 310 MB | 0 |

**Flat at 310 MB from 200 to 4,000 characters**, with the same real-time factor. A 4,000-char reply
that OOMs as one call costs 310 MB as 229 calls.

Control run: after that loop, a *single* 200-char call still jumped 310 MB → 1,372 MB. So the
per-call cost is inherent to CORE's long-utterance path, not an accumulated leak.

### Decisions this forces

1. **The plugin splits before it synthesizes.** `synthesize()` receives whatever Hermes hands it —
   up to 4,000 chars — and must not pass that to `tts()` in one go. Split on `。！？`（plus `、`
   for very long clauses), synthesize each piece, concatenate the WAV payloads, return one file.
   This is not an optimisation; without it a normal long reply OOM-kills the gateway.
2. **`max_text_length` is a plugin concern, not the tool's.** `tts.voicevox.max_text_length` should
   stay modest so Hermes itself chunks earlier and each delivered file stays small.
3. **Hermes' own splitter is useless for Japanese** (`(?<=[.!?])\s+`), so the split above is the
   only one that will happen.
4. **`unload_voice_model` takes the model's UUID** (`VoiceModelId`), not a style id — passing a
   style id raises `TypeError: Expected a uuid.UUID instance`. Measured the hard way at the end of
   the Job.
5. Memory is never returned inside a process, so the *idle* figure is whatever the largest
   single call reached. Sentence chunking keeps that at ~310 MB, which is the real argument for it.
