# Experiments

Measured on the machine that also runs Hermes Agent: Linux x86_64, Python 3.11.16,
cgroup quota **4 CPUs** (`cpu.max = 400000 100000`) on a 32-core host, 2026-11-06.

Sandbox: throwaway venv, `voicevox_core==0.17.1` (the release wheel — not on PyPI), data pulled with
`download-linux-x64`: ONNX Runtime **1.17.3** (the downloader's pick), Open JTalk dict v1.11.1 (103 MB),
`0.vvm` (57 MB) and later `1.vvm` (58 MB).

> The downloader requires interactive agreement to the「VOICEVOX 音声モデル 利用規約」and the
> ONNX Runtime terms. For these experiments the prompt was answered `y` on this machine to obtain
> the artifacts; that acceptance is the operator's, not something a plugin may do on a user's
> behalf. Nothing was redistributed — data stays in `/tmp`.

Reproduce with `exp1.py` … `exp4.py` (not shipped here; the numbers below are what they printed).


## 1. Startup cost — almost nothing until a model is loaded

| step | time | RSS |
|---|---|---|
| `import voicevox_core` | 0.016 s | 21 MB |
| `Onnxruntime.load_once(filename=…)` | 0.012 s | 31 MB |
| `OpenJtalk(dict_dir)` | ~0 s | 32 MB |
| `Synthesizer(ort, jtalk)` | ~0 s | 32 MB |
| `VoiceModelFile.open("0.vvm").metas` | ~0 s | 32 MB |
| `load_voice_model` | 0.66–0.88 s | 175 MB |
| first `tts()` after that | 0.6–3.8 s | 272 MB |

Consequence: **initialising the engine is free; only loading a model costs.** A plugin can defer
everything to the first synthesis (or to `warm()`) and pay 0.7 s + ~145 MB once.

## 2. Listing voices needs no ONNX Runtime at all

`VoiceModelFile.open(path).metas` works with **no ONNX Runtime loaded**
(`Onnxruntime.get() is None` at the time). Building a style→file index over two `.vvm` files
(11 styles) took **0.3 ms**.

Consequence: `list_voices()` and any style→model resolution can run on a cold process with zero
engine cost. Voice discovery must not touch ONNX.

`0.vvm` alone gave 10 styles — 四国めたん ×4, ずんだもん ×4, 春日部つむぎ, 雨晴はう — so
**`0.vvm` is not "ずんだもん only"**; several characters share a file and a style id is not
unique across files by convention, only within the loader. `1.vvm` added 冥鳴ひまり (id 14), total
11 styles and +88 MB with both loaded.

## 3. `cpu_num_threads` is the whole performance story

Real-time factor (wall time ÷ audio duration) for 5 / 26 / 100-character Japanese text,
after a warm-up call:

| `cpu_num_threads` | 5 ch | 26 ch | 100 ch |
|---|---|---|---|
| 0 (documented "adapt to the environment") | 1.03 | 0.71 | 0.74 |
| 1 | 0.89 | 0.76 | 0.74 |
| 2 | 0.50 | 0.45 | 0.43 |
| 4 | **0.31** | **0.28** | **0.28** |

`0` behaved like `1`, not like "all cores" — under a 4-CPU cgroup it produced the worst numbers
in the table. Passing an explicit thread count is ~**2.5–3.5× faster** than the default.

Consequence: the provider must never rely on the default. Default to `min(4, os.cpu_count() or 1)`
(exposed as `tts.voicevox.cpu_num_threads`) and document why. This is the difference between
"0.3 s per sentence" and "0.9 s per sentence".

## 4. Unloading actually returns the memory

| state | RSS |
|---|---|
| models loaded (0.vvm + 1.vvm) | 263 MB |
| after 40 short synthesis calls | 354 MB (flat from call 10 onward — no leak) |
| `unload_voice_model(model.id)` | **150 MB** |

`unload_voice_model` released ~185 MB in 0.095 s. After unloading, `tts()` raises
`StyleNotFoundError` — so the provider must reload transparently, which costs another 0.66 s.

40 sequential calls showed no growth after warm-up (354 MB constant), so the per-call path does
not leak. The graph/allocator caches are what push 263 MB → 354 MB once; they do not come back
down, so the honest idle number after `release()` is ~150 MB, not 30 MB.

## 5. Threads: concurrent `tts()` was safe here

Four threads calling `tts()` on one `Synthesizer` at once: no exception, and all four outputs were
**byte-identical** to a serial run. So the binding serialises or is internally safe for this path.

Consequence: a lock is defence in depth, not a correctness requirement. Keep one (cheap, and
`stream()` may interleave differently), but do not claim CORE is unsafe.

## 6. Streaming is available and WAV-framed

`Synthesizer.streaming_synthesis(audio_query, style_id, segment_length=0.3)` returns a
`SynthesisStream` iterator. Observed for 4.28 s of audio:

- 16 chunks, **205 356 bytes total = 44-byte RIFF header + 205 312 bytes of PCM**
- chunk sizes `{44, 4608, 14336}` — the first chunk is the WAV header, not audio
- the iterator's first element arrives immediately (the method returns at once)
- concatenated payload equals the PCM of the non-streaming `synthesis()` output

Consequence: `streams_pcm = True` is feasible — strip the 44-byte header, yield the rest as
int16 LE mono at 24 000 Hz (`FRAME_RATE 93.75 × 256`). Still deferred out of v1, but the door is
open and this is the shape of the code.

## 7. Failure modes, by exact exception

| situation | exception | message |
|---|---|---|
| dict dir missing | `NotLoadedOpenjtalkDictError` at `OpenJtalk()` | 「OpenJTalkの辞書が読み込まれていません」 |
| style's model not loaded | `StyleNotFoundError` | 「…スタイルが見つかりませんでした。音声モデルが読み込まれていないか…」 |
| same model loaded twice | `ModelAlreadyLoadedError` | names the model id |
| `on_existing="REPLACE"` | `ValueError` | valid values are `ERROR`, `RELOAD`, `SKIP` |
| empty text | no error, 0.19 s of audio | — |
| English text | no error, 6 s of audio | OpenJTalk reads it as Japanese syllables — garbage in, audible garbage out |

Two traps worth encoding in code:

1. **`Onnxruntime.load_once(filename=…)` ignores `filename` after the first success.** In a fresh
   process a bogus path raises, but once the singleton exists the argument is discarded — so a
   test that asserts "bad path raises" must run first or in its own process.
2. **The default ONNX filename is versioned and points at a version the downloader does not
   fetch.** `LIB_RECOMMENDED_VERSIONED_FILENAME` here is `libvoicevox_onnxruntime.so.1.23.2`,
   while the downloader installed `libvoicevox_onnxruntime.so.1.17.3` (≥ the required minor 17,
   ≤ supported 29). Relying on the default fails on a stock downloader layout; the plugin must
   resolve the actual file.

## 8. What this changes in the plan

- Initialisation is lazy by default and `is_available()` can stay a pure filesystem check.
- `list_voices()` reads `.vvm` metas only — no ONNX, no model load, 0.3 ms.
- `cpu_num_threads` is explicit, defaulting to ≤4; the documented default is a performance trap.
- Model policy: load the model for the requested style, unload it on `release()`; ~0.7 s and
  ~145 MB per model, so two characters resident costs ~88 MB more but does buy instant switching.
- `release()` genuinely returns memory, and the honest idle figure is ~150 MB (ONNX Runtime itself
  stays resident — it is a singleton and cannot be unloaded).
- Lock around synthesis: kept, but as insurance rather than a fix.
- Streaming: implementable later by stripping the 44-byte header; not v1.
