# hermes-plugin-voicevox-tts

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) text-to-speech provider that speaks
through [VOICEVOX CORE](https://github.com/VOICEVOX/voicevox_core) — the synthesis library, loaded
in-process.

Not the VOICEVOX **editor**. Not the VOICEVOX **ENGINE** either: no HTTP server, no second process,
no IPC, no port to manage. The plugin is a module Hermes imports, and speech comes back as bytes.

```yaml
tts:
  provider: voicevox
  voicevox:
    core: /opt/voicevox_core      # holds onnxruntime/, dict/, models/
    speaker: 3                    # ずんだもん（ノーマル）; names work too
    cpu_num_threads: 4
    idle_unload_seconds: 300
```

## Why CORE and not the engine

| | VOICEVOX ENGINE (HTTP) | VOICEVOX CORE (in-process, this plugin) |
|---|---|---|
| Extra process | yes, must be running | none |
| Port / IPC | `127.0.0.1:50021` | none |
| Startup | engine boot + model load | model load on first use, dropped when idle |
| Config surface | URL, retries, liveness | a directory path |
| Failure mode | engine offline ⇒ nothing to say | one actionable error |

Hermes already documents a `type: command` provider for the engine, so an engine-shaped plugin
would add little. This is the other half.

## Requirements

- Python 3.10+ (the wheel is `cp310-abi3`)
- Linux or macOS, glibc ≥ 2.34 on Linux (`manylinux_2_34`)
- ffmpeg **optional** — without it replies arrive as audio files instead of voice bubbles

## Install

VOICEVOX CORE's Python wheel is **not on PyPI**, and the voice models require clicking through
their licence, so this plugin ships no dependency and walks you through both.

```bash
# 1. the wheel (pick your platform from the release page)
pip install https://github.com/VOICEVOX/voicevox_core/releases/download/0.17.1/\
voicevox_core-0.17.1-cp310-abi3-manylinux_2_34_x86_64.whl

# 2. ONNX Runtime, the Open JTalk dictionary and voice models (~180 MB)
#    Read the terms first — this downloads nothing:
python bootstrap.py --show-terms --core ~/voicevox_core
#    Then, from a real terminal, accept them and fetch:
python bootstrap.py --accept --core ~/voicevox_core
#    It prints the official terms, requires you to type 「同意して取得する」, relays your answer
#    to the official downloader, and records what was accepted in <core>/.consent.json.

# 3. the plugin
git clone https://github.com/0ppaikoa/hermes-plugin-voicevox-tts /tmp/hpvv
cp -r /tmp/hpvv/hermes_voicevox ~/.hermes/plugins/voicevox
hermes plugins enable voicevox        # plugins are opt-in
```

Then set `tts.provider: voicevox` and `tts.voicevox.core` in `config.yaml`, and restart the
gateway. `hermes plugins enable` alone does nothing until the provider is selected.

## Voices

`speaker` takes an id or a name; `tts.voice` (or a per-call `voice`) overrides it:

```yaml
tts:
  voice: ずんだもん（あまあま）      # or "36", or 四国めたん
```

`hermes tools` lists every style found in your `.vvm` files — read from the model metadata, with
no ONNX Runtime loaded (0.3 ms for two files).

Which styles you get depends on the models you downloaded: `0.vvm` carries 四国めたん ×4,
ずんだもん ×4, 春日部つむぎ and 雨晴はう; `1.vvm` adds 冥鳴ひまり.

## Memory: why replies are synthesised in pieces

One CORE call costs roughly **23–34 MB of resident memory per second of generated audio**, and the
memory is never returned inside the process. Measured in an isolated 7 GiB Job:

| one call per reply | | | sentence-at-a-time | | |
|---|---|---|---|---|---|
| 100 chars | 18 s | 618 MB | 400 chars | 67.6 s | 310 MB |
| 200 chars | 36 s | 1,496 MB | 800 chars | 134.8 s | 310 MB |
| 400 chars | 71.5 s | 3,908 MB | 1,600 chars | 269.5 s | 310 MB |
| 800 chars | — | **OOMKilled** | 4,000 chars | 673.2 s | 310 MB |

So the plugin splits on `。！？` (and hard-splits long clauses) and concatenates the WAVs, keeping
each piece ≤ 40 characters ≈ 200 MB. Hermes' own splitter keys on `(?<=[.!?])\s+`, which never
fires for Japanese, so this is the only splitting that happens — and without it a long reply takes
the gateway down.

Verified live: a 1,000-character reply (166 s of audio) moved RSS by **+134 MB**.

## What it does not do (v1)

- No PCM streaming: `synthesize()` only, so speech starts when the reply is rendered. CORE's
  `streaming_synthesis()` makes it feasible later (it yields the 44-byte RIFF header then PCM at
  24 kHz).
- No custom user dictionary, no singing synthesis, no GPU tuning.
- One model resident at a time; switching character reloads (~0.7 s).

## Licence, and what you owe

The plugin is **MIT** ([LICENSE](LICENSE)). It contains no VOICEVOX code, no voice models, no
dictionary and no ONNX Runtime — it only calls the CORE installation you provide.

[NOTICE](NOTICE) lists every third-party component with its licence or terms: VOICEVOX CORE
(MIT), the voice models and the VOICEVOX ONNX Runtime (their own 利用規約), the Open JTalk
dictionary (NAIST BSD), and ffmpeg (optional, LGPL/GPL depending on build).

Two obligations land on **you**, the operator, and the plugin deliberately does not take them
over:

1. **Accepting the model terms.** The official downloader asks you to accept
   「VOICEVOX 音声モデル 利用規約」 interactively, on purpose. This plugin never answers that
   prompt, never pipes a `y` into it, and never downloads the models — it only tells you the
   command when the directory is empty.
2. **Crediting generated audio**, e.g. `VOICEVOX:ずんだもん`, whenever you publish it.

`bootstrap.py` keeps both of those yours. It is not imported by the plugin, it always shows the
official terms text before anything is fetched, and it writes `<core>/.consent.json` with the
digest of the exact text it showed you. Re-run `python bootstrap.py` at any time to see what is
installed and what was accepted.

It accepts on two channels, and records which one was used:

- `--accept` in a terminal: you type `同意して取得する` yourself.
- `--accept --approved-by <who> --approval-ref <where/when you said yes>` when there is no
  terminal — an agent relaying an instruction you actually gave in conversation. Half an
  authorisation (one flag without the other) is refused outright, and the record names the
  channel, so every acceptance is traceable to a specific "yes".

## Design notes

[docs/plan.md](docs/plan.md) and [docs/experiments.md](docs/experiments.md) carry the measured
reasoning behind each decision, including the ones the measurements reversed.
