# hermes-plugin-voicevox-tts

**Status: planning. This repository contains the design, not an implementation yet.**

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) TTS provider that speaks through
[VOICEVOX CORE](https://github.com/VOICEVOX/voicevox_core) — the synthesis library, loaded
in-process.

Not the VOICEVOX **editor**. Not the VOICEVOX **ENGINE** either: no HTTP server, no extra
process, no IPC, no port to manage. The plugin is a module that Hermes imports, and speech comes
back as bytes.

```yaml
# the intended end state, not yet implemented
tts:
  provider: voicevox
  voicevox:
    core: /opt/voicevox_core        # CORE data dir (onnxruntime, dict, models)
    speaker: 3                      # ずんだもん（ノーマル）
```

## Why CORE and not the engine

| | VOICEVOX ENGINE (HTTP) | VOICEVOX CORE (in-process) |
|---|---|---|
| Extra process | yes, must be running | none |
| Port / IPC | `127.0.0.1:50021` | none |
| Startup | engine boot + model load | model load on first use, unload when idle |
| Config surface | URL, retries, engine liveness | a directory path |
| Failure mode | engine offline ⇒ no speech | import/model error ⇒ one clear message |

Hermes already ships a documented `type: command` provider path for the engine, so an
engine-shaped plugin adds little. CORE is the interesting one.

## Plan

See [docs/plan.md](docs/plan.md) for the implementation plan: architecture, the CORE API surface
this depends on, the Hermes plugin contract it has to satisfy, phases, testing strategy, and the
open questions.

## Licence and credit

The plugin itself will be MIT.

VOICEVOX CORE is MIT (0.16 and later). The **voice models** are not: they ship under the
「VOICEVOX 音声モデル 利用規約」, which permits commercial and non-commercial use *provided*
that generated audio is credited — e.g. `VOICEVOX:ずんだもん`. The plugin will carry that
obligation in its README and refuse to hide it; audio it produces is the user's to credit.
