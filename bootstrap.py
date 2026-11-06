#!/usr/bin/env python3
"""Fetch VOICEVOX CORE data — with the operator's acceptance, never on their behalf.

The official downloader prints the 「VOICEVOX 音声モデル 利用規約」 and waits for a `y`. That
prompt exists so a person reads it. This script keeps it that way:

    python bootstrap.py                 # verify what is installed; say what is missing
    python bootstrap.py --show-terms    # print the official terms text, download nothing
    python bootstrap.py --accept        # show the terms, require a typed phrase, then fetch

`--accept` is still not the plugin accepting anything: it runs the official downloader with the
operator's own instruction relayed, and records what was accepted, when, and against which text.
The plugin itself (`hermes_voicevox`) never imports this file and never downloads anything.

Exit codes: 0 ok, 2 data missing, 3 refused (no TTY / wrong phrase / downloader failed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

DOWNLOADER_URL = ("https://github.com/VOICEVOX/voicevox_core/releases/download/0.17.1/"
                  "download-{os}")
PHASE = "同意して取得する"
DEFAULT_CORE = "~/voicevox_core"
CACHE = "~/.cache/hermes-voicevox"


def downloader_name() -> str:
    if sys.platform.startswith("win"):
        return "download-windows-x64.exe"
    if sys.platform == "darwin":
        return "download-osx-arm64" if os.uname().machine == "arm64" else "download-osx-x64"
    return "download-linux-arm64" if os.uname().machine in ("aarch64", "arm64") else "download-linux-x64"


def ensure_downloader(cache_dir: str) -> str:
    """Fetch the official MIT-licensed downloader binary if it is not cached yet."""
    os.makedirs(cache_dir, exist_ok=True)
    name = downloader_name()
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        url = DOWNLOADER_URL.format(os=name)
        print(f"ダウンローダーを取得します: {url}")
        urllib.request.urlretrieve(url, path)
        os.chmod(path, 0o755)
    return path


def run_downloader(binary: str, core_dir: str, answer: str) -> subprocess.CompletedProcess:
    """Run the official downloader, relaying *answer* to its consent prompt."""
    os.makedirs(core_dir, exist_ok=True)
    return subprocess.run(
        [binary, "--only", "onnxruntime", "--only", "dict", "--only", "models", "-o", core_dir],
        input=answer + "\n", text=True, capture_output=True, check=False,
    )


def show_terms(binary: str, core_dir: str) -> str:
    """Print the official terms without accepting them: answer `n` and keep the output."""
    result = run_downloader(binary, core_dir, "n")
    text = result.stdout + result.stderr
    print(text)
    return text


def data_present(core_dir: str) -> bool:
    vvms = os.path.join(core_dir, "models", "vvms")
    return (os.path.isdir(os.path.join(core_dir, "dict"))
            and os.path.isdir(os.path.join(core_dir, "onnxruntime"))
            and os.path.isdir(vvms) and any(n.endswith(".vvm") for n in os.listdir(vvms)))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def consent_record(core_dir: str, terms_text: str) -> dict:
    """What was accepted, in a form that can be re-checked when the terms change."""
    record = {
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accepted_by": os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown",
        "accepted_via": "bootstrap.py --accept (operator-typed phrase, official downloader prompt)",
        "phrase": PHASE,
        "terms_text_sha256": hashlib.sha256(terms_text.encode("utf-8")).hexdigest(),
        "assets": {},
    }
    for name, path in (
        ("models_terms", os.path.join(core_dir, "models", "TERMS.txt")),
        ("onnxruntime_terms", os.path.join(core_dir, "onnxruntime", "TERMS.txt")),
    ):
        if os.path.exists(path):
            record["assets"][name] = _sha256(path)
    vvms = os.path.join(core_dir, "models", "vvms")
    if os.path.isdir(vvms):
        record["models"] = sorted(n for n in os.listdir(vvms) if n.endswith(".vvm"))
    return record


def accept(binary: str, core_dir: str) -> int:
    if not sys.stdin.isatty():
        print("拒否: --accept は対話端末でのみ実行できます。")
        print("規約を読んで同意するのは人間の作業です。TTY から実行し直してください。",
              file=sys.stderr)
        return 3

    print("以下は公式ダウンローダーが表示する利用規約の原文です（まだ何も取得していません）。\n")
    terms_text = show_terms(binary, core_dir)

    print("\n" + "=" * 72)
    print("内容を確認したうえで、続行するには次の語句をそのまま入力してください:")
    print(f"    {PHASE}")
    print("=" * 72)
    try:
        typed = input("> ").strip()
    except EOFError:
        typed = ""
    if typed != PHASE:
        print("拒否: 入力が一致しません。何も取得していません。", file=sys.stderr)
        return 3

    print("取得します（公式ダウンローダーに、いま入力された同意を渡します）…")
    result = run_downloader(binary, core_dir, "y")
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0 or not data_present(core_dir):
        print("失敗: 取得が完了しませんでした。", file=sys.stderr)
        return 3

    record = consent_record(core_dir, terms_text)
    record_path = os.path.join(core_dir, ".consent.json")
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print(f"\n完了: {core_dir}")
    print(f"同意の記録: {record_path}")
    print(f"  規約文の sha256: {record['terms_text_sha256']}")
    return 0


def verify(core_dir: str) -> int:
    if not data_present(core_dir):
        print(f"{core_dir} に VOICEVOX CORE のデータがありません。\n")
        print("次のいずれかを、あなた自身で実行してください:\n")
        print(f"  python {os.path.basename(__file__)} --show-terms   # 規約を読みだけ（取得しない）")
        print(f"  python {os.path.basename(__file__)} --accept       # 同意して取得（対話端末で）\n")
        print("または、すでに取得済みのディレクトリを tts.voicevox.core に指定してください。")
        return 2

    record_path = os.path.join(core_dir, ".consent.json")
    print(f"データあり: {core_dir}")
    if os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as handle:
            record = json.load(handle)
        print(f"同意の記録あり: {record.get('accepted_at')} / {record.get('accepted_by')}")
        print(f"  規約文 sha256: {record.get('terms_text_sha256')}")
    else:
        print("同意の記録はありません（このスクリプト以外で取得されたデータの可能性があります）。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--core", default=os.environ.get("HERMES_VOICEVOX_CORE_DIR", DEFAULT_CORE),
                        help=f"data directory (default: {DEFAULT_CORE})")
    parser.add_argument("--show-terms", action="store_true", help="print the terms, accept nothing")
    parser.add_argument("--accept", action="store_true", help="accept with a typed phrase, then fetch")
    args = parser.parse_args(argv)

    core_dir = os.path.abspath(os.path.expanduser(args.core))

    if args.show_terms:
        show_terms(ensure_downloader(os.path.expanduser(CACHE)), core_dir)
        print("\n何も取得していません。")
        return 0
    if args.accept:
        return accept(ensure_downloader(os.path.expanduser(CACHE)), core_dir)
    return verify(core_dir)


if __name__ == "__main__":
    sys.exit(main())
