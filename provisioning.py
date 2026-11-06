"""First-run setup for the VOICEVOX CORE data: status, terms, and a consented download.

Hermes has no terminal in the loop, so the operator's "yes" arrives in conversation. This module
is what an agent drives to act on it: it never decides on its own, it refuses a download unless the
call carries the authorisation it came from, and it records that authorisation next to the digest
of the terms text it showed.

The consent itself belongs to the human: `show_terms()` only reads, and `download()` requires both
``approved_by`` and ``approval_ref`` (who said yes, and where/when they said it).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

ENV_CORE_DIR = "HERMES_VOICEVOX_CORE_DIR"
DOWNLOADER_URL = ("https://github.com/VOICEVOX/voicevox_core/releases/download/0.17.1/download-{os}")
PHASE = "同意して取得する"
DEFAULT_CORE = "~/voicevox_core"
CACHE = "~/.cache/hermes-voicevox"
DOWNLOADER_TIMEOUT = 1800


class SetupRefused(RuntimeError):
    """The action needs an authorisation that was not supplied."""


def downloader_name() -> str:
    if sys.platform.startswith("win"):
        return "download-windows-x64.exe"
    if sys.platform == "darwin":
        machine = os.uname().machine
        return "download-osx-arm64" if machine in ("arm64", "aarch64") else "download-osx-x64"
    return "download-linux-arm64" if os.uname().machine in ("aarch64", "arm64") else "download-linux-x64"


def ensure_downloader(cache_dir: Optional[str] = None) -> str:
    """The official MIT-licensed downloader binary; fetched once, cached."""
    cache_dir = os.path.expanduser(cache_dir or CACHE)
    os.makedirs(cache_dir, exist_ok=True)
    name = downloader_name()
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(DOWNLOADER_URL.format(os=name), path)
        os.chmod(path, 0o755)
    return path


def run_downloader(binary: str, core_dir: str, answer: str) -> subprocess.CompletedProcess:
    """Run the official downloader, relaying *answer* to its consent prompt."""
    os.makedirs(core_dir, exist_ok=True)
    return subprocess.run(
        [binary, "--only", "onnxruntime", "--only", "dict", "--only", "models", "-o", core_dir],
        input=answer + "\n", text=True, capture_output=True, check=False,
        timeout=DOWNLOADER_TIMEOUT,
    )


def show_terms(binary: str, core_dir: str) -> str:
    """Return the official terms text without accepting them (the downloader answers `n`)."""
    result = run_downloader(binary, core_dir, "n")
    return (result.stdout or "") + (result.stderr or "")


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


def consent_record(core_dir: str, terms_text: str, *, approved_by: str, approval_ref: str) -> dict:
    record = {
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accepted_by": approved_by,
        "approval_ref": approval_ref,
        "consent_channel": "conversation (agent relayed the operator's instruction)",
        "accepted_via": "hermes_voicevox.setup.download relaying the official downloader prompt",
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


def status(core_dir: str) -> dict[str, Any]:
    """What is installed, and what was accepted. Reads only."""
    core_dir = os.path.abspath(os.path.expanduser(core_dir))
    state: dict[str, Any] = {"core": core_dir, "data_present": data_present(core_dir),
                             "consent": None, "missing": []}
    for name, path in (("onnxruntime", "onnxruntime"), ("dict", "dict"), ("models", "models/vvms")):
        if not os.path.isdir(os.path.join(core_dir, path)):
            state["missing"].append(name)
    record_path = os.path.join(core_dir, ".consent.json")
    if os.path.exists(record_path):
        try:
            with open(record_path, encoding="utf-8") as handle:
                state["consent"] = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive
            state["consent"] = {"unreadable": str(exc)}
    return state


def download(core_dir: str, *, approved_by: str, approval_ref: str,
             terms_text: Optional[str] = None) -> dict[str, Any]:
    """Fetch the data on a recorded authorisation. Refuses without a complete one."""
    core_dir = os.path.abspath(os.path.expanduser(core_dir))
    if not approved_by or not approval_ref:
        raise SetupRefused(
            "this download needs both approved_by and approval_ref — who said yes, and where "
            "and when they said it (for example \"chat 2026-11-06: 'go ahead'\"). Ask the "
            "operator first; do not invent one."
        )

    binary = ensure_downloader()
    if terms_text is None:
        terms_text = show_terms(binary, core_dir)

    result = run_downloader(binary, core_dir, "y")
    if result.returncode != 0 or not data_present(core_dir):
        detail = ((result.stdout or "") + (result.stderr or "")).strip()[-600:]
        raise RuntimeError(f"the download did not complete (exit {result.returncode}): {detail}")

    record = consent_record(core_dir, terms_text, approved_by=approved_by, approval_ref=approval_ref)
    record_path = os.path.join(core_dir, ".consent.json")
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    return {"core": core_dir, "record": record_path, "models": record.get("models", []),
            "terms_sha256": record["terms_text_sha256"]}
