#!/usr/bin/env python3
"""Command-line front end for :mod:`hermes_voicevox.setup`.

    python bootstrap.py                  # what is installed, what was accepted
    python bootstrap.py --show-terms     # print the official terms; download nothing
    python bootstrap.py --accept         # terminal: type the phrase, then fetch
    python bootstrap.py --accept --approved-by "yuki-js" --approval-ref "chat 2026-11-06: 'go'"
                                         # no terminal: relay an instruction that was given

Agents should call the ``voicevox_setup`` tool rather than shelling out — same code, and the tool
documents the authorisation it needs. Nothing here decides anything: the acceptance is the
operator's, and ``<core>/.consent.json`` records which conversation it came from.

Exit codes: 0 ok, 2 data missing, 3 refused (no authorisation / wrong phrase / failed download).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from hermes_voicevox import setup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--core", default=os.environ.get(setup.ENV_CORE_DIR, setup.DEFAULT_CORE),
                        help=f"data directory (default: {setup.DEFAULT_CORE})")
    parser.add_argument("--show-terms", action="store_true", help="print the terms, accept nothing")
    parser.add_argument("--accept", action="store_true", help="accept, then fetch")
    parser.add_argument("--approved-by", default="", help="who authorised the download")
    parser.add_argument("--approval-ref", default="", help="where and when they authorised it")
    args = parser.parse_args(argv)

    core_dir = os.path.abspath(os.path.expanduser(args.core))

    if args.show_terms:
        print(setup.show_terms(setup.ensure_downloader(), core_dir))
        print("\n何も取得していません。")
        return 0

    if args.accept:
        approved_by, approval_ref = args.approved_by, args.approval_ref
        if not approved_by and sys.stdin.isatty():
            print(setup.show_terms(setup.ensure_downloader(), core_dir))
            print("\n続行するには次の語句をそのまま入力してください: " + setup.PHASE)
            try:
                if input("> ").strip() != setup.PHASE:
                    print("拒否: 入力が一致しません。何も取得していません。", file=sys.stderr)
                    return 3
            except EOFError:
                return 3
            approved_by = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
            approval_ref = "terminal confirmation (phrase typed)"
        try:
            result = setup.download(core_dir, approved_by=approved_by, approval_ref=approval_ref)
        except setup.SetupRefused as exc:
            print(f"拒否: {exc}", file=sys.stderr)
            return 3
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    state = setup.status(core_dir)
    if not state["data_present"]:
        print(f"{state['core']} に VOICEVOX CORE のデータがありません"
              f"（不足: {', '.join(state['missing']) or 'models'}）\n")
        print("  --show-terms  規約を読む（何も取得しない）")
        print("  --accept      同意して取得（端末なら語句入力、無ければ --approved-by/--approval-ref）")
        print("  または、取得済みのディレクトリを tts.voicevox.core に指定する")
        return 2

    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
