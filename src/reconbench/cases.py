"""The benchmark's cases and the runs recorded against them."""
from __future__ import annotations

import json
from pathlib import Path

from reconbench.world import DATA

TRACKS = ("core", "messy", "adversarial", "ambiguous", "policy")


def load(path: Path = DATA / "cases.jsonl", tracks: list[str] | None = None, limit: int | None = None) -> list[dict]:
    cases = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if tracks:
        unknown = set(tracks) - set(TRACKS)
        if unknown:
            raise ValueError(f"unknown tracks {sorted(unknown)}; choose from {', '.join(TRACKS)}")
        cases = [c for c in cases if c["track"] in tracks]
    if limit:
        per: dict[str, int] = {}
        kept = []
        for c in cases:
            if per.get(c["track"], 0) < limit:
                kept.append(c)
                per[c["track"]] = per.get(c["track"], 0) + 1
        cases = kept
    return cases


def public(case: dict) -> dict:
    """What an agent may see about a case: never the expected answer."""
    return {k: case[k] for k in ("id", "track", "world", "vendor_code", "deduction_number")}


def read_run(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
