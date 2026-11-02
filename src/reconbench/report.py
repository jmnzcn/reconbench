"""The leaderboard: a Markdown table for the README and a JSON file for the results page."""
from __future__ import annotations

import json
from pathlib import Path

TRACK_COLUMNS = [("core", "Core"), ("messy", "Messy"), ("adversarial", "Adversarial"), ("ambiguous", "Ambiguous"),
                 ("policy", "Policy only")]


def _f(v, fmt="{:.1f}"):
    return "–" if v is None else fmt.format(v)


def label(r: dict) -> str:
    return {"rules": "Rules engine (vendor-recovery-engine)", "scripted": "Procedure as code (control)"}.get(
        r["backend"], r["model"])


def markdown(results: list[dict]) -> str:
    cols = " | ".join(name for _, name in TRACK_COLUMNS)
    lines = [f"| Agent | Score | {cols} | False claims | Right proposal | Cost per case |",
             "|---|---:|" + "---:|" * len(TRACK_COLUMNS) + "---:|---:|---:|"]
    for r in results:
        s, t = r["summary"], r["summary"]["tracks"]
        false = f"{s['false_claims']} (${s['false_claim_usd']:,.0f})" if s["false_claims"] else "0"
        cost = "–" if r["backend"] in ("rules", "scripted", "mcp") else f"${s['cost_per_case_usd']:.4f}"
        tracks = " | ".join(_f(t.get(k, {}).get("exact_pct")) for k, _ in TRACK_COLUMNS)
        lines.append(f"| {label(r)} | {_f(s['score'])} | {tracks} | {false} "
                     f"| {_f(t.get('ambiguous', {}).get('proposal_right_pct'), '{:.0f}%')} | {cost} |")
    return "\n".join(lines)


def write(results: list[dict], cases: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.md").write_text(markdown(results) + "\n", encoding="utf-8")
    slim = [{k: r[k] for k in ("name", "model", "backend", "summary")} | {"label": label(r), "rows": r["rows"]}
            for r in results]
    (out / "results.json").write_text(json.dumps({"cases": len(cases), "results": slim}, indent=1), encoding="utf-8")
