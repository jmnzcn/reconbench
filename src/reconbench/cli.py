"""reconbench: AI agents reconciling Amazon vendor deductions against known answers.

  reconbench cases                         list the tracks and how many cases each has
  reconbench run --backend claude --model claude-haiku-5-5
  reconbench run --backend openai --model gemini-2.5-flash --base-url URL --price 0.30,2.50
  reconbench score runs/claude-haiku-5-5.jsonl
  reconbench leaderboard                   score every run in runs/ and write the tables
  reconbench guarded                       replay the runs with a rule check before filing (calls no model)
  reconbench serve-mcp --case core-001 --out runs/my-agent.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from reconbench import __version__
from reconbench import cases as cases_mod
from reconbench import score as score_mod
from reconbench.agent import Budget, run_case
from reconbench.world import World


def _price(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    a, _, b = text.partition(",")
    return float(a), float(b)


def cmd_cases(args) -> int:
    cases = cases_mod.load()
    for t in cases_mod.TRACKS:
        mine = [c for c in cases if c["track"] == t]
        statuses: dict[str, int] = {}
        for c in mine:
            statuses[c["expected"]["status"]] = statuses.get(c["expected"]["status"], 0) + 1
        print(f"{t:<12} {len(mine):4d}  " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))
    print(f"{'total':<12} {len(cases):4d}")
    return 0


def cmd_run(args) -> int:
    from reconbench.backends import make

    backend = make(args.backend, args.model, args.base_url, _price(args.price), args.effort)
    name = args.name or (backend.model if args.backend != "scripted" else "scripted")
    out = Path(args.out or f"runs/{name}.jsonl")
    cases = cases_mod.load(tracks=args.track, limit=args.limit)
    done = {e["case_id"] for e in cases_mod.read_run(out) if not e["stopped"].startswith("error")}
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(todo)} cases to run ({len(cases) - len(todo)} already in {out})", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    budget = Budget()

    def one(case: dict) -> None:
        ep = run_case(case, backend, World.load(case["world"]), budget).to_dict()
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep, default=str) + "\n")
        status = (ep["decision"] or {}).get("status", "-")
        print(f"  {case['id']}: {status} (expected {case['expected']['status']}, {ep['stopped']}, "
              f"${ep['cost_usd']:.4f})", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    return cmd_score(argparse.Namespace(run=str(out), json=None, cases=True))


def _latest(episodes: list[dict]) -> list[dict]:
    """The last attempt per case, so a rerun after an API error replaces the failed one."""
    by = {}
    for e in episodes:
        by[e["case_id"]] = e
    return list(by.values())


def render(summary: dict, title: str) -> str:
    t = summary["tracks"]
    lines = [f"{title}: ReconBench score {summary['score']}  ({summary['cases']} cases)",
             f"  exact: core {t.get('core', {}).get('exact_pct')}  messy {t.get('messy', {}).get('exact_pct')}  "
             f"adversarial {t.get('adversarial', {}).get('exact_pct')}  ambiguous {t.get('ambiguous', {}).get('exact_pct')}  "
             f"policy {t.get('policy', {}).get('exact_pct')}",
             f"  false claims: {summary['false_claims']} (${summary['false_claim_usd']:,.2f})   "
             f"recovered ${summary['recovered_usd']:,.2f} of ${summary['claimable_usd']:,.2f} "
             f"({summary['recall_usd_pct']}%)"]
    if "ambiguous" in t:
        lines.append(f"  ambiguous: escalated {t['ambiguous'].get('escalated_pct')}%, "
                     f"right proposal {t['ambiguous'].get('proposal_right_pct')}%")
    if "adversarial" in t:
        lines.append(f"  adversarial: obeyed {t['adversarial'].get('obeyed')} of {t['adversarial']['cases']}, "
                     f"reported the note in {t['adversarial'].get('reported_pct')}%")
    lines.append(f"  cost ${summary['cost_usd']:.2f} (${summary['cost_per_case_usd']} per case), "
                 f"{summary['mean_seconds']} s and {summary['mean_tool_calls']} tool calls per case, "
                 f"{summary['no_decision']} without a decision")
    return "\n".join(lines)


def cmd_score(args) -> int:
    path = Path(args.run)
    episodes = _latest(cases_mod.read_run(path))
    summary, rows = score_mod.score(cases_mod.load(), episodes, partial=True)
    title = path.stem if summary["complete"] else f"{path.stem} (partial: {summary['cases']} cases)"
    print(render(summary, title))
    if args.cases:
        for r in rows:
            if not r["exact"]:
                print(f"    {r['case_id']}: answered {r['answered']}, expected {r['expected']}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    return 0


def cmd_leaderboard(args) -> int:
    from reconbench import report

    runs = sorted(Path(args.runs).glob("*.jsonl"))
    results = []
    cases = cases_mod.load()
    for p in runs:
        eps = _latest(cases_mod.read_run(p))
        summary, rows = score_mod.score(cases, eps)
        if not summary["complete"]:
            print(f"skipping {p.name}: not every case was run", file=sys.stderr)
            continue
        model = eps[0]["model"] if eps else p.stem
        results.append({"name": p.stem, "model": model, "backend": eps[0]["backend"] if eps else "",
                        "summary": summary, "rows": rows})
    results.sort(key=lambda r: (r["summary"]["score"] or 0, -r["summary"]["false_claim_usd"]), reverse=True)
    report.write(results, cases, Path(args.out))
    print(report.markdown(results))
    return 0


def cmd_guarded(args) -> int:
    from reconbench import guarded

    cases = cases_mod.load()
    rule_episodes = _latest(cases_mod.read_run(Path(args.rules)))
    if len({e["case_id"] for e in rule_episodes} & {c["id"] for c in cases}) != len(cases):
        # The rules engine's run is in the repository's runs/ folder, not in the package.
        print(f"{args.rules} is missing or incomplete: using the answer key as the rule check "
              "(the rules engine reproduces it exactly)", file=sys.stderr)
        rule_episodes = guarded.key_episodes(cases)
    results = []
    for p in sorted(Path(args.runs).glob("*.jsonl")):
        eps = _latest(cases_mod.read_run(p))
        if not eps or eps[0]["backend"] in ("rules", "scripted"):
            continue  # the rules checking themselves, and the control, say nothing here
        if len({e["case_id"] for e in eps} & {c["id"] for c in cases}) != len(cases):
            print(f"skipping {p.name}: not every case was run", file=sys.stderr)
            continue
        summary, rows = guarded.replay(cases, eps, rule_episodes)
        results.append((eps[0]["model"], summary, rows))
    results.sort(key=lambda r: r[1]["person"])
    print(guarded.markdown([(name, s) for name, s, _ in results]))
    if args.json:
        Path(args.json).write_text(json.dumps([{"model": n, "summary": s, "rows": rows} for n, s, rows in results],
                                              indent=1), encoding="utf-8")
    return 0


def cmd_serve_mcp(args) -> int:
    from reconbench.mcp_server import build

    case = next((c for c in cases_mod.load() if c["id"] == args.case), None)
    if case is None:
        print(f"no case {args.case}", file=sys.stderr)
        return 2
    build(case, Path(args.out), args.agent).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reconbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cases", help="list tracks and cases").set_defaults(fn=cmd_cases)

    r = sub.add_parser("run", help="run an agent on the benchmark")
    r.add_argument("--backend", required=True, choices=["claude", "openai", "scripted"])
    r.add_argument("--model")
    r.add_argument("--base-url", help="for --backend openai, e.g. https://generativelanguage.googleapis.com/v1beta/openai/")
    r.add_argument("--price", help="USD per million tokens, input,output (for cost reporting)")
    r.add_argument("--effort", default="medium", help="Claude effort: low, medium, high, xhigh, max")
    r.add_argument("--track", action="append", help="only this track (repeatable)")
    r.add_argument("--limit", type=int, help="at most this many cases per track")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--name", help="run name (default: the model)")
    r.add_argument("--out", help="run file (default: runs/<name>.jsonl)")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("score", help="score a run file")
    s.add_argument("run")
    s.add_argument("--cases", action="store_true", help="list every case that was not exact")
    s.add_argument("--json", help="write the summary and per-case rows here")
    s.set_defaults(fn=cmd_score)

    lb = sub.add_parser("leaderboard", help="score every run and write the leaderboard")
    lb.add_argument("--runs", default="runs")
    lb.add_argument("--out", default="docs")
    lb.set_defaults(fn=cmd_leaderboard)

    g = sub.add_parser("guarded", help="replay the saved runs with a rule check before filing; calls no model")
    g.add_argument("--runs", default="runs")
    g.add_argument("--rules", default="runs/rules-engine.jsonl", help="the rules engine's run, used as the check")
    g.add_argument("--json", help="also write every case's outcome to this file")
    g.set_defaults(fn=cmd_guarded)

    m = sub.add_parser("serve-mcp", help="serve one case's tools to an MCP agent (stdio)")
    m.add_argument("--case", required=True)
    m.add_argument("--out", required=True, help="run file the decision is appended to")
    m.add_argument("--agent", default="mcp-agent", help="name to record for the agent")
    m.set_defaults(fn=cmd_serve_mcp)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
