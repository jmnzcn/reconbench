"""What happens to a model's decisions when a claim is filed only if a rule check agrees.

This replays the main guardrail of dispute-agent (https://github.com/jmnzcn/dispute-agent) over runs that
are already saved, so it calls no model. For each case the model's decision is compared with the rules
engine's (runs/rules-engine.jsonl):

  file    both say CLAIMABLE, for the same amount to the cent
  close   both give the same other status
  person  the model asked for review, the two disagree, or the model reached no decision

The rules engine is also what defines the answer key, so no false claim can pass this check: that
column is zero by construction and is not the finding. What the replay measures is the price of the
check: how many cases end up with a person, and how much claimable money waits there.
"""
from __future__ import annotations

from reconbench.score import decision_usd_cents, grade

OUTCOMES = ("file", "close", "person")


def check(case: dict, episode: dict | None, rule_episode: dict | None) -> dict:
    row = grade(case, episode)
    d = (episode or {}).get("decision") or {}
    r = (rule_episode or {}).get("decision") or {}
    status = d.get("status")
    agrees = bool(d) and bool(r) and status == r.get("status") and (
        status != "CLAIMABLE" or abs(decision_usd_cents(case, d) - decision_usd_cents(case, r)) <= 1)
    if status == "NEEDS_REVIEW" and agrees:
        outcome, why = "person", "ambiguous reference"
    elif not agrees:
        outcome, why = "person", "no decision" if not d else "disagrees with the rule check"
    else:
        outcome, why = ("file", "") if status == "CLAIMABLE" else ("close", "")
    filed = outcome == "file"
    return {
        "case_id": case["id"], "track": case["track"], "outcome": outcome, "why": why,
        "false_claim_usd": row["false_claim_usd"] if filed else 0.0,
        "recovered_usd": row["recovered_usd"] if filed else 0.0,
        "wrongly_closed": outcome == "close" and not row["exact"],
        "before_false_claim_usd": row["false_claim_usd"], "before_recovered_usd": row["recovered_usd"],
        "claimable_usd": row["claimable_usd"], "exact": row["exact"],
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    person = [r for r in rows if r["outcome"] == "person"]
    claimable = sum(r["claimable_usd"] for r in rows)

    def pct(x: float, of: float) -> float | None:
        return round(100 * x / of, 1) if of else None

    return {
        "cases": n,
        "filed": sum(r["outcome"] == "file" for r in rows), "closed": sum(r["outcome"] == "close" for r in rows),
        "person": len(person), "person_pct": pct(len(person), n),
        "person_ambiguous": sum(r["why"] == "ambiguous reference" for r in person),
        "person_disagreement": sum(r["why"] != "ambiguous reference" for r in person),
        "false_claims_before": sum(r["before_false_claim_usd"] > 0 for r in rows),
        "false_claim_usd_before": round(sum(r["before_false_claim_usd"] for r in rows), 2),
        "false_claims": sum(r["false_claim_usd"] > 0 for r in rows),
        "wrongly_closed": sum(r["wrongly_closed"] for r in rows),
        "recovered_pct_before": pct(sum(r["before_recovered_usd"] for r in rows), claimable),
        "recovered_pct": pct(sum(r["recovered_usd"] for r in rows), claimable),
    }


def replay(cases: list[dict], episodes: list[dict], rule_episodes: list[dict]) -> tuple[dict, list[dict]]:
    by_id = {e["case_id"]: e for e in episodes}
    rules = {e["case_id"]: e for e in rule_episodes}
    rows = [check(c, by_id.get(c["id"]), rules.get(c["id"])) for c in cases]
    return summarize(rows), rows


def markdown(results: list[tuple[str, dict]]) -> str:
    lines = ["| Agent | False claims alone | False claims with the check | Sent to a person | of them, disagreements "
             "| Claimable money filed without a person |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, s in results:
        before = f"{s['false_claims_before']} (${s['false_claim_usd_before']:,.0f})" if s["false_claims_before"] else "0"
        lines.append(f"| {name} | {before} | {s['false_claims']} | {s['person']} of {s['cases']} ({s['person_pct']}%) "
                     f"| {s['person_disagreement']} | {s['recovered_pct']}% |")
    return "\n".join(lines)
