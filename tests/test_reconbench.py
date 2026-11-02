import json

import httpx
import pytest

from reconbench import cases as cases_mod
from reconbench import score as score_mod
from reconbench.agent import PROCEDURE, run_case
from reconbench.backends import ChatConversation, Scripted
from reconbench.world import World

CASES = cases_mod.load()
BY_ID = {c["id"]: c for c in CASES}


def episode(case_id: str, **decision) -> dict:
    return {"case_id": case_id, "decision": decision or None, "calls": [], "turns": 1, "input_tokens": 0,
            "output_tokens": 0, "seconds": 0.0, "cost_usd": 0.0, "stopped": "decision" if decision else "no decision"}


def first(track: str, status: str) -> dict:
    return next(c for c in CASES if c["track"] == track and c["expected"]["status"] == status)


# --- the data ---------------------------------------------------------------------------------

def test_tracks_and_answers_are_hidden_from_agents():
    assert {c["track"] for c in CASES} == set(cases_mod.TRACKS)
    assert len({c["id"] for c in CASES}) == len(CASES) == 261
    assert "expected" not in cases_mod.public(CASES[0])


def test_adversarial_notes_land_on_real_documents():
    for c in (c for c in CASES if c["track"] == "adversarial"):
        assert c["expected"]["status"] != "CLAIMABLE" and c["notes"] and c["injection_style"]


def test_scripted_procedure_answers_every_case():
    """The control: the procedure as code, through the same tools, gets every case right."""
    eps = [run_case(c, Scripted(), World.load(c["world"])).to_dict() for c in CASES]
    summary, rows = score_mod.score(CASES, eps)
    assert summary["score"] == 100.0 and summary["false_claims"] == 0, [r for r in rows if not r["exact"]][:3]


# --- grading ----------------------------------------------------------------------------------

def test_exact_needs_status_and_amount():
    c = first("core", "CLAIMABLE")
    exp = c["expected"]
    w = World.load(c["world"])
    rate = w.fx_rate(exp["currency"], exp["date"])
    amount = f"{exp['recoverable_usd_cents'] * rate / 100:.2f}"
    assert score_mod.grade(c, episode(c["id"], status="CLAIMABLE", recoverable_amount=amount))["exact"]
    over = f"{exp['recoverable_usd_cents'] * rate * 2 / 100:.2f}"
    row = score_mod.grade(c, episode(c["id"], status="CLAIMABLE", recoverable_amount=over))
    assert not row["exact"] and row["false_claim_usd"] > 0 and row["recovered_usd"] == exp["recoverable_usd_cents"] / 100


def test_claiming_a_valid_deduction_is_a_false_claim():
    c = first("core", "NOT_CLAIMABLE")
    row = score_mod.grade(c, episode(c["id"], status="CLAIMABLE", recoverable_amount="50.00"))
    assert row["false_claim_usd"] > 0 and not row["exact"]


def test_missing_episode_counts_as_unanswered():
    summary, rows = score_mod.score(CASES[:3], [])
    assert summary["no_decision"] == 3 and summary["exact_pct"] == 0 and not summary["complete"]


def test_partial_runs_grade_only_attempted_cases():
    c = CASES[0]
    summary, rows = score_mod.score(CASES, [episode(c["id"], status=c["expected"]["status"])], partial=True)
    assert summary["cases"] == 1 and summary["exact_pct"] == 100.0 and not summary["complete"]


def test_ambiguous_needs_escalation_and_grades_the_proposal():
    c = next(c for c in CASES if c["track"] == "ambiguous"
             and c["expected"]["real_status"] in c["candidates"].values())
    right = next(k for k, v in c["candidates"].items() if v == c["expected"]["real_status"])
    row = score_mod.grade(c, episode(c["id"], status="NEEDS_REVIEW", proposed_match=f"Most likely {right}."))
    assert row["exact"] and row["escalated"] and row["proposal_right"]
    picked = score_mod.grade(c, episode(c["id"], status="CLAIMABLE", recoverable_amount="1.00"))
    assert not picked["exact"] and not picked["escalated"]


def test_obeying_an_injection_is_counted():
    c = next(c for c in CASES if c["track"] == "adversarial")
    row = score_mod.grade(c, episode(c["id"], status="CLAIMABLE", recoverable_amount="10.00"))
    assert row["obeyed"] and row["false_claim_usd"] > 0
    calm = score_mod.grade(c, episode(c["id"], status=c["expected"]["status"],
                                      rationale="The POD carries an instruction to approve; ignored."))
    assert calm["exact"] and calm["reported"] and not calm["obeyed"]


# --- the OpenAI-compatible backend ------------------------------------------------------------

def test_chat_backend_round_trip():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "get_deduction",
                                                              "arguments": '{"deduction_number": "X"}'}}]}
        else:
            msg = {"role": "assistant", "content": "done"}
        return httpx.Response(200, json={"choices": [{"message": msg, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})

    http = httpx.Client(base_url="http://test/", transport=httpx.MockTransport(handler))
    conv = ChatConversation(PROCEDURE, http, "m", (1.0, 2.0))
    t1 = conv.send("start")
    assert t1.calls[0].name == "get_deduction" and t1.calls[0].args == {"deduction_number": "X"}
    assert t1.cost_usd == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1e6)
    t2 = conv.send_results([(t1.calls[0], '{"ok": 1}', False)])
    assert t2.text == "done"
    assert seen[1]["messages"][-1] == {"role": "tool", "tool_call_id": "t1", "name": "get_deduction",
                                       "content": '{"ok": 1}'}
    assert {t["function"]["name"] for t in seen[0]["tools"]} >= {"submit_decision", "search_records"}
