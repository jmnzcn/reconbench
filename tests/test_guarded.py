"""The rule-check replay, on made-up cases. Decisions state USD cents directly, so no world is loaded."""
from reconbench.guarded import check, markdown, replay, summarize


def case(cid, status, owed=0, track="core"):
    return {"id": cid, "track": track, "world": "core",
            "expected": {"status": status, "category": "SHORTAGE", "recoverable_usd_cents": owed,
                         "currency": "USD", "date": "2026-01-01"}}


def ep(cid, status, usd=0):
    return {"case_id": cid, "backend": "claude", "model": "m",
            "decision": {"status": status, "category": "SHORTAGE", "recoverable_usd_cents": usd}}


def test_a_claim_both_agree_on_is_filed():
    row = check(case("a", "CLAIMABLE", 1000), ep("a", "CLAIMABLE", 1000), ep("a", "CLAIMABLE", 1000))
    assert (row["outcome"], row["recovered_usd"], row["false_claim_usd"]) == ("file", 10.0, 0.0)


def test_a_claim_the_rules_do_not_support_goes_to_a_person_and_is_not_filed():
    row = check(case("a", "NOT_CLAIMABLE"), ep("a", "CLAIMABLE", 1000), ep("a", "NOT_CLAIMABLE"))
    assert (row["outcome"], row["why"]) == ("person", "disagrees with the rule check")
    assert row["before_false_claim_usd"] == 10.0 and row["false_claim_usd"] == 0.0


def test_a_claim_for_a_different_amount_goes_to_a_person():
    row = check(case("a", "CLAIMABLE", 1000), ep("a", "CLAIMABLE", 1500), ep("a", "CLAIMABLE", 1000))
    assert row["outcome"] == "person" and row["recovered_usd"] == 0.0 and row["before_recovered_usd"] == 10.0


def test_the_same_other_status_is_closed():
    row = check(case("a", "EXPIRED"), ep("a", "EXPIRED"), ep("a", "EXPIRED"))
    assert (row["outcome"], row["wrongly_closed"]) == ("close", False)


def test_review_and_no_decision_go_to_a_person_for_different_reasons():
    amb = case("a", "NEEDS_REVIEW", track="ambiguous") | {"candidates": {}}
    amb["expected"]["real_status"] = "CLAIMABLE"
    assert check(amb, ep("a", "NEEDS_REVIEW"), ep("a", "NEEDS_REVIEW"))["why"] == "ambiguous reference"
    assert check(case("b", "EXPIRED"), None, ep("b", "EXPIRED"))["why"] == "no decision"
    assert check(case("b", "EXPIRED"), ep("b", "NEEDS_REVIEW"), ep("b", "EXPIRED"))["why"] == "disagrees with the rule check"


def test_closing_on_a_wrong_agreement_is_counted_not_assumed_away():
    # If the check itself were wrong, agreeing with it would close a case that should not be closed.
    row = check(case("a", "CLAIMABLE", 1000), ep("a", "EXPIRED"), ep("a", "EXPIRED"))
    assert row["outcome"] == "close" and row["wrongly_closed"]


def test_summary_and_table():
    cases = [case("a", "CLAIMABLE", 1000), case("b", "NOT_CLAIMABLE"), case("c", "EXPIRED"), case("d", "CLAIMABLE", 500)]
    model = [ep("a", "CLAIMABLE", 1000), ep("b", "CLAIMABLE", 700), ep("c", "EXPIRED")]  # d: no decision
    rules = [ep("a", "CLAIMABLE", 1000), ep("b", "NOT_CLAIMABLE"), ep("c", "EXPIRED"), ep("d", "CLAIMABLE", 500)]
    summary, rows = replay(cases, model, rules)
    assert summary == summarize(rows)
    assert (summary["filed"], summary["closed"], summary["person"], summary["person_disagreement"]) == (1, 1, 2, 2)
    assert (summary["false_claims_before"], summary["false_claim_usd_before"], summary["false_claims"]) == (1, 7.0, 0)
    assert summary["recovered_pct_before"] == summary["recovered_pct"] == 66.7
    table = markdown([("m", summary)])
    assert "| m | 1 ($7) | 0 | 2 of 4 (50.0%) | 2 | 66.7% |" in table
