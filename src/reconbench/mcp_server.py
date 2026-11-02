"""Take the benchmark with any MCP-capable agent.

`reconbench serve-mcp --case core-001 --out runs/my-agent.jsonl` serves the case's tools over stdio,
scoped to that case's vendor, with the adversarial notes in place. The agent investigates and calls
submit_decision; the decision and every tool call are appended to the run file, which
`reconbench score` grades like any other run. The expected answer never leaves this process.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from reconbench.agent import PROCEDURE
from reconbench.tools import ToolSet
from reconbench.world import World


def build(case: dict, out: Path, agent_name: str = "mcp-agent") -> MCPServer:
    tools = ToolSet(World.load(case["world"]).with_notes(case.get("notes") or {}), case["vendor_code"])
    started = time.perf_counter()
    mcp = MCPServer("reconbench", instructions=(
        PROCEDURE + f"\n\nYour case: deduction {case['deduction_number']} for vendor {case['vendor_code']}."))

    def run(name: str, **args) -> str:
        c = tools.call(name, args)
        if name == "submit_decision" and tools.decision is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "case_id": case["id"], "backend": "mcp", "model": agent_name, "decision": tools.decision,
                    "calls": [{"name": x.name, "args": x.args, "is_error": x.is_error, "doc_ids": x.doc_ids}
                              for x in tools.calls],
                    "turns": 0, "input_tokens": 0, "output_tokens": 0,
                    "seconds": round(time.perf_counter() - started, 2), "cost_usd": 0.0, "stopped": "decision",
                    "texts": []}) + "\n")
        return c.result

    @mcp.tool()
    def get_deduction(deduction_number: str) -> str:
        """The deduction being investigated, or another one by number."""
        return run("get_deduction", deduction_number=deduction_number)

    @mcp.tool()
    def search_records(reference: str, kind: str) -> str:
        """Invoices (kind='invoice') or purchase orders (kind='purchase_order') a loose reference points to.
        If 'ambiguous' is true, more than one record matches and none may be assumed."""
        return run("search_records", reference=reference, kind=kind)

    @mcp.tool()
    def get_invoice(invoice_number: str) -> str:
        """An invoice with its lines (ASIN, quantity, unit cost) and its purchase order."""
        return run("get_invoice", invoice_number=invoice_number)

    @mcp.tool()
    def get_purchase_order(purchase_order: str) -> str:
        """A purchase order: lines with agreed unit cost, and the ship window end date."""
        return run("get_purchase_order", purchase_order=purchase_order)

    @mcp.tool()
    def get_shipments(purchase_order: str) -> str:
        """Shipments for a PO: shipped date, ASN sent date, bill of lading, cartons shipped."""
        return run("get_shipments", purchase_order=purchase_order)

    @mcp.tool()
    def get_proof_of_delivery(bill_of_lading: str) -> str:
        """The carrier's proof of delivery for a bill of lading: cartons delivered and date."""
        return run("get_proof_of_delivery", bill_of_lading=bill_of_lading)

    @mcp.tool()
    def get_reversals(deduction_number: str) -> str:
        """Amounts Amazon has already paid back on a deduction."""
        return run("get_reversals", deduction_number=deduction_number)

    @mcp.tool()
    def find_similar_deductions(deduction_number: str) -> str:
        """Other deductions with the same type, reason, reference, ASIN, quantity and amount, oldest first."""
        return run("find_similar_deductions", deduction_number=deduction_number)

    @mcp.tool()
    def days_between(a: str, b: str) -> str:
        """Days from date a to date b (YYYY-MM-DD); negative if b is earlier."""
        return run("days_between", a=a, b=b)

    @mcp.tool()
    def calculator(expression: str) -> str:
        """Evaluate arithmetic: + - * / and parentheses, e.g. '125.40 - 20'."""
        return run("calculator", expression=expression)

    @mcp.tool()
    def submit_decision(status: str, category: str, recoverable_amount: str, evidence: list[str],
                        proposed_match: str = "", rationale: str = "") -> str:
        """Final answer for the case. Call exactly once, when the investigation is complete.
        status: CLAIMABLE, NOT_CLAIMABLE, INSUFFICIENT_EVIDENCE, ALREADY_RECOVERED, EXPIRED or NEEDS_REVIEW.
        recoverable_amount: in the deduction's currency, e.g. '125.40'; '0' unless CLAIMABLE.
        evidence: the doc_id of every document the decision relies on."""
        return run("submit_decision", status=status, category=category, recoverable_amount=recoverable_amount,
                   evidence=evidence, proposed_match=proposed_match, rationale=rationale)

    return mcp
