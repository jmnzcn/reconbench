"""A vendor's systems of record, as the agent's tools see them.

Each benchmark world is a set of clean tables (purchase orders, shipments, proofs of delivery,
invoices, deductions) for a handful of synthetic vendors, plus ECB exchange rates. Every record
handed to an agent carries a `doc_id`, so a decision can be traced to the documents it relied on,
and a free-text `notes` field: real carrier and reviewer documents have one, and the adversarial
track hides instructions in it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

DATA = Path(__file__).resolve().parent / "data"
AS_OF = "2026-07-31"
MIN_SUFFIX_DIGITS = 4

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def canonical_id(value: object) -> str:
    """'inv-0012345 ' -> 'INV12345': no separators or case, no leading zeros in the number."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    s = _NON_ALNUM.sub("", str(value).upper())
    m = re.fullmatch(r"([A-Z]*)0*(\d+)", s)
    return m.group(1) + m.group(2) if m else s


def money(cents: int, currency: str) -> str:
    return f"{cents / 100:.2f} {currency}"


def day(ts) -> str:
    return "" if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


@dataclass
class World:
    """Typed tables for every vendor of one world, plus the exchange rates."""
    tables: dict[str, pd.DataFrame]
    rates: pd.DataFrame
    notes: dict[str, str] = field(default_factory=dict)  # doc_id -> text shown as that document's notes

    @classmethod
    def load(cls, name: str, root: Path = DATA) -> World:
        return _load(name, str(root))

    def with_notes(self, notes: dict[str, str]) -> World:
        w = World(tables=self.tables, rates=self.rates, notes=dict(notes))
        w.__dict__.update({k: self.__dict__[k] for k in ("_by_vendor", "_deductions") if k in self.__dict__})
        return w

    @cached_property
    def _by_vendor(self) -> dict[tuple[str, str], pd.DataFrame]:
        return {(table, v): g for table, t in self.tables.items() for v, g in t.groupby("vendor_code")}

    def _t(self, table: str, vendor: str) -> pd.DataFrame:
        hit = self._by_vendor.get((table, vendor))
        return hit if hit is not None else self.tables[table].iloc[0:0]

    def _doc(self, doc_id: str, body: dict) -> dict:
        return {"doc_id": doc_id, **body, "notes": self.notes.get(doc_id, "")}

    @cached_property
    def _deductions(self) -> pd.DataFrame:
        d = self.tables["deductions"].copy()
        d["canon"] = d["deductionNumber"].map(canonical_id)
        d["ref_canon"] = d["referenceNumber"].map(canonical_id)
        return d

    def deduction(self, vendor: str, number: str) -> dict | None:
        d = self._deductions
        row = d[(d["vendor_code"] == vendor) & (d["canon"] == canonical_id(number))]
        if row.empty:
            return None
        r = row.iloc[0]
        return self._doc(f"DED:{r['deductionNumber']}", {
            "deduction_number": r["deductionNumber"], "date": day(r["deductionDate"]),
            "type": r["deductionType"], "reason_code": r["reasonCode"] or "",
            "reference_number": r["referenceNumber"], "asin": r["amazonProductIdentifier"],
            "quantity": int(r["quantity"]), "amount": money(int(r["amount"]), r["currencyCode"]),
            "amount_cents": int(r["amount"]), "currency": r["currencyCode"],
        })

    def search(self, vendor: str, reference: str, kind: str) -> dict:
        """Candidates for a loosely quoted reference, by the first tier that finds any.

        Tiers: exact, canonical (case, separators, leading zeros), suffix (a digits-only reference
        of 4+ digits that ends a record's digits; invoices only) and fuzzy (one edit away; purchase
        orders only). More than one candidate means the reference is ambiguous.
        """
        table, col = {"invoice": ("invoices", "invoiceNumber"),
                      "purchase_order": ("purchase_orders", "purchaseOrderNumber")}[kind]
        ids = pd.Series(self._t(table, vendor)[col].unique())
        ref = str(reference).strip()
        canon = canonical_id(ref)
        tiers = [("exact", ids[ids == ref]), ("canonical", ids[ids.map(canonical_id) == canon])]
        digits = re.sub(r"\D", "", ref)
        if kind == "invoice" and ref.isdigit() and len(digits) >= MIN_SUFFIX_DIGITS:
            tiers.append(("suffix", ids[ids.map(lambda x: re.sub(r"\D", "", x).endswith(digits))]))
        if kind == "purchase_order":
            tiers.append(("fuzzy", ids[ids.map(lambda x: Levenshtein.distance(canonical_id(x), canon) <= 1)]))
        for tier, hits in tiers:
            if len(hits):
                found = sorted(hits.tolist())
                return {"reference": ref, "kind": kind, "tier": tier, "candidates": found,
                        "ambiguous": len(found) > 1}
        return {"reference": ref, "kind": kind, "tier": "none", "candidates": [], "ambiguous": False}

    def invoice(self, vendor: str, number: str) -> dict | None:
        t = self._t("invoices", vendor)
        rows = t[t["invoiceNumber"] == number]
        if rows.empty:
            return None
        h = rows.iloc[0]
        lines = [{"asin": r["amazonProductIdentifier"], "quantity": int(r["invoicedQuantity"]),
                  "unit_cost": money(int(r["netCost"]), r["currencyCode"]), "unit_cost_cents": int(r["netCost"])}
                 for _, r in rows.sort_values("itemSequenceNumber").iterrows()]
        return self._doc(f"INV:{number}", {
            "invoice_number": number, "date": day(h["invoiceDate"]), "purchase_order": h["purchaseOrderNumber"],
            "currency": h["currencyCode"], "lines": lines})

    def purchase_order(self, vendor: str, number: str) -> dict | None:
        t = self._t("purchase_orders", vendor)
        rows = t[t["purchaseOrderNumber"].map(canonical_id) == canonical_id(number)]
        if rows.empty:
            return None
        h = rows.iloc[0]
        po = h["purchaseOrderNumber"]
        lines = [{"asin": r["amazonProductIdentifier"], "ordered_quantity": int(r["orderedQuantity"]),
                  "unit_cost": money(int(r["netCost"]), r["currencyCode"]), "unit_cost_cents": int(r["netCost"])}
                 for _, r in rows.sort_values("itemSequenceNumber").iterrows()]
        return self._doc(f"PO:{po}", {
            "purchase_order": po, "date": day(h["purchaseOrderDate"]),
            "ship_window_end": day(rows["shipWindowEnd"].max()), "currency": h["currencyCode"], "lines": lines})

    def shipments(self, vendor: str, po: str) -> list[dict]:
        t = self._t("shipments", vendor)
        rows = t[t["purchaseOrderNumber"].map(canonical_id) == canonical_id(po)]
        return [self._doc(f"SHP:{r['shipmentIdentifier']}", {
            "shipment_id": r["shipmentIdentifier"], "purchase_order": r["purchaseOrderNumber"],
            "shipped_date": day(r["shippedDate"]), "asn_sent_date": day(r["shipmentConfirmationDate"]),
            "bill_of_lading": r["billOfLadingNumber"], "cartons_shipped": int(r["cartonCount"])})
            for _, r in rows.drop_duplicates("shipmentIdentifier").iterrows()]

    def proof_of_delivery(self, vendor: str, bol: str) -> dict | None:
        t = self._t("proof_of_delivery", vendor)
        rows = t[t["billOfLadingNumber"] == bol]
        if rows.empty:
            return None
        r = rows.iloc[0]
        return self._doc(f"POD:{r['podDocumentId']}", {
            "pod_document_id": r["podDocumentId"], "bill_of_lading": bol, "carrier": r["carrier"],
            "delivered_date": day(r["deliveredDate"]), "cartons_delivered": int(r["cartonsDelivered"])})

    def reversals(self, vendor: str, number: str) -> list[dict]:
        d = self._deductions
        rows = d[(d["vendor_code"] == vendor) & (d["deductionType"] == "REVERSAL")
                 & (d["ref_canon"] == canonical_id(number))]
        return [self._doc(f"DED:{r['deductionNumber']}", {
            "reversal_number": r["deductionNumber"], "date": day(r["deductionDate"]),
            "reverses": r["referenceNumber"], "amount": money(int(r["amount"]), r["currencyCode"]),
            "amount_cents": int(r["amount"]), "currency": r["currencyCode"]}) for _, r in rows.iterrows()]

    def similar_deductions(self, vendor: str, number: str) -> list[dict]:
        d = self._deductions
        me = d[(d["vendor_code"] == vendor) & (d["canon"] == canonical_id(number))]
        if me.empty:
            return []
        m = me.iloc[0]
        same = d[(d["vendor_code"] == vendor) & (d["canon"] != m["canon"]) & (d["deductionType"] == m["deductionType"])
                 & (d["reasonCode"].fillna("") == (m["reasonCode"] or "")) & (d["ref_canon"] == m["ref_canon"])
                 & (d["amazonProductIdentifier"] == m["amazonProductIdentifier"])
                 & (d["quantity"] == m["quantity"]) & (d["amount"] == m["amount"])]
        return [self._doc(f"DED:{r['deductionNumber']}", {
            "deduction_number": r["deductionNumber"], "date": day(r["deductionDate"]),
            "amount": money(int(r["amount"]), r["currencyCode"])})
            for _, r in same.sort_values(["deductionDate", "deductionNumber"]).iterrows()]

    def fx_rate(self, currency: str, date: str) -> float | None:
        """Units of currency per USD: the latest ECB fixing on or before the date."""
        if currency == "USD":
            return 1.0
        r = self.rates[(self.rates["currency"] == currency) & (self.rates["date"] <= pd.Timestamp(date))]
        return None if r.empty else float(r.iloc[-1]["per_usd"])

    def to_usd_cents(self, cents: int, currency: str, date: str) -> int:
        rate = self.fx_rate(currency, date)
        return int(round(cents / rate)) if rate else 0


@cache
def _load(name: str, root: str) -> World:
    folder = Path(root) / "worlds" / name
    if not folder.exists():
        raise FileNotFoundError(f"no world named {name!r} in {Path(root) / 'worlds'}")
    tables = {p.stem: pd.read_parquet(p) for p in folder.glob("*.parquet")}
    rates = pd.read_parquet(Path(root) / "fx_usd.parquet")
    rates["date"] = pd.to_datetime(rates["date"]).astype("datetime64[ns]")
    return World(tables=tables, rates=rates.sort_values("date").reset_index(drop=True))
