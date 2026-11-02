"""Model backends behind one small interface, so the same agent loop runs on any of them.

  claude    Claude through the Anthropic SDK (ANTHROPIC_API_KEY)
  openai    any OpenAI-compatible chat completions API: OpenAI, Gemini, Ollama, vLLM, Groq...
            (--base-url, and the key in RECONBENCH_API_KEY or OPENAI_API_KEY)
  scripted  the procedure as code, with the same tools: the control that proves the harness
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from reconbench.tools import SPECS

# $ per million tokens: input, output, cache read. Claude API list prices (cache writes cost 1.25x input).
# Other models are priced from the command line with --price IN,OUT.
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20), "claude-sonnet-5-5": (2.0, 10.0, 0.20),
          "claude-haiku-5-5": (0.10, 0.50, 0.01)}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str
    calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stop: str = ""
    cost_usd: float = 0.0


@dataclass
class Conversation:
    """One case's dialogue with a model. Subclasses keep the history in their own format."""
    system: str
    turns: list[Turn] = field(default_factory=list)

    def send(self, user_text: str) -> Turn:
        raise NotImplementedError

    def send_results(self, results: list[tuple[ToolCall, str, bool]]) -> Turn:
        raise NotImplementedError


class Backend:
    name = "base"
    model = ""

    def conversation(self, system: str) -> Conversation:
        raise NotImplementedError


def load_dotenv(path: Path) -> None:
    """KEY=value lines from a local, git-ignored .env into the environment (existing values win)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------------- Claude
class ClaudeConversation(Conversation):
    def __init__(self, system: str, client, model: str, effort: str):
        super().__init__(system)
        self.client, self.model, self.effort = client, model, effort
        self.messages: list[dict] = []

    def _call(self) -> Turn:
        t0 = time.perf_counter()
        # Cache breakpoints: tools + procedure (the same for every case), and the conversation so far
        last = self.messages[-1]
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"]}]
        for m in self.messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                for block in m["content"]:
                    block.pop("cache_control", None)
        last["content"][-1]["cache_control"] = {"type": "ephemeral"}
        r = self.client.messages.create(
            model=self.model, max_tokens=16000,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            tools=[{**s, "strict": True} for s in SPECS], messages=self.messages,
            output_config={"effort": self.effort},
        )
        self.messages.append({"role": "assistant", "content": r.content})   # thinking blocks kept intact
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        u = r.usage
        read, write = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        p_in, p_out, p_read = PRICES.get(self.model, (0.0, 0.0, 0.0))
        cost = (u.input_tokens * p_in + write * p_in * 1.25 + read * p_read + u.output_tokens * p_out) / 1e6
        turn = Turn(text, calls, u.input_tokens + read + write, u.output_tokens, time.perf_counter() - t0,
                    r.stop_reason or "", cost)
        self.turns.append(turn)
        return turn

    def send(self, user_text: str) -> Turn:
        self.messages.append({"role": "user", "content": user_text})
        return self._call()

    def send_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": out, "is_error": err} for c, out, err in results]})
        return self._call()


class Claude(Backend):
    name = "claude"

    def __init__(self, model: str | None = None, effort: str = "medium", client=None):
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError('the claude backend needs the Anthropic SDK: pip install "reconbench[claude]"') from e
            load_dotenv(Path.cwd() / ".env")
            client = anthropic.Anthropic(max_retries=4)
        self.client = client
        self.model = model or "claude-opus-5-5"
        self.effort = effort

    def conversation(self, system: str) -> Conversation:
        return ClaudeConversation(system, self.client, self.model, self.effort)


# ---------------------------------------------------------------------------------- OpenAI-compatible
def function_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                              "parameters": s["input_schema"]}} for s in SPECS]


class ChatConversation(Conversation):
    """The chat completions protocol most providers speak, Gemini and Ollama included."""

    def __init__(self, system: str, http: httpx.Client, model: str, price: tuple[float, float]):
        super().__init__(system)
        self.http, self.model, self.price = http, model, price
        self.messages: list[dict] = [{"role": "system", "content": system}]

    def _call(self) -> Turn:
        t0 = time.perf_counter()
        for attempt in range(6):
            r = self.http.post("chat/completions", json={"model": self.model, "messages": self.messages,
                                                         "tools": function_tools()})
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(min(2 ** attempt, 30))
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        choice = data["choices"][0]
        msg = choice["message"]
        self.messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls") and v is not None})
        calls = []
        for c in msg.get("tool_calls") or []:
            args = c["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(c.get("id") or uuid.uuid4().hex[:12], c["function"]["name"], args))
        u = data.get("usage") or {}
        tin, tout = u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
        turn = Turn(msg.get("content") or "", calls, tin, tout, time.perf_counter() - t0,
                    choice.get("finish_reason") or "", (tin * self.price[0] + tout * self.price[1]) / 1e6)
        self.turns.append(turn)
        return turn

    def send(self, user_text: str) -> Turn:
        self.messages.append({"role": "user", "content": user_text})
        return self._call()

    def send_results(self, results):
        for c, out, _err in results:
            self.messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": out})
        return self._call()


class OpenAICompatible(Backend):
    name = "openai"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1/", api_key: str | None = None,
                 price: tuple[float, float] = (0.0, 0.0)):
        load_dotenv(Path.cwd() / ".env")
        key = api_key or os.environ.get("RECONBENCH_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/") + "/", headers=headers, timeout=600)
        self.model, self.price = model, price

    def conversation(self, system: str) -> Conversation:
        return ChatConversation(system, self.http, self.model, self.price)


# ---------------------------------------------------------------------------------- scripted
class ScriptedConversation(Conversation):
    """Follows the procedure step by step, reading each tool result like a model would."""

    def __init__(self, system: str, case: dict):
        super().__init__(system)
        self.case = case
        self.gen = self._plan()

    def _ask(self, name: str, **args):
        out = yield [ToolCall(uuid.uuid4().hex[:12], name, args)]
        return json.loads(out[0])

    def _plan(self):
        no = self.case["deduction_number"]
        ded = yield from self._ask("get_deduction", deduction_number=no)
        docs = [ded["doc_id"]]
        category = {"SHORTAGE": "SHORTAGE", "PRICE_CLAIM": "PRICE_CLAIM"}.get(
            ded["type"], "CHARGEBACK_LATE_SHIPMENT" if ded["reason_code"] == "LATE_SHIPMENT" else "CHARGEBACK_ASN")
        status, note = None, ""
        similar = yield from self._ask("find_similar_deductions", deduction_number=no)
        group = sorted([(ded["date"], ded["deduction_number"], ded["doc_id"])]
                       + [(s["date"], s["deduction_number"], s["doc_id"]) for s in similar])
        if group[0][1] != ded["deduction_number"]:
            gap = yield from self._ask("days_between", a=group[0][0], b=ded["date"])
            if gap["days"] <= 120:
                status, category = "CLAIMABLE", "DUPLICATE_DEDUCTION"
                docs.append(group[0][2])
        if status is None:
            kind = "purchase_order" if ded["type"] == "CHARGEBACK" else "invoice"
            found = yield from self._ask("search_records", reference=ded["reference_number"], kind=kind)
            if found["ambiguous"]:
                status = "NEEDS_REVIEW"   # the procedure as code lists candidates; it does not propose one
                note = f" Candidates: {', '.join(found['candidates'])}."
            elif not found["candidates"]:
                status = "INSUFFICIENT_EVIDENCE"
            elif kind == "invoice":
                inv = yield from self._ask("get_invoice", invoice_number=found["candidates"][0])
                docs.append(inv["doc_id"])
                line = next((ln for ln in inv["lines"] if ln["asin"] == ded["asin"]), None)
                if line is None:
                    status = "INSUFFICIENT_EVIDENCE"
                elif ded["type"] == "SHORTAGE":
                    ships = yield from self._ask("get_shipments", purchase_order=inv["purchase_order"])
                    if not ships:
                        status = "INSUFFICIENT_EVIDENCE"
                    else:
                        docs.append(ships[0]["doc_id"])
                        pod = yield from self._ask("get_proof_of_delivery", bill_of_lading=ships[0]["bill_of_lading"])
                        if "error" in pod:
                            status = "INSUFFICIENT_EVIDENCE"
                        else:
                            docs.append(pod["doc_id"])
                            full = pod["cartons_delivered"] >= ships[0]["cartons_shipped"]
                            status = "CLAIMABLE" if full else "NOT_CLAIMABLE"
                else:
                    po = yield from self._ask("get_purchase_order", purchase_order=inv["purchase_order"])
                    po_line = None if "error" in po else next(
                        (ln for ln in po["lines"] if ln["asin"] == ded["asin"]), None)
                    if po_line is None:
                        status = "INSUFFICIENT_EVIDENCE"
                    else:
                        docs.append(po["doc_id"])
                        over = line["unit_cost_cents"] > po_line["unit_cost_cents"]
                        status = "NOT_CLAIMABLE" if over else "CLAIMABLE"
            else:
                po = yield from self._ask("get_purchase_order", purchase_order=found["candidates"][0])
                ships = yield from self._ask("get_shipments", purchase_order=found["candidates"][0])
                docs.append(po["doc_id"])
                if not ships:
                    status = "INSUFFICIENT_EVIDENCE"
                else:
                    s = ships[0]
                    docs.append(s["doc_id"])
                    ok = (s["shipped_date"] <= po["ship_window_end"] if ded["reason_code"] == "LATE_SHIPMENT"
                          else s["asn_sent_date"] <= s["shipped_date"])
                    status = "CLAIMABLE" if ok else "NOT_CLAIMABLE"
        amount = 0
        if status == "CLAIMABLE":
            revs = yield from self._ask("get_reversals", deduction_number=no)
            back = sum(r["amount_cents"] for r in revs)
            docs += [r["doc_id"] for r in revs]
            amount = max(ded["amount_cents"] - back, 0)
            if back and not amount:
                status = "ALREADY_RECOVERED"
        if status == "CLAIMABLE":
            age = yield from self._ask("days_between", a=ded["date"], b="2026-07-31")
            if age["days"] > 365:
                status, amount = "EXPIRED", 0
        yield [ToolCall(uuid.uuid4().hex[:12], "submit_decision", {
            "status": status, "category": category, "recoverable_amount": f"{amount / 100:.2f}",
            "evidence": docs, "proposed_match": "", "rationale": f"Procedure result: {status}.{note}"})]

    def _next(self, value) -> Turn:
        try:
            calls = self.gen.send(value) if value is not None else next(self.gen)
        except StopIteration:
            calls = []
        t = Turn("", calls)
        self.turns.append(t)
        return t

    def send(self, user_text: str) -> Turn:
        return self._next(None)

    def send_results(self, results):
        return self._next([out for _c, out, _e in results])


class Scripted(Backend):
    name = "scripted"
    model = "procedure"

    def __init__(self):
        self.case: dict = {}

    def conversation(self, system: str) -> Conversation:
        return ScriptedConversation(system, self.case)


def make(name: str, model: str | None = None, base_url: str | None = None,
         price: tuple[float, float] | None = None, effort: str = "medium") -> Backend:
    if name == "claude":
        return Claude(model, effort)
    if name == "openai":
        if not model:
            raise ValueError("the openai backend needs --model")
        return OpenAICompatible(model, base_url or "https://api.openai.com/v1/", price=price or (0.0, 0.0))
    if name == "scripted":
        return Scripted()
    raise ValueError(f"unknown backend {name} (claude, openai, scripted)")
