"""Persistent monetary reservations; ambiguous/failed requests retain their maximum charge."""
from pathlib import Path
import time
from harness.data import atomic_json, read_json
from harness.storage import exclusive, utc_now


class PaidBudget:
    def __init__(self, directory, cap=30):
        self.path = Path(directory) / "cost-ledger.json"
        self.lock = self.path.with_suffix(".lock")
        self.cap = float(cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self):
        return read_json(self.path) if self.path.exists() else {
            "currency": "CNY", "cap": self.cap, "approval": "user cumulative maximum CNY 30 incl failures/retries",
            "entries": [], "billing_basis": "public unit rates; actual invoice unavailable",
        }

    def reserve(self, identity, quantities, prices):
        if any(not isinstance(v, int) or v < 0 for v in quantities.values()):
            raise ValueError("nonnegative exact token quantities required")
        cost = sum(quantities[k] * prices[k] / 1e6 for k in quantities)
        with exclusive(self.lock):
            state = self.read()
            if state["cap"] != self.cap or self.cap > 30:
                raise ValueError("approved cap mismatch")
            prior = next((row for row in state["entries"] if row["id"] == identity), None)
            if prior:
                raise ValueError("request already reserved; inspect saved result, never resend implicitly")
            if sum(row["charged_upper_cny"] for row in state["entries"]) + cost > self.cap:
                raise RuntimeError("approved monetary cap exhausted")
            state["entries"].append({"id": identity, "quantities_reserved": quantities, "prices": prices,
                                     "status": "started", "charged_upper_cny": cost, "started_at": utc_now()})
            atomic_json(self.path, state)
        return cost

    def finish(self, identity, *, actual=None, status="completed", metadata=None):
        with exclusive(self.lock):
            state = self.read()
            row = next(r for r in state["entries"] if r["id"] == identity)
            if actual is not None:
                if any(v < 0 or v > row["quantities_reserved"].get(k, 0) for k, v in actual.items()):
                    raise RuntimeError("provider token usage exceeded reservation; stop all further paid work")
                # Only complete sample usage can release a reservation. Train quantities remain fully charged.
                row["actual_tokens"] = actual
                row["charged_upper_cny"] = sum(v * row["prices"][k] / 1e6 for k, v in actual.items())
            row.update(status=status, metadata=metadata or {}, finished_at=utc_now())
            state["charged_upper_cny"] = sum(x["charged_upper_cny"] for x in state["entries"])
            state["remaining_cny"] = self.cap - state["charged_upper_cny"]
            atomic_json(self.path, state)
        return state


def guarded_call(budget, identity, quantities, prices, fn):
    budget.reserve(identity, quantities, prices)
    start = time.monotonic()
    try:
        result = fn()
    except Exception as exc:
        budget.finish(identity, status="failed_or_ambiguous", metadata={"error_type": type(exc).__name__, "seconds": time.monotonic() - start})
        raise
    budget.finish(identity, metadata={"seconds": time.monotonic() - start})
    return result
