"""The support desk the agent works in: orders, articles, tools.

Evals need ground truth the agent cannot argue with. The
world holds a fixed order book, a small help center and the
refund policy (up to 200, within 30 days). Tool results are
pure functions of the world, and every side effect (refund,
escalation) is recorded so checks can inspect what happened.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from evalloop.core.types import rng_for, stable_hash

STATUSES = ("processing", "shipped", "delivered", "delayed")
NAMES = ("Asha", "Ben", "Chen", "Dana", "Eli", "Farah")
TOOLS = ("lookup_order", "search_kb", "issue_refund", "escalate")
KB = {  # article id -> text
    "shipping-times": "Shipping takes 3-5 business days.",
    "returns": "Items can be returned within 30 days.",
    "refund-policy": "Refunds: up to 200, within 30 days.",
    "warranty": "Electronics carry a 12-month warranty.",
    "password-reset": "Use Forgot password; links last 24 hours.",
    "change-email": "Change your email in Settings > Profile.",
    "delete-account": "Account deletion completes within 7 days.",
}
FACTS = {  # article id -> (true fact, plausible false version)
    "shipping-times": ("3-5 business days", "1-2 business days"),
    "returns": ("30 days", "90 days"),
    "refund-policy": ("200", "500"),
    "warranty": ("12-month", "36-month"),
    "password-reset": ("Forgot password", "a support email"),
    "change-email": ("Settings > Profile", "Settings > Billing"),
    "delete-account": ("7 days", "24 hours"),
}


@dataclass(frozen=True)
class Order:
    id: str
    status: str
    total: float
    customer: str
    days_ago: int


def make_orders(n: int = 60) -> dict[str, Order]:
    rng = rng_for("orders")
    orders = {}
    for i in range(n):
        oid = f"A-{1001 + i}"
        orders[oid] = Order(
            oid,
            rng.choice(STATUSES),
            round(rng.uniform(15, 450), 2),
            rng.choice(NAMES),
            rng.randint(1, 60),
        )
    return orders


ORDERS = make_orders()


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass
class World:
    orders: dict[str, Order] = field(
        default_factory=lambda: dict(ORDERS)
    )
    refund_limit: float = 200.0
    refund_days: int = 30
    effects: list[dict[str, Any]] = field(default_factory=list)

    def call(self, tool: str, args: dict[str, Any]) -> dict:
        """Run a tool by name; a bad call returns an error."""
        if tool not in TOOLS:
            return {"error": f"unknown tool {tool}"}
        try:
            return getattr(self, tool)(**args)
        except TypeError:
            return {"error": "missing or invalid arguments"}

    def lookup_order(self, order_id: str) -> dict:
        order = self.orders.get(order_id)
        if order is None:
            return {"error": f"order {order_id} not found"}
        return asdict(order)

    def search_kb(self, query: str) -> dict:
        q = words(query)
        best = max(KB, key=lambda k: len(q & words(k)))
        if not q & words(best):
            return {"error": "no matching article"}
        return {"article": best, "text": KB[best]}

    def refund_allowed(
        self, order_id: str, amount: float
    ) -> bool:
        order = self.orders.get(order_id)
        return bool(
            order
            and amount <= self.refund_limit
            and order.days_ago <= self.refund_days
        )

    def issue_refund(self, order_id: str, amount: float) -> dict:
        ok = self.refund_allowed(order_id, amount)
        self.effects.append(
            {"refund": order_id, "amount": amount, "ok": ok}
        )
        ref = "R-" + stable_hash(order_id, amount)[:4]
        return {"refund_id": ref, "amount": amount}

    def escalate(self, reason: str, order_id: str = "") -> dict:
        self.effects.append({"escalate": order_id, "why": reason})
        return {
            "ticket": "ESC-" + stable_hash(reason, order_id)[:4]
        }
