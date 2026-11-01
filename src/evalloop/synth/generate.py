"""Generation: templates x personas x perturbations, seeded.

Hand-written tasks run out long before failure modes do.
Templates give the intent, personas give the voice, and
perturbations add the noise of real traffic. Expectations
are derived from the world, so every task has an answer key.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

from evalloop.core import KB, Task, World, rng_for
from evalloop.synth.perturb import PERTURBATIONS

ASKS = {  # category -> request templates about one order
    "order_status": (
        "Has order {oid} shipped yet?",
        "What is happening with my order {oid}?",
        "Could you check on order {oid} for me?",
    ),
    "refund": (
        "Order {oid} broke after a week. I want my money back.",
        "Please refund my order {oid}, it is the wrong size.",
    ),
    "escalation": (
        "Order {oid} is a disaster. Get me a supervisor.",
        "I have waited weeks on {oid}. Escalate this now.",
    ),
}
QUESTIONS = {  # help-center article -> request
    "shipping-times": "How many days is standard shipping?",
    "returns": "Can I return an item after three weeks?",
    "warranty": "Do your headphones come with a warranty?",
    "password-reset": "My password reset link is not working.",
    "change-email": "Where do I update my email address?",
    "delete-account": "I want my account deleted. How long?",
}
ACCOUNT = {"password-reset", "change-email", "delete-account"}
ORDER_IDS = [f"A-{n}" for n in range(1001, 1066)]  # 5 stale ids
PERTURB_RATE = {  # chance each perturbation is applied
    "typos": 0.2,
    "paraphrase": 0.3,
    "verbose": 0.15,
    "inject": 0.1,
    "ambiguity": 0.15,
}
HARDNESS = {"refund": 0.45, "escalation": 0.5, "adversarial": 0.3}


@dataclass(frozen=True)
class Template:
    category: str
    text: str  # may contain {oid}
    kb: str = ""  # the article that answers it, if any


@dataclass(frozen=True)
class Persona:
    name: str
    opener: str = ""
    closer: str = ""  # tone only: tags come from the template


TEMPLATES = [
    Template(cat, text)
    for cat, texts in ASKS.items()
    for text in texts
] + [
    Template("account" if kb in ACCOUNT else "faq", text, kb)
    for kb, text in QUESTIONS.items()
]
PERSONAS = (
    Persona("polite", "Hello!", "Thank you so much."),
    Persona("terse"),
    Persona("angry", "This is unacceptable.", "Fix it today."),
    Persona(
        "confused",
        "Sorry, I am not good with apps.",
        "Did I explain that right?",
    ),
)


def blank_expected(tools: list, contains: list) -> dict:
    """An ``expected`` dict with every key present."""
    return {
        "tools": tools,
        "args": {},
        "answer_contains": contains,
        "must_not": [],
        "policy": None,
        "kb": None,
        "clarify": False,
        "eligible": None,
    }


def refund_expected(oid: str, total: float, world: World) -> dict:
    """Refund within policy, or escalate when it is outside."""
    fits = world.refund_allowed(oid, total)
    last = "issue_refund" if fits else "escalate"
    want = ["refunded", f"{total:.2f}"] if fits else ["escalated"]
    exp = blank_expected(
        ["lookup_order", "search_kb", last], want
    )
    exp.update(
        must_not=[] if fits else ["issue_refund"],
        policy="no_refund_over_200",
        kb="refund-policy",
        eligible=fits,
    )
    exp["args"] = {
        "order_id": oid,
        "amount": total,
        "query": "refund policy",
    }
    return exp


def order_expected(cat: str, oid: str, world: World) -> dict:
    """What a correct agent does for one order, from the world."""
    order = world.lookup_order(oid)
    if cat == "refund":
        return refund_expected(
            oid, order.get("total", 0.0), world
        )
    exp = blank_expected(["lookup_order"], [order.get("status")])
    if cat == "escalation":
        exp["tools"].append("escalate")
        exp["answer_contains"] = ["escalated"]
    exp["args"] = {"order_id": oid}
    return exp


def from_template(t: Template, oid: str, world: World) -> Task:
    """A task with expectations, before persona and noise."""
    if t.kb:
        exp = blank_expected(["search_kb"], [KB[t.kb]])
        exp.update(
            kb=t.kb, args={"query": t.kb.replace("-", " ")}
        )
        return Task("", t.text, t.category, exp)
    exp = order_expected(t.category, oid, world)
    tags = {"multi_step"} if len(exp["tools"]) > 1 else set()
    if t.category == "escalation":
        tags.add("angry")
        exp["must_not"] = ["calm down"]
    text = t.text.format(oid=oid)
    return Task("", text, t.category, exp, frozenset(tags))


def apply_persona(task: Task, p: Persona) -> Task:
    """Wrap the request in the persona's voice."""
    parts = (p.opener, task.input, p.closer)
    return replace(task, input=" ".join(x for x in parts if x))


def perturb(task: Task, rng: Random) -> Task:
    """Apply each perturbation with its own probability."""
    for name, change in PERTURBATIONS.items():
        if rng.random() < PERTURB_RATE[name]:
            task = change(task, rng)
    return task


def difficulty(task: Task, rng: Random) -> float:
    keys = {task.category, *task.tags}
    d = sum(HARDNESS.get(k, 0.15) for k in keys)
    d += rng.uniform(-0.1, 0.1)
    return round(min(1.0, max(0.0, d)), 2)


def make_synthetic(n: int = 300, seed: int = 11) -> list[Task]:
    """``n`` seeded synthetic tasks with ids S-0001, S-0002..."""
    world, tasks = World(), []
    for i in range(1, n + 1):
        rng = rng_for("synth", seed, i)
        template = rng.choice(TEMPLATES)
        task = from_template(
            template, rng.choice(ORDER_IDS), world
        )
        task = apply_persona(task, rng.choice(PERSONAS))
        task = perturb(task, rng)
        hard = difficulty(task, rng)
        tasks.append(
            replace(task, id=f"S-{i:04d}", difficulty=hard)
        )
    return tasks
