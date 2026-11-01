"""Chapter 2: the anatomy of one agent run, end to end.

A single task goes in; a trajectory comes out: tool calls
with arguments and results, a final answer, tokens, latency
and cost. The ground-truth faults show what went wrong, and
the same task under the improved config shows the fix.
"""

from __future__ import annotations

import textwrap

from evalloop.core import (
    BASELINE,
    DEMO_TASKS,
    IMPROVED,
    SimAgent,
    Step,
    make_tasks,
    run_suite,
)

KEYS = ("error", "status", "article", "refund_id", "ticket")


def brief(step: Step) -> str:
    """One line per tool call: name(args) -> key result."""
    args = ", ".join(f"{k}={v}" for k, v in step.args.items())
    shown = next(step.result[k] for k in KEYS if k in step.result)
    return f"{step.tool}({args}) -> {shown}"


def success(config, tasks) -> float:
    runs = run_suite(SimAgent(config), tasks)
    return sum(r.ok for r in runs) / len(runs)


def main() -> None:
    task = next(t for t in DEMO_TASKS if t.id == "T-0012")
    tags = ",".join(sorted(task.tags))
    print(f"Task {task.id} {task.category} [{tags}]")
    tail = task.input[-56:].split(" ", 1)[1]
    print(f'  "...{tail}"')
    want = ", ".join(task.expected["tools"])
    print(f"  expected: {want}")
    c = BASELINE
    print(
        f"Config {c.version}: model={c.model}"
        f" rules={len(c.rules)} shots={c.few_shots}"
        f" temp={c.temperature}"
    )
    run = SimAgent(c).run(task)
    for i, step in enumerate(run.tool_calls, 1):
        print(f" {i} {brief(step)}")
    for line in textwrap.wrap("Answer: " + run.answer, 62):
        print("  " + line)
    print(
        f"Usage: {run.tokens_in} in / {run.tokens_out} out,"
        f" {run.latency_ms} ms, ${run.cost:.6f}"
    )
    print(f"Faults (ground truth): {', '.join(run.faults)}")
    fixed = SimAgent(IMPROVED).run(task)
    tools = ", ".join(s.tool for s in fixed.tool_calls)
    print(f"Improved: {tools}; faults: {fixed.faults or 'none'}")
    for line in textwrap.wrap("Answer: " + fixed.answer, 62):
        print("  " + line)
    tasks = make_tasks(200, seed=7)
    base = success(BASELINE, tasks)
    best = success(IMPROVED, tasks)
    print(
        f"Suite (200 tasks): baseline {base:.1%}"
        f" -> improved {best:.1%}"
    )


if __name__ == "__main__":
    main()
