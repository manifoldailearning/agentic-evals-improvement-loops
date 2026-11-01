"""Core: the simulated support agent, its world and its tasks.

Everything in this repository evaluates or improves one
agent. Its behavior is a deterministic function of the task,
the config and the seed, so a changed config produces a
measurable, repeatable change in eval results.
"""

from evalloop.core.agent import (
    BASELINE,
    IMPROVED,
    SimAgent,
    run_suite,
)
from evalloop.core.failures import (
    FAILURE_MODES,
    MODES,
    FailureMode,
    failure_probs,
)
from evalloop.core.tasks import (
    DEMO_TASKS,
    load_tasks,
    make_tasks,
    save_tasks,
)
from evalloop.core.types import (
    CATEGORIES,
    MODELS,
    RULES,
    AgentConfig,
    Step,
    Task,
    Trajectory,
    rng_for,
    stable_hash,
)
from evalloop.core.world import KB, ORDERS, Order, World

__all__ = [
    "BASELINE",
    "CATEGORIES",
    "DEMO_TASKS",
    "FAILURE_MODES",
    "IMPROVED",
    "KB",
    "MODELS",
    "MODES",
    "ORDERS",
    "RULES",
    "AgentConfig",
    "FailureMode",
    "Order",
    "SimAgent",
    "Step",
    "Task",
    "Trajectory",
    "World",
    "failure_probs",
    "load_tasks",
    "make_tasks",
    "rng_for",
    "run_suite",
    "save_tasks",
    "stable_hash",
]
