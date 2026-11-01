"""Multi-turn conversations between a simulated user and agent.

The agent answers one message at a time, so a loop supplies
the turns: each is a fresh run on the task the user's latest
message implies. Turn 1 replays the single-turn eval; later
turns use other seeds, so a plain retry re-rolls only the
temperature share of the agent's choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evalloop.core import SimAgent, Task, Trajectory
from evalloop.simulate.users import (
    PERSONAS,
    Persona,
    frustration,
    goal_for,
    in_character,
    satisfied,
    user_reply,
)

SEED_STRIDE = 1000  # keeps turn seeds apart across seeds


@dataclass(frozen=True)
class Turn:
    task: Task  # what the agent was asked this turn
    run: Trajectory  # what it did and said


@dataclass
class Conversation:
    goal: Task
    persona: str
    turns: list[Turn] = field(default_factory=list)
    outcome: str = "out_of_turns"  # satisfied | gave_up
    frustration: int = 0


def turn_seed(seed: int, turn: int) -> int:
    """Turn 1 uses the eval seed itself; later turns do not."""
    return seed + SEED_STRIDE * (turn - 1)


def converse(
    agent: SimAgent,
    task: Task,
    persona: Persona,
    goal: Task,
    seed: int = 0,
    max_turns: int = 6,
) -> Conversation:
    """Play one conversation until the user stops."""
    conv = Conversation(goal, persona.name)
    asked = in_character(task, persona)
    for n in range(1, max_turns + 1):
        run = agent.run(asked, seed=turn_seed(seed, n))
        conv.turns.append(Turn(asked, run))
        if satisfied(run.answer, goal):
            conv.outcome = "satisfied"
            break
        conv.frustration += frustration(run.answer)
        if conv.frustration > persona.patience:
            conv.outcome = "gave_up"
            break
        asked = user_reply(asked, goal, persona, n)
    return conv


def simulate_suite(
    agent: SimAgent, tasks, personas=PERSONAS, seed: int = 0
) -> list[Conversation]:
    """One conversation per task, personas assigned in turn."""
    return [
        converse(
            agent,
            task,
            personas[i % len(personas)],
            goal_for(task, tasks),
            seed,
        )
        for i, task in enumerate(tasks)
    ]
