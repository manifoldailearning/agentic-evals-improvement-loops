"""Simulated users and multi-turn evaluation.

A deterministic user simulator with a persona and a hidden
goal talks to the single-turn agent through a turn loop.
Metrics compare turn-level pass rate with conversation-level
goal completion, turns to success and frustration.
"""

from evalloop.simulate.conversation import (
    Conversation,
    Turn,
    converse,
    simulate_suite,
    turn_seed,
)
from evalloop.simulate.metrics import (
    by_persona,
    conversation_metrics,
    goal_met,
    turn_pass_rate,
)
from evalloop.simulate.users import (
    PERSONAS,
    Persona,
    frustration,
    goal_for,
    in_character,
    satisfied,
    user_reply,
)

__all__ = [
    "PERSONAS",
    "Conversation",
    "Persona",
    "Turn",
    "by_persona",
    "conversation_metrics",
    "converse",
    "frustration",
    "goal_for",
    "goal_met",
    "in_character",
    "satisfied",
    "simulate_suite",
    "turn_pass_rate",
    "turn_seed",
    "user_reply",
]
