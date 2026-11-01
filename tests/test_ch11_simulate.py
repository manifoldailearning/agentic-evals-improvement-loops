import pytest

from ch11_simulated_users import main
from evalloop.core import (
    BASELINE,
    IMPROVED,
    SimAgent,
    make_tasks,
)
from evalloop.simulate import (
    PERSONAS,
    by_persona,
    conversation_metrics,
    converse,
    frustration,
    goal_for,
    goal_met,
    in_character,
    satisfied,
    simulate_suite,
    turn_pass_rate,
    user_reply,
)

TASKS = make_tasks(200, seed=7)
BY_ID = {t.id: t for t in TASKS}
P = {p.name: p for p in PERSONAS}
VAGUE = BY_ID["T-0007"]  # "I want a refund for my last order."


def test_goal_for_ambiguous_task_is_a_concrete_one():
    plain = BY_ID["T-0001"]
    assert goal_for(plain, TASKS) is plain
    goal = goal_for(VAGUE, TASKS)
    assert goal.category == VAGUE.category
    assert not goal.expected["clarify"]
    assert goal == goal_for(VAGUE, TASKS)  # deterministic
    with pytest.raises(ValueError):
        goal_for(VAGUE, [plain])


def test_angry_persona_changes_what_the_agent_sees():
    task = BY_ID["T-0001"]
    assert "angry" in in_character(task, P["angry"]).tags
    assert in_character(task, P["patient"]) is task


def test_user_satisfaction_reads_only_the_reply():
    goal = BY_ID["T-0001"]  # order A-1008 is delivered
    assert satisfied("Order A-1008 is delivered.", goal)
    assert not satisfied("Order A-1009 is delivered.", goal)
    assert not satisfied("", goal)
    angry = BY_ID["T-0010"]  # must not say "calm down"
    rude = "Please calm down. Order A-1022: escalated."
    assert not satisfied(rude, angry)
    assert frustration(rude) == 2
    assert frustration("Could you share your order number?") == 0
    assert frustration("Order A-1022 is shipped.") == 1


def test_user_reply_supplies_order_or_repeats_request():
    goal = goal_for(VAGUE, TASKS)
    oid = goal.expected["args"]["order_id"]
    nxt = user_reply(VAGUE, goal, P["patient"], 1)
    assert nxt.id == "T-0007.2" and oid in nxt.input
    assert nxt.expected == goal.expected
    kinds = {
        user_reply(VAGUE, goal, P["confused"], n).id
        for n in range(1, 20)
    }
    assert VAGUE.id in kinds and len(kinds) > 1  # forgets
    plain = BY_ID["T-0001"]
    again = user_reply(plain, plain, P["terse"], 1)
    assert (again.id, again.input) == (plain.id, "Wrong.")


def test_first_turn_replays_the_single_turn_run():
    agent = SimAgent(BASELINE)
    for task in TASKS[:20]:
        conv = converse(
            agent, task, P["patient"], goal_for(task, TASKS), 3
        )
        assert conv.turns[0].run == agent.run(task, seed=3)


def test_clarifying_question_leads_to_the_goal():
    agent = SimAgent(IMPROVED)
    goal = goal_for(VAGUE, TASKS)
    conv = converse(agent, VAGUE, P["patient"], goal)
    first, last = conv.turns[0], conv.turns[-1]
    assert "order number" in first.run.answer
    assert not first.run.tool_calls
    assert conv.outcome == "satisfied" and goal_met(conv)
    assert last.task.expected == goal.expected
    assert conv.frustration == 0


def test_conversations_respect_patience_and_turn_limit():
    for conv in simulate_suite(SimAgent(BASELINE), TASKS):
        patience = P[conv.persona].patience
        assert len(conv.turns) <= 6
        if conv.outcome == "gave_up":
            assert conv.frustration > patience
        last = conv.turns[-1].run.answer
        happy = satisfied(last, conv.goal)
        assert happy == (conv.outcome == "satisfied")


def test_goal_met_agrees_with_ground_truth():
    for config in (BASELINE, IMPROVED):
        for conv in simulate_suite(SimAgent(config), TASKS):
            last = conv.turns[-1]
            truth = (
                last.run.ok and not last.task.expected["clarify"]
            )
            assert goal_met(conv) == truth


def test_conversation_metrics_differ_from_turn_metrics():
    base = simulate_suite(SimAgent(BASELINE), TASKS)
    best = simulate_suite(SimAgent(IMPROVED), TASKS)
    mb, mi = (
        conversation_metrics(base),
        conversation_metrics(best),
    )
    assert mb["goal_met"] > turn_pass_rate(base)  # retries help
    assert mi["goal_met"] > mb["goal_met"]
    assert mi["gave_up"] < mb["gave_up"]
    angry = by_persona(base)["angry"]["goal_met"]
    assert angry < by_persona(base)["terse"]["goal_met"]


def test_example_compares_turn_and_conversation_views(capsys):
    main()
    out = capsys.readouterr().out
    assert "conversation goal met        72.5%     90.5%" in out
    assert "-> satisfied after 3 turns, goal met" in out
    assert all(len(line) <= 64 for line in out.splitlines())
    assert len(out.splitlines()) <= 25
