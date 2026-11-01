from evalloop.core import BASELINE, RULES, make_tasks
from evalloop.loop import evaluate, split
from evalloop.optimize import (
    Point,
    ablate,
    compare_shots,
    diverse_shots,
    frontier,
    greedy_search,
    moves,
    optimism,
    pareto,
    relevant,
    similar_shots,
    similarity,
)

TASKS = make_tasks(200, seed=7)
POOL = make_tasks(60, seed=99)
ALL_RULES = BASELINE.evolve(rules=frozenset(RULES))


def test_ablation_prices_each_rule():
    loss = ablate(ALL_RULES, TASKS)
    assert set(loss) == set(RULES)
    assert list(loss.values()) == sorted(loss.values())[::-1]
    assert next(iter(loss)) == "cite_kb"
    assert all(d >= 0 for d in loss.values())  # rules only help


def test_ablation_of_no_rules_is_empty():
    assert ablate(BASELINE, TASKS[:10]) == {}


def test_similarity_and_similar_shots():
    task = TASKS[0]
    assert similarity(task, task) == 1.0
    shots = similar_shots(task, POOL, 3)
    assert len(shots) == 3
    scores = [similarity(task, s) for s in shots]
    assert scores == sorted(scores, reverse=True)
    assert relevant(task, shots) >= 1


def test_diverse_shots_are_distinct_and_spread():
    shots = diverse_shots(POOL, 5)
    assert len({s.id for s in shots}) == 5
    assert len({s.category for s in shots}) >= 3


def test_similar_shots_prevent_most_tool_errors():
    scores = compare_shots(ALL_RULES, TASKS, POOL, k=6)
    errors = {k: e for k, (_, e) in scores.items()}
    assert errors["similar"] < errors["random"] < errors["none"]


def test_moves_skip_what_config_already_has():
    c = BASELINE.evolve(rules=frozenset({"cite_kb"}), few_shots=6)
    m = moves(c)
    assert "rule:cite_kb" not in m and "rule:verify_order_id" in m
    assert not any(x.startswith("few_shots") for x in m)
    assert f"temperature:{c.temperature}" not in m


def test_greedy_search_improves_and_respects_budget():
    dev, test = split(TASKS)
    found = greedy_search(BASELINE, dev, budget=15)
    assert found.evals <= 15
    assert found.score > evaluate(BASELINE, dev)[0]
    assert found.path[0][0] == "rule:cite_kb"
    gap = optimism(BASELINE, found.config, dev, test)
    assert -1 < gap < 1


def test_pareto_keeps_only_undominated_points():
    pts = [
        Point("a", 1.0, 0.6),
        Point("b", 2.0, 0.5),  # dearer and worse than a
        Point("c", 3.0, 0.9),
        Point("d", 3.0, 0.8),  # same cost, worse than c
    ]
    assert [p.label for p in pareto(pts)] == ["a", "c"]
    assert pareto([]) == []


def test_frontier_uses_trajectory_cost():
    configs = {
        m: BASELINE.evolve(model=m) for m in ("small", "large")
    }
    points, front = frontier(configs, TASKS[:40])
    small, large = points
    assert small.cost < large.cost / 10
    assert large.success >= small.success and front == points


def test_example_main(capsys):
    from ch17_optimization import main

    main()
    out = capsys.readouterr().out
    assert "medium bare is dominated by small-ft +prompt" in out
    assert "overfit warning" in out
    lines = out.splitlines()
    assert len(lines) <= 25 and max(map(len, lines)) <= 64
