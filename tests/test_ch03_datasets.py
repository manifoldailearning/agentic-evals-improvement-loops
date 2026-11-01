from dataclasses import replace

import pytest

import ch03_datasets
from evalloop.core import make_tasks
from evalloop.datasets import (
    DatasetVersion,
    allocate,
    cells,
    coverage,
    dedup,
    diff,
    duplicate_groups,
    golden_set,
    grouped_split,
    jaccard,
    leaks,
    normalize,
    random_split,
    read_version,
    slice_scores,
    slices,
    stratified_sample,
    stratum,
    thin_cells,
    token_set,
    write_version,
)

RAW = make_tasks(200, seed=7)
TASKS, DROPPED = dedup(RAW)


def test_normalize_and_jaccard():
    assert normalize("Where's my ORDER  A-1002?") == (
        "where s my order a 1002"
    )
    a, b = token_set("refund my order"), token_set("my order")
    assert jaccard(a, b) == pytest.approx(2 / 3)
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_dedup_maps_every_drop_to_a_kept_twin():
    kept_ids = {t.id for t in TASKS}
    assert len(TASKS) + len(DROPPED) == len(RAW)
    assert set(DROPPED.values()) <= kept_ids
    texts = [normalize(t.input) for t in TASKS]
    assert len(texts) == len(set(texts))


def test_coverage_counts_untagged_tasks_as_plain():
    task = next(t for t in RAW if not t.tags)
    assert cells(task) == [(task.category, "plain")]
    matrix = coverage(RAW)
    assert matrix[("faq", "angry")] == 0
    assert ("faq", "angry") in thin_cells(
        matrix, 1, [("faq", "angry")]
    )


def test_allocate_is_exact_and_keeps_every_stratum():
    quota = allocate({"a": 50, "b": 3, "c": 1}, 10)
    assert sum(quota.values()) == 10
    assert min(quota.values()) >= 1
    assert quota["a"] > quota["b"]
    with pytest.raises(ValueError):
        allocate({"a": 5, "b": 5}, 1)


def test_stratified_sample_is_seeded_and_covers_strata():
    one = stratified_sample(TASKS, 30, seed=1)
    assert one == stratified_sample(TASKS, 30, seed=1)
    assert len(one) == 30
    assert {stratum(t) for t in one} == {
        stratum(t) for t in TASKS
    }


def test_slices_and_scores():
    named = slices(TASKS)
    assert "category=refund" in named
    assert "tag=adversarial" in named
    score = {t.id: float(t.category == "faq") for t in TASKS}
    means = slice_scores(named, score)
    assert means["category=faq"] == 1.0
    assert means["category=refund"] == 0.0


def test_golden_set_takes_hardest_task_per_cell():
    gold = golden_set(TASKS)
    assert set(coverage(gold)) == set(coverage(TASKS))
    hardest = max(
        (t for t in TASKS if t.category == "order_status"),
        key=lambda t: t.difficulty,
    )
    assert hardest.difficulty == max(
        t.difficulty for t in gold if t.category == "order_status"
    )


def test_grouped_split_has_no_leaks_unlike_random():
    groups = duplicate_groups(RAW)
    assert sum(len(g) for g in groups) == len(RAW)
    dev, test = grouped_split(RAW, seed=3)
    assert len(dev) + len(test) == len(RAW)
    assert leaks(dev, test) == []
    dev, test = random_split(RAW, seed=3)
    assert len(leaks(dev, test)) > len(test) / 2


def test_version_hash_diff_and_roundtrip(tmp_path):
    v1 = DatasetVersion("v1", tuple(TASKS))
    shuffled = DatasetVersion("v1", tuple(reversed(TASKS)))
    assert v1.content_hash == shuffled.content_hash
    edited = replace(TASKS[0], difficulty=0.99)
    extra = replace(TASKS[1], id="T-9999")
    v2 = DatasetVersion("v2", (edited, *TASKS[2:], extra))
    d = diff(v1, v2)
    assert d.changed == [TASKS[0].id]
    assert d.removed == [TASKS[1].id]
    assert d.added == ["T-9999"]
    write_version(v2, tmp_path / "v2")
    assert read_version(tmp_path / "v2").content_hash == (
        v2.content_hash
    )
    (tmp_path / "v2" / "tasks.jsonl").write_text("")
    with pytest.raises(ValueError):
        read_version(tmp_path / "v2")


def test_example_main_prints_leak_comparison(capsys):
    ch03_datasets.main()
    out = capsys.readouterr().out
    assert "Raw: 200 tasks; after dedup: 95" in out
    assert "Grouped split: 0 of" in out
    assert max(len(line) for line in out.splitlines()) <= 64
