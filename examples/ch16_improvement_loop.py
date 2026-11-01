"""Chapter 16: baseline to better, one gated change at a time.

Each iteration triages dev failures, tries the change that
targets the most of them, and keeps it only if the held-out
test set confirms the dev gain. The last step tunes a noisy
knob on dev and shows the test gate catching the overfit.
"""

from __future__ import annotations

from evalloop.core import BASELINE, make_tasks
from evalloop.loop import Loop, evaluate, split

TEMPS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)
HEAD = " # evidence         change                    dev  test"


def row(r) -> str:
    change = r.change.replace("rule:", "+")
    dev = 100 * (r.dev[1] - r.dev[0])
    test = 100 * (r.test[1] - r.test[0])
    return (
        f"{r.iteration:>2} {r.evidence:<16} {change:<25}"
        f"{dev:+5.1f} {test:+5.1f} {r.decision}"
    )


def rates(config, dev, test) -> str:
    d, t = evaluate(config, dev)[0], evaluate(config, test)[0]
    return f"dev {d:.1%}  test {t:.1%}"


def main() -> None:
    dev, test = split(make_tasks(200, seed=12))
    print(f"200 tasks -> dev {len(dev)}, test {len(test)}")
    print(f"baseline: {rates(BASELINE, dev, test)}")
    loop = Loop(dev, test)
    config = loop.run(BASELINE, 7)
    config = loop.sweep(config, "temperature", TEMPS)
    print(HEAD)
    for r in loop.log.records:
        print(row(r))
    kept = len(loop.log.accepted())
    print(f"kept {kept} of {len(loop.log.records)} changes")
    print(f"{config.version}: {rates(config, dev, test)}")
    print("Rejected:")
    print("  7: no dev gain, so its small test gain went unseen")
    print(
        "  8: best of 9 temperatures on dev was luck; test fell"
    )
    print(
        f"All records cite dataset {loop.log.records[0].dataset}"
    )


if __name__ == "__main__":
    main()
