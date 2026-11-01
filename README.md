# Evals and Improvement Loops for Agentic Systems in Production

This is the companion code for the book **Evals and Improvement Loops for Agentic
Systems in Production** by Nachiketh Murthy.

You can't improve an agent you can't measure. This repository shows the whole
practice end to end:
- build an eval set and grade agent runs, using rule-based checks and calibrated
  model judges;
- evaluate trajectories, conversations and robustness;
- use statistics to decide whether a change actually helped;
- gate releases in CI;
- run experiments, and turn failures into improvements in a disciplined loop.

Each package is a small, readable reference implementation for one chapter, with a
runnable example and tests.

Everything runs offline against a **simulated** customer-support agent:
- **The agent** looks up orders, searches a knowledge base, issues refunds and
  escalates.
- **How it fails:** its failure modes (wrong tool, missing citation, policy
  breach, ignored ambiguity, bad tone and others) depend on its configuration:
  model tier, prompt rules, few-shot examples and temperature.
- **Why that matters:** improvement loops are real here. Change the config and the
  eval results change measurably and reproducibly. The baseline config succeeds on
  63.5% of the demo tasks and the improved config on 89.5%.

> The numbers come from a simulator, not a production system. They are there to
> show the methods; they are not benchmarks of any model. Every chapter states the
> assumptions behind its simulation.

## Quick start

```bash
git clone https://github.com/manifoldailearning/agentic-evals-improvement-loops
cd agentic-evals-improvement-loops
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # Python 3.11+, no runtime dependencies
pytest -q
python examples/ch01_vibes_vs_evals.py
```

The `evalloop` command runs the evals CI uses:

```bash
evalloop run --config baseline     # pass rate with a 95% interval
evalloop compare improved v3       # paired comparison on the same tasks
evalloop baseline                  # write evals/baseline.json
evalloop gate                      # regression gate; non-zero exit fails CI
evalloop review                    # full release review: ship or hold
evalloop maturity eval_set graders # where a team stands, what to do next
```

## Chapter map

| Ch | Topic | Package | Example |
|----|-------|---------|---------|
| 1 | Evals, not vibes | - | `ch01_vibes_vs_evals.py` |
| 2 | Anatomy of an agent eval | `evalloop.core` | `ch02_anatomy.py` |
| 3 | Eval datasets | `evalloop.datasets` | `ch03_datasets.py` |
| 4 | Error analysis | `evalloop.analysis` | `ch04_error_analysis.py` |
| 5 | Synthetic data | `evalloop.synth` | `ch05_synthetic.py` |
| 6 | Deterministic graders | `evalloop.graders` | `ch06_graders.py` |
| 7 | Model-graded evals | `evalloop.judge` | `ch07_judges.py` |
| 8 | Calibrating judges | `evalloop.calibration` | `ch08_calibration.py` |
| 9 | Human annotation | `evalloop.annotation` | `ch09_annotation.py` |
| 10 | Trajectory evaluation | `evalloop.trajectory` | `ch10_trajectory.py` |
| 11 | Simulated users | `evalloop.simulate` | `ch11_simulated_users.py` |
| 12 | Robustness | `evalloop.robustness` | `ch12_robustness.py` |
| 13 | Statistics for evals | `evalloop.stats` | `ch13_statistics.py` |
| 14 | Regression gates | `evalloop.gates` | `ch14_regression_gates.py` |
| 15 | Online experiments | `evalloop.online` | `ch15_online.py` |
| 16 | The improvement loop | `evalloop.loop` | `ch16_improvement_loop.py` |
| 17 | Prompt and context optimization | `evalloop.optimize` | `ch17_optimization.py` |
| 18 | The data flywheel | `evalloop.flywheel` | `ch18_flywheel.py` |
| 19 | Eval infrastructure | `evalloop.infra` | `ch19_infrastructure.py` |
| 20 | The operating model | `evalloop.program` | `ch20_operating_model.py` |

## Design notes

- **Ground truth is kept apart from verdicts.** Each simulated run records the
  faults that really happened (`Trajectory.faults`). Graders, judges and loops
  never read them. They exist so the book can measure how accurate each grader
  is.
- **Determinism everywhere.** Randomness is seeded from stable hashes of the task,
  config and seed, so every number in the book can be reproduced exactly.
- **One verdict, many signals.** The release review in chapter 20 reuses one
  graded run of the eval set for quality, the gate, a paired comparison, an
  injection probe and the cost budget.
- **Standard library only.** Every package can be read in an afternoon and runs
  with no setup.

## Learn more

- This book is part of a series with *Agentic AI Deployment Patterns for
  Production*, *Agentic Ops for Production* and *Agentic AI Governance and
  Security*.
- Live program: [Agentic AI Enterprise Mastery Bootcamp](https://manifoldailearning.com/services/agenticaibootcamp)
- [manifoldailearning.com](https://manifoldailearning.com)

## License

MIT. See [LICENSE](LICENSE).
