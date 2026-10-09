# Runnable task notebooks

Run `bash scripts/setup_dev.sh`, then `source scripts/dev_env.sh` and `jupyter lab` from the repository root.

| Task | Notebook | Full pipeline |
| --- | --- | --- |
| 1 — DPO | [task1.ipynb](task1_dpo/task1.ipynb) | Standard, beta, length-control and word-limit experiments |
| 2 — PPO | [task2.ipynb](task2_ppo/task2.ipynb) | Continuation, cached clipping, clipping/KL forks |
| 3 — GRPO | [task3.ipynb](task3_grpo/task3.ipynb) | Continuation, group size, normalization and gradient diagnostics |
| 4 — Safety | [task4.ipynb](task4_safety/task4.ipynb) | Frozen policy generations, fixed AI judge, blind manual audit and aggregation |
| 5 — Feedback | [task5.ipynb](task5_feedback/task5.ipynb) | GSM8K/SVAMP, exact verification, pairwise AI judging and perturbations |
| 6 — Synthesis | [task6.ipynb](task6_synthesis/task6.ipynb) | Cross-task evidence and compute tables; no generated report prose |

All notebooks have executed in CPU diagnostic mode and show figures and examples. Real fixed-data checks are distinguished from hand-authored unit fixtures and from unrun model experiments. Set `RUN_FULL_EXPERIMENTS=True` only when assets, prerequisite adapters and suitable compute are available. See [RUNNING.md](RUNNING.md) for exact commands, the explicit DPO sequence-length override, metrics and remaining asset-download constraints.
