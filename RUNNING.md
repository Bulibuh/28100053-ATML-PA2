# Completed ATML PA2 code

The course starter from [AbDu11aHHH/ATML-PA2-LLM-PostTraining](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining), commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`, was imported into this repository while retaining this repository's Git origin. This is material reuse of the course's model/data plumbing, metric helpers, frozen judges, exact verifier, configurations, and objective scaffolds. `starter_provenance.json` records the SHA256 of every imported file.

**Only three original files changed:**

- `task1_dpo/dpo.py`: subtract the frozen-reference preference margin in the DPO logit.
- `task2_ppo/ppo.py`: use the pessimistic minimum of unclipped and clipped surrogates.
- `task3_grpo/grpo.py`: normalize rewards separately within each prompt group; constant-reward groups have zero advantage.

`BUG_FIXES.patch` contains the complete changes. `python -m scripts.verify_starter` verifies all 55 imported files. Every other starter file, including its README, requirements, fixed prompts, configuration files and unfinished scaffolds, is preserved byte for byte. The completed entry points are the **new `experiments.py` modules**, rather than the preserved TODO entry points in the upstream README.

## Install and assets

Use Python 3.12 and `uv` (or install the two requirements files in an equivalent virtual environment). Run from the repository root:

```bash
bash scripts/setup_dev.sh
source scripts/dev_env.sh
python -m scripts.fetch_course_assets
python -m scripts.validate_assets
```

The new asset helper uses the same pinned release `0b350481fb03f5525a35bcdec4131bd4fe487f98`, preserves original tracked files, verifies LFS SHA256 / Git blob SHA1 against release metadata, and materializes the first 100 official SVAMP examples when absent. It uses the supported standard HTTPS transport where Xet transfers fail. Caches, raw datasets, course checkpoints, and trained adapters stay in the starter's ignored directories. No authentication secret is needed for these public assets/models. Downloads need `huggingface.co` and its LFS/Xet delivery hosts. Do not disable TLS or content verification.

The released DPO total-sequence cap is 768, but the fixed files include prompts as long as **3,563 tokens**. The unchanged starter encoder rejects these before training (54 standard-train, 58 balanced-train, 10 standard-eval and 9 stratified-eval examples). The commands below make an explicit **4,096-token runtime override** for every DPO condition to retain every fixed example and its full prompt. The original config remains unchanged. This is a documented departure from the released token cap and may require course clarification for a strictly identical protocol; use `--max-sequence-length 768` to reproduce the failure instead. The override increases GPU compute/memory. No prompt or dataset row is shortened/filtered.

```bash
python -m scripts.inspect_dpo_data --max-sequence-length 4096
```

Full course training needs a CUDA machine; the supplied 1.5B policy, 0.5B critic, reward model and 3B judge are too costly to execute as a full study in this CPU cloud machine. Test the largest training configuration against your GPU memory before running the complete grid; the exact minimum VRAM has not been measured here. GPU installation uses the course requirements and the GPU's compatible PyTorch build. The current instance has 71 non-LFS course files verified. Large checkpoint and PPO-cache downloads are blocked at `us.aws.cdn.hf.co`; this host has been added to the environment draft but access is not confirmed. `python -m scripts.fetch_course_assets --small-only` supports the verified small-file workflow, while the full command remains necessary before model experiments.

The current CPU validation uses PyTorch 2.6.0+cpu and the exact pinned Transformers/Tokenizers/PEFT/TRL versions.

## Required experiments

Each command loads the common released config and fixed data without depending on interactive notebook state. The standard DPO run trains one epoch; all short DPO forks start with a new adapter on the original base. Every PPO/GRPO run restores the same supplied midpoint, including PPO's matched critic. SFT and all evaluation models are frozen. Task 5 compute files separately record verifier/judge time, cache hits and pairwise requests.

```bash
# Task 1: one-epoch standard, three beta forks, and balanced-length intervention
python -m task1_dpo.experiments --mode standard --max-sequence-length 4096
python -m task1_dpo.experiments --mode beta --max-sequence-length 4096
python -m task1_dpo.experiments --mode length --max-sequence-length 4096

# Task 2: 20-update continuation, cache + 3 clipping forks, 3 KL forks
python -m task2_ppo.experiments --mode standard
python -m task2_ppo.experiments --mode clipping
python -m task2_ppo.experiments --mode kl

# Task 3: 20-update continuation, equal-generation cache analysis, 2 normalization forks
python -m task3_grpo.experiments --mode standard
python -m task3_grpo.experiments --mode groups
python -m task3_grpo.experiments --mode normalization

# Task 4: standard policies only; no model selection based on safety results
python -m task4_safety.experiments --mode generate
python -m task4_safety.experiments --mode judge
python -m task4_safety.experiments --mode audit
# Complete manual_label in results/task4_safety/manual_audit.csv yourself.
# Do not open AI judgments or the audit lookup while labeling.
python -m task4_safety.experiments --mode summarize

# Task 5: frozen SFT/RLVR/RLAIF policies, fixed datasets and fixed AI judge
python -m task5_feedback.experiments --mode gsm
python -m task5_feedback.experiments --mode diagnostics
python -m task5_feedback.experiments --mode transfer
python -m task5_feedback.experiments --mode compare

# Task 6: evidence/compute tables, no new training and no generated report prose
python -m task6_synthesis.summarize
```

Each of Tasks 1–5 also supports `--mode all`. Task 4's `all` produces a blank audit sheet and reports the incomplete human audit honestly. Commands refuse to overwrite existing training runs or manual annotations. Archive previous run directories/adapters intentionally before rerunning. Task 4 judging and Task 5 generation/judging resume only matching persisted data/configuration. Existing standard adapters allow each evaluation notebook to run without executing earlier notebooks.

## Notebooks, visualizations, and examples

There is one notebook in each task directory: `task1_dpo/task1.ipynb` through `task6_synthesis/task6.ipynb`. Open them using:

```bash
source scripts/dev_env.sh
jupyter lab
```

They run independently from either the root or their own directory. Each notebook provides CPU equation/aggregation diagnostics, prints examples where meaningful, lists the exact full-course commands, and renders saved results. **`RUN_FULL_EXPERIMENTS=False` by default.** Set it to `True` when the course assets, prerequisite standard adapters (Task 4), and suitable compute are available. Notebook diagnostic fixtures are clearly labeled and stored in `results/validation/`; they are not course empirical results.

Saved-result plots cover DPO fitting/length strata/word limits; PPO reward, KL, losses, entropy, clipping, gradient norms and lengths; GRPO informativeness, difficulty bins and length-conditioned gradients; safety rates, category distributions and manual confusion matrices; and math accuracy, pairwise wins, format, transfer drops and controlled reward sensitivity. Plots appear inline and are exported to each task's `results/<task>/figures/`. Examples are quoted from actual saved generations, not invented trained-model outputs.

## Validation

```bash
source scripts/dev_env.sh
python -m scripts.verify_starter
python -m pytest -q
python -m scripts.execute_notebooks
```

Tests check objective values and gradients, terminal masking/GAE, equal-generation grouping, safety denominators, missing manual labels, and verifier distractors. Integration tests run the **actual DPO/PPO/GRPO loops on tiny random Qwen2 models with PEFT LoRA**, including policy/critic updates and checkpoint saving. They do not download models or claim assignment results. All six notebooks have also been executed with their CPU diagnostics enabled. CPU checks include the actual 192-completion GRPO cache study and exact-verifier scoring of all 100 fixed controlled diagnostic responses; AI/model evaluation has not run.

## Metric and budget conventions

DPO sums response-token likelihoods and evaluates the reference-corrected margin. Reference estimates use the supplied base policy with the adapter disabled. Reported KL is the released masked **response-token mean** of policy minus reference log probability; the nonnegative GRPO loss penalty is logged separately from this common evaluation convention. Entropy uses categorical token entropy over valid response positions. Length includes generated EOS where present, matching the released mask, and excludes post-EOS padding; mean, standard deviation and quartiles are saved.

Online ratios use dropout-disabled forwards consistently for old/new policies, preventing dropout noise from appearing as policy drift. DPO retains the released LoRA dropout during training. A wrapper copies the released generator's inference tensors before autograd; the original generation code is unchanged. Critics predict `V(s_t)` from the state immediately before action token `t`; terminal GAE bootstraps with zero as specified by the course.

Forks use identical seeds, prompt schedules, update counts, generation caps, optimizer settings, and allocated token ceilings. Actual generated-token counts can differ when policies terminate early; each run saves allocated and realized counts, so this difference is visible rather than silently described as an exact realized-token match. The K-cache comparison reuses the same first eight completions per prompt exactly once for every K. Difficulty is defined once using the median per-prompt K=8 mean reward. Normalization diagnostics measure `|dL/d log pi|` per token and per sequence; the shared short/long cutoff is displayed in the notebook.

The blind audit uses the supplied seed/selection helper: the same 30 safe + 30 unsafe IDs across all four policies (240 response annotations). No human labels are generated automatically. Judge confidence is for auditing only. The supplied pairwise parser maps unparseable responses to `TIE`; saved tie counts therefore include that fallback and cannot separately identify parsing ambiguity without changing the frozen judge.

All runs save hyperparameters, split/prompt IDs, logs, generations, resource usage, and completion status. Course data, cached diagnostics, and checkpoints are never tuned/rewritten. Metrics are written with strict JSON: unavailable numeric values are null. Qualitative examples and manual audit disagreements require your own interpretation. No PDF report prose, empirical result, or manual annotation is automatically fabricated. The report and public GitHub publication are separate user actions; this code has not been pushed.
