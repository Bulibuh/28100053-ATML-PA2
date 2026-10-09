"""Generate one independently runnable notebook per assignment task."""
from pathlib import Path
import nbformat as nbf
from common.data import REPO_ROOT

TASKS={
 1:('task1_dpo','DPO',['python -m task1_dpo.experiments --mode standard --max-sequence-length 4096','python -m task1_dpo.experiments --mode beta --max-sequence-length 4096','python -m task1_dpo.experiments --mode length --max-sequence-length 4096']),
 2:('task2_ppo','PPO',['python -m task2_ppo.experiments --mode standard','python -m task2_ppo.experiments --mode clipping','python -m task2_ppo.experiments --mode kl']),
 3:('task3_grpo','GRPO',['python -m task3_grpo.experiments --mode standard','python -m task3_grpo.experiments --mode groups','python -m task3_grpo.experiments --mode normalization']),
 4:('task4_safety','Safety calibration',['python -m task4_safety.experiments --mode all']),
 5:('task5_feedback','RLVR versus RLAIF',['python -m task5_feedback.experiments --mode all']),
 6:('task6_synthesis','Cross-task evidence',['python -m task6_synthesis.summarize']),
}
BOOTSTRAP='''from pathlib import Path
import os, sys, shlex, subprocess
candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in candidates if (p / "configs/base.yaml").exists()), None)
if ROOT is None:
    raise RuntimeError("Start Jupyter from the repository or a task directory.")
os.chdir(ROOT)
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
for name, value in {
    "HF_HOME": ROOT / ".cache/huggingface", "HF_XET_CACHE": ROOT / ".cache/huggingface/xet",
    "MPLCONFIGDIR": ROOT / ".cache/matplotlib", "IPYTHONDIR": ROOT / ".cache/ipython",
}.items():
    os.environ.setdefault(name, str(value))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
from IPython.display import display
from common.notebook_diagnostics import diagnostic
print("Repository:", ROOT)
print("Python:", sys.executable, "CUDA:", torch.cuda.is_available())
RUN_FULL_EXPERIMENTS = False  # Set True only when the required assets/models and compute are available.
'''


def main():
    for task,(directory,title,commands) in TASKS.items():
        nb=nbf.v4.new_notebook()
        nb.metadata={'kernelspec':{'display_name':'ATML PA2 (Python)','language':'python','name':'python3'},'language_info':{'name':'python'},'atml_task':task}
        intro=f'''# Task {task}: {title}

This notebook runs independently and delegates experiments to task-specific Python modules.
The CPU diagnostics below validate mathematics/aggregation using explicitly labeled hand-authored fixtures;
they are **not** trained-model/course results. Actual course results are loaded from disk in the final cells.
No previous notebook state is required. Installation and full reproduction commands are in `RUNNING.md`.
'''
        if task==1:intro+='\nThe fixed prompts exceed the released 768-token cap. Full commands use a documented 4,096-token runtime override, preserving every prompt/example and the original configs. This increases compute and changes the released token budget; see RUNNING.md.\n'
        if task==4:intro+='\nUse only the standard DPO/PPO/GRPO adapters. Complete the blind `results/task4_safety/manual_audit.csv` yourself, then rerun `--mode summarize`; no AI/manual agreement is inferred from missing labels.\n'
        if task==6:intro+='\nThis notebook assembles evidence and compute tables. Interpretation and report prose remain yours. It does not generate the PDF report.\n'
        nb.cells=[nbf.v4.new_markdown_cell(intro),nbf.v4.new_code_cell(BOOTSTRAP),
                  nbf.v4.new_markdown_cell('## CPU implementation diagnostic\nThese figures show equations or fixture behavior, with no simulated course metrics.'),
                  nbf.v4.new_code_cell(f'diagnostic_result = diagnostic({task})')]
        command_cell='COMMANDS = '+repr(commands)+'''\nfor command in COMMANDS:
    print(command)
if RUN_FULL_EXPERIMENTS:
    for command in COMMANDS:
        args = shlex.split(command)
        args[0] = sys.executable
        subprocess.run(args, cwd=ROOT, check=True)
else:
    print("Full experiment execution is disabled. Set RUN_FULL_EXPERIMENTS=True to run the commands above.")
'''
        if task in (1,3,4,5):
            nb.cells.extend([nbf.v4.new_markdown_cell('## Fixed course data diagnostics\nCPU checks on real downloaded assets; these do not claim trained-policy behavior.'),nbf.v4.new_code_cell(f'from common.notebook_diagnostics import course_asset_diagnostic\ncourse_asset_diagnostic({task})')])
        nb.cells.extend([nbf.v4.new_markdown_cell('## Full experiment commands\nCommands fail on errors; required runs are never replaced by fixture diagnostics. Existing run directories are protected from accidental overwrite.'),nbf.v4.new_code_cell(command_cell)])
        if task<=5:
            code=f'from common.visualization import plot_task\nplot_task({task})'
        else:code='from task6_synthesis.summarize import summarize\nevidence, compute = summarize()\ndisplay(evidence)\ndisplay(compute)'
        nb.cells.extend([nbf.v4.new_markdown_cell('## Saved course results and examples\nMissing results are explicitly reported. Plots use the saved JSON/JSONL logs from actual runs.'),nbf.v4.new_code_cell(code)])
        if task==6:nb.cells.append(nbf.v4.new_markdown_cell('## Questions for your own analysis\n- How do objective strength and reference pressure relate to observed policy drift?\n- Which length and group effects are supported by controlled comparisons?\n- Do reward gains coincide with useful behavior and safety calibration?\n- What do verifier–judge disagreements reveal about feedback coverage and exploitability?\n- What do the measured token budgets, runtime, and VRAM show about compute costs?\n\nUse saved evidence and your own interpretation; do not treat missing experiments as evidence.'))
        path=REPO_ROOT/directory/f'task{task}.ipynb';nbf.write(nb,path);print(path.relative_to(REPO_ROOT))


if __name__=='__main__':main()
