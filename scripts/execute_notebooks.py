"""Execute all task notebooks with the current interpreter; full experiments default off."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from common.data import REPO_ROOT
from scripts.create_notebooks import TASKS


def main():
    for name,subdirectory in {'HF_HOME':'huggingface','HF_XET_CACHE':'huggingface/xet','MPLCONFIGDIR':'matplotlib','IPYTHONDIR':'ipython',
                              'JUPYTER_RUNTIME_DIR':'jupyter/runtime','JUPYTER_DATA_DIR':'jupyter/data','JUPYTER_CONFIG_DIR':'jupyter/config','XDG_CACHE_HOME':'xdg'}.items():
        path=REPO_ROOT/'.cache'/subdirectory
        path.mkdir(parents=True,exist_ok=True)
        os.environ[name]=str(path)
    for task,(directory,_,_) in TASKS.items():
        path=REPO_ROOT/directory/f'task{task}.ipynb'
        notebook=nbformat.read(path,as_version=4)
        NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(path.parent)}}).execute()
        nbformat.write(notebook,path)
        print(f'Executed {path.relative_to(REPO_ROOT)}',flush=True)


if __name__=='__main__':main()
