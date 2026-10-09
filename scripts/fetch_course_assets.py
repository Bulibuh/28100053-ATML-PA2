"""Fetch the pinned course assets over HTTPS without overwriting starter files."""
from __future__ import annotations
import os
import argparse
from pathlib import Path
from common.data import REPO_ROOT

# Use the supported HTTP transport where Xet cannot reach its transfer service.
os.environ.setdefault('HF_HOME',str(REPO_ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_XET_CACHE',str(REPO_ROOT/'.cache/huggingface/xet'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import hashlib
import json
from huggingface_hub import HfApi,snapshot_download
from scripts.download_assets import HF_ASSET_REPO,HF_ASSET_REVISION
from scripts.prepare_transfer_eval import materialize_transfer_eval
from common.experiment_utils import dump


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--small-only',action='store_true',help='Fetch and verify non-LFS data/metadata while weight downloads are blocked')
    args=ap.parse_args()
    protected=set(json.loads((REPO_ROOT/'starter_provenance.json').read_text())['original_sha256'])
    tree=list(HfApi().list_repo_tree(repo_id=HF_ASSET_REPO,repo_type='dataset',revision=HF_ASSET_REVISION,recursive=True))
    files=[f for f in tree if hasattr(f,'blob_id') and f.path.startswith(('data/','cached/','checkpoints/','manifests/')) and f.path not in protected]
    verification=[]
    batches=[[f for f in files if not f.lfs]]
    if not args.small_only:batches.append([f for f in files if f.lfs])
    for batch in batches:
        try:
            snapshot_download(repo_id=HF_ASSET_REPO,repo_type='dataset',revision=HF_ASSET_REVISION,local_dir=str(REPO_ROOT),
                              allow_patterns=[f.path for f in batch],max_workers=4)
        except Exception as exc:
            dump(REPO_ROOT/'results/validation/course_assets.json',{'repository':HF_ASSET_REPO,'revision':HF_ASSET_REVISION,
                 'complete':False,'verified_files':verification,'pending_files':[f.path for f in batch],'error_type':type(exc).__name__})
            raise SystemExit(f'Pinned asset download failed ({type(exc).__name__}); check Hugging Face delivery-host network access. Verified earlier batches are retained.') from None
        for info in batch:
            verification.append(verify_file(info))
        dump(REPO_ROOT/'results/validation/course_assets.json',{'repository':HF_ASSET_REPO,'revision':HF_ASSET_REVISION,
             'complete':len(verification)==len(files),'verified_files':verification,'pending_files':[f.path for f in files if f.path not in {r['path'] for r in verification}]})
    materialize_transfer_eval()
    print(f'Verified {len(verification)} pinned-release asset files; preserved all starter files.')


def verify_file(info):
        p=REPO_ROOT/info.path
        # LFS files have authoritative content SHA256; normal files use Git blob SHA1.
        if info.lfs:
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
            expected=info.lfs.sha256;actual=h.hexdigest();algorithm='sha256'
        else:
            data=p.read_bytes();actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest();expected=info.blob_id;algorithm='git-blob-sha1'
        if actual!=expected:raise RuntimeError(f'Asset checksum mismatch: {info.path}')
        return {'path':info.path,'algorithm':algorithm,'digest':actual,'bytes':p.stat().st_size}


if __name__=='__main__':main()
