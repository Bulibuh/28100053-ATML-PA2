"""Verify that the only changes to supplied files are the three recorded objective fixes."""
import hashlib
import json
from common.data import REPO_ROOT


def main():
    provenance=json.loads((REPO_ROOT/'starter_provenance.json').read_text())
    changed=[];failures=[]
    for name,digest in provenance['original_sha256'].items():
        path=REPO_ROOT/name
        if not path.exists():failures.append(f'Missing original file: {name}');continue
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        expected=provenance['corrected_sha256'].get(name,digest)
        if actual!=expected:failures.append(f'Unexpected change: {name}')
        if actual!=digest:changed.append(name)
    if failures:raise SystemExit('\n'.join(failures))
    print(f"Verified {len(provenance['original_sha256'])} starter files; only {len(changed)} objective files changed:")
    print('\n'.join(changed))


if __name__=='__main__':main()
