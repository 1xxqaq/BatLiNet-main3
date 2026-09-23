"""Download an immutable public dataset snapshot; never select moving 'main'."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--revision', default='154a4026cd188960b9bfc029a9e28ad3f1091910')
    args = p.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        p.error('--revision must be a full immutable snapshot SHA')
    from huggingface_hub import snapshot_download
    root = Path(args.output)
    manifest = root / 'context_grid_snapshot.json'
    expected = dict(repo_id='Battery-Life/BatteryLife_Processed', revision=args.revision)
    if manifest.exists() and json.loads(manifest.read_text()) != expected:
        raise ValueError('Output already contains another recorded dataset snapshot.')
    folders = ['CALCE', 'HNEI', 'HUST', 'ISU_ILCC', 'MATR', 'MICH', 'MICH_EXP',
               'RWTH', 'SNL', 'Stanford', 'Tongji', 'UL_PUR', 'XJTU', 'Life labels', 'READMEs']
    snapshot_download(repo_id=expected['repo_id'], repo_type='dataset', revision=args.revision,
                      local_dir=str(root), allow_patterns=[f'{f}/*' for f in folders] + ['README.md'],
                      max_workers=4)
    manifest.write_text(json.dumps(expected, indent=2), encoding='utf-8')
    print(f'Snapshot ready: {root.resolve()}')


if __name__ == '__main__':
    main()
