"""Delete only the retired single-cell experiment on the Linux server."""
import argparse
import contextlib
import os
import shutil
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_NAME = 'cross_chem_single_cell_v1'
RETIRED_RUNNER = 'run_cross_chem_single_cell.py'
CACHE_MODULES = {
    'scripts/__pycache__': ('run_cross_chem_single_cell', 'test_cross_chem_single_cell'),
    'src/models/rul_predictors/__pycache__': ('cycle_residual_single_cell',),
}
RETIRED_FILES = (
    'configs/cross_chem_single_cell_v1/protocol.yaml',
    'docs/cross_chem_single_cell_v1.md',
    'docs/cross_chem_single_cell_v1_checks.json',
    'scripts/run_cross_chem_single_cell.py',
    'scripts/start_cross_chem_single_cell_nohup.sh',
    'scripts/test_cross_chem_single_cell.py',
    'src/models/rul_predictors/cycle_residual_single_cell.py',
)


def checked_workspace(root):
    root = Path(root).resolve(strict=True)
    target = root / 'workspaces' / WORKSPACE_NAME
    if target.is_symlink() or target.resolve() != target:
        raise RuntimeError(f'结果路径包含符号链接，停止清理：{target}')
    if target.exists() and not target.is_dir():
        raise RuntimeError(f'结果路径不是目录：{target}')
    return target


def is_training_command(argv, cwd, root):
    if Path(cwd) != Path(root) or '--report' in argv or '--progress' in argv:
        return False
    if not argv or not Path(argv[0]).name.startswith(('python', 'pypy')):
        return False
    if '-c' in argv[1:3] or '-m' in argv[1:3]:
        return False
    # The first non-option argument must be this exact script, not a shell string.
    for value in argv[1:]:
        if value.startswith('-'):
            continue
        path = Path(value)
        candidate = path if path.is_absolute() else Path(cwd) / path
        return path.name == RETIRED_RUNNER and candidate.resolve() == Path(root) / 'scripts' / RETIRED_RUNNER
    return False


def owned_process(pid, root):
    try:
        proc = Path('/proc') / str(pid)
        argv = [v.decode('utf-8', errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
        cwd = (proc/'cwd').resolve(strict=True)
        return is_training_command(argv, cwd, root)
    except (OSError, RuntimeError):
        return False


def training_pids(root):
    return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit() and owned_process(int(p.name), root)]


def stop_training(root):
    pids = training_pids(root)
    for pid in pids:
        if owned_process(pid, root):
            print(f'终止旧单电池训练进程：{pid}', flush=True)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while any(owned_process(pid, root) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.2)
    for pid in pids:
        # Recheck ownership, so an unrelated reused PID is never signalled.
        if owned_process(pid, root):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    while any(owned_process(pid, root) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.2)
    if training_pids(root):
        raise RuntimeError('仍有旧单电池训练进程，未删除结果。')


def remove_cache(root):
    count = 0
    for relative, modules in CACHE_MODULES.items():
        folder = Path(root) / relative
        if folder.is_symlink() or folder.resolve() != folder:
            raise RuntimeError(f'编译缓存路径包含符号链接，停止清理：{folder}')
        if not folder.exists():
            continue
        for module in modules:
            for path in folder.glob(module + '.*.pyc'):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                    count += 1
    return count


def remove_results(root):
    target = checked_workspace(root)
    if target.exists():
        shutil.rmtree(target)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='只清理已撤销单电池训练进程、独立结果和编译缓存；组合迁移及MIX保留')
    parser.add_argument('--dry-run', action='store_true', help='仅列出范围，不停止进程或删除文件')
    args = parser.parse_args()
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('本清理入口仅用于Linux服务器。')
    root = ROOT.resolve(strict=True)
    if not (root/'scripts/run_cross_chem_two_models.py').is_file():
        raise RuntimeError('组合迁移入口不存在，项目位置可能不正确。')
    remaining = [name for name in RETIRED_FILES if (root/name).exists()]
    if remaining:
        raise RuntimeError('仍存在旧单电池代码，请先拉取删除提交：' + ', '.join(remaining))
    target = checked_workspace(root)
    print(f'仅清理结果：{target}', flush=True)
    print('组合迁移和MIX工作区保持原样。', flush=True)
    if args.dry_run:
        print(f'匹配的旧训练进程：{training_pids(root)}')
        print(f'结果目录存在：{target.exists()}')
        return
    import fcntl
    with contextlib.ExitStack() as stack:
        def lock(name):
            handle = stack.enter_context((target/name).open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'单电池{name}仍被占用，未删除结果；请稍后重试。') from None
        if target.exists():
            lock('.launcher.lock')
        stop_training(root)
        if target.exists():
            lock('.queue.lock')
        caches = remove_cache(root)
        removed = remove_results(root)
    print(f'清理完成：结果目录{"已删除" if removed else "不存在，无需删除"}；编译缓存删除{caches}个。')


if __name__ == '__main__':
    main()
