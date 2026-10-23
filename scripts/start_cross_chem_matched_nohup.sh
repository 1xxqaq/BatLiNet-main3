#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$ROOT"
[[ -f scripts/run_cross_chem_matched.py && -f scripts/cross_chem_matched_data.py ]] || {
  echo "没有定位到完整的新迁移代码目录：$ROOT" >&2; exit 1;
}

# --report is read-only and runs in the foreground; preparation/training detach.
EXISTING="$ROOT/workspaces/cross_chem_two_models_v2"
WORKSPACE="$ROOT/workspaces/cross_chem_matched_v3"
DEVICE="cuda:0"
MODE="train"
EXTRA=()
while (($#)); do
  case "$1" in
    --existing-workspace|--workspace|--device)
      [[ $# -ge 2 ]] || { echo "$1 缺少参数。" >&2; exit 2; }
      case "$1" in
        --existing-workspace) EXISTING="$2" ;;
        --workspace) WORKSPACE="$2" ;;
        --device) DEVICE="$2" ;;
      esac
      shift 2 ;;
    --prepare-only) [[ "$MODE" == train ]] || { echo '运行模式重复。' >&2; exit 2; }; MODE=prepare; EXTRA+=(--prepare-only); shift ;;
    --pilot-only) [[ "$MODE" == train ]] || { echo '运行模式重复。' >&2; exit 2; }; MODE=pilot; EXTRA+=(--pilot-only); shift ;;
    --report) [[ "$MODE" == train ]] || { echo '运行模式重复。' >&2; exit 2; }; MODE=report; EXTRA+=(--report); shift ;;
    *) echo "不支持的启动参数：$1" >&2; exit 2 ;;
  esac
done

# Activate the named environment without assuming a particular conda install.
if ! command -v conda >/dev/null 2>&1; then
  for CANDIDATE in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                   "$HOME/anaconda3/etc/profile.d/conda.sh" \
                   /opt/conda/etc/profile.d/conda.sh; do
    if [[ -f "$CANDIDATE" ]]; then source "$CANDIDATE"; break; fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo '找不到conda，请先安装或初始化conda。' >&2; exit 1; }
if ! declare -F conda >/dev/null; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate batlinet
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Resolve without creating or touching any file in the legacy workspace.
EXISTING="$("$PYTHON_BIN" -B -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$EXISTING")"
WORKSPACE="$("$PYTHON_BIN" -B -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$WORKSPACE")"
"$PYTHON_BIN" -B -c 'from pathlib import Path; import sys; a,b=map(Path,sys.argv[1:]); assert a!=b and not a.is_relative_to(b) and not b.is_relative_to(a), "新旧工作区必须互相独立"' "$EXISTING" "$WORKSPACE"
[[ -f "$EXISTING/dataset/manifest.json" && -f "$EXISTING/queue.json" ]] || {
  echo "缺少旧队列实际缓存/收据：$EXISTING；不会生成替代名单。" >&2; exit 1;
}
ARGS=(-u -B scripts/run_cross_chem_matched.py --existing-workspace "$EXISTING" \
      --workspace "$WORKSPACE" --device "$DEVICE" "${EXTRA[@]}")
if [[ "$MODE" == report ]]; then
  exec "$PYTHON_BIN" "${ARGS[@]}"
fi

command -v flock >/dev/null || { echo '启动需要flock。' >&2; exit 1; }
mkdir -p -- "$WORKSPACE/logs"
exec 9>"$WORKSPACE/.launcher.lock"
flock -n 9 || { echo '另一条新迁移启动命令正在执行。' >&2; exit 1; }
# Read-only legacy lock inspection: never create/truncate/touch its lock files.
if [[ -f "$EXISTING/.queue.lock" ]]; then
  exec 8<"$EXISTING/.queue.lock"
  flock -n 8 || { echo '旧迁移队列仍在运行，请等待其结束。' >&2; exit 1; }
  flock -u 8
  exec 8<&-
fi
if [[ -f "$WORKSPACE/.queue.lock" ]]; then
  exec 8<"$WORKSPACE/.queue.lock"
  flock -n 8 || { echo '新迁移队列仍在运行，拒绝重复启动。' >&2; exit 1; }
  flock -u 8
  exec 8<&-
fi
if [[ -f "$WORKSPACE/queue.pid" ]]; then
  read -r OLD_PID < "$WORKSPACE/queue.pid" || true
  if [[ "${OLD_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "进程 $OLD_PID 仍在运行，拒绝重复启动。" >&2; exit 1;
  fi
fi
LOG="$WORKSPACE/logs/queue_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PYTHON_BIN" "${ARGS[@]}" >"$LOG" 2>&1 </dev/null 9>&- &
PID=$!
printf '%s\n' "$PID" > "$WORKSPACE/queue.pid"
printf '%s\n' "$LOG" > "$WORKSPACE/latest_log.txt"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  set +e
  wait "$PID"
  EXIT_CODE=$?
  set -e
  echo "后台任务已结束，退出码 $EXIT_CODE，日志：$LOG"
  tail -n 80 -- "$LOG"
  exit "$EXIT_CODE"
fi
printf '新迁移后台进程：%s\n日志：%s\n' "$PID" "$LOG"
printf '旧队列只读来源：%s\n新结果目录：%s\n' "$EXISTING" "$WORKSPACE"
echo '已进入conda环境batlinet，OMP/MKL线程数均为8；无需在外层再加nohup。'
echo '可关闭本地终端；查看进度：tail -f "$(cat workspaces/cross_chem_matched_v3/latest_log.txt)"'
