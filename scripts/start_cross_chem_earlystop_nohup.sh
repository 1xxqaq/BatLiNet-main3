#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$ROOT"
[[ -f scripts/run_cross_chem_earlystop.py ]] || {
  echo "没有定位到新版早停迁移执行脚本：$ROOT" >&2; exit 1;
}

PREVIOUS="$ROOT/workspaces/cross_chem_matched_v3"
WORKSPACE="$ROOT/workspaces/cross_chem_earlystop_v4"
DEVICE="cuda:0"
MODE="train"
EXTRA=()
while (($#)); do
  case "$1" in
    --previous-workspace|--workspace|--device)
      [[ $# -ge 2 ]] || { echo "$1 缺少参数。" >&2; exit 2; }
      case "$1" in
        --previous-workspace) PREVIOUS="$2" ;;
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

PREVIOUS="$("$PYTHON_BIN" -B -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$PREVIOUS")"
WORKSPACE="$("$PYTHON_BIN" -B -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$WORKSPACE")"
for REQUIRED in queue.json selection.json pilot/split_0.json pilot/split_1.json; do
  [[ -f "$PREVIOUS/$REQUIRED" ]] || {
    echo "缺少v3的实际来源材料：$PREVIOUS/$REQUIRED；不会创建替代划分。" >&2; exit 1;
  }
done
EXISTING="$("$PYTHON_BIN" -B -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); q=json.loads((p/"queue.json").read_text(encoding="utf-8")); print(Path(q["existing_workspace"]).resolve())' "$PREVIOUS")"
KNOWN_QUEUES=(cross_chem_matched_v3 cross_chem_two_models_v2 \
              cross_chem_cycle_soft_difference_v1 mix_cycle_soft_difference_v1 \
              three_tasks_three_models_v1 mix20_matched_baselines_v1 \
              mix_branch_ablation_v1 mix_naive_ensemble_v1 \
              mix_soft_difference_refinements_v1)
PROTECTED=("$PREVIOUS" "$EXISTING")
WORKSPACE_PARENTS=("$ROOT/workspaces" "$(dirname -- "$PREVIOUS")")
for PARENT in "${WORKSPACE_PARENTS[@]}"; do
  for NAME in "${KNOWN_QUEUES[@]}"; do PROTECTED+=("$PARENT/$NAME"); done
done
"$PYTHON_BIN" -B -c 'from pathlib import Path; import sys; new=Path(sys.argv[1]); old=[Path(p).resolve() for p in sys.argv[2:]]; assert all(new!=p and not new.is_relative_to(p) and not p.is_relative_to(new) for p in old), "新工作区必须独立于v3、来源缓存及已知旧实验目录"' "$WORKSPACE" "${PROTECTED[@]}"
ARGS=(-u -B scripts/run_cross_chem_earlystop.py --previous-workspace "$PREVIOUS" \
      --workspace "$WORKSPACE" --device "$DEVICE" "${EXTRA[@]}")
if [[ "$MODE" == report ]]; then
  exec "$PYTHON_BIN" "${ARGS[@]}"
fi

command -v flock >/dev/null || { echo '启动需要flock。' >&2; exit 1; }
mkdir -p -- "$WORKSPACE/logs"
exec 9>"$WORKSPACE/.launcher.lock"
flock -n 9 || { echo '另一条v4启动命令正在执行。' >&2; exit 1; }

# Existing lock files are opened read-only and checked with a shared lock.
# Active training holds an exclusive lock and is refused, never interrupted.
check_old_queue() {
  local FOLDER="$1"
  local LOCK="$FOLDER/.queue.lock"
  if [[ -f "$LOCK" ]]; then
    exec 8<"$LOCK"
    flock -s -n 8 || {
      echo "已有训练队列仍在运行：$FOLDER；请等待其结束，未停止任何旧进程。" >&2
      exit 1
    }
    flock -u 8
    exec 8<&-
  fi
}
check_old_queue "$PREVIOUS"
check_old_queue "$EXISTING"
for PARENT in "${WORKSPACE_PARENTS[@]}"; do
  for NAME in "${KNOWN_QUEUES[@]}"; do check_old_queue "$PARENT/$NAME"; done
done
if [[ -f "$WORKSPACE/.queue.lock" ]]; then
  exec 8<"$WORKSPACE/.queue.lock"
  flock -n 8 || { echo 'v4队列仍在运行，拒绝重复启动。' >&2; exit 1; }
  flock -u 8
  exec 8<&-
fi
if [[ -f "$WORKSPACE/queue.pid" ]]; then
  read -r OLD_PID < "$WORKSPACE/queue.pid" || true
  if [[ "${OLD_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "v4进程 $OLD_PID 仍在运行，拒绝重复启动。" >&2; exit 1;
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
printf 'v4后台进程：%s\n日志：%s\n' "$PID" "$LOG"
printf 'v3只读来源：%s\n新结果目录：%s\n' "$PREVIOUS" "$WORKSPACE"
echo '已进入conda环境batlinet，OMP/MKL线程数均为8；无需在外层再加nohup。'
echo '32次LFP开发早停后确定预算；344次正式训练全部完成后统一测试。'
printf '看进度：tail -f %q\n' "$LOG"
