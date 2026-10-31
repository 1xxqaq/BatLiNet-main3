#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$ROOT"
[[ -f scripts/run_cross_chem_earlystop_v6.py ]] || {
  echo "没有定位到v6执行脚本：$ROOT" >&2; exit 1;
}

PREVIOUS="$ROOT/workspaces/cross_chem_matched_v3"
WORKSPACE="$ROOT/workspaces/cross_chem_earlystop_v6"
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
for REQUIRED in queue.json selection.json; do
  [[ -f "$PREVIOUS/$REQUIRED" ]] || {
    echo "缺少v3的实际来源材料：$PREVIOUS/$REQUIRED；不会重新抽取名单或选择学习率。" >&2; exit 1;
  }
done
EXISTING="$("$PYTHON_BIN" -B -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); q=json.loads((p/"queue.json").read_text(encoding="utf-8")); print(Path(q["existing_workspace"]).resolve())' "$PREVIOUS")"
KNOWN_QUEUES=(cross_chem_matched_v3 cross_chem_two_models_v2 \
              cross_chem_earlystop_v4 cross_chem_earlystop_v5 cross_chem_earlystop_v6 \
              cross_chem_cycle_soft_difference_v1 mix_cycle_soft_difference_v1 \
              three_tasks_three_models_v1 mix20_matched_baselines_v1 \
              mix_branch_ablation_v1 mix_naive_ensemble_v1 \
              mix_soft_difference_refinements_v1)
WORKSPACE_PARENTS=("$ROOT/workspaces" "$(dirname -- "$PREVIOUS")")
for REPOSITORY_PARENT in "$(dirname -- "$ROOT")" "$(dirname -- "$(dirname -- "$(dirname -- "$PREVIOUS")")")"; do
  for TREE in BatLiNet-cycle-mixer BatLiNet-cross-chem-earlystop-v4 BatLiNet-cross-chem-earlystop-v5 BatLiNet-cross-chem-earlystop-v6; do
    WORKSPACE_PARENTS+=("$REPOSITORY_PARENT/$TREE/workspaces")
  done
done
PROTECTED=("$PREVIOUS" "$EXISTING")
for PARENT in "${WORKSPACE_PARENTS[@]}"; do
  for NAME in "${KNOWN_QUEUES[@]}"; do
    FOLDER="$PARENT/$NAME"
    if [[ "$NAME" != cross_chem_earlystop_v6 || "$FOLDER" != "$WORKSPACE" ]]; then
      PROTECTED+=("$FOLDER")
    fi
  done
done
"$PYTHON_BIN" -B -c 'from pathlib import Path; import sys; new=Path(sys.argv[1]); old=[Path(p).resolve() for p in sys.argv[2:]]; assert all(new!=p and not new.is_relative_to(p) and not p.is_relative_to(new) for p in old), "新工作区必须独立于v3、来源缓存及已知旧实验目录"' "$WORKSPACE" "${PROTECTED[@]}"
ARGS=(-u -B scripts/run_cross_chem_earlystop_v6.py --previous-workspace "$PREVIOUS" \
      --workspace "$WORKSPACE" --device "$DEVICE" "${EXTRA[@]}")
if [[ "$MODE" == report ]]; then
  exec "$PYTHON_BIN" "${ARGS[@]}"
fi

command -v flock >/dev/null || { echo '启动需要flock。' >&2; exit 1; }
mkdir -p -- "$WORKSPACE/logs"
exec 9>"$WORKSPACE/.launcher.lock"
flock -n 9 || { echo '另一条v6启动命令正在执行。' >&2; exit 1; }

# Old queue lock files are opened read-only. Running queues are not interrupted.
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
for FOLDER in "${PROTECTED[@]}"; do check_old_queue "$FOLDER"; done
if [[ -f "$WORKSPACE/.queue.lock" ]]; then
  exec 8<"$WORKSPACE/.queue.lock"
  flock -n 8 || { echo 'v6队列仍在运行，拒绝重复启动。' >&2; exit 1; }
  flock -u 8
  exec 8<&-
fi
if [[ -f "$WORKSPACE/queue.pid" ]]; then
  read -r OLD_PID < "$WORKSPACE/queue.pid" || true
  if [[ "${OLD_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "v6进程 $OLD_PID 仍在运行，拒绝重复启动。" >&2; exit 1;
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
printf 'v6后台进程：%s\n日志：%s\n' "$PID" "$LOG"
printf 'v3只读来源：%s\n新结果目录：%s\n' "$PREVIOUS" "$WORKSPACE"
echo '已进入conda环境batlinet，OMP/MKL线程数均为8；无需在外层再加nohup。'
echo '每10次更新评估，连续5次未严格改善停止；最早100次、最多1000次更新。'
echo '344次全新训练：单电池方法按训练误差停滞停止，组合按留出电池对关系误差早停。'
echo '不执行来源开发实验，不留出40块LFP；全部训练完成后统一计算336份测试。'
printf '看进度：tail -f %q\n' "$LOG"
