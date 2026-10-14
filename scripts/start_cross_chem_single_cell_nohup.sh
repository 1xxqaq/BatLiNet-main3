#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
BASE_WORKSPACE="${1:-$PWD/workspaces/cross_chem_two_models_v2}"
WORKSPACE="$PWD/workspaces/cross_chem_single_cell_v1"
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null
command -v flock >/dev/null
[[ -f "$BASE_WORKSPACE/dataset/manifest.json" && -f "$BASE_WORKSPACE/queue.json" ]] || {
  echo "缺少已完成迁移的数据和名单：$BASE_WORKSPACE；不要删除旧迁移工作区。"; exit 1;
}
mkdir -p -- "$WORKSPACE/logs"
exec 9>"$WORKSPACE/.launcher.lock"
flock -n 9 || { echo '另一单电池启动命令正在执行。'; exit 1; }
for DIR in "$WORKSPACE" "$BASE_WORKSPACE" "$PWD/workspaces/cross_chem_cycle_soft_difference_v1" "$PWD/workspaces/mix_cycle_soft_difference_v1" "$PWD/workspaces/three_tasks_three_models_v1"; do
  LOCK="$DIR/.queue.lock"
  if [[ -f "$LOCK" ]]; then
    exec 8>"$LOCK"
    flock -n 8 || { echo "队列仍在运行：$DIR；拒绝并发启动。"; exit 1; }
    flock -u 8
    exec 8>&-
  fi
done
LOG="$WORKSPACE/logs/queue_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PYTHON_BIN" -u scripts/run_cross_chem_single_cell.py \
  --base-workspace "$BASE_WORKSPACE" --workspace "$WORKSPACE" \
  >"$LOG" 2>&1 </dev/null 9>&- &
PID=$!
printf '%s\n' "$PID" > "$WORKSPACE/queue.pid"
printf '%s\n' "$LOG" > "$WORKSPACE/latest_log.txt"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "启动失败，日志：$LOG"
  tail -n 80 -- "$LOG"
  exit 1
fi
printf '单电池后台进程已启动：%s\n日志：%s\n' "$PID" "$LOG"
echo '两种对照共224组目标实验，另有8次LFP预训练；全部固定100轮。'
echo '可关闭本地电脑与终端，服务器需保持运行；中断后重用本命令恢复。'
echo '进度与北京时间预计结束：python scripts/run_cross_chem_single_cell.py --progress'
echo '实时日志：tail -f "$(cat workspaces/cross_chem_single_cell_v1/latest_log.txt)"'
echo '最终结果：python scripts/run_cross_chem_single_cell.py --report'
