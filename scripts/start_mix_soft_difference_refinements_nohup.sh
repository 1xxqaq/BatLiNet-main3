#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
HISTORY_ROOT="${1:-/root/autodl-tmp/BatLiNet-main3}"
ORIGINAL_ROOT="${2:-/root/autodl-tmp/BatLiNet-main2}"
WORKSPACE="$PWD/workspaces/mix_soft_difference_refinements_v1"
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null
command -v flock >/dev/null
command -v nohup >/dev/null
mkdir -p -- "$WORKSPACE/logs"
exec 9>"$WORKSPACE/.launcher.lock"
flock -n 9 || { echo '另一启动命令正在执行。'; exit 1; }
# The child inherits this lock for its entire lifetime, including preparation.
exec 8>"$WORKSPACE/.background.lock"
flock -n 8 || { echo '本实验后台进程仍在运行，拒绝重复启动。'; exit 1; }
LOG="$WORKSPACE/logs/queue_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PYTHON_BIN" -u -B scripts/run_mix_soft_difference_refinements.py \
  --history-root "$HISTORY_ROOT" --original-root "$ORIGINAL_ROOT" --workspace "$WORKSPACE" \
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
printf '已后台启动四组对照，共64次训练。进程号：%s\n日志：%s\n' "$PID" "$LOG"
printf '查看进度：tail -f "%s"\n' "$LOG"
echo '可关闭本地终端；服务器须继续运行。正式预检是否通过请查看日志。'
