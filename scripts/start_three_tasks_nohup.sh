#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
SOURCE_ROOT="${1:-/root/autodl-tmp/BatLiNet-main3}"
WORKSPACE="$PWD/workspaces/three_tasks_three_models_v1"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p -- "$WORKSPACE/logs"
command -v "$PYTHON_BIN" >/dev/null
command -v flock >/dev/null
exec 9>"$WORKSPACE/.launcher.lock"
flock -n 9 || { echo '另一启动命令正在执行。'; exit 1; }
if [[ -f "$WORKSPACE/queue.pid" ]]; then
  read -r OLD_PID < "$WORKSPACE/queue.pid" || true
  if [[ "${OLD_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "记录的进程 $OLD_PID 仍在运行，请先查看状态；拒绝重复启动。"
    exit 1
  fi
fi
LOG="$WORKSPACE/logs/queue_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PYTHON_BIN" -u scripts/run_three_tasks.py \
  --source-root "$SOURCE_ROOT" --workspace "$WORKSPACE" \
  >"$LOG" 2>&1 </dev/null 9>&- &
PID=$!
printf '%s\n' "$PID" > "$WORKSPACE/queue.pid"
printf '%s\n' "$LOG" > "$WORKSPACE/latest_log.txt"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "启动失败，日志：$LOG"
  tail -n 40 -- "$LOG"
  exit 1
fi
printf '已后台启动，进程号：%s\n日志：%s\n查看进度：tail -f "%s"\n' "$PID" "$LOG" "$LOG"
echo '可以关闭本地电脑或终端；服务器需保持运行。任务异常会停止，检查日志后重跑此命令可续训。'
