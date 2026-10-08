#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
HISTORY_ROOT="${1:-/root/autodl-tmp/BatLiNet-main3}"
ORIGINAL_ROOT="${2:-/root/autodl-tmp/BatLiNet-main2}"
WORKSPACE="$PWD/workspaces/mix_cycle_soft_difference_v1"
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null
command -v flock >/dev/null
mkdir -p -- "$WORKSPACE/logs"
exec 9>"$WORKSPACE/.launcher.lock"
flock -n 9 || { echo '另一组合模型启动命令正在执行。'; exit 1; }
if [[ -f "$PWD/workspaces/three_tasks_three_models_v1/.queue.lock" ]]; then
  exec 8>"$PWD/workspaces/three_tasks_three_models_v1/.queue.lock"
  flock -n 8 || { echo '暂缓的三任务队列仍在运行。请先停止该队列，再启动MIX组合实验。'; exit 1; }
  flock -u 8
  exec 8>&-
fi
if [[ -f "$WORKSPACE/queue.pid" ]]; then
  read -r OLD_PID < "$WORKSPACE/queue.pid" || true
  if [[ "${OLD_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "组合模型进程 $OLD_PID 仍在运行，拒绝重复启动。"
    exit 1
  fi
fi
LOG="$WORKSPACE/logs/queue_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PYTHON_BIN" -u scripts/run_mix_cycle_soft_difference.py \
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
printf '已后台启动，仅组合版共16次训练。进程号：%s\n日志：%s\n查看：tail -f "%s"\n' "$PID" "$LOG" "$LOG"
echo '可以关闭本地电脑或终端；服务器保持运行。检查失败会停止，正式预检结果请查看日志。'
