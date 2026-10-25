#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
CODE_ROOT="$PWD"
if (( $# > 3 )); then
  echo "用法：bash $0 [历史目录] [原始目录] [本次输出目录]" >&2
  exit 2
fi
HISTORY_ROOT="${1:-/root/autodl-tmp/BatLiNet-main3}"
ORIGINAL_ROOT="${2:-/root/autodl-tmp/BatLiNet-main2}"
WORKSPACE="${3:-$CODE_ROOT/workspaces/mix_branch_ablation_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
for REQUIRED in "$PYTHON_BIN" flock nohup; do
  command -v "$REQUIRED" >/dev/null || {
    echo "启动失败：找不到命令 $REQUIRED。请先激活运行环境。" >&2
    exit 1
  }
done
mkdir -p -- "$WORKSPACE/logs"
WORKSPACE="$(cd -- "$WORKSPACE" && pwd -P)"

# 串行化启动命令；运行状态由内核锁判定，不把复用的旧 PID 当作运行中。
exec 9>"$WORKSPACE/.launcher.lock"
flock -n 9 || { echo '另一分支消融启动命令正在执行。' >&2; exit 1; }
exec 8>"$WORKSPACE/.queue.lock"
flock -n 8 || {
  echo "本输出目录已有队列运行，拒绝重复启动：$WORKSPACE" >&2
  exit 1
}
flock -u 8
exec 8>&-

LOG="$WORKSPACE/logs/queue_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PYTHON_BIN" -u "$CODE_ROOT/scripts/run_mix_branch_ablation.py" \
  --history-root "$HISTORY_ROOT" --original-root "$ORIGINAL_ROOT" \
  --workspace "$WORKSPACE" --device "$DEVICE" \
  >"$LOG" 2>&1 </dev/null 9>&- &
PID=$!
printf '%s\n' "$PID" > "$WORKSPACE/queue.pid"
printf '%s\n' "$LOG" > "$WORKSPACE/latest_log.txt"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  if wait "$PID"; then
    printf '入口已正常退出；请查看结果和日志确认完成状态。\n日志：%s\n' "$LOG"
    tail -n 40 -- "$LOG"
    exit 0
  else
    RESULT=$?
    printf '启动失败（退出码 %s），日志：%s\n' "$RESULT" "$LOG" >&2
    tail -n 80 -- "$LOG" >&2
    exit "$RESULT"
  fi
fi
printf '已后台启动分支消融：48 次训练，64 份评价。进程号：%s\n设备：%s\n日志：%s\n查看：tail -f "%s"\n' \
  "$PID" "$DEVICE" "$LOG" "$LOG"
echo '脚本内部已使用 nohup，可以关闭终端或本地电脑；服务器须保持运行。'
echo '进程存活不等于数据及显存预检通过，请查看日志。相同 GPU 不要同时运行其他训练任务。'
