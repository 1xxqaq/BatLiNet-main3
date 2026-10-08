#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
SOURCE_ROOT="${1:-/root/autodl-tmp/BatLiNet-main3}"
WORKSPACE="$PWD/workspaces/cross_chem_two_models_v2"
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null
command -v flock >/dev/null
mkdir -p -- "$WORKSPACE/logs"
exec 9>"$WORKSPACE/.launcher.lock"
flock -n 9 || { echo '另一迁移启动命令正在执行。'; exit 1; }
for NAME in cross_chem_two_models_v2 cross_chem_cycle_soft_difference_v1 mix_cycle_soft_difference_v1 three_tasks_three_models_v1; do
  LOCK="$PWD/workspaces/$NAME/.queue.lock"
  if [[ -f "$LOCK" ]]; then
    exec 8>"$LOCK"
    flock -n 8 || { echo "队列 $NAME 仍在运行，拒绝并发启动。"; exit 1; }
    flock -u 8
    exec 8>&-
  fi
done
if [[ -f "$WORKSPACE/queue.pid" ]]; then
  read -r OLD_PID < "$WORKSPACE/queue.pid" || true
  if [[ "${OLD_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "双模型迁移进程 $OLD_PID 仍在运行，拒绝重复启动。"
    exit 1
  fi
fi
LOG="$WORKSPACE/logs/queue_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PYTHON_BIN" -u scripts/run_cross_chem_two_models.py \
  --source-root "$SOURCE_ROOT" --workspace "$WORKSPACE" \
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
printf '迁移后台进程已启动：%s\n日志：%s\n' "$PID" "$LOG"
echo '原始迁移＋组合版：三个目标体系、14档、八种子，每模型112组，共224组；原始每组含两个训练阶段。'
echo '真实数据数量和显存检查通过后才正式训练；检查失败停止，请查看日志。'
echo '可以关闭本地电脑和终端，服务器保持运行；中断后重用本命令按最近一轮恢复。'
echo '看进度：tail -f "$(cat workspaces/cross_chem_two_models_v2/latest_log.txt)"'
echo '看结果：python scripts/run_cross_chem_two_models.py --report'
