#!/usr/bin/env bash
# LingBot-VLA 2.0 / RoboTwin 评测入口
# 用法:
#   bash eval.sh serve <checkpoint_dir> [port]
#   bash eval.sh run   <task_name> <demo_clean|demo_randomized> <port> <episodes> [seed]
#
# 说明:
#   - serve 起 WebSocket 策略服务（FP32），默认端口 13400
#   - run   启动一个任务一个 setting 的闭环评测，每回合由任务内置 check_success() 判定
#   - ROBOTWIN_ROOT 指向 RoboTwin 2.0 代码目录
set -euo pipefail

ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-$(pwd)}
PYTHON=${PYTHON:-python}

MODE=${1:-}
shift || true

case "$MODE" in
  serve)
    CKPT=${1:?checkpoint dir required}
    PORT=${2:-13400}
    export QWEN3VL_PATH=${QWEN3VL_PATH:?set QWEN3VL_PATH to Qwen3-VL-4B-Instruct}
    export LINGBOT_DENOISE_STEPS=10
    "$PYTHON" -m deploy.lingbot_vla_v2_policy \
      --model_path "$CKPT" \
      --use_length 50 \
      --use_bf16 False \
      --use_fp32 True \
      --use_compile True \
      --port "$PORT"
    ;;

  run)
    TASK=${1:?task name required}
    SETTING=${2:?demo_clean or demo_randomized}
    PORT=${3:-13400}
    EPISODES=${4:-100}
    SEED=${5:-0}
    cd "$ROBOTWIN_ROOT"
    "$PYTHON" script/eval_policy_client_lingbotvla.py \
      --task_name "$TASK" \
      --task_config "$SETTING" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --device_id 0 \
      --seed "$SEED" \
      --test_num "$EPISODES" \
      --expert_check true \
      --accept_expert_info_on_failure true \
      --eval_batch false \
      --additional_info eval_video_log=false
    ;;

  *)
    echo "usage: bash eval.sh serve <ckpt> [port]" >&2
    echo "       bash eval.sh run <task> <demo_clean|demo_randomized> <port> <episodes> [seed]" >&2
    exit 1
    ;;
esac
