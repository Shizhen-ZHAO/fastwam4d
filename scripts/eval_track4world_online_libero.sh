#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$repo_root/../.." && pwd)
cd "$repo_root"
exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DIFFSYNTH_MODEL_BASE_PATH=/mnt/homes/zhaoshizhen/checkpoints/fastwam/model_base DIFFSYNTH_SKIP_DOWNLOAD=true \
    MUJOCO_GL=egl EGL_DEVICE_ID=0 LIBERO_CONFIG_PATH="$workspace_root/configs/libero_config" \
    PYTHONPATH="$repo_root:$repo_root/src:$workspace_root/.deps/fastwam_track4world:$workspace_root/Track4World:$workspace_root/third_party/libero_base/repos/LIBERO${PYTHONPATH:+:$PYTHONPATH}" \
    OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "${FASTWAM_PYTHON:-$workspace_root/.conda/envs/lf_fastwam/bin/python}" \
    experiments/libero/eval_libero_single.py \
    ckpt=/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224.pt \
    EVALUATION.dataset_stats_path=/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json \
    EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=2 EVALUATION.num_trials=1 \
    EVALUATION.device=cuda:0 EVALUATION.num_inference_steps=20 \
    +EVALUATION.geometry_adapter=outputs/libero_track4world_online_60steps/geometry_adapter.pt \
    EVALUATION.output_dir=outputs/libero_track4world_online_rollout "$@"
