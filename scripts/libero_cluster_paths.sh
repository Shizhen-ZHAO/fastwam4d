#!/usr/bin/env bash
# 训练与评测共用。可以在这里改路径，也可以在调用前 export 覆盖。
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-/new-interaction/share/user_folder/jiahao.ljh/data/LIBERO-fastwam}"
export WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/mnt/new-interaction-p/common/user_folder/shizhen/checkpoints/Wan2.2-TI2V-5B}"
export MODEL_BASE="${MODEL_BASE:-$(dirname "${WAN_CHECKPOINT_DIR}")}"
export MODEL_ID="${MODEL_ID:-$(basename "${WAN_CHECKPOINT_DIR}")}"
# 直接使用 Wan 目录里的 Wan2.2_VAE.pth / models_t5_umt5-xxl-enc-bf16.pth。
export REDIRECT_COMMON_FILES="${REDIRECT_COMMON_FILES:-false}"
# 跟随参考 train_libero_offline_16gpu.sh 及其 model/data 配置的仓库相对路径。
export ACTION_DIT_CHECKPOINT="${ACTION_DIT_CHECKPOINT:-${REPO_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export TEXT_CACHE="${TEXT_CACHE:-${REPO_ROOT}/data/text_embeds_cache/libero}"
# 参考脚本不指定预先计算的 stats：训练时自动计算，保存到 OUTPUT_DIR/dataset_stats.json。
# 严格 run_pair 对照先用 prepare_stats.py 计算一次，再 export DATASET_STATS 供两边共用。
export DATASET_STATS="${DATASET_STATS:-null}"
# tokenizer 沿参考模型配置和模型根目录；绝对 model_id 不受上方 Wan 根目录覆盖影响。
export TOKENIZER_MODEL_ID="${TOKENIZER_MODEL_ID:-/new_interaction_group/common_models/Wan-AI__Wan2.1-I2V-14B-480P}"
# 参考启动链直接调用当前 PATH 的 python/accelerate，没有固定 conda/CUDA 路径。
export FASTWAM_ENV="${FASTWAM_ENV:-}"
export CUDA_HOME="${CUDA_HOME:-}"
