#!/bin/bash
# Tibetan Continual Pretraining with Qwen2.5-3B on 8xH200
# This script trains the model on CUTE Tibetan dataset

set -e

echo "=========================================="
echo "Tibetan Continual Pretraining"
echo "Model: Qwen2.5-3B-base"
echo "Dataset: CUTE Tibetan Corpus"
echo "GPUs: 8xH200"
echo "=========================================="

cd LLaMA-Factory

# Check if preprocessed data exists
if [ ! -f "hf_data/tibetan_datasets/tibetan_cpt_final.json" ]; then
    echo "Error: Preprocessed CPT data not found!"
    echo "Please run 1_preprocess.py first."
    exit 1
fi

# # Create output directory
# mkdir -p /data/private/autodl-tmp/saves/qwen2.5-3b/full/cpt

# # Save original model weights for later comparison
# echo "Saving original model state for weight analysis..."
# python3 -c "
# import torch
# from transformers import AutoModel
# model = AutoModel.from_pretrained('/data/private/autodl-tmp/qwen2-5-3b-base', trust_remote_code=True)
# state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
# torch.save(state_dict, '/data/private/autodl-tmp/saves/qwen2.5-3b/original_weights.pt')
# print('Original weights saved!')
# del model, state_dict
# "

# Launch training with DeepSpeed
echo "Launching CPT training..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5_3b_full_cpt.yaml

echo "=========================================="
echo "CPT Training Complete!"
echo "Model saved to: saves/qwen2.5-3b/full/cpt"
echo "Next step: Run SFT with train_sft.sh"
echo "=========================================="

