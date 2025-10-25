#!/bin/bash
# Tibetan Supervised Fine-Tuning with Qwen2.5-3B on 8xH200
# This script performs SFT on the CPT model using instruction data

set -e

echo "=========================================="
echo "Tibetan Supervised Fine-Tuning"
echo "Model: Qwen2.5-3B (after CPT)"
echo "Dataset: Tibetan-Mix + CUTE Parallel"
echo "GPUs: 8xH200"
echo "=========================================="

cd LLaMA-Factory

# Check if CPT model exists
if [ ! -d "saves/qwen2.5-3b/full/cpt" ]; then
    echo "Error: CPT model not found!"
    echo "Please run train_cpt.sh first."
    exit 1
fi

# Check if preprocessed data exists
if [ ! -f "/data/private/autodl-tmp/hf_data/tibetan_datasets/tibetan_sft_final.json" ]; then
    echo "Error: Preprocessed SFT data not found!"
    echo "Please run 1_preprocess.py first."
    exit 1
fi

# Create output directory
mkdir -p saves/qwen2.5-3b/full/sft

# Save CPT model weights for later comparison
echo "Saving CPT model state for weight analysis..."
python3 -c "
import torch
from transformers import AutoModel
import os
import glob

# Find the latest checkpoint
cpt_dir = 'saves/qwen2.5-3b/full/cpt'
checkpoints = glob.glob(os.path.join(cpt_dir, 'checkpoint-*'))
if checkpoints:
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
    model_path = latest_checkpoint
else:
    model_path = cpt_dir

print(f'Loading CPT model from: {model_path}')
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
torch.save(state_dict, 'saves/qwen2.5-3b/cpt_weights.pt')
print('CPT weights saved!')
del model, state_dict
"

Launch training with DeepSpeed
echo "Launching SFT training..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5_3b_full_sft.yaml

echo "=========================================="
echo "SFT Training Complete!"
echo "Model saved to: saves/qwen2.5-3b/full/sft"
echo "Next step: Run evaluation and weight analysis"
echo "=========================================="

Save SFT model weights for later comparison (using epoch 2 checkpoint based on loss curves)
echo "Saving SFT epoch 2 model state for weight analysis..."
python3 -c "
import torch
from transformers import AutoModel
import os

# Use epoch 2 checkpoint (checkpoint-652) based on loss curve analysis
sft_checkpoint = 'saves/qwen2.5-3b/full/sft/checkpoint-652'
if os.path.exists(sft_checkpoint):
    model_path = sft_checkpoint
    print(f'Loading SFT epoch 2 model from: {model_path}')
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state_dict, 'saves/qwen2.5-3b/sft_weights.pt')
    print('SFT epoch 2 weights saved!')
else:
    print(f'Warning: Epoch 2 checkpoint not found at {sft_checkpoint}')
    print('Using final model instead...')
    sft_dir = 'saves/qwen2.5-3b/full/sft'
    model = AutoModel.from_pretrained(sft_dir, trust_remote_code=True)
    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state_dict, 'saves/qwen2.5-3b/sft_weights.pt')
    print('SFT final weights saved!')
"
del model, state_dict
"

