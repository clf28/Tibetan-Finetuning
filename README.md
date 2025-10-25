# Tibetan Language Finetuning with Qwen2.5-3B

**Complete pipeline for adapting Qwen2.5-3B to Tibetan language using full-parameter finetuning.**

---

## 📊 Dataset Configuration

### Data Sources
- **CUTE Parallel Corpus**: 934K aligned triplets (Tibetan-Chinese-English)
- **CUTE Non-Parallel**: 990K Tibetan-only texts
- **tibetan-mix-instruction-tuning-60K**: 30K Tibetan + 18K Chinese instructions

### Dataset Proportions

**CPT (Continual Pretraining)**:
- 200K Tibetan texts (100% Tibetan)
- Language adaptation focused

**SFT (Supervised Fine-Tuning)**:
- 40K Tibetan examples (80%)
- 10K Chinese examples (20%)
- **Why Chinese anchoring?** Prevents catastrophic forgetting in full-parameter mode

---

## ⚙️ Training Configuration

### Hardware Requirements
- **Recommended**: 8x H200 GPUs
- **Minimum**: 4x high-end GPUs with 24GB+ VRAM each
- **Framework**: LLaMA-Factory with DeepSpeed ZeRO-2

### CPT Configuration
```yaml
batch_size: 4 per GPU × 8 GPUs × 4 accum = 128 effective
learning_rate: 5e-5
epochs: 1
sequence_length: 8192
precision: BF16
```

### SFT Configuration
```yaml
batch_size: 8 per GPU × 8 GPUs × 2 accum = 128 effective
learning_rate: 1e-5 
epochs: 2
sequence_length: 4096
precision: BF16
```


---

## 📈 Evaluation Metrics

### Primary Metrics 
- **Perplexity**: Language modeling quality (lower is better)
- **BLEU**: Translation accuracy (Chinese↔Tibetan, English↔Tibetan)
- **chrF**: Character-level translation quality (higher is better)


## 🚀 Quick Start
Download datasets from Huggingface:
- Pretraining: CMLI-NLP/CUTE-Datasets
- SFT: lightman7/tibetan-mix-instruction-tuning-60K

### Prerequisites
```bash
conda create -n finetune python=3.10
conda activate finetune
cd LLaMA-Factory
pip install -e ".[torch,metrics, deepspeed, tensorboard]" --no-build-isolation
pip install sacrebleu matplotlib seaborn pandas
```

### Step 1: Preprocess Data
```bash
python 1_preprocess.py
```
**What it does:**
- Creates Tibetan CPT dataset (200K texts)
- Creates Tibetan SFT dataset (50K examples: 80% Tibetan + 20% Chinese)
- Filters datasets by token length for training efficiency
- **Output**: `hf_data/tibetan_datasets/*.json`

### Step 2: Continual Pretraining
```bash
bash 2_train_cpt.sh
```
**What it does:**
- Adapts model to Tibetan linguistic patterns
- Uses 200K Tibetan texts
- **Output**: `saves/qwen2.5-3b/full/cpt/`

### Step 3: Supervised Fine-Tuning
```bash
bash 3_train_sft.sh
```
**What it does:**
- Instruction following training
- 20% Chinese anchoring prevents catastrophic forgetting
- **Output**: `saves/qwen2.5-3b/full/sft/`

### Step 4: Evaluate Models
```bash
# Evaluate all three models
python 4_evaluate.py --model_path qwen2-5-3b-base --model_name base
python 4_evaluate.py --model_path saves/qwen2.5-3b/full/cpt --model_name cpt
python 4_evaluate.py --model_path saves/qwen2.5-3b/full/sft/checkpoint-652 --model_name sft_epoch2 --compare
```
**What it evaluates:**
- **Perplexity**: Language modeling quality (40-50% improvement expected)
- **Chinese→Tibetan translation**: BLEU and chrF (3-5x improvement expected)
- **English→Tibetan translation**: Cross-lingual transfer benefits

**What it generates:**
- 5 focused comparison charts (perplexity, translation metrics, heatmap, radar)
- JSON metrics files with significant improvement data
- ASCII visualization summaries
- **Output**: `evaluation_results/*.png` and `*.json`

### Step 5: Analyze Weight Changes
```bash
python 5_analyze_weights.py \
    --original qwen2-5-3b-base \
    --cpt saves/qwen2.5-3b/full/cpt \
    --sft saves/qwen2.5-3b/full/sft/checkpoint-652 \
    --output_dir weight_analysis \
    --top_k 20  # Optional: number of top layers to show (default: 20)
```
**What it analyzes:**
- Layer-wise magnitude of changes during CPT and SFT
- Correlation between training stages
- Which layers changed most/least
- Layer type analysis (attention vs MLP vs embeddings)
- Change distributions and statistics

**What it generates:**
- 7 detailed visualization charts showing layer adaptation patterns
- Comprehensive analysis report with insights into model learning
- JSON data for further research and publication
- **Output**: `weight_analysis/*.png` and `*.json`

**Key Insights:**
- **Which layers adapt most** to Tibetan language patterns
- **How CPT and SFT** affect different layer types differently
- **Correlation patterns** between training stages
- **Layer specialization** (attention vs MLP vs embeddings)

### Step 6: Test the Model
```bash
python 6_test_inference.py --model_path saves/qwen2.5-3b/full/sft/checkpoint-652
```
**Interactive testing:**
- Try Tibetan instructions
- Test Chinese→Tibetan translation
- Verify model capabilities

---

## 📊 Generated Outputs

### Training Outputs
- **CPT Model**: `saves/qwen2.5-3b/full/cpt/`
- **SFT Model**: `saves/qwen2.5-3b/full/sft/`

### Filtered Datasets
- **SFT Training**: `hf_data/tibetan_datasets/tibetan_sft_filtered_4k.json`
- **SFT Test**: `hf_data/tibetan_datasets/tibetan_sft_test_filtered_4k.json`
- **CPT Test**: `hf_data/tibetan_datasets/tibetan_cpt_test_filtered_4k.json`

### Evaluation Results (5 focused visualizations)
- **Metrics**: `evaluation_results/*_evaluation.json` (perplexity, BLEU, chrF)
- **perplexity_comparison.png** - Language modeling performance across stages
- **translation_metrics_comparison.png** - Chinese translation metrics (BLEU & chrF)
- **combined_metrics_overview.png** - All significant metrics in one view
- **metrics_heatmap.png** - Normalized performance comparison
- **radar_chart.png** - Multi-dimensional view of improvements
- **Summary**: `evaluation_results/evaluation_summary.txt` (ASCII charts included)

### Weight Analysis Results (7 visualizations)
- **cpt_layer_changes.png** - Top 20 layers with largest CPT changes
- **sft_layer_changes.png** - Top 20 layers with largest SFT changes
- **cpt_sft_correlation.png** - Correlation between CPT and SFT changes (scatter plot)
- **cpt_layer_types.png** - Changes grouped by layer type (attention, MLP, embedding)
- **sft_layer_types.png** - SFT changes by layer type
- **cpt_distributions.png** - Statistical distributions of CPT changes
- **sft_distributions.png** - Statistical distributions of SFT changes
- **analysis_summary.txt** - Detailed text report with insights
- **weight_analysis_results.json** - Complete JSON data for research

---

## 🗂️ Project Structure

```
├── README.md                              # This file
├── 1_preprocess.py                        # Data preprocessing & filtering
├── 2_train_cpt.sh                         # CPT training script
├── 3_train_sft.sh                         # SFT training script
├── 4_evaluate.py                          # Evaluation with visualizations
├── 5_analyze_weights.py                   # Layer-wise weight change analysis
├── 6_test_inference.py                    # Interactive model testing
│
├── hf_data/                               # Downloaded datasets
│   ├── CUTE-Datasets/                     # Tibetan parallel/non-parallel corpus
│   └── tibetan-mix-instruction-tuning-60K/ # Mixed instruction data
│
├── LLaMA-Factory/                         # Training framework
│   └── examples/train_full/               # YAML configurations
│
└── qwen2-5-3b-base/                       # Base model (5.8GB)
```

---


## 👏 Acknowledge
The training of this project is based on [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory/tree/main)

