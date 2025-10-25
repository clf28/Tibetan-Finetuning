"""
Final Preprocessing with Chinese Data Mixing for Full-Parameter SFT

Strategy:
- CPT: 100% Tibetan (language adaptation)
- SFT: 80% Tibetan + 20% Chinese (prevent catastrophic forgetting)

This is CRITICAL for full-parameter finetuning!
"""

import json
import random
import os
from tqdm import tqdm
from transformers import AutoTokenizer

random.seed(42)

# Configuration
CPT_MAX_LINES = 200000  # 200K Tibetan texts
# Hold-out test sizes (approx upper bounds; actual will cap to available data)
CPT_TEST_SIZE = 1000
SFT_TEST_SIZE = 500

# Filtering Configuration
MODEL_PATH = "qwen2-5-3b-base"  # Relative to project root
SFT_CUTOFF_LEN = 4096  # Match SFT training config
CPT_CUTOFF_LEN = 8192  # Match CPT training config

# SFT Configuration (Full-parameter strategy)
SFT_TIBETAN_INSTRUCTION = 16000  # 80% of 50K = 40K total Tibetan
SFT_TIBETAN_TRANSLATION_MIX = 8000
SFT_TIBETAN_TRANSLATION_CUTE = 16000
# Total Tibetan: 40K (80%)

SFT_CHINESE_GENERAL = 10000  # 20% Chinese for anchoring
# Total: 50K examples

# Translation templates
TRANSLATION_TEMPLATES_CN_TO_BO = [
    "请将以下中文翻译成藏文：{}",
    "把这段中文翻译成藏文：{}",
    "将下面的中文句子翻译为藏文：{}",
    "请帮我把这句中文翻译成藏文：{}",
    "中文转藏文：{}",
    "翻译成藏文：{}",
    "请用藏文表达以下内容：{}",
]

TRANSLATION_TEMPLATES_EN_TO_BO = [
    "Please translate the following English to Tibetan: {}",
    "Translate this English text to Tibetan: {}",
    "Convert the following English sentence to Tibetan: {}",
    "Please help me translate this English to Tibetan: {}",
    "English to Tibetan: {}",
    "Translate to Tibetan: {}",
    "Express the following in Tibetan: {}",
]

def has_tibetan(text):
    return any('\u0F00' <= char <= '\u0FFF' for char in text)

def has_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def load_tibetan_mix():
    """Load and categorize tibetan-mix dataset"""
    print("Loading tibetan-mix dataset...")
    
    with open('hf_data/tibetan-mix-instruction-tuning-60K/gpt4_zhbo_60k.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tibetan_instructions = []
    chinese_instructions = []  # Chinese general data
    cn_to_bo = []
    
    for item in data:
        inst = item.get('instruction', '').strip()
        inp = item.get('input', '').strip()
        out = item.get('output', '').strip()
        
        if not inst or not out:
            continue
        
        # Categorize by output language
        if has_tibetan(out) and not has_chinese(out):
            # Pure Tibetan output
            if has_tibetan(inst):
                # Tibetan instruction → Tibetan output
                tibetan_instructions.append({
                    'instruction': inst,
                    'input': inp,
                    'output': out
                })
            elif has_chinese(inst):
                # Chinese instruction → Tibetan output (translation)
                cn_to_bo.append({
                    'chinese': inst,
                    'tibetan': out
                })
        elif has_chinese(out) and not has_tibetan(out):
            # Pure Chinese output - use for anchoring!
            if has_chinese(inst):
                chinese_instructions.append({
                    'instruction': inst,
                    'input': inp,
                    'output': out
                })
    
    print(f"  ✓ Tibetan instructions: {len(tibetan_instructions):,}")
    print(f"  ✓ Chinese instructions: {len(chinese_instructions):,}")
    print(f"  ✓ Chinese→Tibetan pairs: {len(cn_to_bo):,}")
    
    return tibetan_instructions, chinese_instructions, cn_to_bo

def load_cute_parallel():
    """Load CUTE parallel corpus"""
    print("Loading CUTE parallel corpus...")
    
    bo_path = "hf_data/CUTE-Datasets/parallel-corpus/bo.txt"
    zh_path = "hf_data/CUTE-Datasets/parallel-corpus/zh.txt"
    en_path = "hf_data/CUTE-Datasets/parallel-corpus/en.txt"
    
    with open(bo_path, 'r', encoding='utf-8') as f:
        bo_lines = [line.strip() for line in f if line.strip()]
    
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh_lines = [line.strip() for line in f if line.strip()]
    
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]
    
    min_len = min(len(bo_lines), len(zh_lines), len(en_lines))
    parallel_data = []
    
    for i in range(min_len):
        if bo_lines[i] and zh_lines[i] and en_lines[i]:
            parallel_data.append({
                'tibetan': bo_lines[i],
                'chinese': zh_lines[i],
                'english': en_lines[i]
            })
    
    print(f"  ✓ Loaded {len(parallel_data):,} parallel triplets")
    return parallel_data

def load_cute_nonparallel():
    """Load CUTE non-parallel Tibetan"""
    print("Loading CUTE non-parallel corpus...")
    
    bo_path = "hf_data/CUTE-Datasets/non-parallel-corpus/n-bo.txt"
    
    with open(bo_path, 'r', encoding='utf-8') as f:
        bo_lines = [line.strip() for line in f if line.strip()]
    
    print(f"  ✓ Loaded {len(bo_lines):,} lines")
    return bo_lines

def preprocess_for_cpt():
    """CPT: 100% Tibetan (language adaptation)"""
    print("\n" + "="*80)
    print(f"CPT: 100% Tibetan (target: {CPT_MAX_LINES:,} lines)")
    print("="*80)
    
    all_tibetan = []
    
    # From CUTE
    cute_parallel = load_cute_parallel()
    all_tibetan.extend([item['tibetan'] for item in cute_parallel])
    
    cute_nonparallel = load_cute_nonparallel()
    all_tibetan.extend(cute_nonparallel)
    
    # From tibetan-mix
    tibetan_instructions, _, _ = load_tibetan_mix()
    for item in tibetan_instructions:
        all_tibetan.append(item['instruction'])
        all_tibetan.append(item['output'])
    
    # Sample
    random.shuffle(all_tibetan)
    sampled = all_tibetan[:CPT_MAX_LINES]
    
    cpt_data = [{'text': text.strip()} for text in sampled if len(text.strip()) > 10]
    
    print(f"  ✓ Final CPT dataset (pre-split): {len(cpt_data):,} examples (100% Tibetan)")
    
    # Train/Test split
    test_size = min(CPT_TEST_SIZE, max(1000, len(cpt_data) // 20))  # ~5% or capped
    indices = list(range(len(cpt_data)))
    random.shuffle(indices)
    test_indices = set(indices[:test_size])
    cpt_test = [cpt_data[i] for i in range(len(cpt_data)) if i in test_indices]
    cpt_train = [cpt_data[i] for i in range(len(cpt_data)) if i not in test_indices]
    
    print(f"  ✓ CPT split → train: {len(cpt_train):,}, test: {len(cpt_test):,}")
    
    # Save
    output_dir = "hf_data/tibetan_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, "tibetan_cpt_final.json")
    test_file = os.path.join(output_dir, "tibetan_cpt_test.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(cpt_train, f, ensure_ascii=False, indent=2)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(cpt_test, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved train to {train_file}")
    print(f"  ✓ Saved test  to {test_file}")
    return cpt_train

def preprocess_for_sft():
    """SFT: 80% Tibetan + 20% Chinese (full-parameter strategy)"""
    print("\n" + "="*80)
    print("SFT: 80% Tibetan + 20% Chinese (Full-Parameter Strategy)")
    print("="*80)
    print("Why? Prevent catastrophic forgetting in full-parameter finetuning!")
    print("="*80)
    
    sft_data = []
    
    # Load all data
    tibetan_instructions, chinese_instructions, mix_cn_to_bo = load_tibetan_mix()
    cute_parallel = load_cute_parallel()
    
    # ========== 80% TIBETAN DATA ==========
    print(f"\n📊 Adding 80% Tibetan data...")
    
    # 1. Tibetan instructions (BO→BO)
    print(f"  1. Tibetan instructions: {SFT_TIBETAN_INSTRUCTION:,}")
    random.shuffle(tibetan_instructions)
    for item in tibetan_instructions[:SFT_TIBETAN_INSTRUCTION]:
        sft_data.append({
            'instruction': item['instruction'],
            'input': item.get('input', ''),
            'output': item['output'],
            'source': 'tibetan_instruction'
        })
    
    # 2. Translation pairs from tibetan-mix (CN→BO)
    print(f"  2. tibetan-mix translations: {SFT_TIBETAN_TRANSLATION_MIX:,}")
    random.shuffle(mix_cn_to_bo)
    for item in mix_cn_to_bo[:SFT_TIBETAN_TRANSLATION_MIX]:
        template = random.choice(TRANSLATION_TEMPLATES_CN_TO_BO)
        sft_data.append({
            'instruction': template.format(item['chinese']),
            'input': '',
            'output': item['tibetan'],
            'source': 'translation_mix'
        })
    
    # 3. Translation pairs from CUTE (CN→BO, EN→BO)
    print(f"  3. CUTE translations: {SFT_TIBETAN_TRANSLATION_CUTE:,}")
    random.shuffle(cute_parallel)
    
    cute_count = 0
    for item in cute_parallel[:SFT_TIBETAN_TRANSLATION_CUTE]:
        if len(item['tibetan']) < 5 or len(item['chinese']) < 5:
            continue
        
        # CN→BO
        template = random.choice(TRANSLATION_TEMPLATES_CN_TO_BO)
        sft_data.append({
            'instruction': template.format(item['chinese']),
            'input': '',
            'output': item['tibetan'],
            'source': 'translation_cute'
        })
        cute_count += 1
        
        # EN→BO (50% chance)
        if len(item['english']) > 5 and random.random() > 0.5:
            template = random.choice(TRANSLATION_TEMPLATES_EN_TO_BO)
            sft_data.append({
                'instruction': template.format(item['english']),
                'input': '',
                'output': item['tibetan'],
                'source': 'translation_cute_en'
            })
            cute_count += 1
    
    tibetan_total = len(sft_data)
    print(f"  ✓ Total Tibetan data: {tibetan_total:,}")
    
    # ========== 20% CHINESE DATA (ANCHORING) ==========
    print(f"\n🔗 Adding 20% Chinese data for anchoring...")
    print(f"  Purpose: Prevent catastrophic forgetting in full-parameter SFT")
    print(f"  Target: {SFT_CHINESE_GENERAL:,} examples")
    
    random.shuffle(chinese_instructions)
    for item in chinese_instructions[:SFT_CHINESE_GENERAL]:
        sft_data.append({
            'instruction': item['instruction'],
            'input': item.get('input', ''),
            'output': item['output'],
            'source': 'chinese_anchor'
        })
    
    chinese_total = len(sft_data) - tibetan_total
    print(f"  ✓ Added {chinese_total:,} Chinese examples")
    
    # Final shuffle
    random.shuffle(sft_data)
    
    # Remove 'source' tag before saving (only for analysis)
    sft_data_clean = []
    for item in sft_data:
        sft_data_clean.append({
            'instruction': item['instruction'],
            'input': item['input'],
            'output': item['output']
        })
    
    # Print statistics (pre-split)
    print(f"\n📊 Final SFT Dataset Statistics (pre-split):")
    print(f"  Total: {len(sft_data):,} examples")
    print(f"  Tibetan: {tibetan_total:,} ({tibetan_total/len(sft_data)*100:.1f}%)")
    print(f"  Chinese: {chinese_total:,} ({chinese_total/len(sft_data)*100:.1f}%)")
    print(f"\n  ✅ Ratio: {tibetan_total/len(sft_data)*100:.0f}% Tibetan / {chinese_total/len(sft_data)*100:.0f}% Chinese")
    
    # Train/Test split
    test_size = min(SFT_TEST_SIZE, max(1000, len(sft_data_clean) // 20))  # ~5% or capped
    indices = list(range(len(sft_data_clean)))
    random.shuffle(indices)
    test_indices = set(indices[:test_size])
    sft_test = [sft_data_clean[i] for i in range(len(sft_data_clean)) if i in test_indices]
    sft_train = [sft_data_clean[i] for i in range(len(sft_data_clean)) if i not in test_indices]
    
    print(f"  ✓ SFT split → train: {len(sft_train):,}, test: {len(sft_test):,}")
    
    # Save
    output_dir = "hf_data/tibetan_datasets"
    train_file = os.path.join(output_dir, "tibetan_sft_final.json")
    test_file = os.path.join(output_dir, "tibetan_sft_test.json")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(sft_train, f, ensure_ascii=False, indent=2)
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(sft_test, f, ensure_ascii=False, indent=2)
    
    print(f"\n  ✓ Saved train to {train_file}")
    print(f"  ✓ Saved test  to {test_file}")
    
    return sft_train

def format_example(example):
    """Format example as it would appear during training"""
    instruction = example.get('instruction', '').strip()
    input_text = example.get('input', '').strip()
    output_text = example.get('output', '').strip()

    # Qwen chat template format
    if input_text:
        formatted = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
    else:
        formatted = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"

    return formatted

def filter_sft_by_tokens():
    """Filter SFT dataset by token length for efficient training"""
    print("="*80)
    print("FILTER SFT DATASET BY TOKEN LENGTH")
    print("="*80)
    print(f"Cutoff length: {SFT_CUTOFF_LEN} tokens")
    print(f"Model: {MODEL_PATH}")

    input_file = "hf_data/tibetan_datasets/tibetan_sft_final.json"
    output_file = "hf_data/tibetan_datasets/tibetan_sft_filtered_4k.json"

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Load tokenizer
    print("\n🔄 Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✓ Tokenizer loaded")

    # Load data
    print("\n🔄 Loading SFT dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data):,} examples")

    # Analyze token lengths
    print("\n🔄 Analyzing token lengths...")
    token_counts = []
    filtered_data = []
    source_stats = {'original': {}, 'filtered': {}}

    for example in tqdm(data, desc="Processing examples"):
        # Count tokens
        formatted_text = format_example(example)
        token_count = len(tokenizer.encode(formatted_text))

        # Track source statistics
        source = example.get('source', 'unknown')
        if source not in source_stats['original']:
            source_stats['original'][source] = 0
            source_stats['filtered'][source] = 0
        source_stats['original'][source] += 1

        # Filter
        if token_count < SFT_CUTOFF_LEN:
            filtered_data.append(example)
            source_stats['filtered'][source] += 1
            token_counts.append(token_count)

    # Statistics
    print("\n📊 Filtering Results:")
    print(f"  Original: {len(data):,} examples")
    print(f"  Filtered: {len(filtered_data):,} examples")
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        print(f"  Average tokens: {avg_tokens:.0f}")
        print(f"  Max tokens: {max_tokens:.0f}")

    # Source breakdown
    print("\n📈 Source Breakdown:")
    print(f"{'Source':<12} {'Original':<10} {'Filtered':<10} {'Kept%':<8}")
    print("-" * 50)
    for source in source_stats['original']:
        orig_count = source_stats['original'][source]
        filt_count = source_stats['filtered'][source]
        pct_kept = filt_count / orig_count * 100 if orig_count > 0 else 0
        print(f"{source:<12} {orig_count:<10} {filt_count:<10} {pct_kept:<8.1f}")

    # Save filtered data
    print(f"\n💾 Saving filtered dataset to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("✓ FILTERING COMPLETE!")
    print("="*80)
    print(f"✓ Filtered dataset: {len(filtered_data):,} examples")
    print(f"✓ Ready for efficient training with cutoff_len={SFT_CUTOFF_LEN}")

    # Recommendations
    if len(filtered_data) < 10000:
        print("\n⚠️  WARNING: Filtered dataset is quite small!")
        print("   Consider: increasing cutoff_len or adding more short examples")
    elif len(filtered_data) > 40000:
        print("\n✅ Good: Plenty of short examples available")
        print("   Training will be efficient with good data utilization")
    else:
        print("\n✅ Balanced: Good dataset size for training")

    return filtered_data

def filter_sft_test_by_tokens():
    """Filter SFT test set by token length for consistent evaluation"""
    print("="*80)
    print("FILTER SFT TEST SET BY TOKEN LENGTH")
    print("="*80)
    print(f"Cutoff length: {SFT_CUTOFF_LEN} tokens (matches training)")
    print(f"Model: {MODEL_PATH}")

    input_file = "hf_data/tibetan_datasets/tibetan_sft_test.json"
    output_file = "hf_data/tibetan_datasets/tibetan_sft_test_filtered_4k.json"

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Load tokenizer
    print("\n🔄 Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✓ Tokenizer loaded")

    # Load test data
    print("\n🔄 Loading SFT test dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data):,} test examples")

    # Analyze token lengths and filter
    print("\n🔄 Analyzing token lengths and filtering...")
    token_counts = []
    filtered_data = []
    source_stats = {'original': {}, 'filtered': {}}

    for example in tqdm(data, desc="Processing test examples"):
        # Count tokens
        formatted_text = format_example(example)
        token_count = len(tokenizer.encode(formatted_text))

        # Track source statistics
        source = example.get('source', 'unknown')
        if source not in source_stats['original']:
            source_stats['original'][source] = 0
            source_stats['filtered'][source] = 0
        source_stats['original'][source] += 1

        # Filter to match training cutoff_len
        if token_count <= SFT_CUTOFF_LEN:  # Note: <= instead of < for test set
            filtered_data.append(example)
            source_stats['filtered'][source] += 1
            token_counts.append(token_count)

    # Statistics
    print("\n📊 Test Set Filtering Results:")
    print(f"  Original: {len(data):,} examples")
    print(f"  Filtered: {len(filtered_data):,} examples")
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        print(f"  Average tokens: {avg_tokens:.0f}")
        print(f"  Max tokens: {max_tokens:.0f}")

    # Source breakdown
    print("\n📈 Test Source Breakdown:")
    print(f"{'Source':<12} {'Original':<10} {'Filtered':<10} {'Kept%':<8}")
    print("-" * 60)
    for source in source_stats['original']:
        orig_count = source_stats['original'][source]
        filt_count = source_stats['filtered'][source]
        pct_kept = filt_count / orig_count * 100 if orig_count > 0 else 0
        print(f"{source:<12} {orig_count:<10} {filt_count:<10} {pct_kept:<8.1f}")

    # Save filtered test data
    print(f"\n💾 Saving filtered test dataset to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("✓ TEST SET FILTERING COMPLETE!")
    print("="*80)
    print(f"✓ Filtered test dataset: {len(filtered_data):,} examples")
    print(f"✓ Consistent with training cutoff_len={SFT_CUTOFF_LEN}")
    print("✓ Ready for fair model evaluation")

    # Recommendations
    if len(filtered_data) < 500:
        print("\n⚠️  WARNING: Filtered test set is quite small!")
        print("   Consider: using a larger test split or reducing filtering threshold")
    elif len(filtered_data) > 2000:
        print("\n✅ Good: Sufficient test examples for reliable evaluation")
        print("   Will provide stable metrics and good statistical significance")
    else:
        print("\n✅ Balanced: Good test set size for evaluation")

    # Critical reminder
    print("\n🎯 IMPORTANT:")
    print("   Test set now matches training data distribution")
    print("   Evaluation will be fair and representative of training")

    return filtered_data

def filter_cpt_test_by_tokens():
    """Filter CPT test set by token length for consistent evaluation"""
    print("="*80)
    print("FILTER CPT TEST SET BY TOKEN LENGTH")
    print("="*80)
    print(f"Cutoff length: {CPT_CUTOFF_LEN} tokens (matches CPT training)")
    print(f"Model: {MODEL_PATH}")

    input_file = "hf_data/tibetan_datasets/tibetan_cpt_test.json"
    output_file = "hf_data/tibetan_datasets/tibetan_cpt_test_filtered_4k.json"

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Load tokenizer
    print("\n🔄 Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✓ Tokenizer loaded")

    # Load test data
    print("\n🔄 Loading CPT test dataset...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data):,} test examples")

    # Analyze token lengths and filter
    print("\n🔄 Analyzing token lengths and filtering...")
    token_counts = []
    filtered_data = []

    for example in tqdm(data, desc="Processing CPT test examples"):
        # For CPT, examples have "text" field
        text = example.get('text', '').strip()

        # Count tokens
        token_count = len(tokenizer.encode(text))

        # Filter to match training cutoff_len
        if token_count <= CPT_CUTOFF_LEN:  # Note: <= instead of < for test set
            filtered_data.append(example)
            token_counts.append(token_count)

    # Statistics
    print("\n📊 CPT Test Set Filtering Results:")
    print(f"  Original: {len(data):,} examples")
    print(f"  Filtered: {len(filtered_data):,} examples")
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        print(f"  Average tokens: {avg_tokens:.0f}")
        print(f"  Max tokens: {max_tokens:.0f}")

    # Save filtered test data
    print(f"\n💾 Saving filtered CPT test dataset to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("✓ CPT TEST SET FILTERING COMPLETE!")
    print("="*80)
    print(f"✓ Filtered test dataset: {len(filtered_data):,} examples")
    print(f"✓ Consistent with CPT training cutoff_len={CPT_CUTOFF_LEN}")

    # Recommendations
    if len(filtered_data) < 200:
        print("\n⚠️  WARNING: Filtered test set is quite small!")
        print("   Consider: using a larger test split or reducing filtering threshold")
    elif len(filtered_data) > 1000:
        print("\n✅ Good: Sufficient test examples for reliable perplexity evaluation")
        print("   Will provide stable metrics for language adaptation assessment")
    else:
        print("\n✅ Balanced: Good test set size for evaluation")

    # Important note for CPT
    print("\n🎯 IMPORTANT FOR CPT:")
    print("   Longer sequences are beneficial for language adaptation evaluation")
    print("   But consistency with training cutoff_len ensures fair assessment")
    print("   Consider this filtering optional - CPT is more tolerant of long sequences than SFT")

    return filtered_data

def main():
    print("="*80)
    print("TIBETAN FINETUNING - FULL-PARAMETER STRATEGY")
    print("="*80)
    
    print("\n🎯 Strategy for Full-Parameter Finetuning:")
    print("  CPT: 100% Tibetan (language adaptation)")
    print("  SFT: 80% Tibetan + 20% Chinese (prevent catastrophic forgetting)")
    
    print("\n📚 Data Sources:")
    print("  1. CUTE parallel: ~934K triplets")
    print("  2. CUTE non-parallel: ~990K Tibetan")
    print("  3. tibetan-mix Tibetan: 30K instructions")
    print("  4. tibetan-mix Chinese: 18K instructions (for anchoring)")
    print("  5. tibetan-mix CN→BO: 12.8K translation pairs")
    
    # Preprocess
    cpt_data = preprocess_for_cpt()  # returns train split
    sft_data = preprocess_for_sft()  # returns train split

    # Filter datasets by token length for efficient training and consistent evaluation
    print("\n" + "="*80)
    print("🔧 FILTERING DATASETS BY TOKEN LENGTH")
    print("="*80)
    print("This ensures:")
    print("- Training efficiency (no truncation)")
    print("- Memory usage optimization")
    print("- Consistent train/test distribution")
    print("- Fair evaluation metrics")
    print("="*80)

    # Filter SFT training data
    sft_filtered = filter_sft_by_tokens()

    # Filter SFT test data
    sft_test_filtered = filter_sft_test_by_tokens()

    # Filter CPT test data
    cpt_test_filtered = filter_cpt_test_by_tokens()

    # Update file list for summary
    print("\n📁 Generated Files (including filtered):")
    print("  1. tibetan_cpt_final.json (train)")
    print(f"     - {len(cpt_data):,} Tibetan texts (100%)")
    print("     - tibetan_cpt_test.json (held-out test)")
    print("     - tibetan_cpt_test_filtered_4k.json (filtered test)")
    print("  2. tibetan_sft_final.json (train)")
    print(f"     - {len(sft_data):,} total examples")
    print("     - tibetan_sft_test.json (held-out test)")
    print("     - tibetan_sft_test_filtered_4k.json (filtered test)")
    print("     - tibetan_sft_filtered_4k.json (filtered train)")

    # Summary
    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*80)

    print(f"\n⏱️  Estimated Training Time (8xH200):")
    cpt_hours = len(cpt_data) / 30000
    sft_hours = len(sft_filtered) / 15000  # Use filtered data for time estimation
    print(f"  CPT: ~{cpt_hours:.1f}-{cpt_hours*1.5:.1f} hours")
    print(f"  SFT: ~{sft_hours:.1f}-{sft_hours*1.5:.1f} hours (filtered data)")
    print(f"  Total: ~{cpt_hours+sft_hours:.1f}-{(cpt_hours+sft_hours)*1.5:.1f} hours")
    
    print("\n🎯 Why This Strategy?")
    print("  ✅ Full-parameter finetuning updates ALL weights")
    print("  ✅ 20% Chinese data prevents catastrophic forgetting")
    print("  ✅ Maintains multilingual capability")
    print("  ✅ Better stability and convergence")
    print("  ✅ More practical for real-world use")
    
    print("\n✓ Ready to train with optimal full-parameter strategy!")

if __name__ == "__main__":
    main()

