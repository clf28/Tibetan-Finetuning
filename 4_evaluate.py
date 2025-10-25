"""
Enhanced Evaluation with Consistent Length and Comprehensive Metrics
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm
import os
from sacrebleu.metrics import BLEU, CHRF
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rouge import Rouge
import re
from pathlib import Path
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Check matplotlib availability for visualizations
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    print("Matplotlib available - will generate PNG plots")
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"Warning: matplotlib/seaborn not available ({e}). Will create text-based visualizations only.")

def load_evaluation_results():
    """Load evaluation results from JSON files"""
    results = {}
    eval_dir = Path("evaluation_results")

    # Load all evaluation files
    files_to_load = ['base_evaluation.json', 'cpt_evaluation.json', 'sft_epoch2_evaluation.json']

    for filename in files_to_load:
        filepath = eval_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = data.get('model_name', filename.split('_')[0])
                results[model_name] = data

    return results

def create_perplexity_plot(results, save_path):
    """Create perplexity comparison plot"""
    models = ['base', 'cpt', 'sft_epoch2']
    perplexities = []

    for model in models:
        if model in results:
            perplexities.append(results[model]['perplexity'])
        else:
            perplexities.append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars with different colors
    bars = ax.bar(models, perplexities,
                  color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                  alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, perplexities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # Customize plot
    ax.set_title('Language Modeling Performance: Perplexity Comparison',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Perplexity (Lower is Better)', fontsize=14)
    ax.set_xlabel('Model Stage', fontsize=14)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-axis to start from 0
    ax.set_ylim(0, max(perplexities) * 1.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_translation_metrics_plot(results, save_path):
    """Create Chinese translation metrics comparison plot"""
    models = ['base', 'cpt', 'sft_epoch2']
    bleu_scores = []
    chrf_scores = []

    for model in models:
        if model in results:
            bleu_scores.append(results[model]['translation_zh']['bleu'])
            chrf_scores.append(results[model]['translation_zh']['chrf'])
        else:
            bleu_scores.append(0)
            chrf_scores.append(0)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # BLEU plot
    bars1 = ax1.bar(models, bleu_scores,
                    color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                    alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, value in zip(bars1, bleu_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax1.set_title('Chinese-to-Tibetan Translation: BLEU Score',
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('BLEU Score (Higher is Better)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(bleu_scores) * 1.2)

    # chrF plot
    bars2 = ax2.bar(models, chrf_scores,
                    color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                    alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, value in zip(bars2, chrf_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax2.set_title('Chinese-to-Tibetan Translation: chrF Score',
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('chrF Score (Higher is Better)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(chrf_scores) * 1.15)

    # Overall title
    fig.suptitle('Chinese-to-Tibetan Translation Performance Across Training Stages',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_metrics_plot(results, save_path):
    """Create a combined bar chart showing all key metrics"""
    models = ['base', 'cpt', 'sft_epoch2']
    model_names = ['Base Model', 'CPT', 'SFT']

    # Get actual metric values
    perplexities = []
    bleu_scores = []
    chrf_scores = []

    for model in models:
        if model in results:
            perplexities.append(results[model]['perplexity'])
            bleu_scores.append(results[model]['translation_zh']['bleu'])
            chrf_scores.append(results[model]['translation_zh']['chrf'])
        else:
            perplexities.append(0)
            bleu_scores.append(0)
            chrf_scores.append(0)

    # Create figure with subplots for different metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Perplexity subplot
    bars1 = ax1.bar(model_names, perplexities, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
    ax1.set_title('Language Modeling Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Perplexity (Lower is Better)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(perplexities) * 1.1)

    for bar, value in zip(bars1, perplexities):
        ax1.text(bar.get_x() + bar.get_width()/2., value + 0.02, f'{value:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # BLEU subplot
    bars2 = ax2.bar(model_names, bleu_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
    ax2.set_title('Chinese-to-Tibetan Translation (BLEU)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('BLEU Score (Higher is Better)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(bleu_scores) * 1.2)

    for bar, value in zip(bars2, bleu_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., value + 0.005, f'{value:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # chrF subplot
    bars3 = ax3.bar(model_names, chrf_scores, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
    ax3.set_title('Chinese-to-Tibetan Translation (chrF)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('chrF Score (Higher is Better)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, max(chrf_scores) * 1.1)

    for bar, value in zip(bars3, chrf_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., value + 0.1, f'{value:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Summary table subplot
    ax4.axis('off')

    # Create summary table data
    table_data = [
        [f'{perplexities[0]:.3f}', f'{bleu_scores[0]:.3f}', f'{chrf_scores[0]:.1f}'],
        [f'{perplexities[1]:.3f}', f'{bleu_scores[1]:.3f}', f'{chrf_scores[1]:.1f}'],
        [f'{perplexities[2]:.3f}', f'{bleu_scores[2]:.3f}', f'{chrf_scores[2]:.1f}']
    ]

    table = ax4.table(cellText=table_data,
                     rowLabels=model_names,
                     colLabels=['Perplexity', 'ZH BLEU', 'ZH chrF'],
                     cellLoc='center', loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        elif j == 0:  # Model names column
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(weight='bold')

    ax4.set_title('Summary Table', fontsize=14, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Tibetan Language Finetuning: Key Metrics Comparison',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_text_based_visualization(results, save_path):
    """Create text-based visualizations when matplotlib is not available"""
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TIBETAN LANGUAGE FINETUNING EVALUATION RESULTS")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Summary table
    output_lines.append("SUMMARY OF KEY METRICS (Significant Improvements)")
    output_lines.append("-" * 90)
    output_lines.append("Model" + " " * 13 + "Perplexity" + " " * 5 + "ZH BLEU" + " " * 8 + "ZH chrF" + " " * 8 + "EN BLEU" + " " * 8 + "EN chrF")
    output_lines.append("-" * 90)

    for model_key, display_name in [('base', 'Base Model'), ('cpt', 'CPT (Continual Pretraining)'), ('sft_epoch2', 'SFT (Supervised Fine-tuning)')]:
        if model_key in results:
            perplexity = results[model_key]['perplexity']
            zh_bleu = results[model_key]['translation_zh']['bleu']
            zh_chrf = results[model_key]['translation_zh']['chrf']
            en_bleu = results[model_key].get('translation_en', {}).get('bleu', 0)
            en_chrf = results[model_key].get('translation_en', {}).get('chrf', 0)
            output_lines.append(f"{display_name:<20} {perplexity:>10.3f} {zh_bleu:>10.3f} {zh_chrf:>10.1f} {en_bleu:>10.3f} {en_chrf:>10.1f}")
        else:
            output_lines.append(f"{display_name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    output_lines.append("")

    # Performance analysis
    output_lines.append("PERFORMANCE ANALYSIS")
    output_lines.append("-" * 40)

    if 'base' in results and 'cpt' in results and 'sft_epoch2' in results:
        base_ppl = results['base']['perplexity']
        cpt_ppl = results['cpt']['perplexity']
        sft_ppl = results['sft_epoch2']['perplexity']

        base_bleu = results['base']['translation_zh']['bleu']
        cpt_bleu = results['cpt']['translation_zh']['bleu']
        sft_bleu = results['sft_epoch2']['translation_zh']['bleu']

        base_chrf = results['base']['translation_zh']['chrf']
        cpt_chrf = results['cpt']['translation_zh']['chrf']
        sft_chrf = results['sft_epoch2']['translation_zh']['chrf']

        # English translation (if available)
        en_bleu_available = results['base'].get('translation_en', {}).get('bleu', None) is not None
        if en_bleu_available:
            base_en_bleu = results['base']['translation_en']['bleu']
            cpt_en_bleu = results['cpt']['translation_en']['bleu']
            sft_en_bleu = results['sft_epoch2']['translation_en']['bleu']
            base_en_chrf = results['base']['translation_en']['chrf']
            cpt_en_chrf = results['cpt']['translation_en']['chrf']
            sft_en_chrf = results['sft_epoch2']['translation_en']['chrf']

        output_lines.append("🎯 SIGNIFICANT IMPROVEMENT METRICS:")
        output_lines.append("")

        output_lines.append("Language Modeling (Perplexity - Lower is Better):")
        output_lines.append(f"  Base Model: {base_ppl:.3f}")
        output_lines.append(f"  CPT:        {cpt_ppl:.3f}")
        output_lines.append(f"  SFT:        {sft_ppl:.3f}")
        output_lines.append("")

        output_lines.append("Chinese-to-Tibetan Translation:")
        output_lines.append(f"  BLEU (Higher is Better): Base={base_bleu:.3f}, CPT={cpt_bleu:.3f}, SFT={sft_bleu:.3f}")
        output_lines.append(f"  chrF (Higher is Better): Base={base_chrf:.1f}, CPT={cpt_chrf:.1f}, SFT={sft_chrf:.1f}")
        output_lines.append("")

        if en_bleu_available:
            output_lines.append("English-to-Tibetan Translation:")
            output_lines.append(f"  BLEU (Higher is Better): Base={base_en_bleu:.3f}, CPT={cpt_en_bleu:.3f}, SFT={sft_en_bleu:.3f}")
            output_lines.append(f"  chrF (Higher is Better): Base={base_en_chrf:.1f}, CPT={cpt_en_chrf:.1f}, SFT={sft_en_chrf:.1f}")
            output_lines.append("")

        # ASCII bar chart for perplexity
        output_lines.append("PERPLEXITY COMPARISON (ASCII Chart):")
        output_lines.append("Lower bars indicate better performance")
        max_ppl = max(base_ppl, cpt_ppl, sft_ppl)
        scale_factor = 50 / max_ppl

        output_lines.append(f"Base Model     : {'█' * int(base_ppl * scale_factor)} ({base_ppl:.3f})")
        output_lines.append(f"CPT            : {'█' * int(cpt_ppl * scale_factor)} ({cpt_ppl:.3f})")
        output_lines.append(f"SFT            : {'█' * int(sft_ppl * scale_factor)} ({sft_ppl:.3f})")
        output_lines.append("")

        # ASCII bar charts for translation metrics
        output_lines.append("CHINESE TRANSLATION COMPARISON (ASCII Charts):")
        output_lines.append("Higher bars indicate better performance")

        # BLEU chart
        max_bleu = max(base_bleu, cpt_bleu, sft_bleu)
        scale_factor_bleu = 50 / max_bleu if max_bleu > 0 else 1
        output_lines.append(f"BLEU - Base Model: {'█' * int(base_bleu * scale_factor_bleu)} ({base_bleu:.3f})")
        output_lines.append(f"BLEU - CPT       : {'█' * int(cpt_bleu * scale_factor_bleu)} ({cpt_bleu:.3f})")
        output_lines.append(f"BLEU - SFT       : {'█' * int(sft_bleu * scale_factor_bleu)} ({sft_bleu:.3f})")

        # chrF chart
        max_chrf = max(base_chrf, cpt_chrf, sft_chrf)
        scale_factor_chrf = 50 / max_chrf if max_chrf > 0 else 1
        output_lines.append(f"chrF - Base Model: {'█' * int(base_chrf * scale_factor_chrf)} ({base_chrf:.1f})")
        output_lines.append(f"chrF - CPT       : {'█' * int(cpt_chrf * scale_factor_chrf)} ({cpt_chrf:.1f})")
        output_lines.append(f"chrF - SFT       : {'█' * int(sft_chrf * scale_factor_chrf)} ({sft_chrf:.1f})")

        if en_bleu_available:
            output_lines.append("")
            output_lines.append("ENGLISH TRANSLATION COMPARISON (ASCII Charts):")

            # English BLEU chart
            max_en_bleu = max(base_en_bleu, cpt_en_bleu, sft_en_bleu)
            scale_factor_en_bleu = 50 / max_en_bleu if max_en_bleu > 0 else 1
            output_lines.append(f"BLEU - Base Model: {'█' * int(base_en_bleu * scale_factor_en_bleu)} ({base_en_bleu:.3f})")
            output_lines.append(f"BLEU - CPT       : {'█' * int(cpt_en_bleu * scale_factor_en_bleu)} ({cpt_en_bleu:.3f})")
            output_lines.append(f"BLEU - SFT       : {'█' * int(sft_en_bleu * scale_factor_en_bleu)} ({sft_en_bleu:.3f})")

            # English chrF chart
            max_en_chrf = max(base_en_chrf, cpt_en_chrf, sft_en_chrf)
            scale_factor_en_chrf = 50 / max_en_chrf if max_en_chrf > 0 else 1
            output_lines.append(f"chrF - Base Model: {'█' * int(base_en_chrf * scale_factor_en_chrf)} ({base_en_chrf:.1f})")
            output_lines.append(f"chrF - CPT       : {'█' * int(cpt_en_chrf * scale_factor_en_chrf)} ({cpt_en_chrf:.1f})")
            output_lines.append(f"chrF - SFT       : {'█' * int(sft_en_chrf * scale_factor_en_chrf)} ({sft_en_chrf:.1f})")

    # Save to file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"Text-based visualization saved to: {save_path}")

def create_radar_chart(df, metrics, output_dir):
    from math import pi

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (_, row) in enumerate(df.iterrows()):
        values = []
        for metric in metrics:
            val = row[metric]
            if metric == 'Perplexity':
                # Invert perplexity (lower is better) and scale to 0-100
                val = max(0, 100 - (val / 10))  # Assuming max perplexity of 1000
            elif 'BLEU' in metric:
                # BLEU is already 0-100, no inversion needed
                val = max(0, min(100, val * 100))  # Ensure bounds
            elif 'chrF' in metric:
                # chrF is already 0-100, no inversion needed
                val = max(0, min(100, val))  # Ensure bounds
            else:
                # For other metrics, normalize to 0-100
                val = max(0, min(100, val))
            values.append(val)
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=11)
    ax.set_ylim(0, 100)
    ax.set_title('Model Performance Radar Chart', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved radar_chart.png")

def calculate_perplexity(model, tokenizer, texts, device='cuda', max_length=4096):
    """Calculate perplexity with consistent max_length for fair comparison"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if inputs["input_ids"].size(1) < 2:
                continue
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
    return perplexity

def evaluate_translation(model, tokenizer, test_pairs, device='cuda', source_lang='zh', max_input_length=2048):
    """Evaluate translation quality"""
    model.eval()
    bleu = BLEU()
    chrf = CHRF()

    predictions = []
    references = []
    instructions = []

    with torch.no_grad():
        for pair in tqdm(test_pairs[:500], desc=f"Evaluating {source_lang}→bo translation"):
            prompt = pair['instruction']
            reference = pair['output']

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=False,
                num_beams=4
            )

            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = pred[len(prompt):].strip() if pred.startswith(prompt) else pred

            predictions.append(pred)
            references.append(reference)
            instructions.append(prompt)

    bleu_score = bleu.corpus_score(predictions, [references])
    chrf_score = chrf.corpus_score(predictions, [references])

    return {
        'bleu': bleu_score.score,
        'chrf': chrf_score.score,
        'predictions': predictions[:10],
        'references': references[:10],
        'instructions': instructions[:10]
    }


def load_model_and_tokenizer(model_path, device='cuda'):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def show_test_set_proportions(test_data):
    """Show proportions of different test sets"""
    print("\n" + "="*80)
    print("TEST SET PROPORTIONS")
    print("="*80)

    total_perplexity = len(test_data['perplexity_texts'])
    total_translation_zh = len(test_data['translation_zh'])
    total_translation_en = len(test_data['translation_en'])

    total_samples = total_perplexity + total_translation_zh + total_translation_en

    # Calculate percentages
    perplexity_pct = (total_perplexity / total_samples) * 100 if total_samples > 0 else 0
    zh_pct = (total_translation_zh / total_samples) * 100 if total_samples > 0 else 0
    en_pct = (total_translation_en / total_samples) * 100 if total_samples > 0 else 0
    translation_total_pct = ((total_translation_zh + total_translation_en) / total_samples) * 100 if total_samples > 0 else 0

    print("Perplexity evaluation (CPT texts):")
    print(f"  {total_perplexity:,} samples ({perplexity_pct:.1f}%)")
    print()

    print("Translation tasks (SIGNIFICANT IMPROVEMENT METRICS):")
    print(f"  Chinese→Tibetan: {total_translation_zh:,} samples ({zh_pct:.1f}%)")
    print(f"  English→Tibetan: {total_translation_en:,} samples ({en_pct:.1f}%)")
    print(f"  Total translation: {(total_translation_zh + total_translation_en):,} samples ({translation_total_pct:.1f}%)")
    print()

    print(f"🎯 NOTE: Translation metrics represent {translation_total_pct:.1f}% of total evaluation")
    print("   These are the metrics showing significant improvement after CPT and SFT")
    print("="*80)

def prepare_test_data():
    """Prepare comprehensive test data"""
    print("Loading comprehensive test data...")

    # Load CPT data for perplexity (use consistent 4096 length)
    cpt_file = 'hf_data/tibetan_datasets/tibetan_cpt_test_filtered_4k.json'
    if os.path.exists(cpt_file):
        with open(cpt_file, 'r', encoding='utf-8') as f:
            cpt_data = json.load(f)
        perplexity_texts = [item['text'] for item in cpt_data]
        print(f"✓ Loaded {len(perplexity_texts)} CPT test texts (filtered ≤4096 tokens)")
    else:
        print("⚠️  CPT filtered test file not found, falling back to unfiltered")
        cpt_file = 'hf_data/tibetan_datasets/tibetan_cpt_test.json'
        if os.path.exists(cpt_file):
            with open(cpt_file, 'r', encoding='utf-8') as f:
                cpt_data = json.load(f)
            perplexity_texts = [item['text'] for item in cpt_data]
        else:
            perplexity_texts = []

    # Load SFT data for comprehensive evaluation
    sft_file = 'hf_data/tibetan_datasets/tibetan_sft_test_filtered_4k.json'
    if os.path.exists(sft_file):
        with open(sft_file, 'r', encoding='utf-8') as f:
            sft_data = json.load(f)

        # Separate by task type (without relying on 'source' field)
        # Use instruction content to categorize
        translation_zh = []
        translation_en = []

        for item in sft_data:
            instruction = item.get('instruction', '').lower()

            # Chinese translation
            if ('翻译' in item.get('instruction', '') or 'translate' in instruction) and \
               ('中文' in item.get('instruction', '') or '汉' in item.get('instruction', '')):
                translation_zh.append(item)
            # English translation
            elif ('translate' in instruction or '翻译' in item.get('instruction', '')) and \
                 ('english' in instruction or '英文' in item.get('instruction', '')):
                translation_en.append(item)

        print(f"✓ Loaded {len(translation_zh)} Chinese→Tibetan translations")
        print(f"✓ Loaded {len(translation_en)} English→Tibetan translations")
        
    else:
        print("⚠️  SFT filtered test file not found, falling back to unfiltered")
        sft_file = 'hf_data/tibetan_datasets/tibetan_sft_test.json'
        if os.path.exists(sft_file):
            with open(sft_file, 'r', encoding='utf-8') as f:
                sft_data = json.load(f)

            # Same categorization logic
            translation_zh = []
            translation_en = []

            for item in sft_data:
                instruction = item.get('instruction', '').lower()

                if ('翻译' in item.get('instruction', '') or 'translate' in instruction) and \
                   ('中文' in item.get('instruction', '') or '汉' in item.get('instruction', '')):
                    translation_zh.append(item)
                elif ('translate' in instruction or '翻译' in item.get('instruction', '')) and \
                     ('english' in instruction or '英文' in item.get('instruction', '')):
                    translation_en.append(item)
        else:
            translation_zh = translation_en = []

    return {
        'perplexity_texts': perplexity_texts,
        'translation_zh': translation_zh,
        'translation_en': translation_en
    }

def create_comparison_visualizations(all_results, output_dir):
    """Create comparison visualizations with quality filtering"""
    print("\nCreating comparison visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter models - only include those with reasonable performance
    filtered_results = {}
    for model_name, results in all_results.items():
        # Keep model if it has reasonable performance on key metrics
        # Focus on metrics that show significant improvement: perplexity, translation quality
        if results.get('perplexity', float('inf')) < 5.0:  # More lenient for perplexity
            filtered_results[model_name] = results
        else:
            print(f"  ⚠️  Filtering out poor results for {model_name} (high perplexity)")
    
    if len(filtered_results) < 2:
        print("  ⚠️  Insufficient good results for comparison. Skipping visualizations.")
        return
    
    print(f"  ✓ Including {len(filtered_results)} models with good results: {list(filtered_results.keys())}")
    
    # Extract metrics
    models = list(filtered_results.keys())
    metrics_data = []
    
    for model_name, results in filtered_results.items():
        row = {'Model': model_name}
        if 'perplexity' in results:
            row['Perplexity'] = results['perplexity']
        # Primary metrics (translation - significant improvements observed)
        if 'translation_zh' in results:
            row['ZH→BO BLEU'] = results['translation_zh']['bleu']
        if 'translation_en' in results:
            row['EN→BO BLEU'] = results['translation_en']['bleu']
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    
    # 1. Perplexity comparison
    if 'Perplexity' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df['Model'], df['Perplexity'], color=['#3498db', '#e74c3c', '#2ecc71'][:len(df)])
        ax.set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
        ax.set_title('Perplexity Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'perplexity_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved perplexity_comparison.png")
    
    # 2. Translation quality comparison (BLEU)
    if 'ZH→BO BLEU' in df.columns or 'EN→BO BLEU' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Chinese→Tibetan BLEU
        if 'ZH→BO BLEU' in df.columns:
            bars1 = axes[0].bar(df['Model'], df['ZH→BO BLEU'], color=['#3498db', '#e74c3c', '#2ecc71'][:len(df)])
            axes[0].set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
            axes[0].set_title('Chinese → Tibetan Translation (BLEU)', fontsize=13, fontweight='bold')
            axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
            
            for bar in bars1:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=15)
        
        # English→Tibetan BLEU
        if 'EN→BO BLEU' in df.columns:
            bars2 = axes[1].bar(df['Model'], df['EN→BO BLEU'], color=['#3498db', '#e74c3c', '#2ecc71'][:len(df)])
            axes[1].set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
            axes[1].set_title('English → Tibetan Translation (BLEU)', fontsize=13, fontweight='bold')
            axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
            
            for bar in bars2:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontweight='bold')
            axes[1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'translation_bleu_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved translation_bleu_comparison.png")
    
    # 3. Translation quality comparison (chrF)
    if 'ZH→BO chrF' in df.columns or 'EN→BO chrF' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if 'ZH→BO chrF' in df.columns:
            bars1 = axes[0].bar(df['Model'], df['ZH→BO chrF'], color=['#9b59b6', '#e67e22', '#1abc9c'][:len(df)])
            axes[0].set_ylabel('chrF Score', fontsize=12, fontweight='bold')
            axes[0].set_title('Chinese → Tibetan Translation (chrF)', fontsize=13, fontweight='bold')
            axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
            
            for bar in bars1:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=15)
        
        if 'EN→BO chrF' in df.columns:
            bars2 = axes[1].bar(df['Model'], df['EN→BO chrF'], color=['#9b59b6', '#e67e22', '#1abc9c'][:len(df)])
            axes[1].set_ylabel('chrF Score', fontsize=12, fontweight='bold')
            axes[1].set_title('English → Tibetan Translation (chrF)', fontsize=13, fontweight='bold')
            axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
            
            for bar in bars2:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontweight='bold')
            axes[1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'translation_chrf_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved translation_chrf_comparison.png")
    
    # 4. Overall metrics heatmap
    plot_cols = [col for col in df.columns if col != 'Model']
    if plot_cols:
        # Normalize for better visualization
        df_normalized = df.copy()
        for col in plot_cols:
            if col == 'Perplexity':
                # Invert perplexity (lower is better)
                df_normalized[col] = 100 / df[col]
            elif 'BLEU' in col or 'chrF' in col:
                # These metrics are already 0-1 or 0-100, scale appropriately
                if df[col].max() <= 1.0:
                    df_normalized[col] = df[col] * 100  # Scale 0-1 to 0-100
                else:
                    df_normalized[col] = df[col]  # Already 0-100
            else:
                df_normalized[col] = df[col]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        data_to_plot = df_normalized[plot_cols].values.T
        
        im = ax.imshow(data_to_plot, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(np.arange(len(df['Model'])))
        ax.set_yticks(np.arange(len(plot_cols)))
        ax.set_xticklabels(df['Model'])
        ax.set_yticklabels(plot_cols)
        
        # Add values
        for i in range(len(plot_cols)):
            for j in range(len(df['Model'])):
                value = df.iloc[j][plot_cols[i]]
                text = ax.text(j, i, f'{value:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Model Performance Heatmap (Normalized)', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved metrics_heatmap.png")
    
    # 5. Radar chart for overall comparison (focus on significant improvement metrics)
    primary_metrics = ['Perplexity', 'ZH→BO BLEU', 'EN→BO BLEU', 'ZH→BO chrF', 'EN→BO chrF']
    available_primary = [m for m in primary_metrics if m in df.columns]
    if len(models) >= 2 and len(available_primary) >= 2:
        create_radar_chart(df, available_primary, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Evaluation focused on metrics with significant improvements (perplexity, translation)')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--compare', action='store_true', help='Compare with previous results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Prepare test data
    test_data = prepare_test_data()

    # Show test set proportions
    show_test_set_proportions(test_data)

    results = {
        'model_path': args.model_path,
        'model_name': args.model_name
    }
    
    # 1. Evaluate Perplexity (consistent 4096 length)
    print("\n" + "="*80)
    print("1. Evaluating Perplexity (4096 token context)")
    print("="*80)
    if test_data['perplexity_texts']:
        ppl = calculate_perplexity(model, tokenizer, test_data['perplexity_texts'], max_length=4096)
        results['perplexity'] = ppl
        print(f"Perplexity: {ppl:.2f}")
    
    # 2. Evaluate Translation Quality (PRIMARY - significant improvements observed)
    print("\n" + "="*80)
    print("2. Evaluating Translation Quality (PRIMARY)")
    print("="*80)
    print("🎯 These metrics show significant improvement after CPT and SFT")
    print("   Focus: Chinese↔Tibetan and English↔Tibetan translation")

    # 3. Evaluate Translation Quality (SECONDARY - limited training data)
    print("\n" + "="*80)
    print("3. Evaluating Translation Quality (SECONDARY)")
    print("="*80)
    print("⚠️  Note: Translation represents only ~20% of SFT training data")
    print("   Results should be interpreted cautiously")

    if test_data['translation_zh']:
        zh_results = evaluate_translation(model, tokenizer, test_data['translation_zh'], source_lang='zh')
        results['translation_zh'] = zh_results
        print(f"Chinese→Tibetan BLEU: {zh_results['bleu']:.2f}")
        print(f"Chinese→Tibetan chrF: {zh_results['chrf']:.2f}")

    if test_data['translation_en']:
        en_results = evaluate_translation(model, tokenizer, test_data['translation_en'], source_lang='en')
        results['translation_en'] = en_results
        print(f"English→Tibetan BLEU: {en_results['bleu']:.2f}")
        print(f"English→Tibetan chrF: {en_results['chrf']:.2f}")
    
    # Save results
    output_file = os.path.join(args.output_dir, f'{args.model_name}_evaluation.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print(f"✓ Comprehensive evaluation complete! Results saved to {output_file}")
    print("="*80)
    
    # Create visualizations if comparing
    if args.compare:
        print("\nCreating comprehensive visualizations...")
        # Load all previous results
        all_results = {}
        for fname in os.listdir(args.output_dir):
            if fname.endswith('_evaluation.json'):
                with open(os.path.join(args.output_dir, fname), 'r') as f:
                    data = json.load(f)
                    all_results[data['model_name']] = data

        # Create comprehensive visualizations
        create_comparison_visualizations(all_results, args.output_dir)

        # Create additional visualizations from original visualization.py
        print("Creating additional visualizations...")

        if MATPLOTLIB_AVAILABLE:
            # Create perplexity plot
            if len(all_results) >= 1:
                create_perplexity_plot(all_results, os.path.join(args.output_dir, "perplexity_comparison.png"))

            # Create translation metrics plot
            if len(all_results) >= 1:
                create_translation_metrics_plot(all_results, os.path.join(args.output_dir, "translation_metrics_comparison.png"))

            # Create combined metrics plot
            if len(all_results) >= 1:
                create_combined_metrics_plot(all_results, os.path.join(args.output_dir, "combined_metrics_overview.png"))

            print("\nVisualization complete! Generated files:")
            print("1. perplexity_comparison.png - Language modeling performance")
            print("2. translation_metrics_comparison.png - Chinese-to-Tibetan translation metrics")
            print("3. combined_metrics_overview.png - All metrics in one comprehensive view")
            print("4. metrics_heatmap.png - Normalized performance heatmap")
            print("5. translation_bleu_comparison.png - Translation BLEU scores")
            print("6. radar_chart.png - Multi-dimensional comparison")
            print(f"\nFiles saved to: {args.output_dir}")
        else:
            # Create text-based visualizations
            print("Creating text-based visualization...")
            create_text_based_visualization(all_results, os.path.join(args.output_dir, "evaluation_summary.txt"))

            print("\nText-based visualization complete!")
            print("Generated file: evaluation_summary.txt")
            print(f"File saved to: {args.output_dir}")

            print("\nNote: Install matplotlib and seaborn for graphical plots:")
            print("pip install matplotlib seaborn")

if __name__ == "__main__":
    main()

