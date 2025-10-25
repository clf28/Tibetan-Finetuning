#!/usr/bin/env python3
"""
Layer-wise Weight Change Analysis for Tibetan Language Finetuning

This script analyzes how different layers change during CPT and SFT stages
to understand the model's adaptation mechanisms for Tibetan language learning.

Key Analysis:
1. Layer-wise magnitude of changes (L2 norm)
2. Correlation between CPT and SFT changes
3. Top changed layers for each stage
4. Layer type analysis (attention vs MLP vs embeddings)
5. Change distribution analysis

Usage:
    python 5_analyze_weights.py --original qwen2-5-3b-base --cpt saves/qwen2.5-3b/full/cpt --sft saves/qwen2.5-3b/full/sft
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_model_weights(model_path, model_name="model"):
    """Load model state dict"""
    print(f"Loading {model_name} from {model_path}...")

    # Handle both directory and checkpoint paths
    if os.path.isdir(model_path):
        # Try to find the latest checkpoint or final model
        checkpoints = [f for f in os.listdir(model_path) if f.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            model_path = os.path.join(model_path, latest_checkpoint)
            print(f"  Using latest checkpoint: {latest_checkpoint}")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        state_dict = model.state_dict()
        print(f"  ✓ Loaded {len(state_dict)} parameters")
        return state_dict
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return None

def compute_layer_changes(original_weights, new_weights, layer_name):
    """Compute L2 norm of weight changes for a layer"""
    if layer_name not in original_weights or layer_name not in new_weights:
        return None

    orig_weight = original_weights[layer_name]
    new_weight = new_weights[layer_name]

    if orig_weight.shape != new_weight.shape:
        print(f"  ⚠️  Shape mismatch for {layer_name}: {orig_weight.shape} vs {new_weight.shape}")
        return None

    # Compute L2 norm of the difference
    diff = new_weight - orig_weight
    change_magnitude = torch.norm(diff).item()

    # Compute relative change
    orig_magnitude = torch.norm(orig_weight).item()
    relative_change = change_magnitude / (orig_magnitude + 1e-8)

    return {
        'magnitude': change_magnitude,
        'relative': relative_change,
        'shape': list(orig_weight.shape),
        'param_count': orig_weight.numel()
    }

def analyze_layer_changes(original_weights, cpt_weights, sft_weights):
    """Analyze changes in all layers"""
    print("\n" + "="*80)
    print("LAYER-WISE WEIGHT CHANGE ANALYSIS")
    print("="*80)

    layer_changes = {
        'cpt': {},
        'sft': {},
        'cpt_to_sft': {}  # Changes from CPT to SFT
    }

    # Get all unique layer names
    all_layers = set(original_weights.keys())
    all_layers.update(cpt_weights.keys() if cpt_weights else set())
    all_layers.update(sft_weights.keys() if sft_weights else set())

    print(f"Analyzing {len(all_layers)} layers...")

    # Analyze CPT changes
    if cpt_weights:
        print("\n📊 CPT Changes (Base → CPT):")
        print(f"{'Layer':<50} {'Magnitude':<12} {'Relative%':<10} {'Shape':<15}")
        print("-" * 90)

        for layer_name in tqdm(sorted(all_layers), desc="Analyzing CPT changes"):
            change = compute_layer_changes(original_weights, cpt_weights, layer_name)
            if change:
                layer_changes['cpt'][layer_name] = change
                mag_str = f"{change['magnitude']:.2e}"
                rel_str = f"{change['relative']*100:.2f}%"
                shape_str = "×".join(map(str, change['shape']))
                print(f"{layer_name:<50} {mag_str:<12} {rel_str:<10} {shape_str:<15}")

    # Analyze SFT changes
    if sft_weights:
        print("\n📊 SFT Changes (Base → SFT):")
        print(f"{'Layer':<50} {'Magnitude':<12} {'Relative%':<10} {'Shape':<15}")
        print("-" * 90)

        for layer_name in tqdm(sorted(all_layers), desc="Analyzing SFT changes"):
            change = compute_layer_changes(original_weights, sft_weights, layer_name)
            if change:
                layer_changes['sft'][layer_name] = change
                mag_str = f"{change['magnitude']:.2e}"
                rel_str = f"{change['relative']*100:.2f}%"
                shape_str = "×".join(map(str, change['shape']))
                print(f"{layer_name:<50} {mag_str:<12} {rel_str:<10} {shape_str:<15}")

    # Analyze CPT to SFT changes
    if cpt_weights and sft_weights:
        print("\n📊 CPT→SFT Changes (CPT → SFT):")
        print(f"{'Layer':<50} {'Magnitude':<12} {'Relative%':<10} {'Shape':<15}")
        print("-" * 90)

        for layer_name in tqdm(sorted(all_layers), desc="Analyzing CPT→SFT changes"):
            change = compute_layer_changes(cpt_weights, sft_weights, layer_name)
            if change:
                layer_changes['cpt_to_sft'][layer_name] = change
                mag_str = f"{change['magnitude']:.2e}"
                rel_str = f"{change['relative']*100:.2f}%"
                shape_str = "×".join(map(str, change['shape']))
                print(f"{layer_name:<50} {mag_str:<12} {rel_str:<10} {shape_str:<15}")

    return layer_changes

def categorize_layers(layer_name):
    """Categorize layers by type"""
    name = layer_name.lower()

    if 'embed' in name:
        return 'embedding'
    elif 'attention' in name or 'attn' in name:
        return 'attention'
    elif 'mlp' in name or 'feed_forward' in name:
        return 'mlp'
    elif 'layernorm' in name or 'ln' in name:
        return 'layernorm'
    elif 'lm_head' in name or 'head' in name:
        return 'output'
    else:
        return 'other'

def create_layer_magnitude_plot(changes_dict, title, save_path, top_k=20):
    """Create layer magnitude comparison plot"""
    if not changes_dict:
        print(f"  ⚠️  No data available for {title}")
        return

    # Prepare data
    layers = []
    magnitudes = []
    categories = []
    relative_changes = []

    for layer_name, change in changes_dict.items():
        layers.append(layer_name)
        magnitudes.append(change['magnitude'])
        categories.append(categorize_layers(layer_name))
        relative_changes.append(change['relative'] * 100)

    # Sort by magnitude
    sorted_indices = np.argsort(magnitudes)[::-1]
    layers = [layers[i] for i in sorted_indices]
    magnitudes = [magnitudes[i] for i in sorted_indices]
    categories = [categories[i] for i in sorted_indices]
    relative_changes = [relative_changes[i] for i in sorted_indices]

    # Take top K layers
    if len(layers) > top_k:
        layers = layers[:top_k]
        magnitudes = magnitudes[:top_k]
        categories = categories[:top_k]
        relative_changes = relative_changes[:top_k]

    # Create plot with better spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Magnitude plot (left)
    colors = ['#ff6b6b' if cat == 'embedding' else
              '#4ecdc4' if cat == 'attention' else
              '#45b7d1' if cat == 'mlp' else
              '#96ceb4' if cat == 'layernorm' else
              '#feca57' if cat == 'output' else '#6c5ce7'
              for cat in categories]

    bars1 = ax1.barh(range(len(layers)), magnitudes, color=colors, alpha=0.8, height=0.8)
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels(layers, fontsize=9, fontweight='bold')
    ax1.set_xlabel('Change Magnitude (L2 Norm)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {len(layers)} Layers - {title}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add magnitude values with bounds checking
    for i, (bar, mag) in enumerate(zip(bars1, magnitudes)):
        # Calculate text position - if too far right, place inside bar
        plot_right = ax1.get_xlim()[1]
        text_x = min(mag * 1.02, plot_right * 0.98)

        if text_x > mag * 1.01 or text_x > plot_right * 0.9:  # If text would be outside plot or too close to edge
            text_x = mag * 0.98  # Place inside bar
            ha = 'right'
            color = 'white'
            bbox_style = {}
        else:
            ha = 'left'
            color = 'black'
            bbox_style = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

        ax1.text(text_x, bar.get_y() + bar.get_height()/2,
                f'{mag:.2e}', va='center', ha=ha, fontsize=8,
                fontweight='bold', color=color, bbox=bbox_style)

    # Relative change plot (right)
    bars2 = ax2.barh(range(len(layers)), relative_changes, color=colors, alpha=0.8, height=0.8)
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels([''] * len(layers))  # Hide labels to avoid clutter
    ax2.set_xlabel('Relative Change (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Changes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add relative change values with bounds checking
    for i, (bar, rel) in enumerate(zip(bars2, relative_changes)):
        # Calculate text position - if too far right, place inside bar
        plot_right = ax2.get_xlim()[1]
        text_x = min(rel * 1.02, plot_right * 0.98)

        if text_x > rel * 1.01 or text_x > plot_right * 0.9:  # If text would be outside plot or too close to edge
            text_x = rel * 0.98  # Place inside bar
            ha = 'right'
            color = 'white'
            bbox_style = {}
        else:
            ha = 'left'
            color = 'black'
            bbox_style = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)

        ax2.text(text_x, bar.get_y() + bar.get_height()/2,
                f'{rel:.2f}%', va='center', ha=ha, fontsize=8,
                fontweight='bold', color=color, bbox=bbox_style)

    # Add legend outside the plot area
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc='#ff6b6b', label='Embedding'),
        plt.Rectangle((0,0),1,1, fc='#4ecdc4', label='Attention'),
        plt.Rectangle((0,0),1,1, fc='#45b7d1', label='MLP'),
        plt.Rectangle((0,0),1,1, fc='#96ceb4', label='LayerNorm'),
        plt.Rectangle((0,0),1,1, fc='#feca57', label='Output'),
        plt.Rectangle((0,0),1,1, fc='#6c5ce7', label='Other')
    ]
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
               fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {save_path}")

def create_correlation_plot(cpt_changes, sft_changes, save_path):
    """Create correlation plot between CPT and SFT changes"""
    if not cpt_changes or not sft_changes:
        print("  ⚠️  Insufficient data for correlation analysis")
        return

    # Find common layers
    common_layers = set(cpt_changes.keys()) & set(sft_changes.keys())

    if len(common_layers) < 5:
        print("  ⚠️  Too few common layers for correlation analysis")
        return

    print(f"  Analyzing correlation across {len(common_layers)} common layers")

    # Prepare data
    cpt_magnitudes = []
    sft_magnitudes = []
    layer_names = []
    categories = []

    for layer_name in common_layers:
        cpt_mag = cpt_changes[layer_name]['magnitude']
        sft_mag = sft_changes[layer_name]['magnitude']

        cpt_magnitudes.append(cpt_mag)
        sft_magnitudes.append(sft_mag)
        layer_names.append(layer_name)
        categories.append(categorize_layers(layer_name))

    # Create correlation plot with better layout
    fig, ax = plt.subplots(figsize=(14, 12))

    # Color by category
    colors = ['#ff6b6b' if cat == 'embedding' else
              '#4ecdc4' if cat == 'attention' else
              '#45b7d1' if cat == 'mlp' else
              '#96ceb4' if cat == 'layernorm' else
              '#feca57' if cat == 'output' else '#6c5ce7'
              for cat in categories]

    # Add jitter to prevent exact overlaps
    jitter_amount = 0.02  # Small jitter
    cpt_jittered = [x * (1 + np.random.normal(0, jitter_amount)) for x in cpt_magnitudes]
    sft_jittered = [x * (1 + np.random.normal(0, jitter_amount)) for x in sft_magnitudes]

    # Scatter plot with larger, semi-transparent points
    scatter = ax.scatter(cpt_jittered, sft_jittered, c=colors, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Add labels for top changes with better positioning
    top_indices = np.argsort(np.array(cpt_magnitudes) + np.array(sft_magnitudes))[-15:]  # More labels
    used_positions = set()

    for idx in top_indices:
        x, y = cpt_jittered[idx], sft_jittered[idx]
        layer_name = layer_names[idx][:25] + '...' if len(layer_names[idx]) > 25 else layer_names[idx]

        # Try different offset positions to avoid overlap
        offsets = [(15, 15), (-15, 15), (15, -15), (-15, -15), (0, 20), (0, -20)]
        for dx, dy in offsets:
            if (x + dx, y + dy) not in used_positions:
                used_positions.add((x + dx, y + dy))
                ax.annotate(layer_name,
                           (x, y),
                           xytext=(dx, dy), textcoords='offset points',
                           fontsize=7, alpha=0.9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
                break

    # Add diagonal line
    max_val = max(max(cpt_magnitudes), max(sft_magnitudes))
    min_val = min(min(cpt_magnitudes), min(sft_magnitudes))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2, label='y = x')

    # Calculate correlation
    correlation = np.corrcoef(cpt_magnitudes, sft_magnitudes)[0, 1]

    ax.set_xlabel('CPT Change Magnitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('SFT Change Magnitude', fontsize=12, fontweight='bold')
    ax.set_title(f'CPT vs SFT Change Correlation (r = {correlation:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis limits with padding
    all_vals = cpt_magnitudes + sft_magnitudes
    min_val = min(all_vals) * 0.8
    max_val = max(all_vals) * 1.2
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Add legend outside the plot
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc='#ff6b6b', label='Embedding', alpha=0.8),
        plt.Rectangle((0,0),1,1, fc='#4ecdc4', label='Attention', alpha=0.8),
        plt.Rectangle((0,0),1,1, fc='#45b7d1', label='MLP', alpha=0.8),
        plt.Rectangle((0,0),1,1, fc='#96ceb4', label='LayerNorm', alpha=0.8),
        plt.Rectangle((0,0),1,1, fc='#feca57', label='Output', alpha=0.8),
        plt.Rectangle((0,0),1,1, fc='#6c5ce7', label='Other', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.08, 0.5),
              fontsize=10, framealpha=0.9, borderpad=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {save_path}")
    print(f"  📊 Correlation coefficient: {correlation:.3f}")

def create_layer_type_analysis(changes_dict, title, save_path):
    """Create layer type analysis plot"""
    if not changes_dict:
        print(f"  ⚠️  No data available for {title}")
        return

    # Group by layer type
    type_changes = defaultdict(list)

    for layer_name, change in changes_dict.items():
        layer_type = categorize_layers(layer_name)
        type_changes[layer_type].append(change['magnitude'])

    # Prepare data for plotting
    types = []
    mean_changes = []
    max_changes = []
    layer_counts = []

    for layer_type, changes in type_changes.items():
        types.append(layer_type)
        mean_changes.append(np.mean(changes))
        max_changes.append(np.max(changes))
        layer_counts.append(len(changes))

    # Sort by mean change
    sort_indices = np.argsort(mean_changes)[::-1]
    types = [types[i] for i in sort_indices]
    mean_changes = [mean_changes[i] for i in sort_indices]
    max_changes = [max_changes[i] for i in sort_indices]
    layer_counts = [layer_counts[i] for i in sort_indices]

    # Create plot with better spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Mean changes
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#6c5ce7']
    bars1 = ax1.bar(types, mean_changes, color=colors[:len(types)], alpha=0.8, width=0.8)
    ax1.set_ylabel('Mean Change Magnitude', fontsize=12, fontweight='bold')
    ax1.set_title(f'Mean Changes by Layer Type - {title}', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Add values with better positioning
    for bar, val, count in zip(bars1, mean_changes, layer_counts):
        # Calculate text position - ensure it doesn't go too high
        text_y = min(val * 1.15, ax1.get_ylim()[1] * 0.95)
        ax1.text(bar.get_x() + bar.get_width()/2, text_y,
                f'{val:.2e}\n({count} layers)', ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Max changes
    bars2 = ax2.bar(types, max_changes, color=colors[:len(types)], alpha=0.8, width=0.8)
    ax2.set_ylabel('Max Change Magnitude', fontsize=12, fontweight='bold')
    ax2.set_title(f'Max Changes by Layer Type - {title}', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Add values with better positioning
    for bar, val in zip(bars2, max_changes):
        # Calculate text position - ensure it doesn't go too high
        text_y = min(val * 1.15, ax2.get_ylim()[1] * 0.95)
        ax2.text(bar.get_x() + bar.get_width()/2, text_y,
                f'{val:.2e}', ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Add legend if there are multiple types
    if len(types) > 1:
        legend_elements = [plt.Rectangle((0,0),1,1, fc=color, label=type, alpha=0.8)
                          for color, type in zip(colors[:len(types)], types)]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {save_path}")

    # Print summary
    print(f"  📊 {title} - Layer Type Summary:")
    for t, mean_chg, max_chg, count in zip(types, mean_changes, max_changes, layer_counts):
        print(f"    {t:>10}: {count:>2} layers, mean={mean_chg:.2e}, max={max_chg:.2e}")

def create_change_distributions(changes_dict, title, save_path):
    """Create change distribution plot"""
    if not changes_dict:
        print(f"  ⚠️  No data available for {title}")
        return

    # Prepare data
    magnitudes = [change['magnitude'] for change in changes_dict.values()]
    relative_changes = [change['relative'] * 100 for change in changes_dict.values()]

    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Magnitude distribution (log scale)
    ax1.hist(magnitudes, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.set_xlabel('Change Magnitude (L2 Norm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Magnitude Distribution - {title}')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Magnitude distribution (linear scale, zoomed)
    ax2.hist(magnitudes, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax2.set_xlabel('Change Magnitude (L2 Norm)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Magnitude Distribution (Linear) - {title}')
    ax2.grid(True, alpha=0.3)

    # Relative change distribution
    ax3.hist(relative_changes, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax3.set_xlabel('Relative Change (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Relative Change Distribution - {title}')
    ax3.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_magnitudes = np.sort(magnitudes)
    yvals = np.arange(1, len(sorted_magnitudes) + 1) / len(sorted_magnitudes)
    ax4.plot(sorted_magnitudes, yvals, 'b-', alpha=0.7, linewidth=2)
    ax4.set_xlabel('Change Magnitude (L2 Norm)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title(f'Cumulative Distribution - {title}')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    # Add statistics
    mean_mag = np.mean(magnitudes)
    median_mag = np.median(magnitudes)
    max_mag = np.max(magnitudes)

    ax1.axvline(mean_mag, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_mag:.2e}')
    ax1.axvline(median_mag, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_mag:.2e}')
    ax1.axvline(max_mag, color='orange', linestyle='--', alpha=0.7, label=f'Max: {max_mag:.2e}')
    ax1.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {save_path}")
    print(f"  📊 {title} - Statistics:")
    print(f"    Mean change: {mean_mag:.2e}")
    print(f"    Median change: {median_mag:.2e}")
    print(f"    Max change: {max_mag:.2e}")
    print(f"    Std dev: {np.std(magnitudes):.2e}")

def create_comprehensive_analysis(layer_changes, output_dir, top_k=20):
    """Create comprehensive analysis plots"""
    print("\n🎨 Creating comprehensive visualizations...")

    # 1. Layer magnitude comparison (top K layers)
    if layer_changes['cpt']:
        create_layer_magnitude_plot(layer_changes['cpt'], 'CPT Changes (Base → CPT)',
                                  os.path.join(output_dir, 'cpt_layer_changes.png'), top_k)
    if layer_changes['sft']:
        create_layer_magnitude_plot(layer_changes['sft'], 'SFT Changes (Base → SFT)',
                                  os.path.join(output_dir, 'sft_layer_changes.png'), top_k)

    # 2. Correlation between CPT and SFT changes
    if layer_changes['cpt'] and layer_changes['sft']:
        create_correlation_plot(layer_changes['cpt'], layer_changes['sft'],
                              os.path.join(output_dir, 'cpt_sft_correlation.png'))

    # 3. Layer type analysis
    if layer_changes['cpt']:
        create_layer_type_analysis(layer_changes['cpt'], 'CPT Changes',
                                 os.path.join(output_dir, 'cpt_layer_types.png'))
    if layer_changes['sft']:
        create_layer_type_analysis(layer_changes['sft'], 'SFT Changes',
                                 os.path.join(output_dir, 'sft_layer_types.png'))

    # 4. Change distributions
    if layer_changes['cpt']:
        create_change_distributions(layer_changes['cpt'], 'CPT Changes',
                                  os.path.join(output_dir, 'cpt_distributions.png'))
    if layer_changes['sft']:
        create_change_distributions(layer_changes['sft'], 'SFT Changes',
                                  os.path.join(output_dir, 'sft_distributions.png'))

def save_analysis_results(layer_changes, output_dir):
    """Save detailed analysis results to JSON"""
    print(f"\n💾 Saving detailed analysis results to {output_dir}...")

    results = {
        'analysis_summary': {},
        'layer_details': {},
        'top_changes': {}
    }

    # Summary statistics
    for stage in ['cpt', 'sft', 'cpt_to_sft']:
        if layer_changes[stage]:
            magnitudes = [change['magnitude'] for change in layer_changes[stage].values()]
            results['analysis_summary'][stage] = {
                'total_layers': len(layer_changes[stage]),
                'mean_change': float(np.mean(magnitudes)),
                'median_change': float(np.median(magnitudes)),
                'max_change': float(np.max(magnitudes)),
                'std_change': float(np.std(magnitudes)),
                'total_params_affected': sum(change['param_count'] for change in layer_changes[stage].values())
            }

    # Layer details
    for stage in ['cpt', 'sft', 'cpt_to_sft']:
        if layer_changes[stage]:
            results['layer_details'][stage] = {}
            for layer_name, change in layer_changes[stage].items():
                results['layer_details'][stage][layer_name] = {
                    'magnitude': change['magnitude'],
                    'relative': change['relative'],
                    'shape': change['shape'],
                    'param_count': change['param_count'],
                    'layer_type': categorize_layers(layer_name)
                }

    # Top changes
    for stage in ['cpt', 'sft', 'cpt_to_sft']:
        if layer_changes[stage]:
            sorted_layers = sorted(layer_changes[stage].items(),
                                 key=lambda x: x[1]['magnitude'], reverse=True)
            results['top_changes'][stage] = {
                'top_10': [(name, change['magnitude']) for name, change in sorted_layers[:10]],
                'bottom_10': [(name, change['magnitude']) for name, change in sorted_layers[-10:]]
            }

    # Save to JSON
    output_file = os.path.join(output_dir, 'weight_analysis_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved detailed results to {output_file}")

    # Save summary report
    report_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("TIBETAN LANGUAGE FINETUNING - WEIGHT ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        for stage in ['cpt', 'sft', 'cpt_to_sft']:
            if layer_changes[stage]:
                f.write(f"{stage.upper()} ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                summary = results['analysis_summary'][stage]
                f.write(f"Total layers analyzed: {summary['total_layers']}\n")
                f.write(f"Mean change magnitude: {summary['mean_change']:.2e}\n")
                f.write(f"Median change magnitude: {summary['median_change']:.2e}\n")
                f.write(f"Max change magnitude: {summary['max_change']:.2e}\n")
                f.write(f"Total parameters affected: {summary['total_params_affected']:,}\n")
                f.write(f"\nTop 5 changed layers:\n")
                for i, (name, mag) in enumerate(results['top_changes'][stage]['top_10'][:5]):
                    f.write(f"  {i+1}. {name}: {mag:.2e}\n")
                f.write("\n")

    print(f"  ✓ Saved summary report to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Layer-wise weight change analysis for Tibetan finetuning')
    parser.add_argument('--original', type=str, required=True, help='Path to original model')
    parser.add_argument('--cpt', type=str, help='Path to CPT model')
    parser.add_argument('--sft', type=str, help='Path to SFT model')
    parser.add_argument('--output_dir', type=str, default='weight_analysis', help='Output directory for analysis results')
    parser.add_argument('--top_k', type=int, default=20, help='Number of top layers to show in plots (default: 20)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model weights
    original_weights = load_model_weights(args.original, "Original Model")
    cpt_weights = load_model_weights(args.cpt, "CPT Model") if args.cpt else None
    sft_weights = load_model_weights(args.sft, "SFT Model") if args.sft else None

    if not original_weights:
        print("❌ Error: Could not load original model weights")
        return

    # Analyze layer changes
    layer_changes = analyze_layer_changes(original_weights, cpt_weights, sft_weights)

    # Create comprehensive visualizations
    create_comprehensive_analysis(layer_changes, args.output_dir, args.top_k)

    # Save detailed results
    save_analysis_results(layer_changes, args.output_dir)

    print("\n" + "="*80)
    print("✓ WEIGHT ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("📊 cpt_layer_changes.png - Top CPT layer changes")
    print("📊 sft_layer_changes.png - Top SFT layer changes")
    print("📊 cpt_sft_correlation.png - Correlation between stages")
    print("📊 cpt_layer_types.png - CPT changes by layer type")
    print("📊 sft_layer_types.png - SFT changes by layer type")
    print("📊 cpt_distributions.png - CPT change distributions")
    print("📊 sft_distributions.png - SFT change distributions")
    print("📋 weight_analysis_results.json - Detailed JSON data")
    print("📋 analysis_summary.txt - Human-readable summary")

if __name__ == "__main__":
    main()
