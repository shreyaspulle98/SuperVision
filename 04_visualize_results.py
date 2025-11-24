#!/usr/bin/env python3
"""
04_visualize_results.py

Comprehensive visualization script for comparing DINOv3 and ALIGNN models.

This script:
- Loads training history and predictions from both models
- Creates detailed visualizations of model performance
- Generates comparison charts
- Saves all figures to results/figures/

Usage:
    python 04_visualize_results.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Color palette
COLORS = {
    'dinov3': '#2E86AB',  # Blue
    'alignn': '#A23B72',  # Purple
    'target': '#F18F01',  # Orange
}


def load_model_results(model_name):
    """
    Load training history, predictions, and metrics for a model.

    Args:
        model_name (str): 'dinov3' or 'alignn'

    Returns:
        dict: Dictionary containing history, predictions, metrics, config
    """
    results = {}

    # Load training history
    history_file = f"results/{model_name}_training_history.csv"
    if Path(history_file).exists():
        results['history'] = pd.read_csv(history_file)
        print(f"✓ Loaded {model_name} training history: {len(results['history'])} epochs")
    else:
        print(f"✗ Training history not found: {history_file}")
        results['history'] = None

    # Load predictions
    pred_file = f"results/{model_name}_predictions.csv"
    if Path(pred_file).exists():
        results['predictions'] = pd.read_csv(pred_file)
        print(f"✓ Loaded {model_name} predictions: {len(results['predictions'])} samples")
    else:
        print(f"✗ Predictions not found: {pred_file}")
        results['predictions'] = None

    # Load metrics
    metrics_file = f"results/{model_name}_metrics.json"
    if Path(metrics_file).exists():
        with open(metrics_file, 'r') as f:
            results['metrics'] = json.load(f)
        print(f"✓ Loaded {model_name} metrics")
    else:
        print(f"✗ Metrics not found: {metrics_file}")
        results['metrics'] = None

    # Load config
    config_file = f"results/{model_name}_config.json"
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            results['config'] = json.load(f)
        print(f"✓ Loaded {model_name} config")
    else:
        print(f"✗ Config not found: {config_file}")
        results['config'] = None

    return results


def plot_training_curves(dinov3_results, alignn_results, save_dir):
    """
    Plot training and validation loss/MAE curves for both models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # DINOv3 Loss
    if dinov3_results['history'] is not None:
        hist = dinov3_results['history']
        ax = axes[0, 0]
        ax.plot(hist['epoch'], hist['train_loss'],
                label='Train Loss', color=COLORS['dinov3'], linewidth=2)
        ax.plot(hist['epoch'], hist['val_loss'],
                label='Val Loss', color=COLORS['dinov3'], linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('DINOv3 Training Curves - Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # DINOv3 MAE
    if dinov3_results['history'] is not None:
        hist = dinov3_results['history']
        ax = axes[0, 1]
        ax.plot(hist['epoch'], hist['train_mae'],
                label='Train MAE', color=COLORS['dinov3'], linewidth=2)
        ax.plot(hist['epoch'], hist['val_mae'],
                label='Val MAE', color=COLORS['dinov3'], linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (K)')
        ax.set_title('DINOv3 Training Curves - MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ALIGNN Loss
    if alignn_results['history'] is not None:
        hist = alignn_results['history']
        ax = axes[1, 0]
        ax.plot(hist['epoch'], hist['train_loss'],
                label='Train Loss', color=COLORS['alignn'], linewidth=2)
        ax.plot(hist['epoch'], hist['val_loss'],
                label='Val Loss', color=COLORS['alignn'], linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('ALIGNN Training Curves - Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ALIGNN MAE
    if alignn_results['history'] is not None:
        hist = alignn_results['history']
        ax = axes[1, 1]
        ax.plot(hist['epoch'], hist['train_mae'],
                label='Train MAE', color=COLORS['alignn'], linewidth=2)
        ax.plot(hist['epoch'], hist['val_mae'],
                label='Val MAE', color=COLORS['alignn'], linestyle='--', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (K)')
        ax.set_title('ALIGNN Training Curves - MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training_curves.png")


def plot_comparison_metrics(dinov3_results, alignn_results, save_dir):
    """
    Create bar charts comparing key metrics between models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics_to_compare = ['mae', 'rmse', 'r2']
    metric_labels = ['MAE (K)', 'RMSE (K)', 'R²']

    for idx, (metric, label) in enumerate(zip(metrics_to_compare, metric_labels)):
        ax = axes[idx]

        # Get test metrics (try test_ prefix first, then fallback to plain name)
        if dinov3_results['metrics']:
            dinov3_val = dinov3_results['metrics'].get(f'test_{metric}', dinov3_results['metrics'].get(metric, 0))
        else:
            dinov3_val = 0
        if alignn_results['metrics']:
            alignn_val = alignn_results['metrics'].get(f'test_{metric}', alignn_results['metrics'].get(metric, 0))
        else:
            alignn_val = 0

        models = ['DINOv3', 'ALIGNN']
        values = [dinov3_val, alignn_val]
        colors_list = [COLORS['dinov3'], COLORS['alignn']]

        bars = ax.bar(models, values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel(label)
        ax.set_title(f'Test {label}')
        ax.grid(True, alpha=0.3, axis='y')

        # For R², lower is not better, so mark the best differently
        if metric == 'r2':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        # Highlight best performer
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(save_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metric_comparison.png")


def plot_predictions_scatter(dinov3_results, alignn_results, save_dir):
    """
    Create scatter plots of predicted vs actual Tc for both models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # DINOv3
    if dinov3_results['predictions'] is not None:
        df = dinov3_results['predictions']
        ax = axes[0]

        ax.scatter(df['actual_tc'], df['predicted_tc'],
                  alpha=0.5, s=30, color=COLORS['dinov3'], edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(df['actual_tc'].min(), df['predicted_tc'].min())
        max_val = max(df['actual_tc'].max(), df['predicted_tc'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, label='Perfect prediction', alpha=0.7)

        # Add regression line
        z = np.polyfit(df['actual_tc'], df['predicted_tc'], 1)
        p = np.poly1d(z)
        ax.plot(df['actual_tc'], p(df['actual_tc']),
               color=COLORS['target'], linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}', alpha=0.7)

        # Calculate R²
        if dinov3_results['metrics']:
            r2 = dinov3_results['metrics'].get('test_r2', dinov3_results['metrics'].get('r2', 0))
            mae = dinov3_results['metrics'].get('test_mae', dinov3_results['metrics'].get('mae', 0))
        else:
            r2 = 0
            mae = 0

        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.2f} K',
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Actual Tc (K)')
        ax.set_ylabel('Predicted Tc (K)')
        ax.set_title('DINOv3 Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ALIGNN
    if alignn_results['predictions'] is not None:
        df = alignn_results['predictions']
        ax = axes[1]

        ax.scatter(df['actual_tc'], df['predicted_tc'],
                  alpha=0.5, s=30, color=COLORS['alignn'], edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(df['actual_tc'].min(), df['predicted_tc'].min())
        max_val = max(df['actual_tc'].max(), df['predicted_tc'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, label='Perfect prediction', alpha=0.7)

        # Add regression line
        z = np.polyfit(df['actual_tc'], df['predicted_tc'], 1)
        p = np.poly1d(z)
        ax.plot(df['actual_tc'], p(df['actual_tc']),
               color=COLORS['target'], linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}', alpha=0.7)

        # Calculate R²
        if alignn_results['metrics']:
            r2 = alignn_results['metrics'].get('test_r2', alignn_results['metrics'].get('r2', 0))
            mae = alignn_results['metrics'].get('test_mae', alignn_results['metrics'].get('mae', 0))
        else:
            r2 = 0
            mae = 0

        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.2f} K',
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Actual Tc (K)')
        ax.set_ylabel('Predicted Tc (K)')
        ax.set_title('ALIGNN Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions_scatter.png")


def plot_residuals(dinov3_results, alignn_results, save_dir):
    """
    Plot residual distributions and residuals vs predicted values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # DINOv3 residual distribution
    if dinov3_results['predictions'] is not None:
        df = dinov3_results['predictions']
        residuals = df['actual_tc'] - df['predicted_tc']

        ax = axes[0, 0]
        ax.hist(residuals, bins=50, color=COLORS['dinov3'], alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.set_xlabel('Residual (Actual - Predicted) [K]')
        ax.set_ylabel('Frequency')
        ax.set_title(f'DINOv3 Residual Distribution (μ={residuals.mean():.2f}, σ={residuals.std():.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # DINOv3 residuals vs predicted
        ax = axes[0, 1]
        ax.scatter(df['predicted_tc'], residuals, alpha=0.5, s=30,
                  color=COLORS['dinov3'], edgecolors='black', linewidth=0.5)
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.set_xlabel('Predicted Tc (K)')
        ax.set_ylabel('Residual (K)')
        ax.set_title('DINOv3 Residuals vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ALIGNN residual distribution
    if alignn_results['predictions'] is not None:
        df = alignn_results['predictions']
        residuals = df['actual_tc'] - df['predicted_tc']

        ax = axes[1, 0]
        ax.hist(residuals, bins=50, color=COLORS['alignn'], alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.set_xlabel('Residual (Actual - Predicted) [K]')
        ax.set_ylabel('Frequency')
        ax.set_title(f'ALIGNN Residual Distribution (μ={residuals.mean():.2f}, σ={residuals.std():.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ALIGNN residuals vs predicted
        ax = axes[1, 1]
        ax.scatter(df['predicted_tc'], residuals, alpha=0.5, s=30,
                  color=COLORS['alignn'], edgecolors='black', linewidth=0.5)
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.set_xlabel('Predicted Tc (K)')
        ax.set_ylabel('Residual (K)')
        ax.set_title('ALIGNN Residuals vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved residuals.png")


def plot_error_by_tc_range(dinov3_results, alignn_results, save_dir):
    """
    Plot error distribution across different Tc ranges.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define Tc bins
    tc_bins = [0, 20, 40, 60, 80, 100, 150, 200]
    bin_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-150', '150+']

    # DINOv3
    if dinov3_results['predictions'] is not None:
        df = dinov3_results['predictions']
        df['tc_bin'] = pd.cut(df['actual_tc'], bins=tc_bins, labels=bin_labels)
        df['abs_error'] = np.abs(df['actual_tc'] - df['predicted_tc'])

        ax = axes[0]
        df.boxplot(column='abs_error', by='tc_bin', ax=ax,
                   patch_artist=True,
                   boxprops=dict(facecolor=COLORS['dinov3'], alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_xlabel('Tc Range (K)')
        ax.set_ylabel('Absolute Error (K)')
        ax.set_title('DINOv3 Error by Tc Range')
        plt.sca(ax)
        plt.xticks(rotation=45)
        ax.get_figure().suptitle('')  # Remove auto title

    # ALIGNN
    if alignn_results['predictions'] is not None:
        df = alignn_results['predictions']
        df['tc_bin'] = pd.cut(df['actual_tc'], bins=tc_bins, labels=bin_labels)
        df['abs_error'] = np.abs(df['actual_tc'] - df['predicted_tc'])

        ax = axes[1]
        df.boxplot(column='abs_error', by='tc_bin', ax=ax,
                   patch_artist=True,
                   boxprops=dict(facecolor=COLORS['alignn'], alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_xlabel('Tc Range (K)')
        ax.set_ylabel('Absolute Error (K)')
        ax.set_title('ALIGNN Error by Tc Range')
        plt.sca(ax)
        plt.xticks(rotation=45)
        ax.get_figure().suptitle('')  # Remove auto title

    plt.tight_layout()
    plt.savefig(save_dir / 'error_by_tc_range.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved error_by_tc_range.png")


def plot_learning_rate_comparison(dinov3_results, alignn_results, save_dir):
    """
    Compare validation MAE convergence between models.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    if dinov3_results['history'] is not None:
        hist = dinov3_results['history']
        ax.plot(hist['epoch'], hist['val_mae'],
               label='DINOv3', color=COLORS['dinov3'], linewidth=2.5, marker='o', markersize=4)

    if alignn_results['history'] is not None:
        hist = alignn_results['history']
        ax.plot(hist['epoch'], hist['val_mae'],
               label='ALIGNN', color=COLORS['alignn'], linewidth=2.5, marker='s', markersize=4)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MAE (K)')
    ax.set_title('Model Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved convergence_comparison.png")


def generate_summary_report(dinov3_results, alignn_results, save_dir):
    """
    Generate a text summary report comparing both models.
    """
    report = []
    report.append("=" * 80)
    report.append("MODEL COMPARISON SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")

    # DINOv3 Summary
    report.append("DINOv3 (Vision Transformer + LoRA)")
    report.append("-" * 80)
    if dinov3_results['metrics']:
        m = dinov3_results['metrics']
        mae_val = m.get('test_mae', m.get('mae', 'N/A'))
        rmse_val = m.get('test_rmse', m.get('rmse', 'N/A'))
        r2_val = m.get('test_r2', m.get('r2', 'N/A'))
        if mae_val != 'N/A':
            report.append(f"  Test MAE:  {mae_val:.4f} K")
        else:
            report.append(f"  Test MAE:  N/A")
        if rmse_val != 'N/A':
            report.append(f"  Test RMSE: {rmse_val:.4f} K")
        else:
            report.append(f"  Test RMSE: N/A")
        if r2_val != 'N/A':
            report.append(f"  Test R²:   {r2_val:.4f}")
        else:
            report.append(f"  Test R²:   N/A")
    if dinov3_results['config']:
        c = dinov3_results['config']
        report.append(f"  Architecture: {c.get('model_name', 'DINOv3-base')}")
        trainable_params = c.get('trainable_params', 'N/A')
        if trainable_params != 'N/A':
            report.append(f"  Trainable params: {trainable_params:,}")
        else:
            report.append(f"  Trainable params: N/A")
        report.append(f"  Learning rate: {c.get('learning_rate', 'N/A')}")
        report.append(f"  Batch size: {c.get('batch_size', 'N/A')}")
    if dinov3_results['history'] is not None:
        report.append(f"  Epochs trained: {len(dinov3_results['history'])}")
    report.append("")

    # ALIGNN Summary
    report.append("ALIGNN (Graph Neural Network)")
    report.append("-" * 80)
    if alignn_results['metrics']:
        m = alignn_results['metrics']
        mae_val = m.get('test_mae', m.get('mae', 'N/A'))
        rmse_val = m.get('test_rmse', m.get('rmse', 'N/A'))
        r2_val = m.get('test_r2', m.get('r2', 'N/A'))
        if mae_val != 'N/A':
            report.append(f"  Test MAE:  {mae_val:.4f} K")
        else:
            report.append(f"  Test MAE:  N/A")
        if rmse_val != 'N/A':
            report.append(f"  Test RMSE: {rmse_val:.4f} K")
        else:
            report.append(f"  Test RMSE: N/A")
        if r2_val != 'N/A':
            report.append(f"  Test R²:   {r2_val:.4f}")
        else:
            report.append(f"  Test R²:   N/A")
    if alignn_results['config']:
        c = alignn_results['config']
        report.append(f"  Pretrained model: {c.get('pretrained_model_name', 'N/A')}")
        trainable_params = c.get('trainable_params', 'N/A')
        if trainable_params != 'N/A':
            report.append(f"  Trainable params: {trainable_params:,}")
        else:
            report.append(f"  Trainable params: N/A")
        report.append(f"  Backbone LR: {c.get('backbone_lr', 'N/A')}")
        report.append(f"  Head LR: {c.get('head_lr', 'N/A')}")
        report.append(f"  Batch size: {c.get('batch_size', 'N/A')}")
    if alignn_results['history'] is not None:
        report.append(f"  Epochs trained: {len(alignn_results['history'])}")
    report.append("")

    # Comparison
    report.append("COMPARISON")
    report.append("-" * 80)
    if dinov3_results['metrics'] and alignn_results['metrics']:
        d_mae = dinov3_results['metrics'].get('test_mae', dinov3_results['metrics'].get('mae', float('inf')))
        a_mae = alignn_results['metrics'].get('test_mae', alignn_results['metrics'].get('mae', float('inf')))

        d_r2 = dinov3_results['metrics'].get('test_r2', dinov3_results['metrics'].get('r2', 0))
        a_r2 = alignn_results['metrics'].get('test_r2', alignn_results['metrics'].get('r2', 0))

        better_mae = "DINOv3" if d_mae < a_mae else "ALIGNN"
        better_r2 = "DINOv3" if d_r2 > a_r2 else "ALIGNN"

        report.append(f"  Better MAE: {better_mae} (Δ = {abs(d_mae - a_mae):.4f} K)")
        report.append(f"  Better R²:  {better_r2} (Δ = {abs(d_r2 - a_r2):.4f})")
        report.append(f"  MAE improvement: {(abs(d_mae - a_mae) / max(d_mae, a_mae)) * 100:.2f}%")

    report.append("")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    with open(save_dir / 'summary_report.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✓ Saved summary_report.txt")


def main():
    """
    Main visualization pipeline.
    """
    print("=" * 80)
    print("MODEL VISUALIZATION AND COMPARISON")
    print("=" * 80)
    print()

    # Create output directory
    save_dir = Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {save_dir}")
    print()

    # Load results
    print("Loading DINOv3 results...")
    dinov3_results = load_model_results('dinov3')
    print()

    print("Loading ALIGNN results...")
    alignn_results = load_model_results('alignn')
    print()

    # Generate visualizations
    print("=" * 80)
    print("Generating visualizations...")
    print("=" * 80)

    print("\n1. Training curves...")
    plot_training_curves(dinov3_results, alignn_results, save_dir)

    print("\n2. Metric comparison...")
    plot_comparison_metrics(dinov3_results, alignn_results, save_dir)

    print("\n3. Prediction scatter plots...")
    plot_predictions_scatter(dinov3_results, alignn_results, save_dir)

    print("\n4. Residual analysis...")
    plot_residuals(dinov3_results, alignn_results, save_dir)

    print("\n5. Error by Tc range...")
    plot_error_by_tc_range(dinov3_results, alignn_results, save_dir)

    print("\n6. Convergence comparison...")
    plot_learning_rate_comparison(dinov3_results, alignn_results, save_dir)

    print("\n7. Summary report...")
    generate_summary_report(dinov3_results, alignn_results, save_dir)

    print("\n" + "=" * 80)
    print(f"✓ All visualizations saved to {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
