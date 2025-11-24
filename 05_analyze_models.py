#!/usr/bin/env python3
"""
05_analyze_models.py

Deep analysis script for identifying model weaknesses and suggesting improvements.

This script:
- Performs detailed error analysis
- Identifies problematic samples and outliers
- Analyzes overfitting/underfitting
- Suggests hyperparameter improvements
- Generates actionable recommendations
- Analyzes feature importance (where applicable)

Usage:
    python 05_analyze_models.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelAnalyzer:
    """
    Comprehensive model analysis class.
    """

    def __init__(self, model_name):
        """
        Initialize analyzer for a specific model.

        Args:
            model_name (str): 'dinov3' or 'alignn'
        """
        self.model_name = model_name
        self.results_dir = Path("results")
        self.analysis_dir = Path("results/analysis") / model_name
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.history = self._load_history()
        self.predictions = self._load_predictions()
        self.metrics = self._load_metrics()
        self.config = self._load_config()

    def _load_history(self):
        """Load training history."""
        file_path = self.results_dir / f"{self.model_name}_training_history.csv"
        if file_path.exists():
            return pd.read_csv(file_path)
        return None

    def _load_predictions(self):
        """Load test predictions."""
        file_path = self.results_dir / f"{self.model_name}_predictions.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Add error columns
            df['error'] = df['actual_tc'] - df['predicted_tc']
            df['abs_error'] = np.abs(df['error'])
            df['percent_error'] = (df['abs_error'] / df['actual_tc']) * 100
            return df
        return None

    def _load_metrics(self):
        """Load metrics."""
        file_path = self.results_dir / f"{self.model_name}_metrics.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

    def _load_config(self):
        """Load config."""
        file_path = self.results_dir / f"{self.model_name}_config.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

    def analyze_overfitting(self):
        """
        Analyze overfitting/underfitting from training curves.
        """
        print(f"\n{'='*80}")
        print(f"OVERFITTING ANALYSIS - {self.model_name.upper()}")
        print(f"{'='*80}")

        if self.history is None:
            print("No training history available.")
            return

        # Get final epoch metrics
        final_epoch = self.history.iloc[-1]
        train_mae = final_epoch['train_mae']
        val_mae = final_epoch['val_mae']

        train_loss = final_epoch['train_loss']
        val_loss = final_epoch['val_loss']

        # Calculate gaps
        mae_gap = val_mae - train_mae
        loss_gap = val_loss - train_loss

        mae_gap_percent = (mae_gap / train_mae) * 100
        loss_gap_percent = (loss_gap / train_loss) * 100

        print(f"\nFinal Epoch Metrics:")
        print(f"  Train MAE: {train_mae:.4f} K")
        print(f"  Val MAE:   {val_mae:.4f} K")
        print(f"  MAE Gap:   {mae_gap:.4f} K ({mae_gap_percent:.2f}%)")
        print()
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Loss Gap:   {loss_gap:.4f} ({loss_gap_percent:.2f}%)")

        # Diagnosis
        print(f"\nDiagnosis:")
        if mae_gap_percent > 20:
            print(f"  ⚠ OVERFITTING DETECTED (MAE gap > 20%)")
            print(f"     Model performs {mae_gap_percent:.1f}% worse on validation data")
        elif mae_gap_percent > 10:
            print(f"  ⚠ Mild overfitting (MAE gap > 10%)")
        else:
            print(f"  ✓ Good generalization (MAE gap < 10%)")

        if train_mae > 10:
            print(f"  ⚠ High training error suggests underfitting")
            print(f"     Model may need more capacity or better features")

        # Trend analysis
        if len(self.history) > 5:
            recent_val_mae = self.history['val_mae'].iloc[-5:].values
            if np.all(np.diff(recent_val_mae) > 0):
                print(f"  ⚠ Validation MAE increasing (last 5 epochs)")
                print(f"     Early stopping should have triggered")

        # Recommendations
        print(f"\nRecommendations:")
        if mae_gap_percent > 20:
            print(f"  • Increase regularization (dropout, weight decay)")
            print(f"  • Reduce model complexity")
            print(f"  • Add more training data")
            print(f"  • Use data augmentation")
        elif train_mae > 10:
            print(f"  • Increase model capacity")
            print(f"  • Increase learning rate")
            print(f"  • Train for more epochs")
            print(f"  • Add more features/improve feature engineering")

    def identify_worst_predictions(self, top_n=20):
        """
        Identify materials with worst predictions.
        """
        print(f"\n{'='*80}")
        print(f"WORST PREDICTIONS - {self.model_name.upper()}")
        print(f"{'='*80}")

        if self.predictions is None:
            print("No predictions available.")
            return

        # Sort by absolute error
        worst = self.predictions.nlargest(top_n, 'abs_error')

        print(f"\nTop {top_n} worst predictions:")
        print(f"{'Rank':<6}{'Material ID':<15}{'Actual Tc':<12}{'Predicted Tc':<15}{'Error':<12}{'% Error':<10}")
        print("-" * 80)

        for idx, (rank, row) in enumerate(worst.iterrows(), 1):
            mat_id = row.get('material_id', f'idx_{rank}')
            actual = row['actual_tc']
            predicted = row['predicted_tc']
            error = row['error']
            pct_error = row['percent_error']

            print(f"{idx:<6}{str(mat_id):<15}{actual:<12.2f}{predicted:<15.2f}{error:<12.2f}{pct_error:<10.2f}")

        # Save to CSV
        worst.to_csv(self.analysis_dir / 'worst_predictions.csv', index=False)
        print(f"\n✓ Saved worst predictions to {self.analysis_dir / 'worst_predictions.csv'}")

        # Analyze patterns
        print(f"\nPattern Analysis:")
        high_tc_errors = worst[worst['actual_tc'] > 100]
        low_tc_errors = worst[worst['actual_tc'] < 20]

        print(f"  High Tc (>100K) materials: {len(high_tc_errors)}/{top_n}")
        print(f"  Low Tc (<20K) materials: {len(low_tc_errors)}/{top_n}")

        if len(high_tc_errors) > top_n / 2:
            print(f"  ⚠ Model struggles with high-Tc materials")
        if len(low_tc_errors) > top_n / 2:
            print(f"  ⚠ Model struggles with low-Tc materials")

    def analyze_error_distribution(self):
        """
        Analyze error distribution across Tc ranges.
        """
        print(f"\n{'='*80}")
        print(f"ERROR DISTRIBUTION ANALYSIS - {self.model_name.upper()}")
        print(f"{'='*80}")

        if self.predictions is None:
            print("No predictions available.")
            return

        # Define bins
        bins = [0, 20, 40, 60, 80, 100, 150, 200]
        labels = ['0-20K', '20-40K', '40-60K', '60-80K', '80-100K', '100-150K', '150K+']

        self.predictions['tc_bin'] = pd.cut(self.predictions['actual_tc'], bins=bins, labels=labels)

        print(f"\nError statistics by Tc range:")
        print(f"{'Tc Range':<15}{'Count':<10}{'Mean MAE':<15}{'Median MAE':<15}{'Std MAE':<15}{'Max Error':<15}")
        print("-" * 90)

        for label in labels:
            subset = self.predictions[self.predictions['tc_bin'] == label]
            if len(subset) == 0:
                continue

            count = len(subset)
            mean_mae = subset['abs_error'].mean()
            median_mae = subset['abs_error'].median()
            std_mae = subset['abs_error'].std()
            max_error = subset['abs_error'].max()

            print(f"{label:<15}{count:<10}{mean_mae:<15.2f}{median_mae:<15.2f}{std_mae:<15.2f}{max_error:<15.2f}")

        # Identify problematic ranges
        grouped = self.predictions.groupby('tc_bin')['abs_error'].mean()
        worst_range = grouped.idxmax()
        worst_mae = grouped.max()

        print(f"\nMost problematic range: {worst_range} (MAE = {worst_mae:.2f} K)")

    def analyze_bias(self):
        """
        Analyze systematic bias in predictions.
        """
        print(f"\n{'='*80}")
        print(f"BIAS ANALYSIS - {self.model_name.upper()}")
        print(f"{'='*80}")

        if self.predictions is None:
            print("No predictions available.")
            return

        mean_error = self.predictions['error'].mean()
        median_error = self.predictions['error'].median()

        print(f"\nSystematic bias:")
        print(f"  Mean error:   {mean_error:.4f} K")
        print(f"  Median error: {median_error:.4f} K")

        if abs(mean_error) > 2:
            if mean_error > 0:
                print(f"  ⚠ Model systematically UNDERPREDICTS by {mean_error:.2f} K")
            else:
                print(f"  ⚠ Model systematically OVERPREDICTS by {abs(mean_error):.2f} K")
        else:
            print(f"  ✓ No significant systematic bias")

        # Test for normality of errors
        _, p_value = stats.normaltest(self.predictions['error'])
        print(f"\nError distribution normality test:")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  ⚠ Errors are NOT normally distributed")
            print(f"     This suggests systematic issues in the model")
        else:
            print(f"  ✓ Errors are approximately normally distributed")

    def suggest_hyperparameter_improvements(self):
        """
        Suggest hyperparameter improvements based on training curves.
        """
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER RECOMMENDATIONS - {self.model_name.upper()}")
        print(f"{'='*80}")

        if self.config is None:
            print("No config available.")
            return

        print(f"\nCurrent hyperparameters:")
        for key, value in self.config.items():
            if key not in ['timestamp', 'device']:
                print(f"  {key}: {value}")

        print(f"\nRecommendations:")

        # Learning rate analysis
        if self.history is not None and len(self.history) > 0:
            val_mae_values = self.history['val_mae'].values

            # Check if stuck in plateau
            if len(val_mae_values) > 10:
                recent_std = np.std(val_mae_values[-10:])
                if recent_std < 0.1:
                    print(f"\n• LEARNING RATE:")
                    print(f"  Current: {self.config.get('learning_rate', 'N/A')}")
                    print(f"  ⚠ Validation MAE plateaued (std={recent_std:.4f})")
                    print(f"  → Try learning rate scheduling:")
                    print(f"     - ReduceLROnPlateau (factor=0.5, patience=5)")
                    print(f"     - Cosine annealing")

            # Check convergence speed
            if len(val_mae_values) >= 5:
                initial_mae = val_mae_values[0]
                epoch_5_mae = val_mae_values[4]
                improvement = (initial_mae - epoch_5_mae) / initial_mae

                if improvement < 0.1:
                    print(f"\n• SLOW CONVERGENCE:")
                    print(f"  Only {improvement*100:.1f}% improvement in first 5 epochs")
                    print(f"  → Increase learning rate by 2-5x")
                    print(f"  → Or use warmup schedule")

        # Batch size
        batch_size = self.config.get('batch_size', 32)
        print(f"\n• BATCH SIZE:")
        print(f"  Current: {batch_size}")
        if batch_size < 32:
            print(f"  → Consider increasing to 32-64 for more stable gradients")
        elif batch_size > 128:
            print(f"  → Large batch size may reduce generalization")
            print(f"  → Try 32-64 with learning rate adjustment")

        # Model-specific recommendations
        if self.model_name == 'dinov3':
            print(f"\n• DINOV3-SPECIFIC:")
            print(f"  → Try different LoRA ranks: 8, 16, 32, 64")
            print(f"  → Experiment with LoRA alpha (currently: {self.config.get('lora_alpha', 'N/A')})")
            print(f"  → Try unfreezing more layers for full fine-tuning")

        elif self.model_name == 'alignn':
            backbone_lr = self.config.get('backbone_lr', 'N/A')
            head_lr = self.config.get('head_lr', 'N/A')
            print(f"\n• ALIGNN-SPECIFIC:")
            print(f"  → Adjust differential learning rates:")
            print(f"     Current backbone LR: {backbone_lr}")
            print(f"     Current head LR: {head_lr}")
            print(f"  → Try backbone_lr = head_lr / 100 (1e-5 to 1e-3)")

        # Regularization
        print(f"\n• REGULARIZATION:")
        weight_decay = self.config.get('weight_decay', 0)
        print(f"  Current weight decay: {weight_decay}")
        if weight_decay == 0:
            print(f"  → Add weight decay: 1e-5 to 1e-4")
        print(f"  → Add dropout to head: 0.1 to 0.3")
        print(f"  → Try label smoothing: 0.1")

    def generate_improvement_checklist(self):
        """
        Generate actionable checklist for improving model.
        """
        print(f"\n{'='*80}")
        print(f"IMPROVEMENT CHECKLIST - {self.model_name.upper()}")
        print(f"{'='*80}")

        checklist = []

        # Data improvements
        checklist.append("\n[ ] DATA IMPROVEMENTS:")
        checklist.append("  [ ] Add data augmentation")
        checklist.append("  [ ] Collect more training data")
        checklist.append("  [ ] Balance dataset across Tc ranges")
        checklist.append("  [ ] Clean outliers and mislabeled data")

        # Model architecture
        checklist.append("\n[ ] ARCHITECTURE:")
        if self.model_name == 'dinov3':
            checklist.append("  [ ] Try different LoRA configurations")
            checklist.append("  [ ] Experiment with different DINOv3 variants (small, base, large)")
            checklist.append("  [ ] Add attention pooling instead of average pooling")
        else:
            checklist.append("  [ ] Try deeper graph networks")
            checklist.append("  [ ] Add edge features")
            checklist.append("  [ ] Experiment with different aggregation functions")

        # Hyperparameters
        checklist.append("\n[ ] HYPERPARAMETERS:")
        checklist.append("  [ ] Tune learning rate (grid search or Bayesian optimization)")
        checklist.append("  [ ] Implement learning rate scheduling")
        checklist.append("  [ ] Adjust batch size")
        checklist.append("  [ ] Add/tune regularization (dropout, weight decay)")

        # Training strategy
        checklist.append("\n[ ] TRAINING STRATEGY:")
        checklist.append("  [ ] Increase epochs (if underfitting)")
        checklist.append("  [ ] Implement progressive unfreezing")
        checklist.append("  [ ] Try different optimizers (AdamW, SGD with momentum)")
        checklist.append("  [ ] Add gradient clipping")

        # Ensemble methods
        checklist.append("\n[ ] ENSEMBLE:")
        checklist.append("  [ ] Combine DINOv3 and ALIGNN predictions")
        checklist.append("  [ ] Train multiple models with different seeds")
        checklist.append("  [ ] Use weighted averaging based on Tc range")

        # Evaluation
        checklist.append("\n[ ] EVALUATION:")
        checklist.append("  [ ] Implement k-fold cross-validation")
        checklist.append("  [ ] Analyze per-material-class performance")
        checklist.append("  [ ] Test on external validation set")

        for item in checklist:
            print(item)

        # Save checklist
        with open(self.analysis_dir / 'improvement_checklist.txt', 'w') as f:
            f.write('\n'.join(checklist))

        print(f"\n✓ Saved checklist to {self.analysis_dir / 'improvement_checklist.txt'}")

    def run_full_analysis(self):
        """
        Run complete analysis pipeline.
        """
        print(f"\n{'#'*80}")
        print(f"# COMPREHENSIVE MODEL ANALYSIS: {self.model_name.upper()}")
        print(f"{'#'*80}")

        self.analyze_overfitting()
        self.identify_worst_predictions()
        self.analyze_error_distribution()
        self.analyze_bias()
        self.suggest_hyperparameter_improvements()
        self.generate_improvement_checklist()

        print(f"\n{'='*80}")
        print(f"Analysis complete! Results saved to: {self.analysis_dir}")
        print(f"{'='*80}")


def compare_models():
    """
    Compare DINOv3 and ALIGNN to identify which performs better where.
    """
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS: DINOV3 vs ALIGNN")
    print(f"{'='*80}")

    # Load predictions from both models
    dinov3_pred_file = Path("results/dinov3_predictions.csv")
    alignn_pred_file = Path("results/alignn_predictions.csv")

    if not dinov3_pred_file.exists() or not alignn_pred_file.exists():
        print("⚠ Both models must be trained for comparison")
        return

    dinov3_df = pd.read_csv(dinov3_pred_file)
    alignn_df = pd.read_csv(alignn_pred_file)

    # Merge predictions
    merged = pd.merge(
        dinov3_df[['material_id', 'actual_tc', 'predicted_tc']],
        alignn_df[['material_id', 'predicted_tc']],
        on='material_id',
        suffixes=('_dinov3', '_alignn')
    )

    merged['error_dinov3'] = np.abs(merged['actual_tc'] - merged['predicted_tc_dinov3'])
    merged['error_alignn'] = np.abs(merged['actual_tc'] - merged['predicted_tc_alignn'])
    merged['better_model'] = merged.apply(
        lambda row: 'DINOv3' if row['error_dinov3'] < row['error_alignn'] else 'ALIGNN',
        axis=1
    )

    print(f"\nPer-sample comparison:")
    dinov3_wins = (merged['better_model'] == 'DINOv3').sum()
    alignn_wins = (merged['better_model'] == 'ALIGNN').sum()
    total = len(merged)

    print(f"  DINOv3 better: {dinov3_wins}/{total} ({dinov3_wins/total*100:.1f}%)")
    print(f"  ALIGNN better: {alignn_wins}/{total} ({alignn_wins/total*100:.1f}%)")

    # Analyze where each model excels
    bins = [0, 20, 40, 60, 80, 100, 150, 200]
    labels = ['0-20K', '20-40K', '40-60K', '60-80K', '80-100K', '100-150K', '150K+']
    merged['tc_bin'] = pd.cut(merged['actual_tc'], bins=bins, labels=labels)

    print(f"\nPerformance by Tc range:")
    for label in labels:
        subset = merged[merged['tc_bin'] == label]
        if len(subset) == 0:
            continue

        dinov3_wins_range = (subset['better_model'] == 'DINOv3').sum()
        total_range = len(subset)

        print(f"  {label}: DINOv3 wins {dinov3_wins_range}/{total_range} ({dinov3_wins_range/total_range*100:.1f}%)")

    # Ensemble suggestion
    print(f"\nENSEMBLE RECOMMENDATION:")
    ensemble_pred = (merged['predicted_tc_dinov3'] + merged['predicted_tc_alignn']) / 2
    ensemble_mae = mean_absolute_error(merged['actual_tc'], ensemble_pred)

    dinov3_mae = mean_absolute_error(merged['actual_tc'], merged['predicted_tc_dinov3'])
    alignn_mae = mean_absolute_error(merged['actual_tc'], merged['predicted_tc_alignn'])

    print(f"  DINOv3 MAE:  {dinov3_mae:.4f} K")
    print(f"  ALIGNN MAE:  {alignn_mae:.4f} K")
    print(f"  Ensemble MAE: {ensemble_mae:.4f} K")

    if ensemble_mae < min(dinov3_mae, alignn_mae):
        improvement = (min(dinov3_mae, alignn_mae) - ensemble_mae) / min(dinov3_mae, alignn_mae) * 100
        print(f"  ✓ Ensemble improves by {improvement:.2f}%!")
    else:
        print(f"  ✗ Ensemble does not improve performance")


def main():
    """
    Main analysis pipeline.
    """
    print("=" * 80)
    print("MODEL ANALYSIS AND IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)

    # Analyze DINOv3
    print("\n" + "="*80)
    print("Analyzing DINOv3...")
    print("="*80)
    dinov3_analyzer = ModelAnalyzer('dinov3')
    dinov3_analyzer.run_full_analysis()

    # Analyze ALIGNN
    print("\n" + "="*80)
    print("Analyzing ALIGNN...")
    print("="*80)
    alignn_analyzer = ModelAnalyzer('alignn')
    alignn_analyzer.run_full_analysis()

    # Compare models
    try:
        compare_models()
    except Exception as e:
        print(f"\n⚠ Could not run comparison: {e}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAnalysis results saved to:")
    print("  - results/analysis/dinov3/")
    print("  - results/analysis/alignn/")


if __name__ == "__main__":
    main()
