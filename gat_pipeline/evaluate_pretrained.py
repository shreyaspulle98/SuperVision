"""
gat_pipeline/evaluate_pretrained.py

Evaluate pre-trained ALIGNN model (jv_supercon_tc_alignn) on test set.

This script:
- Loads the pre-trained ALIGNN model trained on JARVIS superconductor data
- Evaluates on our 3DSC test set WITHOUT fine-tuning
- Provides a zero-shot baseline for comparison

Purpose:
Compare pre-trained vs fine-tuned ALIGNN to quantify transfer learning value.

Usage:
    python -m gat_pipeline.evaluate_pretrained
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_pretrained_model():
    """
    Load pre-trained ALIGNN model for superconductor Tc prediction.

    Uses jv_supercon_tc_alignn from JARVIS database.

    Returns:
        tuple: (model, config)
    """
    print("\n" + "="*70)
    print("Loading Pre-trained ALIGNN Model")
    print("="*70)
    print("\nModel: jv_supercon_tc_alignn")
    print("Source: JARVIS superconductor database")
    print("Task: Critical temperature (Tc) prediction")

    try:
        from alignn.pretrained import get_figshare_model

        print("\nDownloading model from figshare...")
        print("(This may take 1-2 minutes on first run)")

        model = get_figshare_model("jv_supercon_tc_alignn")

        print("\n✓ Model loaded successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        # Set to eval mode
        model.eval()

        return model

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify ALIGNN is installed: pip install alignn")
        print("  3. Try running again (downloads can be flaky)")
        raise


def load_test_data(csv_path="data/processed/test.csv", max_samples=None):
    """
    Load test dataset structures.

    Args:
        csv_path (str): Path to test CSV
        max_samples (int): Limit samples (for testing)

    Returns:
        tuple: (structures, tc_values, material_ids)
    """
    print("\n" + "="*70)
    print("Loading Test Dataset")
    print("="*70)

    df = pd.read_csv(csv_path)

    if max_samples is not None:
        df = df.head(max_samples)
        print(f"\n⚠ Limited to {max_samples} samples for testing")

    print(f"\nDataset: {csv_path}")
    print(f"Samples: {len(df)}")
    print(f"Tc range: {df['tc'].min():.2f} - {df['tc'].max():.2f} K")
    print(f"Mean Tc: {df['tc'].mean():.2f} K")

    structures = []
    tc_values = []
    material_ids = []

    print("\nLoading structures from CIF files...")

    from pymatgen.core import Structure

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
        try:
            cif_path = row['cif']

            # Try multiple path variations
            possible_paths = [
                Path(cif_path),
                Path("data/final/MP/cifs") / Path(cif_path).name,
                Path("data/final") / Path(cif_path).name,
                Path("..") / "data" / "final" / "MP" / "cifs" / Path(cif_path).name,
            ]

            structure = None
            for path in possible_paths:
                if path.exists():
                    structure = Structure.from_file(str(path))
                    break

            if structure is None:
                continue

            structures.append(structure)
            tc_values.append(float(row['tc']))
            material_ids.append(row.get('material_id_2', f"material_{idx}"))

        except Exception as e:
            continue

    print(f"\n✓ Loaded {len(structures)} structures successfully")

    if len(structures) == 0:
        raise ValueError("No structures loaded! Check CIF file paths.")

    return structures, tc_values, material_ids


def predict_single(model, structure):
    """
    Make prediction for a single structure.

    Args:
        model: Pre-trained ALIGNN model
        structure: Pymatgen Structure object

    Returns:
        float: Predicted Tc value
    """
    from jarvis.core.atoms import Atoms as JarvisAtoms
    from alignn.graphs import Graph
    import warnings

    # Suppress CGCNN warnings temporarily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Convert pymatgen Structure to jarvis Atoms
        lattice_mat = structure.lattice.matrix
        coords = structure.frac_coords
        elements = [site.species_string for site in structure]

        jarvis_atoms = JarvisAtoms(
            lattice_mat=lattice_mat,
            coords=coords,
            elements=elements,
            cartesian=False
        )

        # Convert to ALIGNN graph
        g, lg = Graph.atom_dgl_multigraph(jarvis_atoms)

        # Get lattice parameters (a, b, c, alpha, beta, gamma)
        lat_params = structure.lattice.parameters
        lat = torch.tensor([lat_params], dtype=torch.float32)

        # Make prediction (pre-trained model needs lattice params)
        with torch.no_grad():
            prediction = model([g, lg, lat])

    return prediction.item()


def evaluate_pretrained(
    model,
    structures,
    tc_values,
    material_ids,
    save_path="results/pretrained_alignn_predictions.csv"
):
    """
    Evaluate pre-trained model on test set.

    Args:
        model: Pre-trained ALIGNN model
        structures: List of pymatgen Structure objects
        tc_values: List of true Tc values
        material_ids: List of material identifiers
        save_path: Where to save predictions

    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*70)
    print("Running Evaluation")
    print("="*70)

    predictions = []
    actuals = []
    ids = []
    failed_count = 0

    print(f"\nPredicting {len(structures)} structures...")

    for structure, tc_true, mat_id in tqdm(
        zip(structures, tc_values, material_ids),
        total=len(structures),
        desc="Predicting"
    ):
        try:
            tc_pred = predict_single(model, structure)

            predictions.append(tc_pred)
            actuals.append(tc_true)
            ids.append(mat_id)

        except Exception as e:
            print(f"\n⚠ Failed on {mat_id}: {e}")
            failed_count += 1
            continue

    if failed_count > 0:
        print(f"\n⚠ Failed to predict {failed_count}/{len(structures)} structures")

    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # R² score
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Print results
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"\nModel: Pre-trained ALIGNN (jv_supercon_tc_alignn)")
    print(f"Test samples: {len(predictions)}")
    print(f"\nPerformance:")
    print(f"  MAE:  {mae:.2f} K")
    print(f"  RMSE: {rmse:.2f} K")
    print(f"  R²:   {r2:.4f}")

    # Save predictions
    results_df = pd.DataFrame({
        'material_id': ids,
        'tc_true': actuals,
        'tc_pred': predictions,
        'error': predictions - actuals,
        'abs_error': np.abs(predictions - actuals)
    })

    # Create results directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(save_path, index=False)
    print(f"\n✓ Predictions saved to: {save_path}")

    # Save summary metrics
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'n_samples': len(predictions),
        'n_failed': failed_count
    }

    import json
    metrics_path = str(save_path).replace('.csv', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics saved to: {metrics_path}")

    return metrics


def compare_with_finetuned():
    """
    Compare pre-trained results with fine-tuned model results.
    """
    print("\n" + "="*70)
    print("Comparison with Fine-tuned Models")
    print("="*70)

    # Load pre-trained results
    pretrained_path = "results/pretrained_alignn_predictions.csv"
    if not Path(pretrained_path).exists():
        print("\n⚠ Pre-trained results not found")
        return

    pretrained_df = pd.read_csv(pretrained_path)
    pretrained_mae = pretrained_df['abs_error'].mean()

    print(f"\nPre-trained ALIGNN:")
    print(f"  MAE: {pretrained_mae:.2f} K")

    # Load fine-tuned ALIGNN results if available
    finetuned_path = "results/alignn_test_predictions.csv"
    if Path(finetuned_path).exists():
        finetuned_df = pd.read_csv(finetuned_path)
        finetuned_mae = finetuned_df['abs_error'].mean()

        print(f"\nFine-tuned ALIGNN:")
        print(f"  MAE: {finetuned_mae:.2f} K")

        improvement = ((pretrained_mae - finetuned_mae) / pretrained_mae) * 100
        print(f"\nFine-tuning improvement: {improvement:.1f}%")

    # Load DINOv3 results if available
    dinov3_path = "results/dinov3_lora_test_predictions.csv"
    if Path(dinov3_path).exists():
        dinov3_df = pd.read_csv(dinov3_path)
        dinov3_mae = dinov3_df['abs_error'].mean()

        print(f"\nFine-tuned DINOv3:")
        print(f"  MAE: {dinov3_mae:.2f} K")

        comparison = ((pretrained_mae - dinov3_mae) / pretrained_mae) * 100
        print(f"\nDINOv3 vs pre-trained ALIGNN: {comparison:.1f}% improvement")

    print("\n" + "="*70)


def main():
    """Main evaluation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained ALIGNN on superconductor test set"
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default='data/processed/test.csv',
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit number of test samples (for quick testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/pretrained_alignn_predictions.csv',
        help='Where to save predictions'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Pre-trained ALIGNN Evaluation Pipeline")
    print("="*70)
    print("\nThis script evaluates the pre-trained ALIGNN model")
    print("(jv_supercon_tc_alignn) on our test set WITHOUT fine-tuning.")
    print("\nPurpose: Establish a zero-shot baseline for comparison.")

    try:
        # Load pre-trained model
        model = load_pretrained_model()

        # Load test data
        structures, tc_values, material_ids = load_test_data(
            args.test_csv,
            max_samples=args.max_samples
        )

        # Evaluate
        metrics = evaluate_pretrained(
            model,
            structures,
            tc_values,
            material_ids,
            save_path=args.output
        )

        # Compare with fine-tuned models
        compare_with_finetuned()

        print("\n" + "="*70)
        print("✓ Evaluation Complete!")
        print("="*70)

        print("\nNext steps:")
        print("  1. Compare with fine-tuned ALIGNN results")
        print("  2. Analyze where pre-trained model struggles")
        print("  3. Update README with baseline comparison")

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
