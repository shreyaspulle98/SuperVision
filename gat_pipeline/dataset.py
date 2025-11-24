"""
gat_pipeline/dataset.py

Graph dataset preparation for ALIGNN model fine-tuning.

This module:
- Loads crystal structures from CSV files
- Converts structures to ALIGNN-compatible DGL graphs
- Creates atom graphs and line graphs (for bond angles)
- Prepares data loaders for training

ALIGNN Graph Representation:
- Atom graph (g): Nodes are atoms, edges are bonds
- Line graph (lg): Nodes are bonds, edges connect bonds sharing an atom
- Node features: Atomic properties (Z, electronegativity, radius, etc.)
- Edge features: Bond distances, bond types

Data Split:
- Uses same train/val/test split as DINOv3 pipeline
- Loads from data/processed/{train,val,test}.csv
- Each sample has: material_id, formula, tc, CIF structure
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_dataset(csv_path, max_samples=None):
    """
    Load superconductor dataset from CSV.

    Args:
        csv_path (str): Path to CSV file (train.csv, val.csv, or test.csv)
        max_samples (int): Maximum number of samples to load (for debugging)

    Returns:
        tuple: (structures, tc_values, material_ids)
    """
    print(f"Loading dataset from {csv_path}")

    df = pd.read_csv(csv_path)

    if max_samples is not None:
        df = df.head(max_samples)
        print(f"  Limited to {max_samples} samples for debugging")

    print(f"  Loaded {len(df)} samples")
    print(f"  Tc range: {df['tc'].min():.2f} - {df['tc'].max():.2f} K")
    print(f"  Mean Tc: {df['tc'].mean():.2f} K")

    # Extract structures, Tc values, and material IDs
    structures = []
    tc_values = []
    material_ids = []

    print(f"  Parsing CIF files...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading structures"):
        try:
            # Load structure from CIF file
            cif_path = row['cif']

            # Handle relative paths - prepend ../ since we run from gat_pipeline/
            if not Path(cif_path).is_absolute():
                cif_path = Path("..") / "data" / "final" / Path(cif_path).name

            if not Path(cif_path).exists():
                # Try alternative path
                cif_path = Path("..") / row['cif']

            from pymatgen.core import Structure
            structure = Structure.from_file(str(cif_path))

            structures.append(structure)
            tc_values.append(float(row['tc']))
            material_ids.append(row.get('material_id_2', f"material_{idx}"))

        except Exception as e:
            # Skip structures that can't be loaded
            continue

    print(f"  Successfully loaded {len(structures)} structures")

    return structures, tc_values, material_ids


def structure_to_alignn_graphs(structure, tc_value):
    """
    Convert pymatgen Structure to ALIGNN-compatible DGL graphs.

    ALIGNN uses two graphs:
    1. Atom graph: nodes=atoms, edges=bonds
    2. Line graph: nodes=bonds, edges=angle connections

    Args:
        structure: Pymatgen Structure object
        tc_value (float): Critical temperature label

    Returns:
        tuple: (atom_graph, line_graph, tc_tensor)
    """
    try:
        from jarvis.core.atoms import Atoms as JarvisAtoms
        from alignn.graphs import Graph
        import numpy as np

        # Convert pymatgen Structure to jarvis Atoms
        lattice_mat = structure.lattice.matrix
        coords = structure.frac_coords
        elements = [site.species_string for site in structure]

        # Create jarvis Atoms object
        jarvis_atoms = JarvisAtoms(
            lattice_mat=lattice_mat,
            coords=coords,
            elements=elements,
            cartesian=False
        )

        # Convert to ALIGNN graph format
        alignn_graph = Graph.atom_dgl_multigraph(jarvis_atoms)

        # Extract atom graph and line graph
        g = alignn_graph[0]  # Atom graph (DGL graph)
        lg = alignn_graph[1]  # Line graph (DGL graph)

        # Add target label as graph-level feature
        g.ndata['target'] = torch.tensor([tc_value] * g.number_of_nodes(), dtype=torch.float32)

        return g, lg, torch.tensor([tc_value], dtype=torch.float32)

    except Exception as e:
        print(f"Error converting structure to ALIGNN graph: {e}")
        import traceback
        traceback.print_exc()
        raise


class SuperconductorALIGNNDataset:
    """
    Dataset class for superconductor structures compatible with ALIGNN.

    Returns DGL graphs (atom graph + line graph) for each structure.
    """

    def __init__(self, csv_path, max_samples=None):
        """
        Initialize dataset.

        Args:
            csv_path (str): Path to CSV file
            max_samples (int): Maximum samples to load (for debugging)
        """
        self.csv_path = csv_path

        # Load structures and labels
        self.structures, self.tc_values, self.material_ids = load_dataset(
            csv_path, max_samples=max_samples
        )

        # Pre-convert all structures to graphs (for faster training)
        print(f"  Converting structures to ALIGNN graphs...")
        self.graphs = []
        self.line_graphs = []
        self.targets = []

        failed_count = 0
        for i, (structure, tc) in enumerate(tqdm(
            zip(self.structures, self.tc_values),
            total=len(self.structures),
            desc="Creating graphs"
        )):
            try:
                g, lg, target = structure_to_alignn_graphs(structure, tc)
                self.graphs.append(g)
                self.line_graphs.append(lg)
                self.targets.append(target)
            except Exception as e:
                failed_count += 1
                continue

        if failed_count > 0:
            print(f"  ⚠ Failed to convert {failed_count}/{len(self.structures)} structures")

        print(f"  ✓ Created {len(self.graphs)} graph pairs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            tuple: (atom_graph, line_graph, target)
        """
        return self.graphs[idx], self.line_graphs[idx], self.targets[idx]


def collate_alignn(samples):
    """
    Collate function for batching ALIGNN graphs.

    Args:
        samples: List of (g, lg, target) tuples

    Returns:
        tuple: (batched_g, batched_lg, batched_targets)
    """
    import dgl

    graphs, line_graphs, targets = zip(*samples)

    # Batch DGL graphs
    batched_g = dgl.batch(graphs)
    batched_lg = dgl.batch(line_graphs)
    batched_targets = torch.stack(targets)

    return batched_g, batched_lg, batched_targets


def get_dataloader(
    csv_path,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    max_samples=None
):
    """
    Create DataLoader for ALIGNN training.

    Args:
        csv_path (str): Path to CSV file (train.csv, val.csv, test.csv)
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        max_samples (int): Max samples to load (for debugging)

    Returns:
        DataLoader: PyTorch DataLoader with ALIGNN graphs
    """
    dataset = SuperconductorALIGNNDataset(csv_path, max_samples=max_samples)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_alignn
    )

    return loader


def get_dataloaders(
    data_root="data/processed",
    batch_size=32,
    num_workers=0,
    max_samples=None
):
    """
    Create train, val, and test data loaders.

    Args:
        data_root (str): Root directory with train.csv, val.csv, test.csv
        batch_size (int): Batch size
        num_workers (int): Number of workers
        max_samples (int): Max samples per split (for debugging)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data_root = Path(data_root)

    print("Creating data loaders...")
    print(f"  Data root: {data_root}")
    print(f"  Batch size: {batch_size}")

    train_loader = get_dataloader(
        str(data_root / "train.csv"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_samples=max_samples
    )

    val_loader = get_dataloader(
        str(data_root / "val.csv"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_samples=max_samples
    )

    test_loader = get_dataloader(
        str(data_root / "test.csv"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_samples=max_samples
    )

    print(f"\n✓ Data loaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_loader.dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_loader.dataset)} samples)")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset creation
    print("=" * 70)
    print("Testing ALIGNN Dataset")
    print("=" * 70)

    try:
        # Test loading a small sample
        print("\nTesting with 10 samples...")
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root="data/processed",
            batch_size=4,
            max_samples=10
        )

        # Test batch loading
        print("\nTesting batch loading...")
        for g, lg, targets in train_loader:
            print(f"  Atom graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
            print(f"  Line graph: {lg.number_of_nodes()} nodes, {lg.number_of_edges()} edges")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Targets: {targets.squeeze()}")
            break

        print("\n✓ Dataset loading successful!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure ALIGNN and dependencies are installed:")
        print("  pip install alignn dgl pymatgen")
        print("=" * 70)
