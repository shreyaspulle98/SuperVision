#!/usr/bin/env python3
"""
ALIGNN Setup Diagnostic Script

Checks all dependencies and potential issues before training ALIGNN.
Run this before starting ALIGNN training to catch problems early.

Usage:
    python check_alignn_setup.py
"""

import sys
import os
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text.center(70)}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

def print_check(name, passed, message=""):
    status = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    print(f"{status} {name:<50} {message}")
    return passed

def check_python_version():
    """Check Python version >= 3.7"""
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 7
    print_check(
        "Python version (>= 3.7)",
        passed,
        f"v{version.major}.{version.minor}.{version.micro}"
    )
    return passed

def check_imports():
    """Check all required imports"""
    print_header("Checking Python Packages")

    all_passed = True

    # Core packages
    packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
    ]

    for module, name in packages:
        try:
            __import__(module)
            version = None
            try:
                version = __import__(module).__version__
            except:
                pass
            print_check(name, True, f"v{version}" if version else "installed")
        except ImportError:
            print_check(name, False, "NOT INSTALLED - run: pip install " + module)
            all_passed = False

    # PyMatGen (critical for structure loading)
    try:
        import pymatgen
        from pymatgen.core import Structure
        try:
            version = pymatgen.__version__
        except AttributeError:
            # Older pymatgen versions don't have __version__
            version = "installed"
        print_check("PyMatGen", True, f"v{version}" if version != "installed" else "installed")
    except ImportError:
        print_check("PyMatGen", False, "NOT INSTALLED - run: pip install pymatgen")
        all_passed = False

    # DGL (critical for graph operations)
    try:
        import dgl
        print_check("DGL (graph library)", True, f"v{dgl.__version__}")
    except ImportError:
        print_check("DGL", False, "NOT INSTALLED - run: pip install dgl")
        all_passed = False

    # ALIGNN (the main model)
    try:
        import alignn
        from alignn.pretrained import get_figshare_model
        from alignn.graphs import Graph
        print_check("ALIGNN", True, "installed with pretrained support")

        # Check if we can access the Graph class
        try:
            hasattr(Graph, 'atom_dgl_multigraph')
            print_check("ALIGNN Graph API", True, "atom_dgl_multigraph available")
        except:
            print_check("ALIGNN Graph API", False, "atom_dgl_multigraph not found")
            all_passed = False

    except ImportError as e:
        print_check("ALIGNN", False, f"NOT INSTALLED - run: pip install alignn")
        print(f"  {YELLOW}Error: {e}{RESET}")
        all_passed = False

    return all_passed

def check_data_files():
    """Check if data files exist"""
    print_header("Checking Data Files")

    all_passed = True
    base_path = Path("data/processed")

    # Check CSV files
    for split in ["train", "val", "test"]:
        csv_path = base_path / f"{split}.csv"
        exists = csv_path.exists()
        if exists:
            # Count lines
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                print_check(
                    f"{split}.csv",
                    True,
                    f"{len(df)} materials"
                )
            except Exception as e:
                print_check(f"{split}.csv", False, f"Error reading: {e}")
                all_passed = False
        else:
            print_check(f"{split}.csv", False, "NOT FOUND")
            all_passed = False

    return all_passed

def check_cif_files():
    """Check CIF file accessibility"""
    print_header("Checking CIF Files")

    try:
        import pandas as pd
        train_csv = Path("data/processed/train.csv")

        if not train_csv.exists():
            print_check("CIF files", False, "train.csv not found")
            return False

        df = pd.read_csv(train_csv)

        if 'cif' not in df.columns:
            print_check("CIF column", False, "'cif' column not in CSV")
            return False

        # Sample 10 CIF paths
        sample_cifs = df['cif'].head(10).tolist()
        found_count = 0
        missing_paths = []

        for cif_path in sample_cifs:
            cif_full = Path(cif_path)
            if not cif_full.exists():
                # Try alternative path
                cif_full = Path("data/final") / Path(cif_path).name

            if cif_full.exists():
                found_count += 1
            else:
                missing_paths.append(cif_path)

        passed = found_count >= 8  # At least 80% should exist
        print_check(
            "CIF file accessibility",
            passed,
            f"{found_count}/10 sample files found"
        )

        if missing_paths and found_count < 8:
            print(f"  {YELLOW}Missing CIF files (sample):{RESET}")
            for path in missing_paths[:3]:
                print(f"    - {path}")

        return passed

    except Exception as e:
        print_check("CIF files", False, f"Error: {e}")
        return False

def test_structure_loading():
    """Test loading a single structure"""
    print_header("Testing Structure Loading")

    try:
        import pandas as pd
        from pymatgen.core import Structure

        train_csv = Path("data/processed/train.csv")
        df = pd.read_csv(train_csv)

        # Try to load first structure
        cif_path = df['cif'].iloc[0]

        # Try multiple paths
        for attempt_path in [Path(cif_path), Path("data/final") / Path(cif_path).name]:
            if attempt_path.exists():
                structure = Structure.from_file(str(attempt_path))
                print_check(
                    "Structure loading",
                    True,
                    f"Loaded {structure.formula} ({len(structure)} atoms)"
                )
                return True

        print_check("Structure loading", False, f"CIF not found: {cif_path}")
        return False

    except Exception as e:
        print_check("Structure loading", False, f"Error: {e}")
        return False

def test_graph_conversion():
    """Test converting structure to ALIGNN graph"""
    print_header("Testing Graph Conversion")

    try:
        import pandas as pd
        from pymatgen.core import Structure
        from alignn.graphs import Graph
        import dgl

        train_csv = Path("data/processed/train.csv")
        df = pd.read_csv(train_csv)

        # Load a structure
        cif_path = df['cif'].iloc[0]
        for attempt_path in [Path(cif_path), Path("data/final") / Path(cif_path).name]:
            if attempt_path.exists():
                structure = Structure.from_file(str(attempt_path))
                break
        else:
            print_check("Graph conversion", False, "No structure to test")
            return False

        # Convert to graph
        g, lg = Graph.atom_dgl_multigraph(structure)

        print_check(
            "Graph conversion",
            True,
            f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges"
        )
        print_check(
            "Line graph creation",
            True,
            f"{lg.number_of_nodes()} nodes, {lg.number_of_edges()} edges"
        )

        return True

    except Exception as e:
        print_check("Graph conversion", False, f"Error: {e}")
        print(f"  {YELLOW}This might indicate ALIGNN API changes{RESET}")
        return False

def test_alignn_model():
    """Test loading pre-trained ALIGNN model"""
    print_header("Testing ALIGNN Model Loading")

    try:
        from alignn.pretrained import get_figshare_model

        print(f"  {YELLOW}Downloading pre-trained ALIGNN model (may take 1-2 min)...{RESET}")
        model, config = get_figshare_model("mp_e_form")

        print_check(
            "Pre-trained ALIGNN download",
            True,
            "Successfully loaded from figshare"
        )

        # Check model structure
        param_count = sum(p.numel() for p in model.parameters())
        print_check(
            "Model parameters",
            True,
            f"{param_count:,} parameters"
        )

        return True

    except Exception as e:
        print_check("ALIGNN model loading", False, f"Error: {e}")
        print(f"  {YELLOW}Check internet connection or try again later{RESET}")
        return False

def check_disk_space():
    """Check available disk space"""
    print_header("Checking System Resources")

    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024 ** 3)

        passed = free_gb >= 5  # Need at least 5 GB
        print_check(
            "Disk space",
            passed,
            f"{free_gb} GB free (need 5+ GB)"
        )

        return passed
    except:
        return True  # Skip if can't check

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_check("GPU", True, f"{gpu_name}")
            return True
        else:
            print_check("GPU", False, "Not available (will use CPU - slower)")
            return True  # Not a failure, just slower
    except:
        print_check("GPU", False, "PyTorch CUDA not available")
        return True

def provide_recommendations(results):
    """Provide actionable recommendations"""
    print_header("Recommendations")

    if all(results.values()):
        print(f"{GREEN}✓ All checks passed! You're ready to train ALIGNN.{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print("  1. Run: python -m gat_pipeline.train")
        print("  2. Monitor results in results/alignn_*.{csv,json}")
        print("  3. Compare with DINOv3 results")
    else:
        print(f"{YELLOW}⚠ Some checks failed. Please fix the issues above.{RESET}")
        print(f"\n{BLUE}Quick fixes:{RESET}")

        if not results['imports']:
            print("\n  Install missing packages:")
            print("    pip install alignn dgl pymatgen torch numpy pandas sklearn tqdm")

        if not results['data_files']:
            print("\n  Data files missing:")
            print("    Make sure data/processed/{train,val,test}.csv exist")

        if not results['cif_files']:
            print("\n  CIF files not accessible:")
            print("    Check that CIF paths in CSV are correct")

        if not results['model_loading']:
            print("\n  ALIGNN model download failed:")
            print("    Check internet connection")
            print("    Try running again (downloads can be flaky)")

def main():
    print_header("ALIGNN Setup Diagnostic")
    print(f"Checking setup for ALIGNN fine-tuning on superconductor Tc prediction\n")

    results = {}

    # Run all checks
    print_check("Python version", check_python_version())
    results['imports'] = check_imports()
    results['data_files'] = check_data_files()
    results['cif_files'] = check_cif_files()
    results['structure_loading'] = test_structure_loading()
    results['graph_conversion'] = test_graph_conversion()
    results['model_loading'] = test_alignn_model()
    results['disk_space'] = check_disk_space()
    results['gpu'] = check_gpu()

    # Summary
    provide_recommendations(results)

    # Exit code
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()
