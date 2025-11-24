"""
03_render_images.py

Renders physics-informed 2D images from 3D crystal structures for the vision pipeline.

The script:
- Loads crystal structures from processed data
- Generates multiple viewing angles and zoom levels for each structure
- Creates physics-informed visualizations with multi-channel encoding:
  * R channel: d-orbital density (correlation effects)
  * G channel: valence electrons (metallicity/conductivity)  
  * B channel: inverse mass (phonon frequency proxy)
- Includes bonding information as visual context
- Renders high-quality images suitable for DINOv3

Diversity strategy:
- 3 supercell sizes: 1×1×1, 2×2×1, 2×2×2 (close-up, medium, far)
- 3 crystallographic views: a, b, c axes
- 1-3 random rotations per supercell size
- Total: 12-21 images per material

Output:
- data/images/train/: Training set images
- data/images/val/: Validation set images
- data/images/test/: Test set images
- data/images/image_metadata.csv: Image paths and labels
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.transformations.standard_transformations import RotationTransformation
import warnings
warnings.filterwarnings('ignore')


# Element property lookups
D_ELECTRONS = {
    'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 5, 'Mn': 5, 
    'Fe': 6, 'Co': 7, 'Ni': 8, 'Cu': 9, 'Zn': 10,
    'Y': 1, 'Zr': 2, 'Nb': 4, 'Mo': 5, 'Tc': 6,
    'Ru': 7, 'Rh': 8, 'Pd': 10, 'Ag': 10, 'Cd': 10,
    'La': 1, 'Hf': 2, 'Ta': 3, 'W': 4, 'Re': 5,
    'Os': 6, 'Ir': 7, 'Pt': 9, 'Au': 10, 'Hg': 10,
}


def get_d_electron_count(element):
    """Return number of d-electrons for transition metals."""
    # Handle both Element/Species and Composition objects
    if hasattr(element, 'symbol'):
        return D_ELECTRONS.get(str(element.symbol), 0)
    else:
        # For Composition objects, get the first element
        from pymatgen.core import Composition
        if isinstance(element, Composition):
            elements = list(element.elements)
            if elements:
                return D_ELECTRONS.get(str(elements[0].symbol), 0)
        return 0


def get_valence_electrons(element):
    """
    Return approximate number of valence electrons.
    Simplified model based on periodic table position.
    """
    if hasattr(element, 'group') and element.group:
        group = element.group
        if group <= 2:  # s-block
            return group
        elif group >= 13:  # p-block
            return group - 10
        else:  # d-block transition metals
            d_count = get_d_electron_count(element)
            if d_count > 0:
                return min(d_count, 2)  # Rough approximation
    return 1  # Default fallback


def load_structures_from_csv(csv_path):
    """
    Loads crystal structures from the 3DSC dataset CSV.
    
    Args:
        csv_path: Path to CSV file with material data
        
    Returns:
        list: List of (structure, tc_value, material_id) tuples
    """
    print(f"Loading structures from {csv_path}...")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} entries in CSV")
    
    structures = []
    
    # Check what columns are available
    print(f"Columns in CSV: {df.columns.tolist()}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading structures"):
        try:
            # Use formula as material_id if no ID column
            material_id = row.get('material_id', row.get('formula', f'material_{idx}'))
            tc_value = row['tc']
            
            # Try different methods to get structure
            structure = None

            # Method 1: Check if there's a 'cif' column with file path
            if 'cif' in df.columns and pd.notna(row.get('cif')):
                cif_value = row['cif']

                # The CSV contains relative paths like "data/final/MP/cifs/filename.cif"
                # Extract just the filename and look in known locations
                cif_filename = Path(cif_value).name

                # Try multiple possible file locations
                possible_paths = [
                    Path(f"/Users/shrey/Downloads/MP 3/cifs/{cif_filename}"),
                    Path(f"/Users/shrey/Downloads/MP/cifs/{cif_filename}"),
                    Path(cif_value),  # Try original path
                    Path(f"data/raw/{cif_filename}"),
                ]

                for cif_path in possible_paths:
                    if cif_path.exists():
                        structure = Structure.from_file(str(cif_path))
                        break

            # Method 2: Check if there's a 'structure' column with JSON
            if structure is None and 'structure' in df.columns and pd.notna(row.get('structure')):
                import json
                struct_dict = json.loads(row['structure'])
                structure = Structure.from_dict(struct_dict)

            # Method 3: Try to load from file using material_id
            if structure is None:
                # Try multiple possible file locations based on material_id
                possible_paths = [
                    Path(f"data/raw/structures/{material_id}.cif"),
                    Path(f"data/raw/cif/{material_id}.cif"),
                    Path(f"data/structures/{material_id}.cif"),
                ]

                for cif_path in possible_paths:
                    if cif_path.exists():
                        structure = Structure.from_file(str(cif_path))
                        break
            
            if structure is not None:
                structures.append((structure, tc_value, str(material_id)))
            else:
                if idx < 5:  # Only print first few warnings
                    print(f"Warning: No structure data found for {material_id}")
                
        except Exception as e:
            if idx < 5:  # Only print first few errors
                print(f"Error loading structure {idx}: {e}")
            continue
    
    print(f"Successfully loaded {len(structures)} structures")
    return structures


def load_structures(split="train"):
    """
    Loads crystal structures for a given split.

    Args:
        split (str): Data split ('train', 'val', or 'test')

    Returns:
        list: List of (structure, tc_value, material_id) tuples
    """
    csv_path = Path(f"data/processed/{split}.csv")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {csv_path}. "
            f"Please run 02_prepare_data.py first."
        )

    return load_structures_from_csv(csv_path)


def apply_random_rotation(structure):
    """
    Apply a random rotation to a structure.
    
    Args:
        structure: Pymatgen Structure object
        
    Returns:
        Rotated structure
    """
    # Generate random axis
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    
    # Random angle between 0 and 360 degrees
    angle = np.random.uniform(0, 360)
    
    # Apply rotation transformation
    rotation = RotationTransformation(axis, angle, angle_in_radians=False)
    rotated_structure = rotation.apply_transformation(structure)
    
    return rotated_structure


def project_to_2d(structure, view_axis='c', supercell_size=(2, 2, 1)):
    """
    Project 3D crystal structure to 2D plane with supercell expansion.
    
    Args:
        structure: Pymatgen Structure object
        view_axis: Which axis to look down ('a', 'b', or 'c')
        supercell_size: Tuple for supercell expansion
        
    Returns:
        supercell: Expanded structure
        coords_2d: Nx2 array of 2D coordinates
        coord_mapping: Axis indices used for projection
    """
    # Create supercell to show periodicity
    supercell = structure.copy()
    supercell.make_supercell(supercell_size)
    
    # Get Cartesian coordinates
    coords_3d = supercell.cart_coords
    
    # Choose projection axes based on viewing direction
    if view_axis == 'c':
        coord_mapping = [0, 1]  # x, y (looking down z)
    elif view_axis == 'a':
        coord_mapping = [1, 2]  # y, z (looking down x)
    elif view_axis == 'b':
        coord_mapping = [0, 2]  # x, z (looking down y)
    else:
        coord_mapping = [0, 1]  # default
    
    coords_2d = coords_3d[:, coord_mapping]
    
    return supercell, coords_2d, coord_mapping


def normalize_coords_to_image(coords_2d, img_size=224, padding=20):
    """
    Normalize 2D coordinates to fit in image space with padding.
    
    Args:
        coords_2d: Nx2 array of coordinates
        img_size: Image dimension (square)
        padding: Padding in pixels
        
    Returns:
        Normalized coordinates as integer pixel positions
    """
    # Find bounding box
    coords_min = coords_2d.min(axis=0)
    coords_max = coords_2d.max(axis=0)
    coords_range = coords_max - coords_min
    
    # Avoid division by zero
    coords_range[coords_range == 0] = 1.0
    
    # Scale to fit in image with padding
    drawable_size = img_size - 2 * padding
    scale = drawable_size / coords_range.max()
    
    # Center and scale
    coords_centered = coords_2d - coords_min
    coords_scaled = coords_centered * scale + padding
    
    return coords_scaled.astype(int)


def get_physics_color(element):
    """
    Get RGB color encoding physical properties.

    R channel: d-orbital density (correlation effects)
    G channel: valence electrons (metallicity)
    B channel: inverse mass (phonon frequency)

    Args:
        element: Pymatgen Element, Species, or Composition object

    Returns:
        Tuple (R, G, B) with values 0-255
    """
    # Handle Composition objects by using the first element
    from pymatgen.core import Composition
    if isinstance(element, Composition):
        elements = list(element.elements)
        if not elements:
            return (128, 128, 128)  # Gray for empty composition
        element = elements[0]  # Use dominant/first element

    # RED: d-electron count (0-10 range)
    d_count = get_d_electron_count(element)
    red = int((d_count / 10.0) * 255)

    # GREEN: valence electrons (0-12 range)
    valence = get_valence_electrons(element)
    green = int((valence / 12.0) * 255)

    # BLUE: inverse mass (light atoms = high blue = high phonon freq)
    mass = element.atomic_mass
    mass_normalized = 1.0 - min(mass / 238.0, 1.0)  # Normalize by uranium
    blue = int(mass_normalized * 255)

    return (red, green, blue)


def get_atomic_radius(element, scale_factor=3.0):
    """
    Get scaled atomic radius for visualization.

    Args:
        element: Pymatgen Element, Species, or Composition object
        scale_factor: Scaling factor for visibility

    Returns:
        Radius in pixels
    """
    # Handle Composition objects
    from pymatgen.core import Composition
    if isinstance(element, Composition):
        elements = list(element.elements)
        if not elements:
            return 5  # Default radius
        element = elements[0]

    # Use atomic radius or default
    if hasattr(element, 'atomic_radius') and element.atomic_radius:
        radius = element.atomic_radius * scale_factor
    else:
        radius = 5.0  # Default radius

    return max(int(radius), 3)  # Minimum 3 pixels


def draw_bonds(img, supercell, coords_2d, max_atoms=200):
    """
    Draw bonds between atoms using CrystalNN for bonding detection.
    
    Args:
        img: Image array to draw on (modified in place)
        supercell: Pymatgen Structure
        coords_2d: Nx2 array of 2D atomic positions
        max_atoms: Maximum atoms to process for performance
    """
    try:
        bonding = CrystalNN()
        n_atoms = len(supercell)
        
        # Only process subset to avoid performance issues
        if n_atoms > max_atoms:
            indices_to_process = np.random.choice(
                n_atoms, 
                max_atoms, 
                replace=False
            )
        else:
            indices_to_process = range(n_atoms)
        
        for i in indices_to_process:
            try:
                neighbors = bonding.get_nn_info(supercell, i)
                
                for neighbor in neighbors:
                    j = neighbor['site_index']
                    
                    # Only draw each bond once
                    if j > i:
                        pos1 = tuple(coords_2d[i])
                        pos2 = tuple(coords_2d[j])
                        
                        # Check if both positions are valid
                        if (0 <= pos1[0] < img.shape[1] and 
                            0 <= pos1[1] < img.shape[0] and
                            0 <= pos2[0] < img.shape[1] and 
                            0 <= pos2[1] < img.shape[0]):
                            
                            # Bond color: gray
                            bond_color = (80, 80, 80)
                            cv2.line(img, pos1, pos2, bond_color, 1)
            except:
                continue
                
    except Exception as e:
        # If bonding detection fails, skip bonds entirely
        pass


def render_structure_image(structure, output_path, view_axis='c', 
                          supercell_size=(2, 2, 1), image_size=224, 
                          include_bonds=True):
    """
    Renders a physics-informed 2D image from a 3D crystal structure.

    Args:
        structure: Pymatgen Structure object
        output_path: Path to save the rendered image
        view_axis: Viewing direction ('a', 'b', 'c', or 'random')
        supercell_size: Tuple for supercell expansion
        image_size: Output image dimension (square)
        include_bonds: Whether to draw bonds between atoms

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Apply random rotation if requested
        if view_axis == 'random':
            structure = apply_random_rotation(structure)
            view_axis = 'c'  # View along c-axis after rotation
        
        # Create blank image (RGB)
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Project structure to 2D
        supercell, coords_3d, _ = project_to_2d(structure, view_axis, supercell_size)
        coords_2d = normalize_coords_to_image(coords_3d, image_size)
        
        # STEP 1: Draw bonds (background layer)
        if include_bonds:
            draw_bonds(img, supercell, coords_2d)
        
        # STEP 2: Draw atoms (foreground layer) with physics encoding
        for i, site in enumerate(supercell):
            element = site.species if hasattr(site, 'species') else site.specie
            pos_2d = tuple(coords_2d[i])
            
            # Check if position is valid
            if not (0 <= pos_2d[0] < image_size and 0 <= pos_2d[1] < image_size):
                continue
            
            # Get physics-informed color (R=d-orbitals, G=valence, B=inverse mass)
            color = get_physics_color(element)
            
            # Get atomic radius
            radius = get_atomic_radius(element)
            
            # Draw atom as filled circle
            cv2.circle(img, pos_2d, radius, color, -1)
            
            # Add thin black outline for visibility
            cv2.circle(img, pos_2d, radius, (0, 0, 0), 1)
        
        # Save image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error rendering image: {e}")
        return False


def render_multiple_views(structure, material_id, output_dir, 
                          supercell_sizes=[(1,1,1), (2,2,1), (2,2,2)],
                          crystallographic_views=["a", "b", "c"],
                          n_random_rotations=1,
                          include_bonds=True):
    """
    Renders multiple viewing angles and zoom levels of a structure.

    Args:
        structure: Pymatgen Structure
        material_id: Unique material identifier
        output_dir: Directory to save images
        supercell_sizes: List of supercell size tuples
        crystallographic_views: List of viewing directions ('a', 'b', 'c')
        n_random_rotations: Number of random rotations per supercell
        include_bonds: Whether to draw bonds

    Returns:
        list: Metadata dicts for successfully rendered images
    """
    metadata = []
    
    for supercell_size in supercell_sizes:
        # Format supercell size as string for filename
        supercell_str = f"{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}"
        
        # Render crystallographic views
        for view in crystallographic_views:
            filename = f"{material_id}_supercell_{supercell_str}_view_{view}.png"
            output_path = output_dir / filename
            
            success = render_structure_image(
                structure, 
                output_path, 
                view_axis=view,
                supercell_size=supercell_size,
                image_size=224,
                include_bonds=include_bonds
            )
            
            if success:
                metadata.append({
                    'image_path': str(output_path),
                    'material_id': material_id,
                    'supercell_size': supercell_str,
                    'view_type': 'crystallographic',
                    'view': view
                })
        
        # Render random rotations
        for i in range(n_random_rotations):
            filename = f"{material_id}_supercell_{supercell_str}_view_random{i+1}.png"
            output_path = output_dir / filename
            
            success = render_structure_image(
                structure, 
                output_path, 
                view_axis='random',
                supercell_size=supercell_size,
                image_size=224,
                include_bonds=include_bonds
            )
            
            if success:
                metadata.append({
                    'image_path': str(output_path),
                    'material_id': material_id,
                    'supercell_size': supercell_str,
                    'view_type': 'random',
                    'view': f'random{i+1}'
                })
    
    return metadata


def create_image_dataset(split="train",
                        supercell_sizes=[(1,1,1), (2,2,1), (2,2,2)],
                        crystallographic_views=["a", "b", "c"],
                        n_random_rotations=1,
                        include_bonds=True):
    """
    Creates the complete image dataset for a given split.

    Args:
        split: Data split ('train', 'val', or 'test')
        supercell_sizes: List of supercell size tuples
        crystallographic_views: List of crystallographic viewing directions
        n_random_rotations: Number of random rotations per supercell size
        include_bonds: Whether to draw bonds (set False for speed)

    Returns:
        pd.DataFrame: Metadata with image paths and labels
    """
    print(f"\nCreating {split} image dataset...")
    
    # Calculate expected images per material
    n_images_per_material = len(supercell_sizes) * (
        len(crystallographic_views) + n_random_rotations
    )
    print(f"Will generate {n_images_per_material} images per material")

    # Load structures
    structures = load_structures(split)
    
    if len(structures) == 0:
        print(f"No structures found for {split} split!")
        return pd.DataFrame()
    
    # Create output directory
    output_dir = Path(f"data/images/{split}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    # Render each structure
    for structure, tc_value, material_id in tqdm(structures, 
                                                   desc=f"Rendering {split}"):
        # Render multiple views and zoom levels
        image_metadata = render_multiple_views(
            structure, 
            material_id, 
            output_dir,
            supercell_sizes=supercell_sizes,
            crystallographic_views=crystallographic_views,
            n_random_rotations=n_random_rotations,
            include_bonds=include_bonds
        )

        # Add tc value to each metadata entry
        for meta in image_metadata:
            meta['tc'] = tc_value
            meta['split'] = split

        all_metadata.extend(image_metadata)

    return pd.DataFrame(all_metadata)


def main():
    """Main execution function."""
    print("=" * 60)
    print("Step 3: Rendering Physics-Informed Images")
    print("=" * 60)
    print("\nPhysics encoding:")
    print("  R channel: d-orbital density (correlation effects)")
    print("  G channel: valence electrons (metallicity)")
    print("  B channel: inverse mass (phonon frequency)")
    print("=" * 60)

    # ==================== CONFIGURATION ====================
    
    # Supercell sizes (zoom levels)
    supercell_sizes = [
        (1, 1, 1),  # Close-up: local coordination
        (2, 2, 1),  # Medium: in-plane periodicity  
        (2, 2, 2),  # Far: full 3D periodicity
    ]
    
    # Crystallographic viewing directions
    crystallographic_views = ["a", "b", "c"]
    
    # Number of random rotations per supercell size
    # Options:
    #   1 → 12 images per material (3 supercells × 4 views)
    #   2 → 15 images per material (3 supercells × 5 views)
    #   3 → 18 images per material (3 supercells × 6 views)
    n_random_rotations = 3  # Generates 18 diverse images per material
    
    # Include bonds (set False to speed up rendering on laptop)
    include_bonds = True  # <-- Set to False if too slow
    
    # =======================================================
    
    n_images_per_material = len(supercell_sizes) * (
        len(crystallographic_views) + n_random_rotations
    )
    print(f"\nConfiguration:")
    print(f"  Supercell sizes: {supercell_sizes}")
    print(f"  Views per supercell: {len(crystallographic_views)} crystallographic + {n_random_rotations} random")
    print(f"  Images per material: {n_images_per_material}")
    print(f"  Include bonds: {include_bonds}")
    print("=" * 60)
    
    # Create image datasets for each split
    all_metadata = []

    for split in ["train", "val", "test"]:
        try:
            metadata = create_image_dataset(
                split, 
                supercell_sizes=supercell_sizes,
                crystallographic_views=crystallographic_views,
                n_random_rotations=n_random_rotations,
                include_bonds=include_bonds
            )
            
            if len(metadata) > 0:
                all_metadata.append(metadata)
                print(f"\n{split.capitalize()} set: {len(metadata)} images rendered")
                print(f"  Unique materials: {metadata['material_id'].nunique()}")
                print(f"  Tc range: {metadata['tc'].min():.2f}K - {metadata['tc'].max():.2f}K")
                
                # Show breakdown by supercell size
                print(f"  Images by supercell size:")
                for size in metadata['supercell_size'].unique():
                    count = len(metadata[metadata['supercell_size'] == size])
                    print(f"    {size}: {count} images")
            else:
                print(f"\nWarning: No images rendered for {split} split")
                
        except Exception as e:
            print(f"\nError processing {split} split: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_metadata) == 0:
        print("\nError: No images were rendered!")
        return

    # Combine and save metadata
    full_metadata = pd.concat(all_metadata, ignore_index=True)

    # Create metadata directory
    metadata_dir = Path("data/images")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = metadata_dir / "image_metadata.csv"
    full_metadata.to_csv(metadata_path, index=False)

    print("\n" + "=" * 60)
    print("Image rendering complete!")
    print("=" * 60)
    print(f"Total images: {len(full_metadata)}")
    print(f"Unique materials: {full_metadata['material_id'].nunique()}")
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Estimate file size
    estimated_size_mb = len(full_metadata) * 0.05  # ~50KB per image
    print(f"Estimated storage: ~{estimated_size_mb:.1f} MB")
    
    # Print sample of color encoding
    print("\n" + "=" * 60)
    print("Example color encodings:")
    print("=" * 60)
    from pymatgen.core import Element
    
    examples = [
        ('Cu', 'Copper (cuprates)'),
        ('Fe', 'Iron (Fe-based SC)'),
        ('B', 'Boron (MgB2)'),
        ('H', 'Hydrogen (hydrides)'),
        ('O', 'Oxygen (ligand)'),
        ('La', 'Lanthanum (rare earth)')
    ]
    
    for symbol, description in examples:
        elem = Element(symbol)
        color = get_physics_color(elem)
        d_count = get_d_electron_count(elem)
        valence = get_valence_electrons(elem)
        mass = elem.atomic_mass
        print(f"{symbol:3s} ({description:20s}): RGB{color} | d={d_count}, val={valence}, mass={mass:.1f}")

    # Visualize example images
    print("\n" + "=" * 60)
    print("Visualizing Example Images")
    print("=" * 60)
    visualize_example_images(full_metadata)


def visualize_example_images(metadata_df, n_materials=2, max_views=6):
    """
    Displays example rendered images in a grid.

    Args:
        metadata_df: DataFrame with image metadata
        n_materials: Number of materials to show
        max_views: Maximum views per material to display
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Select a few materials randomly
    unique_materials = metadata_df['material_id'].unique()
    if len(unique_materials) > n_materials:
        import random
        random.seed(42)
        selected_materials = random.sample(list(unique_materials), n_materials)
    else:
        selected_materials = unique_materials[:n_materials]

    print(f"\nDisplaying example images for {len(selected_materials)} materials...")

    for mat_id in selected_materials:
        # Get images for this material
        mat_images = metadata_df[metadata_df['material_id'] == mat_id]

        # Get Tc value
        tc_value = mat_images['tc'].iloc[0]

        # Select up to max_views images
        if len(mat_images) > max_views:
            mat_images = mat_images.sample(n=max_views, random_state=42)

        # Create figure
        n_images = len(mat_images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes

        fig.suptitle(f'Material: {mat_id} | Tc = {tc_value:.2f} K', fontsize=14, fontweight='bold')

        for idx, (_, row) in enumerate(mat_images.iterrows()):
            if idx >= len(axes):
                break

            img_path = row['image_path']
            supercell = row.get('supercell_size', 'unknown')
            view = row.get('view', 'unknown')

            try:
                img = mpimg.imread(img_path)
                axes[idx].imshow(img)
                axes[idx].set_title(f'{supercell} | {view}', fontsize=10)
                axes[idx].axis('off')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error loading\n{img_path}',
                             ha='center', va='center', fontsize=8)
                axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Save figure
        output_path = Path(f"data/images/visualization_{mat_id}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")

        # Show figure
        plt.show()
        plt.close()

    print(f"\nVisualization complete!")


if __name__ == "__main__":
    main()