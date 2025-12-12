import os
import numpy as np
import pandas as pd
import trimesh
from sklearn.model_selection import train_test_split


def load_off(off_path):
    """Load OFF mesh and return XYZ vertices."""
    mesh = trimesh.load(off_path, force='mesh')
    if hasattr(mesh, 'vertices'):
        return np.asarray(mesh.vertices, dtype=np.float32)
    raise ValueError(f"Cannot load OFF file: {off_path}")


def main():
    csv_path = "ModelNet40/metadata_modelnet40.csv"   # Update this if needed
    data_dir = "ModelNet40/ModelNet40"
    output_dir = "ModelNet40_npz"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # --- 1. Split GLOBAL train/val from original train ---
    full_train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=full_train_df["class"]      # ensures balanced per class
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # --- 2. Prepare directory tree ---
    for split_name in ["train", "val", "test"]:
        for cls in df["class"].unique():
            os.makedirs(os.path.join(output_dir, split_name, cls), exist_ok=True)

    # --- 3. Mapping helper ---
    split_to_df = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    # --- 4. Save NPZ files ---
    for split_name, split_df in split_to_df.items():
        print(f"Processing {split_name}...")

        for _, row in split_df.iterrows():
            class_name = row["class"]
            obj_id = row["object_id"]
            off_path = row["object_path"]

            # Load OFF mesh
            xyz = load_off(os.path.join(data_dir, off_path))

            # Save npz
            out_path = os.path.join(output_dir, split_name, class_name, f"{obj_id}.npz")
            np.savez(out_path, xyz=xyz)

    print("All done! Dataset prepared successfully.")


if __name__ == "__main__":
    main()
