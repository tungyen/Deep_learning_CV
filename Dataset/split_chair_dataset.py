import os
import random
import shutil

root = "Chair_dataset"
train_dir = os.path.join(root, "train")
val_dir = os.path.join(root, "val")

label_dir = os.path.join(train_dir, "label")
pts_dir = os.path.join(train_dir, "pts")

os.makedirs(os.path.join(val_dir, "label"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "pts"), exist_ok=True)

label_files = sorted(os.listdir(label_dir))
random.shuffle(label_files)


split_idx = int(0.8 * len(label_files))
train_files = label_files[:split_idx]
val_files = label_files[split_idx:]

print(f"Total samples: {len(label_files)}")
print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")

for f in val_files:
    base = os.path.splitext(f)[0]

    src_label = os.path.join(label_dir, f)
    dst_label = os.path.join(val_dir, "label", f)
    shutil.move(src_label, dst_label)

    src_pts = os.path.join(pts_dir, base + ".pts")
    dst_pts = os.path.join(val_dir, "pts", base + ".pts")
    shutil.move(src_pts, dst_pts)

print("Done! Splitted dataset into 80/20 train/val.")