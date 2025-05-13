import os
import glob
import shutil
import numpy as np



def load_shape_names(dataset_dir):
    filepath = os.path.join(dataset_dir, 'modelnet40_shape_names.txt')
    with open(filepath) as f:
        shape_names = f.read().splitlines()
    shape_names = [c.rstrip() for c in shape_names]
    return shape_names


def load_txt(txt_path):
    pcloud = np.loadtxt(txt_path, delimiter=',')
    pcloud = np.reshape(pcloud, [-1, 6]).astype(np.float32)
    xyz, normal = pcloud[:, 0:3], pcloud[:, 3:6]
    return xyz, normal


def main():
    input_dir = "ModelNet40"
    output_dir = 'ModelNet40_npz'
    os.makedirs(output_dir, exist_ok=True)

    # Convert dataset from `.txt` to `.npz`
    for shape_name in load_shape_names(input_dir):
        os.makedirs(os.path.join(output_dir, shape_name), exist_ok=True)
        for txt_path in glob.glob(os.path.join(input_dir, shape_name, '*.txt')):
            model_idx = os.path.basename(txt_path).split('.')[0]
            output_path = os.path.join(output_dir, shape_name, '%s.npz' % model_idx)
            xyz, normal = load_txt(txt_path)
            np.savez(output_path, xyz=xyz, normal=normal)

    # Copy other files
    for txt_path in glob.glob(os.path.join(input_dir, '*.txt')):
        shutil.copy2(txt_path, output_dir)


if __name__ == '__main__':
    main()
    print('All done.')