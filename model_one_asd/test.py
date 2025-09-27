import nibabel as nib
import numpy as np
from pathlib import Path


def count_corrupted_nii_files(directory):
    """
    Count the number of corrupted .nii.gz files in the specified directory.
    A file is considered corrupted if it contains NaN or inf values.
    """
    directory = Path(directory)
    directory = directory.resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    corrupted_count = 0

    for nii_file in directory.glob("*.nii.gz"):
        try:
            img = nib.load(nii_file)
            data = img.get_fdata()

            if np.isnan(data).any() or np.isinf(data).any():
                print(f"Fail: {nii_file} (contains NaN or inf values)")
                corrupted_count += 1
            else:
                print(f"Pass: {nii_file} (valid file)")
        except Exception as e:
            print(f"Fail: {nii_file} (error loading file: {e})")
            corrupted_count += 1

    print(f"Total corrupted files: {corrupted_count}")
    return corrupted_count


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    nii_directory = base_dir / "abide" / "ABIDE_pcp" / "cpac" / "nofilt_noglobal"

    corrupted_files = count_corrupted_nii_files(nii_directory)
    print(f"Number of corrupted files: {corrupted_files}")
