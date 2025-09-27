import nibabel as nib
import numpy as np
from pathlib import Path

def count_corrupted_nii_files(directory):
    """
    Count the number of corrupted .nii.gz files in the specified directory.
    A file is considered corrupted if it contains NaN or inf values.
    """
    # Convert the directory to a Path object
    directory = Path(directory)

    # Resolve the absolute path
    directory = directory.resolve()

    # Check if the directory exists
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Initialize the count of corrupted files
    corrupted_count = 0

    # Iterate through all .nii.gz files in the directory
    for nii_file in directory.glob("*.nii.gz"):
        try:
            # Load the .nii.gz file
            img = nib.load(nii_file)
            data = img.get_fdata()

            # Check for NaN or inf values
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"Fail: {nii_file} (contains NaN or inf values)")
                corrupted_count += 1
            else:
                print(f"Pass: {nii_file} (valid file)")
        except Exception as e:
            # If there's an error loading the file, consider it corrupted
            print(f"Fail: {nii_file} (error loading file: {e})")
            corrupted_count += 1

    # Print the total number of corrupted files
    print(f"Total corrupted files: {corrupted_count}")
    return corrupted_count


# Example usage
if __name__ == "__main__":
    # Specify the directory containing .nii.gz files
    # Use a relative path and resolve it dynamically
    base_dir = Path(__file__).parent  # Get the directory of the current script
    nii_directory = base_dir / "abide" / "ABIDE_pcp" / "cpac" / "nofilt_noglobal"

    # Count corrupted files
    corrupted_files = count_corrupted_nii_files(nii_directory)
    print(f"Number of corrupted files: {corrupted_files}")