import os
import logging
from nilearn import datasets

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_abide_dataset():
    """
    Downloads the ABIDE dataset if it doesn't already exist.
    Returns the path to the downloaded dataset directory.
    """
    data_dir = "./abide"
    logging.info(f"Checking if dataset directory exists: {data_dir}")
    if not os.path.exists(data_dir):
        logging.info(f"Directory not found. Creating directory: {data_dir}")
        os.makedirs(data_dir)
        logging.info("Downloading ABIDE dataset...")
        datasets.fetch_abide_pcp(
            data_dir=data_dir,
            pipeline="cpac",
            derivatives=["func_preproc"],
            n_subjects=100,
        )
        logging.info("Dataset download complete.")
    else:
        logging.info("Dataset directory already exists. Skipping download.")

    return data_dir


if __name__ == "__main__":
    logging.info("Starting dataset download script...")
    download_abide_dataset()
    logging.info("Dataset download script execution complete.")
