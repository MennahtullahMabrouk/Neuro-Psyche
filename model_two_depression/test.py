import numpy as np
from pathlib import Path
import nibabel as nib
import logging
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img
import joblib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DepressionClassifier:
    def __init__(self):
        """Initialize classifier with empty attributes"""
        self.model = None
        self.atlas = None
        self.label_encoder = None
        self._is_loaded = False

    def load_model_package(self, model_path):
        """Load pre-trained model package from specified path"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found at: {model_path}")

            logging.info(f"Loading model package from: {model_path}")

            params = joblib.load(str(model_path / "depression_model.joblib"))
            self.model = params["model"]
            self.label_encoder = params["label_encoder"]

            self.atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
            self._is_loaded = True
            logging.info("Model package loaded successfully")
            return True

        except Exception as e:
            logging.error(f"Error loading model package: {str(e)}")
            self._is_loaded = False
            return False

    def preprocess_file(self, file_path):
        """Preprocess fMRI file for prediction"""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model_package() first.")

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")

            logging.info(f"Preprocessing file: {file_path}")

            fmri_img = nib.load(str(file_path))
            atlas_resampled = resample_to_img(
                self.atlas.maps, fmri_img, interpolation="nearest", force_resample=True, copy_header=True
            )
            masker = NiftiLabelsMasker(labels_img=atlas_resampled, standardize=True)
            time_series = masker.fit_transform(fmri_img)

            time_series_flat = time_series.mean(axis=0).reshape(1, -1)
            return time_series_flat

        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise

    def predict_and_verity(self, file_path):
        """Make prediction and return formatted result with status and confidence"""
        try:
            if not self._is_loaded:
                raise RuntimeError("Model not loaded. Call load_model_package() first.")

            processed_data = self.preprocess_file(file_path)
            prob = self.model.predict_proba(processed_data)[0]
            pred_class = self.model.predict(processed_data)[0]

            status = self.label_encoder.inverse_transform([pred_class])[0]
            confidence = prob[1] * 100 if pred_class == 1 else prob[0] * 100

            if pred_class == 1:
                return f"Depressed (confidence: {confidence:.1f}%)"
            else:
                return f"Healthy (confidence: {confidence:.1f}%)"

        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            logging.error(error_msg)
            return error_msg


def main():
    """Test function for standalone execution"""
    try:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "model_two_depression" / "depression_model"
        input_file_path = (
                project_root
                / "model_two_depression"
                / "Data"
                / "sub-01"
                / "func"
                / "sub-01_task-rest_bold.nii.gz"
        )

        classifier = DepressionClassifier()
        if not classifier.load_model_package(model_path):
            print("Failed to load model package")
            return

        result = classifier.predict_and_verity(input_file_path)
        print("\nPrediction Result:")
        print(result)

    except Exception as e:
        logging.error(f"Test execution failed: {str(e)}")


if __name__ == "__main__":
    main()