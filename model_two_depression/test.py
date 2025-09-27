import numpy as np
from pathlib import Path
import nibabel as nib
import tensorflow as tf
import logging
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DepressionClassifier:
    def __init__(self):
        """Initialize classifier with empty attributes"""
        self.model = None
        self.atlas = None
        self.max_rois = None
        self.max_length = None
        self.label_encoder = None
        self._is_loaded = False

    def load_model_package(self, model_path):
        """Load pre-trained model package from specified path"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found at: {model_path}")

            logging.info(f"Loading model package from: {model_path}")
            self.model = tf.keras.models.load_model(str(model_path / 'model.h5'))

            params = joblib.load(str(model_path / 'preprocessing_params.joblib'))
            self.max_rois = params['max_rois']
            self.max_length = params['max_length']
            self.label_encoder = params['label_encoder']

            self.atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
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
            atlas_resampled = resample_to_img(self.atlas.maps, fmri_img, interpolation='nearest')
            masker = NiftiLabelsMasker(labels_img=atlas_resampled, standardize=True)
            time_series = masker.fit_transform(fmri_img)

            if time_series.shape[1] < self.max_rois:
                missing_rois = self.max_rois - time_series.shape[1]
                time_series = np.hstack([time_series, np.zeros((time_series.shape[0], missing_rois))])

            time_series = pad_sequences([time_series],
                                        maxlen=self.max_length,
                                        dtype='float32',
                                        padding='post',
                                        value=0)
            return time_series.reshape(1, self.max_length, self.max_rois)

        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise

    def predict_and_verity(self, file_path):
        """Make prediction and return formatted result with status and confidence"""
        try:
            if not self._is_loaded:
                raise RuntimeError("Model not loaded. Call load_model_package() first.")

            processed_data = self.preprocess_file(file_path)
            prob = self.model.predict(processed_data, verbose=0)[0][0]
            pred_class = int(prob > 0.5)

            status = self.label_encoder.inverse_transform([pred_class])[0]
            confidence = prob * 100 if pred_class == 1 else (1 - prob) * 100

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
        input_file_path = project_root / "model_two_depression" / "Data" / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz"

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