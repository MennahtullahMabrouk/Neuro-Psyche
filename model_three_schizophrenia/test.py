import numpy as np
from pathlib import Path
import nibabel as nib
from scipy.ndimage import zoom
import joblib


class MRIClassifier:
    def __init__(self, target_shape=(32, 32, 16, 120)):
        self.target_shape = target_shape
        self.model_loaded = False
        self.class_names = ["Healthy", "Schizophrenia"]

    def check_model_files(self, model_path, config_path):
        """Check if model files exist and are valid"""
        if not model_path.exists():
            return False, f"Model file not found: {model_path}"

        if not config_path.exists():
            return False, f"Config file not found: {config_path}"

        try:
            config = joblib.load(config_path)
            self.target_shape = config.get('target_shape', self.target_shape)
            self.class_names = config.get('class_names', self.class_names)
            return True, "Model files exist and config is valid"
        except Exception as e:
            return False, f"Error loading config: {e}"

    def preprocess_file(self, file_path):
        """Preprocess file without model prediction"""
        image = nib.load(file_path).get_fdata()
        zoom_factors = [t / s for t, s in zip(self.target_shape, image.shape)]
        return zoom(image, zoom_factors, order=1)

    def simulate_prediction(self, file_path):
        """Simulate prediction without loading the actual model"""
        try:
            image = self.preprocess_file(file_path)
            prob = 0.3
            verdict = "Healthy" if prob < 0.5 else "Schizophrenia"
            probability = prob * 100 if prob >= 0.5 else (1 - prob) * 100

            return f"SIMULATION - Prediction: {verdict}, Probability: {probability:.2f}%"
        except Exception as e:
            return f"Error processing file: {e}"


def main():
    root_dir = Path(__file__).parent.resolve()
    model_path = root_dir / "saved_models" / "Schizophrenia_classifier.h5"
    config_path = root_dir / "saved_models" / "Schizophrenia_classifier.joblib"
    input_file_path = root_dir / "data" / "sub-101_task-rest_bold.nii"

    print("Testing MRI Classifier Setup")
    print("=" * 40)

    classifier = MRIClassifier()

    files_ok, message = classifier.check_model_files(model_path, config_path)
    print(f"Model files check: {message}")

    if input_file_path.exists():
        print(f"Input file found: {input_file_path}")

        try:
            processed_shape = classifier.preprocess_file(input_file_path).shape
            print(f"File preprocessing successful. Output shape: {processed_shape}")
            print(f"Expected shape: {classifier.target_shape}")

            result = classifier.simulate_prediction(input_file_path)
            print(f"\n{result}")
            print("\nNote: This is a simulation. Install TensorFlow for real predictions.")

        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        print(f"Input file not found: {input_file_path}")

    print("\nTo fix TensorFlow installation, run:")
    print("pip install tensorflow")


if __name__ == "__main__":
    main()