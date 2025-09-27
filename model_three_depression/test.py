import numpy as np
from pathlib import Path
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MRIClassifier:
    def __init__(self, target_shape=(32, 32, 16, 120)):
        self.target_shape = target_shape
        self.model = None

    def load_model(self, model_path):
        """Load the pre-trained model"""
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_file(self, file_path):
        """Load and preprocess a single NIfTI file"""
        image = nib.load(file_path).get_fdata()
        zoom_factors = [t / s for t, s in zip(self.target_shape, image.shape)]
        return zoom(image, zoom_factors, order=1)

    def predict_and_verity(self, file_path):
        """Predict and output verdict with probability"""
        image = self.preprocess_file(file_path)
        prob = self.model.predict(np.expand_dims(image, axis=0))[0][0]
        verdict = "Schizophrenia" if prob >= 0.5 else "Healthy"
        probability = prob * 100
        return f"Input file: {file_path}\nPrediction: {verdict}, Probability: {probability:.2f}%"


def main():
    root_dir = Path(__file__).parent.parent.resolve()
    model_path = (
        root_dir / "model_three_depression" / "saved_models" / "depression_classifier.h5"
    )
    input_file_path = (
        root_dir / "model_three_depression" / "data" / "sub-102_task-rest_bold.nii"
    )

    classifier = MRIClassifier(target_shape=(32, 32, 16, 120))
    classifier.load_model(model_path)

    result = classifier.predict_and_verity(input_file_path)
    print(result)


if __name__ == "__main__":
    main()
