import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv3D,
    MaxPooling3D,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import joblib
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MRIClassifier:
    def __init__(self, target_shape=(32, 32, 16, 120)):
        self.target_shape = target_shape
        self.model = None
        self.class_names = ["Healthy", "Diseased"]
        self.data_dir = None

    def set_data_directory(self, data_dir):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_file_paths_and_labels(self):
        if self.data_dir is None:
            raise ValueError("Data directory not set. Call set_data_directory() first.")
        file_paths, labels = [], []
        for file_path in self.data_dir.glob("*"):
            if file_path.suffix in [".nii", ".gz"] or ".nii.gz" in file_path.name:
                file_paths.append(file_path)
                if "sub-" in file_path.name:
                    try:
                        subject_number = int(file_path.name.split("-")[1].split("_")[0])
                        if 301 <= subject_number <= 312:
                            labels.append(0)
                        elif 101 <= subject_number <= 125:
                            labels.append(1)
                        else:
                            logging.warning(f"Subject {subject_number} not categorized")
                    except (IndexError, ValueError) as e:
                        logging.warning(
                            f"Could not parse subject number from {file_path.name}: {e}"
                        )
        if not file_paths:
            raise FileNotFoundError(f"No valid NIfTI files found in {self.data_dir}")
        return file_paths, np.array(labels)

    def resample_image(self, image):
        zoom_factors = [t / s for t, s in zip(self.target_shape, image.shape)]
        return zoom(image, zoom_factors, order=1)

    def preprocess_file(self, file_path):
        image = nib.load(file_path).get_fdata()
        return self.resample_image(image)

    def build_model(self):
        model = Sequential(
            [
                Conv3D(
                    16,
                    kernel_size=(3, 3, 3),
                    activation="relu",
                    input_shape=self.target_shape,
                ),
                MaxPooling3D(pool_size=(2, 2, 2)),
                BatchNormalization(),
                Dropout(0.25),
                Conv3D(32, kernel_size=(3, 3, 3), activation="relu"),
                MaxPooling3D(pool_size=(2, 2, 2)),
                BatchNormalization(),
                Dropout(0.25),
                Flatten(),
                Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        logging.info(f"Model loaded from {model_path}")
        return self

    def load_weights(self, weights_path):
        if self.model is None:
            self.build_model()
        self.model.load_weights(weights_path)
        logging.info(f"Weights loaded from {weights_path}")
        return self

    def train_and_evaluate(self, epochs=50, batch_size=8, test_size=0.2):
        file_paths, labels = self.load_file_paths_and_labels()
        train_files, test_files, y_train, y_test = train_test_split(
            file_paths, labels, test_size=test_size, random_state=42, stratify=labels
        )
        train_data = self.create_dataset(train_files, y_train, batch_size)
        test_data = self.create_dataset(test_files, y_test, batch_size)
        self.model = self.build_model()
        logging.info("Starting training...")
        history = self.model.fit(
            train_data,
            steps_per_epoch=len(train_files) // batch_size,
            validation_data=test_data,
            validation_steps=len(test_files) // batch_size,
            epochs=epochs,
            verbose=1,
        )
        logging.info("Evaluating on test set...")
        test_results = self.evaluate(test_files, y_test, batch_size)
        return history, test_results

    def create_dataset(self, file_paths, labels, batch_size):
        num_samples = len(file_paths)
        while True:
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]
                batch_X = []
                batch_y = []
                for i in batch_indices:
                    image = self.preprocess_file(file_paths[i])
                    batch_X.append(image)
                    batch_y.append(labels[i])
                yield np.array(batch_X), np.array(batch_y)

    def evaluate(self, file_paths, labels, batch_size):
        X = np.array([self.preprocess_file(str(fp)) for fp in file_paths])
        y = np.array(labels)
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        y_pred = self.model.predict(X)
        y_pred_class = (y_pred > 0.5).astype(int)
        return {
            "loss": loss,
            "accuracy": accuracy,
            "predictions": y_pred_class,
            "true_labels": y,
            "probabilities": y_pred,
        }

    def predict_and_verity(self, file_path):
        image = self.preprocess_file(file_path)
        prob = self.model.predict(np.expand_dims(image, axis=0))[0][0]
        if prob >= 0.5:
            return f"Prediction: Schizophrenia, Schizophrenia Probability: {prob * 100:.2f}%"
        else:
            return f"Prediction: Healthy, Healthy Probability: {(1 - prob) * 100:.2f}%"

    def save_model(self, output_dir, model_name="depression_classifier"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_name}.h5"
        config_path = output_dir / f"{model_name}.joblib"
        self.model.save(model_path)
        joblib.dump(
            {"target_shape": self.target_shape, "class_names": self.class_names},
            config_path,
        )
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Config saved to {config_path}")
        return str(model_path), str(config_path)


def main():
    classifier = MRIClassifier(target_shape=(32, 32, 16, 120))
    root_dir = Path(__file__).parent.parent.resolve()
    data_dir = root_dir / "model_three_depression" / "data"
    classifier.set_data_directory(data_dir)
    history, test_results = classifier.train_and_evaluate(
        epochs=5, batch_size=2, test_size=0.2
    )
    print("\nEvaluation Results:")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    output_dir = root_dir / "model_three_depression" / "saved_models"
    model_path, config_path = classifier.save_model(output_dir)
    print(f"\nModel saved to: {model_path}")
    print(f"Config saved to: {config_path}")

    test_file = str(data_dir / "sub-101_task-rest_bold.nii")
    print(classifier.predict_and_verity(test_file))


if __name__ == "__main__":
    main()
