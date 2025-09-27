import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import logging
from sklearn.impute import SimpleImputer
from nilearn import input_data, datasets
import random
import joblib
from pathlib import Path
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_abide_dataset(data_dir):
    """
    Load and preprocess the ABIDE dataset to extract functional connectivity features.
    Args:
        data_dir (str): Directory path to download/load ABIDE dataset.
    Returns:
        tuple: Feature matrix (X), labels (y), and fitted NiftiMasker object.
    """
    logging.info("Loading and preprocessing ABIDE dataset...")
    logging.info(f"Fetching dataset from {data_dir}...")
    abide = datasets.fetch_abide_pcp(data_dir=data_dir, pipeline='cpac', derivatives=['func_preproc'])
    logging.info(f"Type of phenotypic data: {type(abide.phenotypic)}")
    logging.info(f"Columns in phenotypic data: {abide.phenotypic.columns}")
    logging.info(f"First entry of phenotypic data:\n{abide.phenotypic.iloc[0]}")

    masker = input_data.NiftiMasker(standardize=True, mask_strategy='epi')
    X = []
    y = []
    total_files = len(abide.func_preproc)
    logging.info(f"Found {total_files} functional files to process.")

    max_features = 0

    for i, (func_file, phenotypic) in enumerate(zip(abide.func_preproc, abide.phenotypic.iloc)):
        try:
            logging.info(f"Processing file {i + 1}/{total_files}: {func_file}")
            logging.info(f"Phenotypic data for file {func_file}:\n{phenotypic}")

            time_series = masker.fit_transform(func_file)
            logging.info(f"Time series shape: {time_series.shape}")

            correlation_matrix = np.corrcoef(time_series.T)
            logging.info(f"Correlation matrix shape: {correlation_matrix.shape}")

            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            logging.info(f"Feature length for file {func_file}: {len(upper_triangle)}")
            X.append(upper_triangle)
            max_features = max(max_features, len(upper_triangle))

            if 'DX_GROUP' in phenotypic:
                y.append(phenotypic['DX_GROUP'])
            else:
                logging.warning(f"Phenotypic data for file {func_file} does not contain 'DX_GROUP'. Skipping this file.")
                continue
        except Exception as e:
            logging.error(f"Error processing file {func_file}: {e}")
            continue

    logging.info(f"Normalizing feature lengths to {max_features}...")
    X = [np.pad(features, (0, max_features - len(features))) for features in X]
    X = np.array(X)
    y = np.array(y) - 1
    logging.info(f"Successfully preprocessed dataset with {len(X)} samples.")
    return X, y, masker


def augment_data(X, y, num_samples=1000):
    """
    Generate synthetic samples by linear interpolation between random pairs of existing samples.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        num_samples (int): Number of synthetic samples to generate.
    Returns:
        tuple: Synthetic feature matrix and labels.
    """
    logging.info(f"Generating {num_samples} synthetic samples...")
    synthetic_X, synthetic_y = [], []
    for _ in range(num_samples):
        idx1, idx2 = random.sample(range(len(X)), 2)
        alpha = np.random.rand()
        new_sample = alpha * X[idx1] + (1 - alpha) * X[idx2]
        synthetic_X.append(new_sample)
        synthetic_y.append(y[idx1])
    logging.info("Synthetic data generation complete.")
    return np.array(synthetic_X), np.array(synthetic_y)


def save_model(model, pca, imputer, masker, filename='model.joblib'):
    """
    Save the trained model and preprocessing objects to disk.
    Args:
        model: Trained classifier model.
        pca: PCA object used for dimensionality reduction.
        imputer: Imputer object for missing value handling.
        masker: NiftiMasker object used for fMRI preprocessing.
        filename (str): Path to save the model.
    """
    logging.info(f"Saving model and preprocessing objects to {filename}...")
    joblib.dump({'model': model, 'pca': pca, 'imputer': imputer, 'masker': masker}, filename)
    logging.info("Model and preprocessing objects saved.")


def load_model(filename='model.joblib'):
    """
    Load the trained model and preprocessing objects from disk.
    Args:
        filename (str): Path to load the model from.
    Returns:
        tuple: model, pca, imputer, masker objects.
    """
    logging.info(f"Loading model and preprocessing objects from {filename}...")
    data = joblib.load(filename)
    return data['model'], data['pca'], data['imputer'], data['masker']


def preprocess_single_file(nii_file_path, masker, imputer, pca):
    """
    Preprocess a single NIfTI fMRI file for model prediction.
    Args:
        nii_file_path (str): Path to the .nii.gz file.
        masker: Pre-fitted NiftiMasker object.
        imputer: Pre-fitted SimpleImputer object.
        pca: Pre-fitted PCA object.
    Returns:
        np.ndarray or None: Preprocessed feature vector after PCA or None on failure.
    """
    logging.info(f"Preprocessing single file: {nii_file_path}")
    try:
        if not Path(nii_file_path).exists():
            logging.error(f"File not found at {nii_file_path}.")
            return None

        import nibabel as nib
        try:
            img = nib.load(nii_file_path)
            logging.info(f"NIfTI file loaded successfully. Shape: {img.shape}")
        except Exception as e:
            logging.error(f"Error loading NIfTI file with nibabel: {e}")
            return None

        time_series = masker.transform(nii_file_path)
        logging.info(f"Time series shape: {time_series.shape}")

        correlation_matrix = np.corrcoef(time_series.T)
        logging.info(f"Correlation matrix shape: {correlation_matrix.shape}")

        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        logging.info(f"Feature length: {len(upper_triangle)}")

        max_features = pca.n_features_in_
        features = np.pad(upper_triangle, (0, max_features - len(upper_triangle)))

        features = imputer.transform([features])
        features_encoded = pca.transform(features)
        return features_encoded
    except Exception as e:
        logging.error(f"Error preprocessing file {nii_file_path}: {e}")
        return None


def save_predictions(nii_file_path, features, prediction, prediction_proba, output_file='predictions.csv'):
    """
    Save prediction results along with features and file path to a CSV file.
    Args:
        nii_file_path (str): Input file path.
        features (np.ndarray): Feature vector used for prediction.
        prediction (np.ndarray): Predicted class label.
        prediction_proba (np.ndarray): Prediction probabilities.
        output_file (str): CSV filename to append/save the results.
    """
    try:
        data = {
            'file_path': [nii_file_path],
            'features': [features.tolist()],
            'prediction': [prediction[0]],
            'confidence_asd': [prediction_proba[1]],
            'confidence_no_asd': [prediction_proba[0]]
        }
        df = pd.DataFrame(data)
        if Path(output_file).exists():
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, index=False)
        logging.info(f"Prediction saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving prediction: {e}")


def predict_single_file(nii_file_path, model, pca, imputer, masker, save_to_file=True):
    """
    Preprocess a single .nii.gz file and predict ASD diagnosis.
    Optionally saves the prediction to a CSV file.
    Args:
        nii_file_path (str): Path to the input .nii.gz file.
        model: Trained classifier.
        pca: PCA object.
        imputer: Imputer object.
        masker: NiftiMasker object.
        save_to_file (bool): Whether to save prediction results.
    Returns:
        str: Prediction message or error.
    """
    logging.info(f"Predicting for single file: {nii_file_path}")
    try:
        if not Path(nii_file_path).exists():
            logging.error(f"File not found at {nii_file_path}.")
            return "Error: File not found."

        preprocessed_data = preprocess_single_file(nii_file_path, masker, imputer, pca)
        if preprocessed_data is None:
            return "Error: Unable to preprocess the file."

        prediction = model.predict(preprocessed_data)
        prediction_proba = model.predict_proba(preprocessed_data)[0]

        if save_to_file:
            save_predictions(nii_file_path, preprocessed_data, prediction, prediction_proba)

        if prediction[0] == 1:
            return f"Patient likely has ASD (Confidence: {prediction_proba[1] * 100:.2f}%)"
        else:
            return f"Patient does not have ASD (Confidence: {prediction_proba[0] * 100:.2f}%)"
    except Exception as e:
        logging.error(f"Error predicting for file {nii_file_path}: {e}")
        return "Error: Unable to make a prediction."


def load_predictions(prediction_file='predictions.csv'):
    """
    Load saved predictions from CSV file.
    Args:
        prediction_file (str): Path to the CSV file.
    Returns:
        pd.DataFrame or None: DataFrame with predictions or None on failure.
    """
    try:
        df = pd.read_csv(prediction_file)
        logging.info(f"Loaded {len(df)} predictions from {prediction_file}")
        return df
    except Exception as e:
        logging.error(f"Error loading predictions: {e}")
        return None


def retrain_model_with_predictions(X_train, y_train, prediction_file='predictions.csv'):
    """
    Augment training data with saved predictions for retraining.
    Args:
        X_train (np.ndarray): Original training features.
        y_train (np.ndarray): Original training labels.
        prediction_file (str): CSV file containing saved predictions.
    Returns:
        tuple: Combined features and labels.
    """
    predictions_df = load_predictions(prediction_file)
    if predictions_df is None:
        logging.warning("No predictions found. Retraining with original dataset only.")
        return X_train, y_train

    X_new = np.array([eval(features) for features in predictions_df['features']])
    y_new = predictions_df['prediction'].values

    X_train_combined = np.vstack((X_train, X_new))
    y_train_combined = np.hstack((y_train, y_new))

    logging.info(f"Combined dataset shape: X_train = {X_train_combined.shape}, y_train = {y_train_combined.shape}")
    return X_train_combined, y_train_combined


def main():
    """
    Main training and evaluation routine.
    Loads ABIDE dataset, preprocesses data, performs data augmentation,
    trains MLP classifier with PCA, and saves the model.
    """
    logging.info("Starting training script...")

    data_dir = './abide'
    X, y, masker = preprocess_abide_dataset(data_dir)
    latent_dim = 5
    logging.info(f"Dataset shape: X = {X.shape}, y = {y.shape}")

    logging.info("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    logging.info(f"Train set shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
    logging.info(f"Test set shape: X_test = {X_test.shape}, y_test = {y_test.shape}")

    X_train_combined, y_train_combined = retrain_model_with_predictions(X_train, y_train)

    synthetic_X, synthetic_y = augment_data(X_train_combined, y_train_combined, num_samples=len(X_train_combined))
    X_train_aug = np.vstack((X_train_combined, synthetic_X))
    y_train_aug = np.hstack((y_train_combined, synthetic_y))
    logging.info(f"Augmented train set shape: X_train_aug = {X_train_aug.shape}, y_train_aug = {y_train_aug.shape}")

    logging.info("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_train_aug = imputer.fit_transform(X_train_aug)
    X_test = imputer.transform(X_test)

    logging.info(f"Reducing to {latent_dim} components with PCA...")
    pca = PCA(n_components=latent_dim)
    X_train_encoded = pca.fit_transform(X_train_aug)
    X_test_encoded = pca.transform(X_test)
    logging.info(f"Encoded train set shape: X_train_encoded = {X_train_encoded.shape}")
    logging.info(f"Encoded test set shape: X_test_encoded = {X_test_encoded.shape}")

    logging.info("Building and training SLP classifier...")
    slp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, activation='relu',
                        solver='adam', alpha=0.01, random_state=42, early_stopping=True)

    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(slp, X_train_encoded, y_train_aug, cv=5)
    logging.info(f"Cross-validation scores: {cv_scores}")
    logging.info(f"Mean CV Accuracy: {np.mean(cv_scores) * 100:.2f}%")

    slp.fit(X_train_encoded, y_train_aug)
    logging.info("SLP classifier training complete.")

    y_pred = slp.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    logging.info("Classification Report:")
    print(classification_report(y_test, y_pred))

    save_model(slp, pca, imputer, masker, filename='model_one_asd.joblib')

    nii_file_path = 'model_one_asd/abide/ABIDE_pcp/cpac/nofilt_noglobal/Caltech_0051461_func_preproc.nii.gz'
    result = predict_single_file(nii_file_path, slp, pca, imputer, masker)
    print(result)

    logging.info("Training script execution complete.")


if __name__ == "__main__":
    main()
