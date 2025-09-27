import os
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).parent.parent
dataset_path = PROJECT_ROOT / 'model_two_depression' / 'Data'
participants_file = os.path.join(dataset_path, 'participants.tsv')

participants_data = pd.read_csv(participants_file, sep='\t')

atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')

X = []
y = []

for subject_id in participants_data['participant_id']:
    func_dir = os.path.join(dataset_path, subject_id, 'func')
    if not os.path.exists(func_dir):
        continue

    subject_file = os.path.join(func_dir, f"{subject_id}_task-rest_bold.nii.gz")

    if not os.path.exists(subject_file):
        continue

    try:
        fmri_img = nib.load(subject_file)
        atlas_resampled = resample_to_img(atlas.maps, fmri_img, interpolation='nearest', force_resample=True,
                                          copy_header=True)
        masker = NiftiLabelsMasker(labels_img=atlas_resampled, standardize=True)
        time_series = masker.fit_transform(fmri_img)

        time_series_flat = time_series.mean(axis=0)

        X.append(time_series_flat)
        label = participants_data.loc[participants_data['participant_id'] == subject_id, 'group'].values[0]
        y.append(label)
    except Exception as e:
        continue

if len(X) == 0:
    raise ValueError("No valid fMRI data found!")

max_rois = max([x.shape[0] for x in X])
X_padded = []
for x in X:
    if x.shape[0] < max_rois:
        padded = np.pad(x, (0, max_rois - x.shape[0]), mode='constant')
        X_padded.append(padded)
    else:
        X_padded.append(x)

X = np.array(X_padded)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

logging.info(f"Data Shape: {X.shape}")
logging.info(f"Labels Shape: {y_encoded.shape}")
logging.info(f"Classes: {np.unique(y, return_counts=True)}")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")


def save_model_package(model, save_path, label_encoder):
    os.makedirs(save_path, exist_ok=True)
    joblib.dump({
        'model': model,
        'label_encoder': label_encoder
    }, os.path.join(save_path, 'depression_model.joblib'))
    logging.info(f"Model package saved to {save_path}")


save_path = 'model_two_depression/depression_model'
save_model_package(model, save_path, label_encoder)


def predict_on_files(model, file_paths, atlas, participants_data, label_encoder):
    predictions = []
    confidences = []
    true_labels = []

    for file_path in file_paths:
        try:
            fmri_img = nib.load(file_path)
            subject_id = os.path.basename(file_path).split('_')[0]

            atlas_resampled = resample_to_img(atlas.maps, fmri_img, interpolation='nearest', force_resample=True,
                                              copy_header=True)
            masker = NiftiLabelsMasker(labels_img=atlas_resampled, standardize=True)
            time_series = masker.fit_transform(fmri_img)

            time_series_flat = time_series.mean(axis=0).reshape(1, -1)

            prob = model.predict_proba(time_series_flat)[0]
            pred_class = model.predict(time_series_flat)[0]

            predictions.append(pred_class)
            confidences.append(prob[1])

            if subject_id in participants_data['participant_id'].values:
                true_label = participants_data.loc[participants_data['participant_id'] == subject_id, 'group'].values[0]
                true_labels.append(label_encoder.transform([true_label])[0])
            else:
                true_labels.append(None)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            predictions.append(None)
            confidences.append(None)
            true_labels.append(None)

    return predictions, confidences, true_labels


valid_subjects = [sid for sid in participants_data['participant_id']
                  if os.path.exists(os.path.join(dataset_path, sid, 'func', f"{sid}_task-rest_bold.nii.gz"))]

if len(valid_subjects) > 0:
    test_file = os.path.join(dataset_path, valid_subjects[0], 'func', f"{valid_subjects[0]}_task-rest_bold.nii.gz")
    if os.path.exists(test_file):
        preds, confs, true = predict_on_files(model, [test_file], atlas, participants_data, label_encoder)
        logging.info(f"Test prediction - File: {os.path.basename(test_file)}")
        logging.info(f"Prediction: {'Depressed' if preds[0] == 1 else 'Control'} (confidence: {confs[0]:.2f})")
        if true[0] is not None:
            logging.info(f"True Label: {label_encoder.inverse_transform([true[0]])[0]}")