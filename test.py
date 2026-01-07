import os
import warnings
import kagglehub
import numpy as np
import pandas as pd
import torch

from config import Config
from data.loader import prepare_data, create_dataloaders
from models.cax_models import load_cvt_model, load_autoencoder
from models.feature_extractor import extract_features, encode_features
from evaluate import calculate_metrics, print_classification_report
from utils.visualize import (
    plot_sample_predictions
)
from utils.save_load import load_models

warnings.filterwarnings("ignore")

np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

print(f"Using device: {Config.DEVICE}")


def main():
    """Testing / inference pipeline for Kidney CT classification"""

    # === DATA LOADING (Kidney CT) === #
    print("\n" + "=" * 60)
    print("LOADING TEST DATA : KIDNEY CT")
    print("=" * 60)

    dataset_path = kagglehub.dataset_download(
        "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone"
    )

    x_train, y_train, x_test, y_test = prepare_data(dataset_path)
    _, test_loader = create_dataloaders(
        x_train, y_train, x_test, y_test
    )

    # === LOAD TRAINED MODELS === #
    print("\n" + "=" * 60)
    print("LOADING TRAINED MODELS")
    print("=" * 60)

    cvt_model = load_cvt_model(
        device=Config.DEVICE,
        pretrained=False
    )

    autoencoder = load_autoencoder(
        Config.INPUT_DIM,
        Config.LATENT_DIM,
        Config.DEVICE
    )

    cvt_model, autoencoder, xgb_classifier, scaler = load_models(
        cvt_model,
        autoencoder,
        load_dir=Config.MODELS_DIR,
        device=Config.DEVICE
    )

    # === FEATURE EXTRACTION === #
    print("\n" + "=" * 60)
    print("EXTRACTING FEATURES (CvT)")
    print("=" * 60)

    test_features, test_labels = extract_features(
        cvt_model,
        test_loader,
        Config.DEVICE,
        "Testing"
    )

    print(f"Extracted Features Shape: {test_features.shape}")

    # === FEATURE ENCODING === #
    print("\n" + "=" * 60)
    print("ENCODING FEATURES (AUTOENCODER)")
    print("=" * 60)

    test_encoded = encode_features(
        autoencoder,
        test_features,
        Config.DEVICE
    )

    print(f"Encoded Features Shape: {test_encoded.shape}")

    test_encoded_scaled = scaler.transform(test_encoded)

    # === INFERENCE === #
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE (XGBOOST)")
    print("=" * 60)

    test_pred = xgb_classifier.predict(test_encoded_scaled)
    test_proba = xgb_classifier.predict_proba(test_encoded_scaled)

    test_results = calculate_metrics(
        test_labels,
        test_pred,
        test_proba,
        "TEST SET"
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("=" * 60)
    print_classification_report(test_labels, test_pred)

    # === VISUALIZATION === #
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    results_dir = os.path.join(Config.RESULTS_DIR, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    plot_sample_predictions(
        x_test,
        test_labels,
        test_pred,
        save_path=results_dir
    )

    # === SUMMARY === #
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED (KIDNEY CT)")
    print("=" * 60)

    print(f"Accuracy : {test_results['accuracy']:.2f}%")
    print(f"Precision: {test_results['precision']:.2f}%")
    print(f"Recall   : {test_results['recall']:.2f}%")
    print(f"F1-Score : {test_results['f1_score']:.2f}%")
    print(f"AUC      : {test_results['auc']:.2f}%")

    # === SAVE PREDICTIONS === #
    predictions_df = pd.DataFrame({
        "True_Label": test_labels,
        "Predicted_Label": test_pred,
        "Prob_Normal": test_proba[:, 0],
        "Prob_Cyst": test_proba[:, 1],
        "Prob_Tumor": test_proba[:, 2],
        "Prob_Stone": test_proba[:, 3],
    })

    predictions_path = os.path.join(
        results_dir, "predictions.csv"
    )
    predictions_df.to_csv(predictions_path, index=False)

    print(f"\n✓ Predictions saved to {predictions_path}")
    print(f"✓ Visualizations saved to {results_dir}/")


if __name__ == "__main__":
    main()
