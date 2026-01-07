import os
import torch
import warnings
import kagglehub
import numpy as np
import pandas as pd

from config import Config
from utils.save_load import save_models
from train_models.xgboost import train_xgboost
from train_models.autoencoder import train_autoencoder
from data.loader import prepare_data, create_dataloaders
from models.cax_models import load_cvt_model, load_autoencoder
from models.feature_extractor import extract_features, encode_features
from evaluate import calculate_metrics, print_classification_report
from utils.visualize import (
    plot_confusion_matrices,
    plot_performance_comparison,
    plot_roc_curve,
    plot_feature_importance,
    plot_sample_predictions
)

warnings.filterwarnings("ignore")

np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

print(f"Using device: {Config.DEVICE}")


def main():
    # === DATA LOADING (Kidney CT Dataset) === #
    print("\n" + "=" * 60)
    print("DATA LOADING : KIDNEY CT (NORMAL / CYST / TUMOR / STONE)")
    print("=" * 60)

    dataset_path = kagglehub.dataset_download(
        "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone"
    )

    x_train, y_train, x_test, y_test = prepare_data(dataset_path)
    train_loader, test_loader = create_dataloaders(
        x_train, y_train, x_test, y_test
    )

    # === FEATURE EXTRACTION (CvT) === #
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION WITH CvT")
    print("=" * 60)

    cvt_model = load_cvt_model(
        device=Config.DEVICE,
        pretrained=True
    )

    train_features, train_labels = extract_features(
        cvt_model, train_loader, Config.DEVICE, "Training"
    )
    test_features, test_labels = extract_features(
        cvt_model, test_loader, Config.DEVICE, "Testing"
    )

    print("\nExtracted Feature Shapes")
    print(f"  Train: {train_features.shape}")
    print(f"  Test : {test_features.shape}")

    # === AUTOENCODER (Feature Compression) === #
    print("\n" + "=" * 60)
    print(
        f"TRAINING AUTOENCODER ({train_features.shape[1]} → {Config.LATENT_DIM})"
    )
    print("=" * 60)

    input_dim = train_features.shape[1]
    autoencoder = load_autoencoder(
        input_dim,
        Config.LATENT_DIM,
        Config.DEVICE
    )

    autoencoder = train_autoencoder(
        autoencoder,
        train_features
    )

    # === LATENT SPACE ENCODING === #
    print("\n" + "=" * 60)
    print("ENCODING FEATURES INTO LATENT SPACE")
    print("=" * 60)

    train_encoded = encode_features(
        autoencoder, train_features, Config.DEVICE
    )
    test_encoded = encode_features(
        autoencoder, test_features, Config.DEVICE
    )

    print(f"Encoded Train Shape: {train_encoded.shape}")
    print(f"Encoded Test  Shape: {test_encoded.shape}")

    # === XGBOOST CLASSIFIER === #
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST CLASSIFIER (MULTI-CLASS)")
    print("=" * 60)

    xgb_classifier, scaler = train_xgboost(
        train_encoded,
        train_labels
    )

    # === EVALUATION === #
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    train_scaled = scaler.transform(train_encoded)
    test_scaled = scaler.transform(test_encoded)

    train_pred = xgb_classifier.predict(train_scaled)
    train_proba = xgb_classifier.predict_proba(train_scaled)

    test_pred = xgb_classifier.predict(test_scaled)
    test_proba = xgb_classifier.predict_proba(test_scaled)

    train_results = calculate_metrics(
        train_labels, train_pred, train_proba, "TRAIN"
    )
    test_results = calculate_metrics(
        test_labels, test_pred, test_proba, "TEST"
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("=" * 60)
    print_classification_report(test_labels, test_pred)

    # === VISUALIZATION === #
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    plot_confusion_matrices(
        train_results, test_results, save_path=Config.RESULTS_DIR
    )
    plot_performance_comparison(
        train_results, test_results, save_path=Config.RESULTS_DIR
    )
    plot_roc_curve(
        train_results,
        test_results,
        train_labels,
        test_labels,
        save_path=Config.RESULTS_DIR
    )
    plot_feature_importance(
        xgb_classifier, save_path=Config.RESULTS_DIR
    )
    plot_sample_predictions(
        x_test, test_labels, test_pred, save_path=Config.RESULTS_DIR
    )

    # === SAVE MODELS === #
    print("\n" + "=" * 60)
    print("SAVING TRAINED MODELS")
    print("=" * 60)

    save_models(
        cvt_model,
        autoencoder,
        xgb_classifier,
        scaler
    )

    # === FINAL SUMMARY === #
    print("\n" + "=" * 60)
    print("CAX-NET TRAINING COMPLETED (KIDNEY CT)")
    print("=" * 60)

    results_df = pd.DataFrame({
        "Dataset": ["Training", "Testing"],
        "Accuracy (%)": [
            train_results["accuracy"],
            test_results["accuracy"]
        ],
        "Precision (%)": [
            train_results["precision"],
            test_results["precision"]
        ],
        "Recall (%)": [
            train_results["recall"],
            test_results["recall"]
        ],
        "F1-Score (%)": [
            train_results["f1_score"],
            test_results["f1_score"]
        ],
        "AUC (%)": [
            train_results["auc"],
            test_results["auc"]
        ],
    })

    print("\n" + results_df.to_string(index=False))
    print("\n✓ Models saved to models/")
    print("✓ Results saved to results/")


if __name__ == "__main__":
    main()
