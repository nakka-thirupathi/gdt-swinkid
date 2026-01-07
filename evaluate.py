import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize

from config import Config


def calculate_metrics(y_true, y_pred, y_proba, dataset_name=""):
    """
    Calculate evaluation metrics for multi-class classification
    (Kidney CT: Normal / Cyst / Tumor / Stone)
    """

    acc = accuracy_score(y_true, y_pred) * 100

    prec = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    ) * 100

    rec = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    ) * 100

    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    ) * 100

    # Multi-class AUC (One-vs-Rest)
    y_true_bin = label_binarize(
        y_true, classes=list(range(Config.NUM_CLASSES))
    )

    auc_score = roc_auc_score(
        y_true_bin,
        y_proba,
        average="macro",
        multi_class="ovr"
    ) * 100

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy : {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall   : {rec:.2f}%")
    print(f"  F1-Score : {f1:.2f}%")
    print(f"  AUC      : {auc_score:.2f}%")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc_score,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "proba": y_proba
    }


def evaluate_model(classifier, scaler, test_features, test_labels, dataset_name="TEST"):
    """
    Evaluate trained classifier on test data
    """

    test_features_scaled = scaler.transform(test_features)
    test_pred = classifier.predict(test_features_scaled)
    test_proba = classifier.predict_proba(test_features_scaled)

    results = calculate_metrics(
        test_labels,
        test_pred,
        test_proba,
        dataset_name
    )

    return results


def print_classification_report(y_true, y_pred, labels=None):
    """
    Print detailed per-class report
    """
    if labels is None:
        labels = Config.LABELS

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0
        )
    )


def get_roc_data(y_true, y_proba, class_idx):
    """
    Get ROC curve data for a specific class (One-vs-Rest)
    """
    y_true_bin = (y_true == class_idx).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_proba[:, class_idx])
    return fpr, tpr, thresholds
