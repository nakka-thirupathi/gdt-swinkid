import numpy as np
import xgboost as xgb
import torch
from sklearn.preprocessing import StandardScaler

from config import Config


def train_xgboost(train_features, train_labels, xgb_params=None, device=Config.DEVICE):
    """
    * Train XGBoost classifier
    
    Args:
        train_features: Training features (numpy array)
        train_labels: Training labels (numpy array)
        xgb_params: XGBoost parameters (dict)
        device: Computation device
    
    Returns:
        Tuple of (trained_model, scaler)
    """
    print("\nTraining XGBoost Classifier")
    print("="*60)
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    
    # scale_pos_weight for imbalanced data
    neg_count = np.sum(train_labels == 0)
    pos_count = np.sum(train_labels == 1)
    scale_pos_weight = neg_count / pos_count
    
    print(f"Class balance: Pneumonia={neg_count}, Normal={pos_count}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    if xgb_params is None:
        xgb_params = Config.XGB_PARAMS.copy()
    
    xgb_params['scale_pos_weight'] = scale_pos_weight
    xgb_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\nTraining with optimized parameters...")
    xgb_classifier = xgb.XGBClassifier(**xgb_params)
    xgb_classifier.fit(train_features_scaled, train_labels, verbose=True)
    
    print("\nXGBoost training completed!")
    
    return xgb_classifier, scaler
