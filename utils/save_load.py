"""
Model saving and loading utilities
"""
import os
import torch
import joblib
from config import Config


def save_models(cvt_model, autoencoder, xgb_classifier, scaler, 
                save_dir=Config.MODELS_DIR):
    """
    Save all trained models
    
    Args:
        cvt_model: CvT feature extractor
        autoencoder: Trained autoencoder
        xgb_classifier: Trained XGBoost classifier
        scaler: Fitted StandardScaler
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save PyTorch models
    torch.save(cvt_model.state_dict(), os.path.join(save_dir, 'cvt_model.pth'))
    torch.save(autoencoder.state_dict(), os.path.join(save_dir, 'autoencoder.pth'))
    
    # Save XGBoost and scaler
    xgb_classifier.save_model(os.path.join(save_dir, 'xgb_classifier.json'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    
    print(f"\nModels saved to {save_dir}/")
    print("  - cvt_model.pth")
    print("  - autoencoder.pth")
    print("  - xgb_classifier.json")
    print("  - scaler.pkl")


def load_models(cvt_model, autoencoder, load_dir=Config.MODELS_DIR, device=Config.DEVICE):
    """
    Load trained models
    
    Args:
        cvt_model: CvT model instance (architecture)
        autoencoder: Autoencoder model instance (architecture)
        load_dir: Directory containing saved models
        device: Device to load models to
    
    Returns:
        Tuple of (cvt_model, autoencoder, xgb_classifier, scaler)
    """
    import xgboost as xgb
    
    # Load PyTorch models
    cvt_model.load_state_dict(torch.load(os.path.join(load_dir, 'cvt_model.pth'), 
                                         map_location=device))
    autoencoder.load_state_dict(torch.load(os.path.join(load_dir, 'autoencoder.pth'),
                                           map_location=device))
    
    # Load XGBoost and scaler
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.load_model(os.path.join(load_dir, 'xgb_classifier.json'))
    scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
    
    cvt_model.eval()
    autoencoder.eval()
    
    print(f"\nModels loaded from {load_dir}/")
    
    return cvt_model, autoencoder, xgb_classifier, scaler


def save_checkpoint(epoch, model, optimizer, loss, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device=Config.DEVICE):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: {filepath} (Epoch {epoch}, Loss {loss:.6f})")
    
    return model, optimizer, epoch, loss