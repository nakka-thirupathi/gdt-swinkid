import numpy as np
import torch

from config import Config


def extract_features(model, dataloader, device=Config.DEVICE, desc="Extracting"):
    """
    * Extract features using CvT model
    
    Args:
        model: Feature extraction model
        dataloader: PyTorch DataLoader
        device: Computation device
        desc: Description for logging
    
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    model.eval()
    all_features = []
    all_labels = []
    
    print(f"\n{desc} features...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {(batch_idx + 1) * dataloader.batch_size} samples")
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"  Completed! Shape: {features.shape}")
    
    return features, labels


def encode_features(autoencoder, features, device=Config.DEVICE):
    """
    * Encode features to latent space using autoencoder
    
    Args:
        autoencoder: Trained autoencoder model
        features: Features to encode
        device: Computation device
    
    Returns:
        Encoded features as numpy array
    """
    autoencoder.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(device)
        encoded = autoencoder.encode(features_tensor).cpu().numpy()
    
    return encoded
