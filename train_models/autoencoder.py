import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config

def train_autoencoder(autoencoder, train_features, num_epochs=Config.AE_EPOCHS,batch_size=Config.AE_BATCH_SIZE,learning_rate=Config.AE_LEARNING_RATE,device=Config.DEVICE):
    """
    * Train autoencoder for dimensionality reduction
    
    Args:
        autoencoder: Autoencoder model
        train_features: Training features (numpy array)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Computation device
    
    Returns:
        Trained autoencoder model
    """
    print(f"\nTraining Autoencoder ({train_features.shape[1]}-dim → {autoencoder.encoder[-1].out_features}-dim)")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    
    features_tensor = torch.FloatTensor(train_features).to(device)
    dataset_ae = torch.utils.data.TensorDataset(features_tensor)
    loader_ae = DataLoader(dataset_ae, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    autoencoder.train()
    print("\nTraining...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in loader_ae:
            batch_features = batch[0]
            
            reconstructed, _ = autoencoder(batch_features)
            loss = criterion(reconstructed, batch_features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader_ae)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    print("\nAutoencoder training completed!")
    return autoencoder
