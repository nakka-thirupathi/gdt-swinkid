import timm
import torch
import torch.nn as nn

from config import Config


class CvTFeatureExtractor(nn.Module):
    """
    * Convolutional Vision Transformer for feature extraction
    """
    
    def __init__(self, pretrained=True, model_name='vit_base_patch16_224'):
        super(CvTFeatureExtractor, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0
        )
        self.feature_dim = self.model.num_features
        print(f"ViT loaded with feature dimension: {self.feature_dim}")
    
    def forward(self, x):
        return self.model(x)


class AutoencoderADE(nn.Module):
    """
    * Autoencoder-Decoder Embedding for dimensionality reduction
    """
    
    def __init__(self, input_dim=Config.INPUT_DIM, latent_dim=Config.LATENT_DIM):
        super(AutoencoderADE, self).__init__()
        
        # encoder: input_dim → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, latent_dim)
        )
        
        # decoder: latent_dim → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


def load_cvt_model(device=Config.DEVICE, pretrained=True):
    """
    * Load and return CvT feature extractor
    """
    model = CvTFeatureExtractor(pretrained=pretrained).to(device)
    model.eval()
    return model


def load_autoencoder(input_dim, latent_dim, device=Config.DEVICE):
    """
    * Load and return Autoencoder
    """
    model = AutoencoderADE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    return model
