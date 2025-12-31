from __future__ import annotations
import pandas as pd
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

def run_pca(df: pd.DataFrame, n_components: int) -> Tuple[pd.DataFrame, PCA]:    
    """
    Function for performing PCA on the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input scaled and transformed DataFrame with omics features.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    Tuple[pd.DataFrame, PCA]
        A tuple containing:
        - pd.DataFrame: Transformed DataFrame with principal components.
        - PCA: Fitted PCA object.
    """
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(df.values)
    columns = [f"PC{i+1}" for i in range(transformed.shape[1])]
    df_pca = pd.DataFrame(transformed, index=df.index, columns=columns)
    return df_pca, pca


class Autoencoder(nn.Module):

    """
    Autoencoder model for dimensionality reduction.
    Args:
        input_dim (int): Dimension of input features.
        latent_dim (int): Dimension of latent space (default is 10).
    Returns:
        torch.Tensor: Reconstructed input.
        torch.Tensor: Latent representation.
    """

    def __init__(self, input_dim, latent_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)  # latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent

def train_autoencoder(df_clean, epochs=50, batch_size=256, latent_dim=10, patience=5, lr=1e-3, verbose=True):
    """
    Train Autoencoder and return latent features.
    
    Args:
        df_clean: pandas DataFrame or numpy array (n_samples, n_features)
        epochs: Maximum training epochs
        batch_size: Training batch size
        latent_dim: Output latent space dimension
        patience: Early stopping patience
        lr: Learning rate
    
    Returns:
        numpy array: Latent features (n_samples, latent_dim)
        list: Training history with losses per epoch
    """
    # Convert to tensor
    X_tensor = torch.tensor(
        df_clean.values if hasattr(df_clean, 'values') else df_clean,                               # Handle both DataFrame and ndarray
        dtype=torch.float32
    )
    
    # Train/validation split
    X_train, X_val = train_test_split(X_tensor, test_size=0.1, random_state=42)
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, X_train), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=batch_size)
    
    # Model setup
    input_dim = X_tensor.shape[1]
    model = Autoencoder(input_dim, latent_dim)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Early stopping variables
    best_val_loss = np.inf
    counter = 0
    best_model_state = None
    training_history = []  # Collect losses for progress
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, _ in val_loader:
                val_out, _ = model(val_x)
                val_loss += criterion(val_out, val_x).item()
        avg_val_loss = val_loss / len(val_loader)
        
        if verbose:
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            # Progress update (called by Streamlit)
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Load best model and extract features
    if best_model_state:
        model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        _, X_latent = model(X_tensor)
        return X_latent.numpy(), training_history