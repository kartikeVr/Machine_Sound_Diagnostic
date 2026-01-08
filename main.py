import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import matplotlib.pyplot as plt 
import numpy as np
import os
import esc50_loader 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class autoEncoder(nn.Module):
    def __init__(self):
        super(autoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # Layer 1: 1 -> 16
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: [16, 32, 156]

            # Layer 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: [32, 16, 78]

            # Layer 3: 32 -> 64 (Bottleneck starts tightening)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: [64, 8, 39]  <-- LATENT REPRESENTATION
        )

        # --- DECODER (Reconstructing) ---
        # We must reverse the Encoder steps exactly
        self.decoder = nn.Sequential(
            # Layer 3 Reverse: 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            
            # Layer 2 Reverse: 32 -> 16
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),

            # Layer 1 Reverse: 16 -> 1
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid() # Output must be 0-1 range to match input
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if decoded.shape !=x.shape:
            decoded = torch.nn.functional.interpolate(decoded, size=x.shape[2:])
        
        return decoded

def evaluate_model(model, test_loader):
    criterion = nn.MSELoss()
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for spectrogram in test_loader:
            spectrogram = spectrogram.to(device)
            outputs = model(spectrogram)
            loss = criterion(outputs, spectrogram)
            running_loss += loss.item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = (1 - np.sqrt(avg_loss)) * 100
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Pre-computed spectrograms are loaded directly
    
    best_accuracy = -float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, spectrogram in enumerate(train_loader):
            spectrogram = spectrogram.to(device)
            
            # spectrogram is already processed (log-scaled, normalized)
            # Shape: (Batch, 1, n_mels, time)

            optimizer.zero_grad()
            outputs = model(spectrogram)
            loss = criterion(outputs, spectrogram)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = (1 - np.sqrt(epoch_loss)) * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, test_loader)
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_autoencoder.pth')
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    # Select classes - Train ONLY on 'Normal' class (e.g., engine)
    target_classes = ["engine"]
    print(f"Loading data for Normal class: {target_classes}...")
    
    try:
        train_loader, test_loader = esc50_loader.get_machine_dataloaders(
            target_classes, 
            batch_size=16,
            root_path="." 
        )
        
        print("Initializing model...")
        model = autoEncoder().to(device)
        
        print(f"Starting training on device: {device}")
        train_model(model, train_loader, test_loader, num_epochs=15)
        
        print("Training complete.")
        print("Evaluating on test set...")
        evaluate_model(model, test_loader)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    