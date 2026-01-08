import torch
import matplotlib.pyplot as plt
import os
import argparse
import torchaudio
import esc50_loader
from main import autoEncoder  # Import the model class

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_best_model(model_path='best_autoencoder.pth'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Train the model first.")
    
    model = autoEncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    return model

def preprocess_audio(file_path, target_sr=16000, fixed_length=16000*5):
    """
    Preprocess a single audio file to match the training data format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        waveform, sr = torchaudio.load(file_path)
        
        # Resample
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        
        # Pad / Crop to fixed length
        if waveform.shape[1] < fixed_length:
            pad_len = fixed_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :fixed_length]
            
        waveform = waveform.to(device)

        # Generate Spectrogram matches generate_spectrograms.py
        spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_mels=64
        ).to(device)

        spectrogram = spec_transform(waveform) # Shape: (1, n_mels, time)
        
        # Log transform
        spectrogram = torch.log(spectrogram + 1e-9)
        
        # Normalize to [0, 1] per sample
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        if spec_max - spec_min > 1e-9:
            spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
        else:
            spectrogram = spectrogram - spec_min

        # Add batch and channel dimensions if needed
        # Current shape is (1, 64, time) or (Channels, 64, time). 
        # Model expects (Batch, 1, 64, time)
        
        # If waveform was mono, shape is (1, ...). If stereo, likely (2, ...).
        # We only want one channel.
        if spectrogram.shape[0] > 1:
             spectrogram = spectrogram[0:1, :, :]
             
        # Add batch dimension -> (1, 1, 64, time)
        spectrogram = spectrogram.unsqueeze(0)
        
        return spectrogram

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_anomaly(model, file_path, threshold=0.035):
    """
    Predict if a file is an anomaly.
    Threshold is a heuristic MSE loss value. 
    Lower loss = Normal. Higher loss = Anomaly.
    """
    spectrogram = preprocess_audio(file_path)
    if spectrogram is None:
        return

    with torch.no_grad():
        reconstruction = model(spectrogram)
        loss = torch.nn.functional.mse_loss(reconstruction, spectrogram)
        loss_val = loss.item()
        accuracy = (1 - (loss_val**0.5)) * 100
        
    print(f"\n--- Prediction Result for {os.path.basename(file_path)} ---")
    print(f"Reconstruction Loss: {loss_val:.4f}")
    print(f"Reconstruction Accuracy: {accuracy:.2f}%")
    
    if loss_val > threshold:
        print(f"Result: ANOMALY DETECTED (Loss > {threshold})")
    else:
        print(f"Result: NORMAL (Loss <= {threshold})")

def visualize_reconstruction(model, test_loader, num_samples=3):
    model.eval()
    with torch.no_grad():
        # Get a single batch
        data_iter = iter(test_loader)
        images = next(data_iter)
        images = images.to(device)
        
        # Reconstruct
        outputs = model(images)
        
        # Calculate individual losses for display
        mse_loss = torch.nn.functional.mse_loss(outputs, images, reduction='none')
        # mse_loss shape: (batch, 1, h, w). Mean over dimensions 1,2,3
        sample_losses = mse_loss.view(images.size(0), -1).mean(dim=1)

        # Plot
        images = images.cpu().numpy()
        outputs = outputs.cpu().numpy()
        
        # Limit samples
        num_samples = min(num_samples, images.shape[0])
        
        fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(4 * num_samples, 6))
        
        for i in range(num_samples):
            # Original
            ax = axes[0, i] if num_samples > 1 else axes[0]
            ax.imshow(images[i][0], cmap='inferno', origin='lower')
            ax.set_title("Original")
            ax.axis('off')
            
            # Reconstructed
            ax = axes[1, i] if num_samples > 1 else axes[1]
            ax.imshow(outputs[i][0], cmap='inferno', origin='lower')
            loss_val = sample_losses[i].item()
            acc_val = (1 - (loss_val**0.5)) * 100
            ax.set_title(f"Reconstructed\nLoss: {loss_val:.4f} | Acc: {acc_val:.1f}%")
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig('reconstruction_results.png')
        print("Saved visualization to reconstruction_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fault Sound Classifier Inference")
    parser.add_argument('--file', type=str, help="Path to a single .wav file to test")
    args = parser.parse_args()

    # 1. Load Model
    try:
        model = load_best_model()
    except Exception as e:
        print(e)
        exit(1)

    if args.file:
        # Single File Inference
        predict_anomaly(model, args.file)
    else:
        # Existing Batch Evaluation Logic
        
        # Define Normal (trained) vs Anomaly classes
        normal_class = ["engine","chainsaw","keyboard_typing","mouse_click","vacuum_cleaner","washing_machine"]
        anomaly_class = ["footsteps"] # A fault sound or completely different sound
        
        try:
            # 2. Evaluate on Normal Data
            print(f"\n--- Evaluating on NORMAL data {normal_class} ---")
            _, normal_loader = esc50_loader.get_machine_dataloaders(
                normal_class, 
                batch_size=8,
                root_path="." 
            )
            
            # We can calculate average loss/accuracy for the whole normal set
            normal_losses = []
            model.eval()
            criterion = torch.nn.MSELoss()
            with torch.no_grad():
                for batch in normal_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    loss = criterion(out, batch)
                    normal_losses.append(loss.item())
            
            avg_normal_loss = sum(normal_losses) / len(normal_losses)
            avg_normal_acc = (1 - (avg_normal_loss**0.5)) * 100
            print(f"Normal Average Loss: {avg_normal_loss:.4f}")
            print(f"Normal Average Accuracy: {avg_normal_acc:.2f}%")


            # 3. Evaluate on Anomaly Data
            print(f"\n--- Evaluating on ANOMALY data {anomaly_class} ---")
            # Note: We use the 'train' loader or 'test' loader for anomaly, 
            # doesn't matter since the model has never seen this class.
            # We'll use the test split for consistency.
            _, anomaly_loader = esc50_loader.get_machine_dataloaders(
                anomaly_class, 
                batch_size=8,
                root_path="." 
            )
            
            anomaly_losses = []
            with torch.no_grad():
                for batch in anomaly_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    loss = criterion(out, batch)
                    anomaly_losses.append(loss.item())
                    
            avg_anomaly_loss = sum(anomaly_losses) / len(anomaly_losses)
            avg_anomaly_acc = (1 - (avg_anomaly_loss**0.5)) * 100
            print(f"Anomaly Average Loss: {avg_anomaly_loss:.4f}")
            print(f"Anomaly Average Accuracy: {avg_anomaly_acc:.2f}%")
            
            print(f"\nDiff in Accuracy: {avg_normal_acc - avg_anomaly_acc:.2f}% (Higher is better for detection)")

            # 4. Visualize Comparison
            print("\nVisualizing comparison...")
            
            # Get one batch from each
            normal_batch = next(iter(normal_loader)).to(device)
            anomaly_batch = next(iter(anomaly_loader)).to(device)
            
            with torch.no_grad():
                normal_recon = model(normal_batch)
                anomaly_recon = model(anomaly_batch)
                
            # Move to cpu
            nb = normal_batch.cpu()
            nr = normal_recon.cpu()
            ab = anomaly_batch.cpu()
            ar = anomaly_recon.cpu()
            
            # Plot
            fig, axes = plt.subplots(4, 3, figsize=(10, 12))
            
            for i in range(3):
                # Normal Original
                axes[0, i].imshow(nb[i][0], cmap='inferno', origin='lower')
                axes[0, i].set_title("Normal (Original)")
                axes[0, i].axis('off')
                
                # Normal Recon
                loss_n = torch.nn.functional.mse_loss(nr[i], nb[i]).item()
                acc_n = (1 - loss_n**0.5)*100
                axes[1, i].imshow(nr[i][0], cmap='inferno', origin='lower')
                axes[1, i].set_title(f"Recon (Acc: {acc_n:.1f}%)")
                axes[1, i].axis('off')

                # Anomaly Original
                axes[2, i].imshow(ab[i][0], cmap='inferno', origin='lower')
                axes[2, i].set_title("Anomaly (Original)")
                axes[2, i].axis('off')
                
                # Anomaly Recon
                loss_a = torch.nn.functional.mse_loss(ar[i], ab[i]).item()
                acc_a = (1 - loss_a**0.5)*100
                axes[3, i].imshow(ar[i][0], cmap='inferno', origin='lower')
                axes[3, i].set_title(f"Recon (Acc: {acc_a:.1f}%)")
                axes[3, i].axis('off')

            plt.tight_layout()
            plt.savefig('anomaly_detection_results.png')
            print("Saved comparison to anomaly_detection_results.png")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

