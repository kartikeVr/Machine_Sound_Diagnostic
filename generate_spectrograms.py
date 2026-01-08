import os
import pandas as pd
import torch
import torchaudio
from torchvision.utils import save_image
from tqdm import tqdm

def generate_spectrograms(csv_path, audio_dir, output_dir, target_sr=16000, fixed_length=16000*5):
    # 1. Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load Metadata
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} audio files in metadata.")

    # 3. Define Transform
    # Same parameters as in main.py
    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_mels=64
    ).to(device)

    # 4. Process Loop
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Spectrograms"):
        filename = row['filename']
        category = row['category']
        
        # Create category folder
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        # Load Audio
        file_path = os.path.join(audio_dir, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # Resample
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            
            # Pad / Crop to fixed length (5 seconds)
            if waveform.shape[1] < fixed_length:
                pad_len = fixed_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            else:
                waveform = waveform[:, :fixed_length]
                
            waveform = waveform.to(device)

            # Generate Spectrogram
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

            # Save as image
            # Replace .wav with .png
            out_filename = filename.replace('.wav', '.png')
            out_path = os.path.join(cat_dir, out_filename)
            
            save_image(spectrogram, out_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Paths based on current project structure
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "meta", "esc50.csv")
    AUDIO_DIR = os.path.join(BASE_DIR, "audio", "audio", "audio") # Based on file listing
    OUTPUT_DIR = os.path.join(BASE_DIR, "spectrograms")

    print("Starting Spectrogram Generation...")
    generate_spectrograms(CSV_PATH, AUDIO_DIR, OUTPUT_DIR)
    print(f"\nDone! Spectrograms saved to {OUTPUT_DIR}")
