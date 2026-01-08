import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


class ESC50SpectrogramDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        target_classes,
        folds
    ):
        self.img_dir = img_dir
        
        # 1. Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find CSV at {csv_path}")

        df = pd.read_csv(csv_path)

        # 2. Validate classes
        available_classes = set(df["category"].unique())
        target_classes = list(target_classes)

        invalid = set(target_classes) - available_classes
        if invalid:
            raise ValueError(
                f"Invalid class(es): {invalid}\nAvailable: {sorted(available_classes)}"
            )

        # 3. Filter by category
        df = df[df["category"].isin(target_classes)]

        # 4. Filter by folds
        df = df[df["fold"].isin(folds)].reset_index(drop=True)

        self.df = df
        print(
            f"Loaded {len(self.df)} spectrograms "
            f"for classes {target_classes} (Folds: {folds})"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        category = row["category"]
        
        # Construct path: img_dir / category / filename.png
        img_name = filename.replace('.wav', '.png')
        file_path = os.path.join(self.img_dir, category, img_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Spectrogram not found: {file_path}")

        # Load image (C, H, W) - uint8
        img = read_image(file_path)
        
        # Convert to float [0, 1]
        img = img.float() / 255.0
        
        # Ensure 1 channel (if saved as RGB by mistake, take first)
        if img.shape[0] > 1:
            img = img[0:1, :, :]
            
        return img


# --- Helper ---
def get_machine_dataloaders(class_names, batch_size=16, root_path="."):
    csv_path = os.path.join(root_path, "meta", "esc50.csv")
    # Point to the generated spectrograms directory
    spectrogram_path = os.path.join(root_path, "spectrograms")

    if not os.path.exists(spectrogram_path):
         raise FileNotFoundError(f"Spectrograms directory not found at {spectrogram_path}. Run generate_spectrograms.py first.")

    train_ds = ESC50SpectrogramDataset(
        csv_path,
        spectrogram_path,
        class_names,
        folds=[1, 2, 3, 4],
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = ESC50SpectrogramDataset(
        csv_path,
        spectrogram_path,
        class_names,
        folds=[5],
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# --- Usage ---
if __name__ == "__main__":
    SELECTED_CLASSES = ["vacuum_cleaner", "chainsaw"]

    print(f"--- Preparing Data for {SELECTED_CLASSES} ---")

    try:
        train_loader, test_loader = get_machine_dataloaders(
            SELECTED_CLASSES, batch_size=8
        )

        batch = next(iter(train_loader))
        print(f"\nSuccess! Batch shape: {batch.shape}")
        print("(Batch, Channels, Samples)")

    except Exception as e:
        print(f"\nError: {e}")
