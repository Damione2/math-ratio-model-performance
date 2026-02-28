# core/guardian_dataset_live.py
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

# Import centralized config
try:
    from config import ARTIFACTS_DIR
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import ARTIFACTS_DIR

class GuardianLiveDataset(Dataset):
    """
    Spider-Triangulation Dataset: Loads 8-leg features from NVMe memmap
    Shape: (N, 29, 8, 1536) - preserves temporal/syntactic directionality
    """
    def __init__(self, split: str = "train"):
        """
        Args:
            split: "train", "val", or "test" - determines which .bin file to load
        """
        self.split = split
        
        # Memmap binary file (produced by guardian_extract_memmap.py)
        self.bin_path = ARTIFACTS_DIR / f"{split}_features.bin"
        self.meta_path = ARTIFACTS_DIR / f"{split}_metadata.pkl"

        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"Memmap file not found: {self.bin_path}. "
                f"Run extraction for {split} split first."
            )

        # Load metadata
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        # Domain mapping for integer conversion
        self.domain_map = {
            "math": 0,
            "code": 1,
            "real_world": 2,
            "real": 2,  # Alternative name
            "default": 2
        }

        # Open memmap with Spider-Triangulation shape
        # (Samples, Layers, Legs, Dimensions)
        self.data = np.memmap(
            self.bin_path, 
            dtype='float16', 
            mode='r', 
            shape=(len(self.metadata), 29, 8, 1536)
        )
        
        print(f"✅ Fast-Load Map Ready: {split} | Samples: {len(self.metadata)} | Shape: {self.data.shape}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Load from NVMe: (29, 8, 1536)
        features = torch.from_numpy(self.data[idx].astype(np.float32))
        
        # ✅ FIXED: Return as a single Tensor (29, 8, 1536).
        # The Trainer needs to call .to(device) on the batch, which requires a tensor.
        # The Model (GuardianVisionNet) will detect this is a tensor and 
        # slice it into the 29 layers internally.
        
        meta = self.metadata[idx]
        domain_str = meta.get('domain', 'real_world')
        domain_int = self.domain_map.get(domain_str, 2)
        
        return {
            'hidden_states': features,  # ✅ Tensor format
            'label': torch.tensor(meta['label'], dtype=torch.long),
            'domain': torch.tensor(domain_int, dtype=torch.long)
        }

class GuardianLiveDataLoader:
    """
    Wrapper to create DataLoader for Spider-Triangulation features
    Compatible with pipeline's step 3 training loop
    """
    def __init__(self, data_path: Optional[Path] = None, batch_size: int = 16, 
                 split: str = "train", **kwargs):
        """
        Args:
            data_path: Path to the split .pkl file (e.g., "02_train.pkl"). 
                       The split name is extracted from filename.
            batch_size: Batch size for training
            split: Which split to load ("train", "val", "test") - used if data_path is None
            **kwargs: Additional args for compatibility (ignored)
        """
        self.batch_size = batch_size
        
        # Parse split name from data_path if provided
        if data_path:
            # Extract split name from filename like "02_train.pkl"
            filename = Path(data_path).name
            # Handle both "02_train.pkl" and "train_features.bin" formats
            if '_' in filename:
                self.split = filename.split('_')[-1].replace('.pkl', '')
            else:
                self.split = filename.replace('_features.bin', '').replace('_metadata.pkl', '')
        else:
            self.split = split
            
        self.dataset = GuardianLiveDataset(split=self.split)
    
    def get_dataloader(self, shuffle: bool = True):
        """Returns PyTorch DataLoader"""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Essential for Windows/Unsloth stability
            pin_memory=True,  # Allow trainer to move to GPU efficiently
            drop_last=False
        )
    
    def clear_cache(self):
        """Placeholder for training loop compatibility"""
        pass