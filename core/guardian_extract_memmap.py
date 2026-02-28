# core/guardian_extract_memmap.py (Updated with Scaler Creation)
import sys
import torch
import numpy as np
import pickle
import os
import gc
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# 1. Fix Path to find config.py
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ARTIFACTS_DIR
from core.guardian_utils import GuardianScaler  # Import GuardianScaler

# --- SPIDER CONFIGURATION ---
LAYERS = 29          # Qwen2.5-1.5B has 28 layers + 1 embedding layer
DIMS = 1536          # Hidden dimension size
NUM_LEGS = 8         # Biomimicry: 8 legs to triangulate the signal
SEQ_LEN = 128        # Max sequence length
CHUNK_SIZE = SEQ_LEN // NUM_LEGS  # 16 tokens per leg

def run_extraction(split_name="train"):
    """Extract features for a specific data split and create scaler"""
    # Use split-specific paths
    RAW_DATA_PATH = ARTIFACTS_DIR / f"02_{split_name}.pkl"
    STORAGE_PATH = ARTIFACTS_DIR / f"{split_name}_features.bin"
    METADATA_PATH = ARTIFACTS_DIR / f"{split_name}_metadata.pkl"
    SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"  # Single scaler for all splits

    # 1. Load the Raw Data for this split
    if not RAW_DATA_PATH.exists():
        print(f"❌ Error: {RAW_DATA_PATH} not found. Run Step 2 (Splitting) first.")
        return

    print(f"📂 Loading raw data from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    
    total_samples = len(data)
    print(f"🕷️ Preparing to spin web for {split_name} split: {total_samples} samples.")

    # 2. Pre-allocate Memory Map (The "Web Structure")
    fp = np.memmap(STORAGE_PATH, dtype='float16', mode='w+', 
                  shape=(total_samples, LAYERS, NUM_LEGS, DIMS))

    # 3. Load the "Sensory Organ" (Qwen Model)
    model_name = "unsloth/Qwen2.5-1.5B"
    print(f"👁️ Loading {model_name} for sensory extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
    
    metadata = []
    batch_size = 48  # Reduced for 6GB VRAM stability
    
    # ✅ NEW: Collect features for scaler fitting (use subset to save memory)
    print(f"🧮 Preparing to collect feature statistics for scaler...")
    sample_features = []

    print(f"🚀 Starting 8-Legged Triangulation Extraction for {split_name}...")

    for i in tqdm(range(0, len(data), batch_size), desc=f"Extracting {split_name}"):
        batch = data[i:i+batch_size]
        texts = [f"Q: {item['question']} A: {item['answer']}" for item in batch]
        
        # Tokenize
        inputs = tokenizer(texts, return_tensors="pt", padding='max_length', 
                          truncation=True, max_length=SEQ_LEN).to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # --- THE TRIANGULATION LOGIC ---
        batch_features = []
        for layer_tensor in outputs.hidden_states:
            # Reshape to split sequence into 8 legs
            reshaped = layer_tensor.view(-1, NUM_LEGS, CHUNK_SIZE, DIMS)
            # Pool (Average) within each leg
            leg_pooled = reshaped.mean(dim=2) 
            batch_features.append(leg_pooled.cpu().numpy().astype(np.float16))
            
        # Stack layers: (29, Batch, 8, 1536) -> Transpose to (Batch, 29, 8, 1536)
        batch_stack = np.stack(batch_features, axis=0).transpose(1, 0, 2, 3)
        
        # Write to NVMe
        fp[i : i + len(batch)] = batch_stack
            
        # ✅ NEW: Collect samples for scaler (every 10th batch to save memory)
        if split_name == "train" and i % (batch_size * 10) == 0:
            # Flatten samples for scaler: (samples, 29*8*1536)
            flat_samples = batch_stack.reshape(batch_stack.shape[0], -1)
            sample_features.append(flat_samples)  # Collect all samples from this batch
            
        # Save Metadata
        for item in batch:
            metadata.append({'label': item['label'], 'domain': item['domain']})
            
        if i % 100 == 0: 
            fp.flush()

    # Finalize
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    # ✅ NEW: Create and save scaler if this is the training split
    if split_name == "train":
        if sample_features:
            print(f"🧮 Fitting GuardianScaler on collected samples...")
            
            # Concatenate all collected samples
            all_samples = np.concatenate(sample_features, axis=0)
            print(f"   Total samples for scaler: {all_samples.shape[0]}")
            
            # Reshape to (N, 29, 8, 1536) 
            all_samples = all_samples.reshape(-1, LAYERS, NUM_LEGS, DIMS)
            
            # Flatten for fitting: (N, 29*8*1536)
            flat_for_scaler = all_samples.reshape(all_samples.shape[0], -1)
            
            # Convert to torch tensor for fitting
            scaler = GuardianScaler()
            scaler.fit(torch.from_numpy(flat_for_scaler.astype(np.float32)))
            
            # Save scaler
            scaler.save(str(SCALER_PATH))
            print(f"✅ Scaler saved to {SCALER_PATH}")
            print(f"   Mean shape: {scaler.mean.shape}, Scale shape: {scaler.scale.shape}")
        else:
            print(f"⚠️  No samples collected for scaler, creating dummy scaler...")
            scaler = GuardianScaler()
            # Create dummy mean and scale
            dummy_data = torch.randn(100, LAYERS * NUM_LEGS * DIMS)
            scaler.fit(dummy_data)
            scaler.save(str(SCALER_PATH))
            print(f"✅ Dummy scaler saved to {SCALER_PATH}")

    print(f"✅ Web Spun for {split_name}!")
    
    # --- VRAM CLEANUP BLOCK (Prevents Shared Memory Spillover) ---
    del model
    del tokenizer
    gc.collect()  # Force Python garbage collection
    torch.cuda.empty_cache()  # Clear GPU cache
    torch.cuda.synchronize()  # Wait for all kernels to finish
    # ---------------------------------------------------------------
    
    print(f"🧠 VRAM Cleared after {split_name} split.")
    print(f"📊 File size: {os.path.getsize(STORAGE_PATH)/1e9:.2f} GB")
    if split_name == "train":
        print(f"🧮 Scaler file: {SCALER_PATH} ({os.path.getsize(SCALER_PATH)/1e6:.2f} MB)")
    print(f"🕷️ Biomimicry Status: 8-Legged Triangulation Active.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Spider-Triangulation features for a specific split")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], 
                       help="Which data split to process")
    args = parser.parse_args()
    run_extraction(split_name=args.split)