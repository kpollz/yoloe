"""
Pre-compute and cache text vocabulary embeddings for YOLOE Prompt-Free mode.

This script:
1. Loads each YOLOE model
2. Computes text embeddings for all classes in ram_tag_list.txt
3. Saves the vocabulary to a cache file

Usage:
    python cache_vocab.py

The cached vocabularies will be saved to ./cached_vocab/
"""

import os
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLOE

# Configuration
MODEL_IDS = [
    "yoloe-v8s",
    "yoloe-v8m",
    "yoloe-v8l",
    "yoloe-11s",
    "yoloe-11m",
    "yoloe-11l",
]

VOCAB_FILE = "tools/ram_tag_list.txt"
CACHE_DIR = "cached_vocab"


def load_texts(vocab_file):
    """Load class names from vocabulary file."""
    with open(vocab_file, 'r') as f:
        texts = [x.strip() for x in f.readlines()]
    return texts


def init_model(model_id):
    """Initialize YOLOE model for prompt-free mode."""
    filename = f"{model_id}-seg-pf.pt"
    path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


def cache_vocab_for_model(model_id, texts, cache_dir):
    """Compute and cache vocabulary for a specific model."""
    print(f"\n{'='*60}")
    print(f"Processing model: {model_id}")
    print(f"{'='*60}")
    
    # Initialize model (non-pf version to get text embeddings)
    filename = f"{model_id}-seg.pt"
    path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"Computing text embeddings for {len(texts)} classes...")
    
    # Get text embeddings and vocabulary
    vocab = model.get_vocab(texts)
    
    # Extract vocabulary weights
    vocab_weights = []
    for v in vocab:
        vocab_weights.append(v.weight.data.cpu())
    
    # Create cache dictionary
    cache = {
        "model_id": model_id,
        "texts": texts,
        "vocab_weights": vocab_weights,
    }
    
    # Save cache
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{model_id}_vocab.pt")
    torch.save(cache, cache_path)
    
    print(f"Cached vocabulary saved to: {cache_path}")
    print(f"  - Number of classes: {len(texts)}")
    print(f"  - Vocab heads: {len(vocab_weights)}")
    print(f"  - File size: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")
    
    return cache_path


def main():
    print("YOLOE Vocabulary Caching Tool")
    print("=" * 60)
    
    # Load class names
    print(f"\nLoading class names from {VOCAB_FILE}...")
    texts = load_texts(VOCAB_FILE)
    print(f"Total classes: {len(texts)}")
    print(f"First 5 classes: {texts[:5]}")
    
    # Cache vocabulary for each model
    cached_files = []
    for model_id in MODEL_IDS:
        try:
            cache_path = cache_vocab_for_model(model_id, texts, CACHE_DIR)
            cached_files.append(cache_path)
        except Exception as e:
            print(f"ERROR caching {model_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Caching complete!")
    print(f"{'='*60}")
    print(f"\nCached files:")
    for f in cached_files:
        print(f"  - {f}")
    print(f"\nYou can now use these cached vocabularies in the app for instant loading.")


if __name__ == "__main__":
    main()
