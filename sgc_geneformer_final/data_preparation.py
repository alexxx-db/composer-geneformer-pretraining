# Databricks notebook source
# MAGIC %md
# MAGIC # Geneformer Data Preparation
# MAGIC 
# MAGIC This notebook prepares the data for Geneformer pretraining:
# MAGIC 1. Downloads the Genecorpus-30M dataset from HuggingFace
# MAGIC 2. Downloads the token dictionary
# MAGIC 3. Converts to MDS (Mosaic Data Shard) format for streaming
# MAGIC 
# MAGIC **Run this notebook with a CPU cluster before starting GPU training.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC 
# MAGIC Update these values to match your Databricks environment.

# COMMAND ----------

# ============================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================

# Databricks Volume Configuration
CATALOG = "main"
SCHEMA = "guanyu_chen"  
VOLUME_NAME = "sgc"

# Data paths (relative to volume root)
TOKEN_DICT_PATH = "geneformer/token_dictionary.pkl"
SOURCE_DATASET_PATH = "geneformer/data/dataset/genecorpus_30M_2048.dataset"
STREAMING_DATASET_PATH = "geneformer/data/dataset/streaming/genecorpus_30M_2048.dataset"

# Data preparation settings
TEST_SPLIT_RATIO = 0.1
RANDOM_SEED = 42

# HuggingFace source URLs
HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main"
TOKEN_DICT_URL = f"{HUGGINGFACE_BASE_URL}/token_dictionary.pkl"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Setup and Path Construction

# COMMAND ----------

import os
import shutil
import subprocess

# Construct volume path
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

# Construct full paths
PATHS = {
    'volume': VOLUME_PATH,
    'token_dictionary': f"{VOLUME_PATH}/{TOKEN_DICT_PATH}",
    'source_dataset': f"{VOLUME_PATH}/{SOURCE_DATASET_PATH}",
    'streaming_dataset': f"{VOLUME_PATH}/{STREAMING_DATASET_PATH}",
    'train_dir': f"{VOLUME_PATH}/{STREAMING_DATASET_PATH}/train",
    'test_dir': f"{VOLUME_PATH}/{STREAMING_DATASET_PATH}/test",
}

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Volume Path: {VOLUME_PATH}")
print(f"Token Dictionary: {PATHS['token_dictionary']}")
print(f"Source Dataset: {PATHS['source_dataset']}")
print(f"Streaming Dataset: {PATHS['streaming_dataset']}")
print(f"  - Train: {PATHS['train_dir']}")
print(f"  - Test: {PATHS['test_dir']}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Helper Functions

# COMMAND ----------

def check_mds_valid(path: str) -> bool:
    """Check if MDS dataset directory is valid (has index.json)."""
    index_file = os.path.join(path, 'index.json')
    return os.path.exists(index_file)


def get_dir_size(path: str) -> str:
    """Get human-readable directory size."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def download_file(url: str, dest: str, desc: str = None):
    """Download a file with progress."""
    desc = desc or os.path.basename(dest)
    print(f"Downloading {desc}...")
    
    # Create parent directory
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # Use curl for download with progress
    cmd = f'curl -L "{url}" -o "{dest}" --progress-bar'
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0 and os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"  ✅ Downloaded: {dest} ({size:,} bytes)")
        return True
    else:
        print(f"  ❌ Failed to download: {dest}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check Current Status

# COMMAND ----------

def check_status():
    """Check current status of all data components."""
    print("=" * 60)
    print("CURRENT STATUS")
    print("=" * 60)
    
    status = {}
    
    # Check token dictionary
    token_exists = os.path.exists(PATHS['token_dictionary'])
    status['token_dictionary'] = token_exists
    icon = "✅" if token_exists else "❌"
    print(f"{icon} Token Dictionary: {PATHS['token_dictionary']}")
    
    # Check source dataset
    source_exists = os.path.exists(PATHS['source_dataset'])
    if source_exists:
        # Check for required files
        required = ['dataset.arrow', 'dataset_info.json', 'state.json']
        files = os.listdir(PATHS['source_dataset']) if os.path.isdir(PATHS['source_dataset']) else []
        source_valid = all(f in files for f in required)
    else:
        source_valid = False
    status['source_dataset'] = source_valid
    icon = "✅" if source_valid else "❌"
    print(f"{icon} Source Dataset: {PATHS['source_dataset']}")
    
    # Check train MDS
    train_valid = check_mds_valid(PATHS['train_dir'])
    status['train_mds'] = train_valid
    icon = "✅" if train_valid else "❌"
    size_str = f" ({get_dir_size(PATHS['train_dir'])})" if train_valid else ""
    print(f"{icon} Train MDS: {PATHS['train_dir']}{size_str}")
    
    # Check test MDS
    test_valid = check_mds_valid(PATHS['test_dir'])
    status['test_mds'] = test_valid
    icon = "✅" if test_valid else "❌"
    size_str = f" ({get_dir_size(PATHS['test_dir'])})" if test_valid else ""
    print(f"{icon} Test MDS: {PATHS['test_dir']}{size_str}")
    
    print("=" * 60)
    
    # Summary
    if status['train_mds'] and status['test_mds'] and status['token_dictionary']:
        print("\n🎉 All data is ready for training!")
        print("You can skip to the Summary section for the configuration.")
    else:
        print("\n⚠️  Some data is missing. Run the cells below to prepare it.")
    
    return status

status = check_status()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Download Token Dictionary

# COMMAND ----------

def download_token_dictionary():
    """Download token dictionary from HuggingFace."""
    if os.path.exists(PATHS['token_dictionary']):
        print(f"✅ Token dictionary already exists: {PATHS['token_dictionary']}")
        return True
    
    return download_file(TOKEN_DICT_URL, PATHS['token_dictionary'], "token_dictionary.pkl")

# Run download
download_token_dictionary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Download Source Dataset
# MAGIC 
# MAGIC Download the Genecorpus-30M dataset from HuggingFace.
# MAGIC 
# MAGIC **⚠️ This may take 15-30 minutes depending on your network speed (~15GB).**

# COMMAND ----------

def download_source_dataset():
    """Download source dataset from HuggingFace."""
    source_path = PATHS['source_dataset']
    required_files = ['dataset.arrow', 'dataset_info.json', 'state.json']
    
    # Check if already exists
    if os.path.exists(source_path) and os.path.isdir(source_path):
        existing = os.listdir(source_path)
        if all(f in existing for f in required_files):
            size = get_dir_size(source_path)
            print(f"✅ Source dataset already exists: {source_path} ({size})")
            return True
    
    print("=" * 60)
    print("DOWNLOADING SOURCE DATASET")
    print("=" * 60)
    print("⚠️  This may take 15-30 minutes (dataset is ~15GB)")
    print()
    
    os.makedirs(source_path, exist_ok=True)
    
    dataset_files = [
        ("genecorpus_30M_2048.dataset/dataset.arrow", "dataset.arrow"),
        ("genecorpus_30M_2048.dataset/dataset_info.json", "dataset_info.json"),
        ("genecorpus_30M_2048.dataset/state.json", "state.json"),
    ]
    
    all_success = True
    for remote_file, local_file in dataset_files:
        url = f"{HUGGINGFACE_BASE_URL}/{remote_file}"
        dest = f"{source_path}/{local_file}"
        success = download_file(url, dest, local_file)
        if not success:
            all_success = False
    
    if all_success:
        size = get_dir_size(source_path)
        print(f"\n✅ Source dataset ready: {source_path} ({size})")
    else:
        print(f"\n❌ Some files failed to download")
    
    return all_success

# Run download
download_source_dataset()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Convert to MDS Format
# MAGIC 
# MAGIC Convert the HuggingFace dataset to MDS (Mosaic Data Shard) format for efficient streaming during training.
# MAGIC 
# MAGIC **⚠️ This may take 30-60 minutes for 30M samples.**

# COMMAND ----------

# Install required packages if not available
# %pip install datasets mosaicml-streaming tqdm

# COMMAND ----------

def write_mds_fast(dataset, output_path: str, name: str = ""):
    """
    Write dataset to MDS format using pandas conversion for speed.
    This loads all data to memory first, then iterates fast.
    """
    from streaming import MDSWriter
    from tqdm import tqdm
    import numpy as np
    
    columns = {'input_ids': "ndarray", 'length': 'int'}
    
    # Convert to pandas then to list of dicts - MUCH faster iteration than HF dataset
    print(f"  Converting {len(dataset):,} samples to pandas (this loads to memory)...")
    dataset_list = dataset.to_pandas().to_dict('records')
    print(f"  Converted! Now writing MDS...")
    
    with MDSWriter(out=output_path, columns=columns, compression='zstd') as out:
        for x in tqdm(dataset_list, total=len(dataset_list), desc=name):
            # input_ids is already numpy array from pandas conversion
            out.write({"input_ids": x["input_ids"], "length": x["length"]})


def prepare_mds_dataset(force_recreate: bool = False):
    """
    Convert HuggingFace dataset to MDS format.
    Uses pandas conversion for fast iteration (loads data to memory).
    
    Args:
        force_recreate: If True, recreate even if MDS already exists
    """
    from datasets import load_from_disk
    
    # Check if already exists (skip if valid and not forcing recreate)
    mds_exists = check_mds_valid(PATHS['train_dir']) and check_mds_valid(PATHS['test_dir'])
    
    if mds_exists and not force_recreate:
        train_size = get_dir_size(PATHS['train_dir'])
        test_size = get_dir_size(PATHS['test_dir'])
        print(f"✅ MDS dataset already exists:")
        print(f"   Train: {PATHS['train_dir']} ({train_size})")
        print(f"   Test: {PATHS['test_dir']} ({test_size})")
        return True
    
    print("=" * 60)
    print("CONVERTING TO MDS FORMAT")
    print("=" * 60)
    if mds_exists:
        print("⚠️  force_recreate=True, will overwrite existing data")
    print("⚠️  This loads data to memory via pandas for fast processing")
    print()
    
    # Check source dataset exists
    if not os.path.exists(PATHS['source_dataset']):
        print(f"❌ Source dataset not found: {PATHS['source_dataset']}")
        print("   Please run the download cell first.")
        return False
    
    # Clean output directories before writing
    for path in [PATHS['train_dir'], PATHS['test_dir']]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    
    # Load source dataset
    print(f"Loading source dataset: {PATHS['source_dataset']}")
    dataset = load_from_disk(PATHS['source_dataset'])
    print(f"Dataset loaded: {dataset}")
    print(f"Total samples: {len(dataset):,}")
    
    # Split dataset
    print(f"\nSplitting dataset (test_size={TEST_SPLIT_RATIO}, seed={RANDOM_SEED})...")
    split = dataset.train_test_split(test_size=TEST_SPLIT_RATIO, seed=RANDOM_SEED)
    train_ds, test_ds = split["train"], split["test"]
    print(f"Train samples: {len(train_ds):,}")
    print(f"Test samples: {len(test_ds):,}")
    
    # Write train MDS (converts to pandas for fast iteration)
    print(f"\nWriting train MDS: {PATHS['train_dir']}")
    write_mds_fast(train_ds, PATHS['train_dir'], name="Train")
    
    train_size = get_dir_size(PATHS['train_dir'])
    print(f"✅ Train MDS complete: {train_size}")
    
    # Write test MDS
    print(f"\nWriting test MDS: {PATHS['test_dir']}")
    write_mds_fast(test_ds, PATHS['test_dir'], name="Test")
    
    test_size = get_dir_size(PATHS['test_dir'])
    print(f"✅ Test MDS complete: {test_size}")
    
    print("\n" + "=" * 60)
    print("✅ MDS CONVERSION COMPLETE!")
    print("=" * 60)
    return True

# Run MDS conversion
# Set force_recreate=True to recreate even if exists
prepare_mds_dataset(force_recreate=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Final Verification & Training Configuration

# COMMAND ----------

def print_summary():
    """Print final summary and training configuration."""
    print("=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    
    all_ready = True
    
    # Check all components
    checks = [
        ("Token Dictionary", PATHS['token_dictionary'], os.path.exists(PATHS['token_dictionary'])),
        ("Train MDS", PATHS['train_dir'], check_mds_valid(PATHS['train_dir'])),
        ("Test MDS", PATHS['test_dir'], check_mds_valid(PATHS['test_dir'])),
    ]
    
    for name, path, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}: {path}")
        if not status:
            all_ready = False
    
    print("=" * 60)
    
    if all_ready:
        print("\n🎉 ALL DATA IS READY FOR TRAINING!")
        print("\n" + "=" * 60)
        print("COPY THIS TO parameters.yaml:")
        print("=" * 60)
        print(f"""
# ============================================
# Databricks Volume Configuration
# ============================================
volume:
  catalog: {CATALOG}
  schema: {SCHEMA}
  volume_name: {VOLUME_NAME}

# Data paths relative to the volume root
data:
  source_dataset: {SOURCE_DATASET_PATH}
  streaming_dataset: {STREAMING_DATASET_PATH}
  token_dictionary: {TOKEN_DICT_PATH}
  test_split_ratio: {TEST_SPLIT_RATIO}

# Checkpoint path relative to volume root
checkpoints:
  folder: geneformer/checkpoints
""")
        print("=" * 60)
        print("\nNow you can run the training with:")
        print("  composer train.py parameters.yaml")
    else:
        print("\n❌ Some data is missing. Run the cells above to prepare it.")

print_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. (Optional) Cleanup Source Dataset
# MAGIC 
# MAGIC After MDS conversion, you can optionally delete the source HuggingFace dataset to save space.
# MAGIC **Only run this if MDS conversion was successful!**

# COMMAND ----------

def cleanup_source_dataset():
    """Remove source dataset to save space (optional)."""
    if not (check_mds_valid(PATHS['train_dir']) and check_mds_valid(PATHS['test_dir'])):
        print("❌ MDS dataset is not valid. Cannot cleanup source dataset.")
        return
    
    if not os.path.exists(PATHS['source_dataset']):
        print("Source dataset already removed.")
        return
    
    size = get_dir_size(PATHS['source_dataset'])
    print(f"Removing source dataset: {PATHS['source_dataset']} ({size})")
    shutil.rmtree(PATHS['source_dataset'])
    print("✅ Source dataset removed.")

# Uncomment to run cleanup
# cleanup_source_dataset()
