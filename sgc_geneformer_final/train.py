##1 Set parameters

import datetime
import os
import pickle
import random
import subprocess
import sys
from typing import cast

import numpy as np
import pytz

import torch
from torch.utils.data import DataLoader

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

import geneformer
from geneformer.pretrainer import GeneformerPreCollator

from composer.models.huggingface import HuggingFaceModel
from composer.utils import reproducibility, dist
from composer import Trainer
from composer import Callback, Event, Logger, State

from streaming import StreamingDataset

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from cfgutils import *


def build_volume_path(cfg: DictConfig) -> str:
    """Build the Databricks volume path from catalog, schema, and volume_name."""
    volume_cfg = cfg.volume
    return f"/Volumes/{volume_cfg.catalog}/{volume_cfg.schema}/{volume_cfg.volume_name}"


def get_data_paths(cfg: DictConfig, volume_path: str) -> dict:
    """Get all data paths based on volume path and config."""
    data_cfg = cfg.data
    return {
        'source_dataset': f"{volume_path}/{data_cfg.source_dataset}",
        'streaming_dataset': f"{volume_path}/{data_cfg.streaming_dataset}",
        'token_dictionary': f"{volume_path}/{data_cfg.token_dictionary}",
        'train_dir': f"{volume_path}/{data_cfg.streaming_dataset}/train",
        'test_dir': f"{volume_path}/{data_cfg.streaming_dataset}/test",
    }


def check_streaming_dataset_exists(paths: dict) -> bool:
    """Check if the streaming dataset (train and test) already exists."""
    train_exists = os.path.exists(paths['train_dir']) and os.path.isdir(paths['train_dir'])
    test_exists = os.path.exists(paths['test_dir']) and os.path.isdir(paths['test_dir'])
    
    if train_exists and test_exists:
        # Check if directories have actual data (index.json is a good indicator)
        train_has_data = os.path.exists(os.path.join(paths['train_dir'], 'index.json'))
        test_has_data = os.path.exists(os.path.join(paths['test_dir'], 'index.json'))
        return train_has_data and test_has_data
    
    return False


# HuggingFace base URL for Genecorpus-30M dataset
HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/resolve/main"


def download_token_dictionary(paths: dict):
    """Download the token dictionary from HuggingFace if not present."""
    import urllib.request
    
    token_dictionary_path = paths['token_dictionary']
    
    print("=" * 60)
    print("Downloading token dictionary from HuggingFace...")
    print("=" * 60)
    
    token_dict_url = f"{HUGGINGFACE_BASE_URL}/token_dictionary.pkl?download=true"
    os.makedirs(os.path.dirname(token_dictionary_path), exist_ok=True)
    print(f"Downloading to: {token_dictionary_path}")
    urllib.request.urlretrieve(token_dict_url, token_dictionary_path)
    print("Token dictionary downloaded.")
    print("=" * 60)


def download_source_dataset(paths: dict, volume_path: str):
    """Download the source dataset from HuggingFace if not present."""
    import urllib.request
    
    source_dataset_path = paths['source_dataset']
    
    print("=" * 60)
    print("Downloading source dataset from HuggingFace...")
    print("=" * 60)
    
    print(f"Downloading source dataset to: {source_dataset_path}")
    os.makedirs(source_dataset_path, exist_ok=True)
    
    # Files to download for the dataset
    dataset_files = [
        ("genecorpus_30M_2048.dataset/dataset.arrow", "dataset.arrow"),
        ("genecorpus_30M_2048.dataset/dataset_info.json", "dataset_info.json"),
        ("genecorpus_30M_2048.dataset/state.json", "state.json"),
    ]
    
    for remote_file, local_file in dataset_files:
        url = f"{HUGGINGFACE_BASE_URL}/{remote_file}"
        local_path = os.path.join(source_dataset_path, local_file)
        print(f"  Downloading {local_file}...")
        urllib.request.urlretrieve(url, local_path)
    
    print("Source dataset downloaded.")
    print("=" * 60)


def prepare_streaming_dataset(cfg: DictConfig, paths: dict, volume_path: str):
    """Prepare the streaming dataset from the source HuggingFace dataset."""
    from datasets import load_from_disk
    from streaming import MDSWriter
    from tqdm import tqdm
    
    print("=" * 60)
    print("Streaming dataset not found. Preparing MDS dataset...")
    print("=" * 60)
    
    source_dataset_path = paths['source_dataset']
    streaming_dataset_path = paths['streaming_dataset']
    test_split_ratio = cfg.data.get('test_split_ratio', 0.1)
    
    print(f"Source dataset: {source_dataset_path}")
    print(f"Streaming dataset: {streaming_dataset_path}")
    print(f"Test split ratio: {test_split_ratio}")
    
    # Check if source dataset exists, download if not
    if not os.path.exists(source_dataset_path):
        print(f"\nSource dataset not found. Downloading from HuggingFace...")
        download_source_dataset(paths, volume_path)
    
    # Define columns for MDS
    columns = {
        'input_ids': "ndarray",
        'length': 'int'
    }
    
    # Load source dataset
    print(f"\nLoading source dataset from: {source_dataset_path}")
    dataset = load_from_disk(source_dataset_path)
    print(f"Dataset loaded: {dataset}")
    
    # Split dataset into train and test sets
    print(f"\nSplitting dataset with test_size={test_split_ratio}")
    train_test_split = dataset.train_test_split(test_size=test_split_ratio)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create output directories
    os.makedirs(paths['train_dir'], exist_ok=True)
    os.makedirs(paths['test_dir'], exist_ok=True)
    
    # Prepare training dataset
    print("\nPreparing training dataset...")
    train_dataset_list = train_dataset.to_pandas().to_dict('records')
    with MDSWriter(out=paths['train_dir'], columns=columns, compression='zstd') as out:
        for x in tqdm(train_dataset_list, total=len(train_dataset_list), desc="Writing train"):
            out.write({
                "input_ids": x["input_ids"],
                "length": x["length"]
            })
    
    # Prepare test dataset
    print("\nPreparing test dataset...")
    test_dataset_list = test_dataset.to_pandas().to_dict('records')
    with MDSWriter(out=paths['test_dir'], columns=columns, compression='zstd') as out:
        for x in tqdm(test_dataset_list, total=len(test_dataset_list), desc="Writing test"):
            out.write({
                "input_ids": x["input_ids"],
                "length": x["length"]
            })
    
    print("\n" + "=" * 60)
    print("Streaming dataset preparation complete!")
    print("=" * 60)


def main(cfg: DictConfig):
    # Build volume path from config
    volume_path = build_volume_path(cfg)
    print(f"Using Databricks volume: {volume_path}")
    
    # Get all data paths
    paths = get_data_paths(cfg, volume_path)
    print(f"Token dictionary: {paths['token_dictionary']}")
    print(f"Streaming dataset: {paths['streaming_dataset']}")
    
    # Build checkpoint save folder from volume config
    checkpoint_folder = f"{volume_path}/{cfg.checkpoints.folder}"
    print(f"Checkpoint folder: {checkpoint_folder}")
    
    # Seed and reproducibility
    seed_val = cfg.seed_val
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    working_dir = cfg.working_dir
    
    # batch size for training and eval
    train_batch_size = cfg.train_batch_size
    eval_batch_size = cfg.eval_batch_size
    mlm_probability = cfg.mlm_probability
    
    #############################################
    ### Start processing
    reproducibility.configure_deterministic_mode()
    reproducibility.seed_all(seed_val)

    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers', {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks', {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]
    
    # Check and download token dictionary if not exists (only on rank 0 before dist init)
    if not os.path.exists(paths['token_dictionary']):
        print(f"\nToken dictionary not found. Downloading from HuggingFace...")
        download_token_dictionary(paths)
    
    # Read the token dictionary file
    print(f"\nLoading token dictionary from: {paths['token_dictionary']}")
    with open(paths['token_dictionary'], 'rb') as f:
        token_dictionary = pickle.load(f)
    print(f"Token dictionary loaded with {len(token_dictionary)} tokens")

    ### Load model
    model_config = build_model_config(cfg, token_dictionary)

    print("\n=============================")
    print("Model Configuration:")
    print(model_config)
    print("=============================\n")

    config = BertConfig(**model_config)
    model = BertForMaskedLM(config)
    tokenizer = GeneformerPreCollator(token_dictionary=token_dictionary)
    model.train()
    print(model)

    # Initialize distributed training before creating StreamingDataset
    dist.initialize_dist(device=cfg.get("device", "gpu"))
    print(f"\nDistributed initialized: world_size={dist.get_world_size()}, rank={dist.get_global_rank()}")

    # Check if streaming dataset exists, prepare if not (only on rank 0)
    if dist.get_global_rank() == 0:
        if not check_streaming_dataset_exists(paths):
            prepare_streaming_dataset(cfg, paths, volume_path)
        else:
            print(f"\nStreaming dataset already exists at: {paths['streaming_dataset']}")
            print("Skipping MDS preparation.")
    
    # Synchronize all ranks after potential dataset preparation
    dist.barrier()

    # Create streaming dataset
    print(f"\nLoading streaming dataset from: {paths['streaming_dataset']}")
    streaming_dataset_train = StreamingDataset(
        local=paths['train_dir'],
        batch_size=train_batch_size
    )
    streaming_dataset_eval = StreamingDataset(
        local=paths['test_dir'],
        batch_size=eval_batch_size
    )

    # Prepare composer model
    composer_model = HuggingFaceModel(model)

    # Build optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Data collator - wrapped to convert input_ids to Long dtype
    base_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=mlm_probability
    )
    
    def data_collator(features):
        """Wrapper collator that ensures input_ids are Long dtype before MLM masking."""
        for feature in features:
            if 'input_ids' in feature:
                if isinstance(feature['input_ids'], torch.Tensor):
                    feature['input_ids'] = feature['input_ids'].long()
                elif isinstance(feature['input_ids'], np.ndarray):
                    feature['input_ids'] = feature['input_ids'].astype(np.int64)
        return base_collator(features)

    train_dataloader = DataLoader(
        streaming_dataset_train,
        shuffle=False, 
        drop_last=False, 
        collate_fn=data_collator,
        batch_size=train_batch_size,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True
    )

    eval_dataloader = DataLoader(
        streaming_dataset_eval,
        shuffle=False, 
        drop_last=False, 
        collate_fn=data_collator,
        batch_size=eval_batch_size,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True
    )

    # Create Trainer Object
    trainer = Trainer(
        model=composer_model, 
        algorithms=algorithms,
        train_dataloader=train_dataloader,    
        eval_dataloader=eval_dataloader,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        optimizers=optimizer,
        schedulers=[scheduler],
        device=cfg.get("device", "gpu"),
        device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
        save_folder=checkpoint_folder,  # Use volume-based checkpoint folder
        save_interval=cfg.get("save_interval", "5ep"),
        save_overwrite=cfg.get("save_overwrite", False),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", 1),
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        autoresume=cfg.get("autoresume", False),
        python_log_level=cfg.get("python_log_level", None),
        seed=seed_val,        
        fsdp_config=cfg.get("fsdp_config", None),
        loggers=loggers,
        callbacks=callbacks,
    )
    
    # Start training
    trainer.fit()

    print(trainer.state.train_metrics)
    print(trainer.state.eval_metrics)

    print("*************Done")


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)
    main(cfg)
