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


# ============================================
# Failure Injection Callback for Testing Auto-Resume
# ============================================
class FailureInjectionCallback(Callback):
    """
    Callback to simulate failures for testing auto-resume functionality.
    
    This callback will intentionally crash the training after a specified number
    of batches for the first N attempts. On subsequent attempts, training continues
    normally from the checkpoint.
    
    Args:
        attempt_file: Path to file tracking attempt count (should be on persistent volume)
        max_failures: Number of times to fail before allowing training to complete (default: 3)
        fail_at_batch: Which batch to fail at (default: 50)
    """
    
    def __init__(self, attempt_file: str, max_failures: int = 3, fail_at_batch: int = 50):
        self.attempt_file = attempt_file
        self.max_failures = max_failures
        self.fail_at_batch = fail_at_batch
        self.attempt_number = self._get_or_increment_attempt()
        
    def _get_or_increment_attempt(self) -> int:
        """Read current attempt count and increment it."""
        attempt = 1
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.attempt_file), exist_ok=True)
        
        # Read existing attempt count
        if os.path.exists(self.attempt_file):
            try:
                with open(self.attempt_file, 'r') as f:
                    attempt = int(f.read().strip()) + 1
            except (ValueError, IOError):
                attempt = 1
        
        # Write new attempt count
        with open(self.attempt_file, 'w') as f:
            f.write(str(attempt))
        
        return attempt
    
    def reset_attempts(self):
        """Reset the attempt counter (call this after successful completion)."""
        if os.path.exists(self.attempt_file):
            os.remove(self.attempt_file)
            print(f"[FailureInjection] Reset attempt counter")
    
    def init(self, state: State, logger: Logger) -> None:
        """Called when the trainer initializes."""
        if dist.get_global_rank() == 0:
            print(f"\n{'='*60}")
            print(f"[FailureInjection] FAILURE INJECTION ENABLED")
            print(f"[FailureInjection] Attempt: {self.attempt_number} / {self.max_failures + 1}")
            print(f"[FailureInjection] Will fail at batch: {self.fail_at_batch}")
            if self.attempt_number <= self.max_failures:
                print(f"[FailureInjection] ⚠️  This run WILL FAIL (attempt {self.attempt_number})")
            else:
                print(f"[FailureInjection] ✅ This run will SUCCEED (past failure threshold)")
            print(f"{'='*60}\n")
    
    def batch_end(self, state: State, logger: Logger) -> None:
        """Called at the end of each batch."""
        current_batch = int(state.timestamp.batch)
        
        # Only fail if we haven't exceeded max_failures
        if self.attempt_number <= self.max_failures:
            if current_batch == self.fail_at_batch:
                if dist.get_global_rank() == 0:
                    print(f"\n{'='*60}")
                    print(f"[FailureInjection] 💥 INJECTING FAILURE at batch {current_batch}")
                    print(f"[FailureInjection] Attempt {self.attempt_number} of {self.max_failures}")
                    print(f"[FailureInjection] Training will auto-resume from checkpoint")
                    print(f"{'='*60}\n")
                
                # Synchronize all ranks before failing
                dist.barrier()
                
                # Raise an exception to simulate failure
                raise RuntimeError(
                    f"[FailureInjection] Intentional failure at batch {current_batch} "
                    f"(attempt {self.attempt_number}/{self.max_failures}). "
                    f"Set max_retries >= {self.max_failures} in train.yaml to test auto-resume."
                )
    
    def fit_end(self, state: State, logger: Logger) -> None:
        """Called when training completes successfully."""
        if dist.get_global_rank() == 0:
            print(f"\n{'='*60}")
            print(f"[FailureInjection] ✅ Training completed successfully!")
            print(f"[FailureInjection] Total attempts: {self.attempt_number}")
            print(f"[FailureInjection] Resetting attempt counter...")
            print(f"{'='*60}\n")
            self.reset_attempts()


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


def verify_data_exists(paths: dict):
    """
    Verify that all required data exists.
    Raises FileNotFoundError with instructions if data is missing.
    """
    errors = []
    
    # Check token dictionary
    if not os.path.exists(paths['token_dictionary']):
        errors.append(f"Token dictionary not found: {paths['token_dictionary']}")
    
    # Check train MDS directory and index
    train_index = os.path.join(paths['train_dir'], 'index.json')
    if not os.path.exists(train_index):
        errors.append(f"Train MDS dataset not found: {paths['train_dir']}")
    
    # Check test MDS directory and index
    test_index = os.path.join(paths['test_dir'], 'index.json')
    if not os.path.exists(test_index):
        errors.append(f"Test MDS dataset not found: {paths['test_dir']}")
    
    if errors:
        error_list = "\n".join([f"  - {e}" for e in errors])
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"DATA NOT FOUND\n"
            f"{'='*60}\n"
            f"The following required data is missing:\n{error_list}\n\n"
            f"Please run data_preparation.py notebook first to:\n"
            f"  1. Download the dataset from HuggingFace\n"
            f"  2. Download the token dictionary\n"
            f"  3. Convert to MDS format\n\n"
            f"Run the notebook with a CPU cluster before starting GPU training.\n"
            f"{'='*60}"
        )


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
    
    # Add failure injection callback if enabled (for testing auto-resume)
    failure_injection_cfg = cfg.get('failure_injection', None)
    if failure_injection_cfg and failure_injection_cfg.get('enabled', False):
        attempt_file = f"{checkpoint_folder}/failure_injection_attempts.txt"
        failure_callback = FailureInjectionCallback(
            attempt_file=attempt_file,
            max_failures=failure_injection_cfg.get('max_failures', 3),
            fail_at_batch=failure_injection_cfg.get('fail_at_batch', 50),
        )
        callbacks.append(failure_callback)

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]
    
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
    world_size = dist.get_world_size()
    global_rank = dist.get_global_rank()
    print(f"\nDistributed initialized: world_size={world_size}, rank={global_rank}")
    
    # For multi-node training, calculate num_canonical_nodes (number of physical nodes)
    # Assuming 8 GPUs per node for H100 clusters
    gpus_per_node = 8
    num_canonical_nodes = max(1, world_size // gpus_per_node)
    print(f"Using num_canonical_nodes={num_canonical_nodes} for StreamingDataset")

    # Verify all required data exists (only on rank 0, then sync)
    if dist.get_global_rank() == 0:
        verify_data_exists(paths)
        print(f"\n✅ All data verified at: {paths['streaming_dataset']}")
    
    # Synchronize all ranks
    dist.barrier()

    # Create streaming dataset - using remote for source data and local for SSD cache
    print(f"\nLoading streaming dataset from: {paths['streaming_dataset']}")
    streaming_dataset_train = StreamingDataset(
        remote=paths['train_dir'],              # Read compressed data from here
        local="/local_disk0/streaming_cache/train",  # Cache decompressed data here (local SSD)
        batch_size=train_batch_size,
        num_canonical_nodes=num_canonical_nodes,
        shuffle=True,  # Enable shuffling for training
    )
    streaming_dataset_eval = StreamingDataset(
        remote=paths['test_dir'],               # Read compressed data from here
        local="/local_disk0/streaming_cache/test",   # Cache decompressed data here (local SSD)
        batch_size=eval_batch_size,
        num_canonical_nodes=num_canonical_nodes,
        shuffle=False,  # No shuffling for eval
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
