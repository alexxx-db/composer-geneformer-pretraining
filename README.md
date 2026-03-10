# Geneformer Pretraining with Serverless GPU CLI (SGCLI)

This repository contains examples for running distributed training workloads on Databricks Serverless GPU compute using SGCLI.

## Repository Structure

```
├── SGC_hello_world/          # Simple hello world example (A10 GPUs)
├── SGC_geneformer/           # Geneformer pretraining (H100 GPUs)
├── sgcli_wheel/              # SGCLI wheel package
└── README.md
```

---

## Part 1: SGCLI Setup

### Prerequisites

- macOS or Linux
- Python 3.10+
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/)

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd composer_geneformer_pretrain
```

### Step 2: Install Databricks CLI

```bash
# macOS
brew install databricks

# Or via pip
pip install databricks-cli
```

### Step 3: Authenticate to Databricks

```bash
databricks auth login --host https://your-workspace.cloud.databricks.com
```

This creates a `~/.databrickscfg` file with your credentials:

```ini
[DEFAULT]
host      = https://your-workspace.cloud.databricks.com
auth_type = databricks-cli
```

### Step 4: Set Your Profile (Optional)

If you have multiple profiles, set the active one:

```bash
export DATABRICKS_CONFIG_PROFILE=DEFAULT
```

### Step 5: Install SGCLI

```bash
pip install sgcli_wheel/databricks_serverless_gpu_cli-0.0.5-py3-none-any.whl --force-reinstall
```

Verify installation:

```bash
sgcli --help
```

---

## Part 2: Run Hello World Example

A simple test to verify SGCLI and GPU access are working.

### Step 1: Update Configuration

Edit `SGC_hello_world/train_workload.yaml`:

```yaml
experiment_name: torchrun-hello-world-<your-name>  # Change this
code_source:
  type: snapshot
  snapshot:
    repo_path: /path/to/your/local/repo  # Update to your local path
```

### Step 2: Submit the Workload

```bash
cd SGC_hello_world
sgcli run -f train_workload.yaml --watch
```

The `--watch` flag streams logs to your terminal.

### Expected Output

You should see:
- CUDA device detection
- Matrix multiplication test
- "CUDA is working!" message

---

## Part 3: Geneformer Pretraining

Full distributed pretraining of Geneformer on H100 GPUs.

### Overview

1. **Prepare Data** (CPU cluster in Databricks notebook)
2. **Configure Parameters** (local)
3. **Submit Training Job** (SGCLI)

---

### Step 1: Prepare Data (One-time Setup)

The data preparation runs on a **Databricks CPU cluster** (not via SGCLI).

#### 1.1 Create a Unity Catalog Volume

In Databricks, create a volume to store data and checkpoints:

```
/Volumes/<catalog>/<schema>/<volume_name>/
```

Example: `/Volumes/main/guanyu_chen/sgc/`

#### 1.2 Run Data Preparation Notebook

Import `SGC_geneformer/data_preparation.py` as a Databricks notebook and run it on a CPU cluster.

**Before running, update the configuration at the top:**

```python
# ============================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================
CATALOG = "main"              # Your catalog
SCHEMA = "your_schema"        # Your schema
VOLUME_NAME = "sgc"           # Your volume name
```

The notebook will:
1. Download the Genecorpus-30M dataset from HuggingFace (~30M samples)
2. Download the token dictionary
3. Convert to MDS (Mosaic Data Shard) format for efficient streaming

**Expected output structure:**
```
/Volumes/main/your_schema/sgc/geneformer/
├── data/
│   ├── token_dictionary.pkl
│   └── dataset/
│       └── streaming/
│           └── genecorpus_30M_2048.dataset/
│               ├── train/
│               │   └── index.json (+ shard files)
│               └── test/
│                   └── index.json (+ shard files)
└── checkpoints/
```

---

### Step 2: Configure Parameters

#### 2.1 Update `parameters.yaml`

Edit `SGC_geneformer/parameters.yaml`:

```yaml
# ============================================
# Databricks Volume Configuration
# ============================================
volume:
  catalog: main              # Your catalog
  schema: your_schema        # Your schema  
  volume_name: sgc           # Your volume name

# Data paths relative to the volume root
data:
  source_dataset: geneformer/data/dataset/genecorpus_30M_2048.dataset
  streaming_dataset: geneformer/data/dataset/streaming/genecorpus_30M_2048.dataset
  token_dictionary: geneformer/data/token_dictionary.pkl
  test_split_ratio: 0.1

# Checkpoint path relative to volume root
checkpoints:
  folder: geneformer/checkpoints

# ============================================
# Training Configuration
# ============================================
train_batch_size: 16          # Per-device batch size
eval_batch_size: 16
max_duration: 20ep            # Number of epochs
eval_interval: 5ep            # Evaluate every N epochs
save_interval: 5ep            # Save checkpoint every N epochs

# For quick testing, use subset of batches (-1 = use all)
train_subset_num_batches: 100  # Set to -1 for full training
eval_subset_num_batches: 10    # Set to -1 for full eval
```

#### 2.2 Update `train.yaml`

Edit `SGC_geneformer/train.yaml`:

```yaml
experiment_name: geneformer-<your-name>  # Change this

compute:
  gpus: 16                    # Number of GPUs (8 = 1 node, 16 = 2 nodes)
  gpu_type: h100              # GPU type

code_source:
  type: snapshot
  snapshot:
    repo_path: /path/to/your/local/repo  # Update to your local path
```

#### 2.3 Configure MLflow Logging (Optional)

To log metrics to MLflow, update `parameters.yaml`:

```yaml
loggers:
  mlflow:
    tracking_uri: databricks
    experiment_name: mlflow_experiments/geneformer_pretraining
```

---

### Step 3: Submit Training Job

```bash
cd SGC_geneformer
sgcli run -f train.yaml --watch
```

### Monitoring

- **Terminal**: Logs stream with `--watch`
- **MLflow**: View metrics in Databricks MLflow UI
- **Checkpoints**: Saved to your volume at `geneformer/checkpoints/`

---

## Part 4: Testing Failure Recovery (Optional)

This section describes how to test checkpoint recovery and autoresume functionality by intentionally failing training at a specific epoch.

### Overview

The `FailureTestCallback` allows you to:
- Intentionally crash training at a specified epoch
- Test that SGCLI correctly retries the job
- Verify that training resumes from the last checkpoint
- Confirm the job completes successfully after N failures

### Step 1: Enable Failure Testing

Edit `SGC_geneformer/parameters.yaml`:

```yaml
# Auto resume (required for failure recovery)
autoresume: True

# ============================================
# Failure Test Configuration
# ============================================
failure_test:
  enabled: true           # Enable failure testing
  fail_at_epoch: 7        # Fail at epoch 7 (0-indexed)
  max_failures: 3         # Fail 3 times, then continue on 4th attempt
```

### Step 2: Configure SGCLI Retries

Edit `SGC_geneformer/train.yaml`:

```yaml
max_retries: 3  # Must be >= max_failures for automatic recovery
```

**Important**: Set `max_retries` >= `max_failures` so SGCLI automatically restarts the job after each failure.

### Step 3: Configure Checkpoints

Ensure checkpoints are saved before the failure epoch:

```yaml
# parameters.yaml
save_interval: 5ep        # Save checkpoint every 5 epochs
max_duration: 20ep        # Total training duration
```

With `fail_at_epoch: 7` and `save_interval: 5ep`, a checkpoint is saved at epoch 5 before the failure at epoch 7.

### Step 4: Submit and Watch

```bash
cd SGC_geneformer
sgcli run -f train.yaml --watch
```

### Expected Behavior

| Attempt | What Happens |
|---------|--------------|
| 1 | Train epochs 0-5, save checkpoint, fail at epoch 7 |
| 2 | Resume from epoch 5, fail at epoch 7 |
| 3 | Resume from epoch 5, fail at epoch 7 |
| 4 | Resume from epoch 5, **skip failure**, complete training |

### Console Output

**On failure (attempts 1-3):**
```
============================================================
💥 INTENTIONAL FAILURE (Test Mode)
============================================================
  Epoch: 7
  Failure count: 2/3
  Remaining failures: 1
============================================================
```

**On successful continue (attempt 4):**
```
============================================================
✅ FAILURE TEST: Skipping failure (already failed 3 times)
   Training will continue normally from checkpoint
============================================================
```

### How It Works

1. **Failure counter**: Stored in `{checkpoint_folder}/failure_counter.json`
2. **Distributed sync**: Uses `torch.distributed.broadcast` to ensure all ranks fail/continue together
3. **Persistence**: Counter persists across job restarts via the shared volume
4. **Auto-reset**: Counter resets when training completes successfully

### Disable After Testing

Remember to disable failure testing for production runs:

```yaml
failure_test:
  enabled: false
```

---

## Configuration Reference

### train.yaml (Workload Definition)

| Field | Description |
|-------|-------------|
| `experiment_name` | Unique name for your experiment |
| `compute.gpus` | Number of GPUs (8 per node for H100) |
| `compute.gpu_type` | `a10` or `h100` |
| `code_source.snapshot.repo_path` | Local path to repository |
| `environment.dependencies` | Path to dependencies.yaml |

### parameters.yaml (Training Config)

| Field | Description |
|-------|-------------|
| `volume.*` | Databricks volume location |
| `data.*` | Data paths relative to volume |
| `train_batch_size` | Per-device batch size |
| `max_duration` | Training duration (e.g., `20ep`, `1000ba`) |
| `fsdp_config` | FSDP sharding configuration |

### dependencies.yaml (Python Environment)

```yaml
version: "4"
dependencies:
  - mosaicml==0.32.1
  - mosaicml-streaming==0.13.0
  - transformers==4.44.0
  - mlflow>=3.6.0
  # ... other packages

# Additionally installed via commands.sh:
# torch==2.8.0, torchvision==0.23.0, torchaudio==2.8.0 (CUDA 12.6)
```

---

## Troubleshooting

### NCCL Timeout Errors

For multi-node training, adjust timeouts in `train.yaml`:

```yaml
environment:
  env_variables:
    NCCL_TIMEOUT: "1800"                      # 30 min
    TORCH_DIST_INIT_BARRIER_TIMEOUT: "1800"   # 30 min
```

### Data Not Found

If training fails with "DATA NOT FOUND":
1. Verify the data preparation notebook completed successfully
2. Check that paths in `parameters.yaml` match your volume structure
3. Ensure the MDS `index.json` files exist in train/ and test/ directories

### Checking Job Status

```bash
# List recent jobs
sgcli list

# Get job details
sgcli status <job-id>

# Cancel a job
sgcli cancel <job-id>
```

---

## Quick Start Checklist

- [ ] Install Databricks CLI and authenticate
- [ ] Install SGCLI wheel
- [ ] Run Hello World to verify setup
- [ ] Create Unity Catalog volume
- [ ] Run data preparation notebook (CPU cluster)
- [ ] Update `parameters.yaml` with your volume paths
- [ ] Update `train.yaml` with your repo path
- [ ] Submit training: `sgcli run -f train.yaml --watch`
- [ ] (Optional) Test failure recovery with `failure_test.enabled: true`

---

## Resources

- [Databricks Serverless GPU Docs](https://docs.databricks.com/aws/en/compute/serverless/gpu)
- [MosaicML Composer Docs](https://docs.mosaicml.com/projects/composer/)
- [Geneformer Paper](https://www.nature.com/articles/s41586-023-06139-9)

---

## SGCLI Changelog

### Version 0.0.5

- [Bug-fix] Remove print_error from env_secrets
- [CNXT-1859] Remove python script from sgcli
- [Bug-fix] Fix uncommitted snapshot code path directories
- [Bug-fix] Fix call to experiment creation
- [CNXT-1778] Add color for sgcli
- [CNXT-1844] Remove git clone from cli src
- [CNXT-1727] Fix bug in setting permissions in config
- [CNXT-1832] Improve performance of `sgcli get runs` command
- [CNXT-1784] Provide yaml pointers from sgcli tool
- [CNXT-1638] Add dry run command
- [CNXT-1828] Add budget policy attribution field

### Version 0.0.4

- [CNXT-1817] Fix uncommitted changes not being captured in snapshot. Also allows include_paths when changes are outside those paths.
- [CNXT-1809] Update log syntax and allow for downloading to a specified directory
- [CNXT-1806] Add support for automatic permission granting after job submission with `grant_permissions` field.
- [CNXT-1638] Make logs from dependency installation unbuffered

### Version 0.0.3

- [CNXT-1727] Allow uncommitted changes and simplify git UX.
- [CNXT-1759] Add support non git folders.
- [CNXT-1713] Add support for variable interpretation in local and remote.
- [Bug-fix] Fix hardcoded email from databricks.com to actual user email.

### Version 0.0.2

- [CNXT-1727] Support for subfolders in snapshot via git archive. This is a major update to speed up snapshot especially in large repos.
- [CNXT-1716] Remove unique suffix from job run names. Experiment corresponds exactly to Job Run name.

### Version 0.0.1

- [CNXT-1727] Add changelog command.
- [CNXT-1706] Fix get runs hyperlinks and add get status hyperlinks.
- [CNXT-1713] Add experiment_name validation and reduce log level.
- [CNXT-1624] Validate fields and gpu num during workload submission.
- [CNXT-1538] Add Streaming Logs capability.
