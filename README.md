# Geneformer Pretraining with Serverless GPU CLI (SGCLI)

This repository contains examples for running distributed training workloads on Databricks Serverless GPU compute using SGCLI.

## Repository Structure

```
├── SGC_hello_world/          # Simple hello world example (A10 GPUs)
├── sgc_geneformer_final/     # Geneformer pretraining (H100 GPUs)
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
pip install sgcli_wheel/databricks_serverless_gpu_cli-0.0.2-py3-none-any.whl --force-reinstall
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

Import `sgc_geneformer_final/data_preparation.py` as a Databricks notebook and run it on a CPU cluster.

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

Edit `sgc_geneformer_final/parameters.yaml`:

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

Edit `sgc_geneformer_final/train.yaml`:

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
cd sgc_geneformer_final
sgcli run -f train.yaml --watch
```

### Monitoring

- **Terminal**: Logs stream with `--watch`
- **MLflow**: View metrics in Databricks MLflow UI
- **Checkpoints**: Saved to your volume at `geneformer/checkpoints/`

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
  - mosaicml==0.23.5
  - mosaicml-streaming==0.8.0
  - transformers==4.44.0
  # ... other packages
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

---

## Resources

- [Databricks Serverless GPU Docs](https://docs.databricks.com/aws/en/compute/serverless/gpu)
- [MosaicML Composer Docs](https://docs.mosaicml.com/projects/composer/)
- [Geneformer Paper](https://www.nature.com/articles/s41586-023-06139-9)
