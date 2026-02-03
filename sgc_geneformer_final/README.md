# Geneformer Training - Final Version

This is the final version of the Geneformer pretraining code with configurable Databricks volume paths.

## Two-Step Workflow

### Step 1: Data Preparation (CPU Cluster)

Run `data_preparation.py` as a Databricks notebook with a **CPU cluster** to:
1. Download the Genecorpus-30M dataset from HuggingFace (~15GB)
2. Download the token dictionary
3. Convert to MDS (Mosaic Data Shard) format for streaming

```
📓 data_preparation.py → Run on CPU cluster (takes 30-60 mins)
```

### Step 2: Training (GPU Cluster)

After data preparation, run training with a **GPU cluster**:

```bash
composer train.py parameters.yaml
```

```
🚀 train.py → Run on GPU cluster with SGC CLI
```

## Configuration

### Volume Configuration (parameters.yaml)

```yaml
volume:
  catalog: main           # Databricks catalog name
  schema: guanyu_chen     # Databricks schema name  
  volume_name: sgc        # Databricks volume name
```

Volume path: `/Volumes/{catalog}/{schema}/{volume_name}`

### Data Paths (relative to volume root)

```yaml
data:
  source_dataset: geneformer/data/dataset/genecorpus_30M_2048.dataset
  streaming_dataset: geneformer/data/dataset/streaming/genecorpus_30M_2048.dataset
  token_dictionary: geneformer/token_dictionary.pkl
  test_split_ratio: 0.1
```

### Checkpoint Path

```yaml
checkpoints:
  folder: geneformer/checkpoints
```

## Files

| File | Description |
|------|-------------|
| `data_preparation.py` | Databricks notebook for data download & MDS conversion (run first) |
| `train.py` | Main training script |
| `cfgutils.py` | Configuration utilities |
| `parameters.yaml` | Training parameters and volume configuration |
| `train.yaml` | SGC CLI job configuration |
| `dependencies.yaml` | Python dependencies |
| `commands.sh` | Setup commands |

## Quick Start

1. **Update Configuration**
   - Edit `data_preparation.py` - update CATALOG, SCHEMA, VOLUME_NAME
   - Edit `parameters.yaml` - update volume section to match

2. **Run Data Preparation**
   - Import `data_preparation.py` as a Databricks notebook
   - Attach a CPU cluster
   - Run all cells (takes 30-60 mins)

3. **Run Training**
   - Use SGC CLI: `sgcli train submit -f train.yaml`
   - Or run directly: `composer train.py parameters.yaml`

## Data Flow

```
HuggingFace Dataset
        ↓
  data_preparation.py (CPU)
        ↓
  MDS Format in Volume
        ↓
    train.py (GPU)
        ↓
  Checkpoints in Volume
```
