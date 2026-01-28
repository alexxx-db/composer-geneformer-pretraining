# Geneformer Training - Final Version

This is the final version of the Geneformer pretraining code with configurable Databricks volume paths and automatic MDS dataset preparation.

## Features

- **Configurable Volume Paths**: Set catalog, schema, and volume_name in `parameters.yaml`
- **Automatic MDS Preparation**: If train/test streaming datasets don't exist, they are automatically prepared from the source HuggingFace dataset
- **Volume-based Checkpoints**: Checkpoints are saved to the configured Databricks volume

## Configuration

### Volume Configuration (parameters.yaml)

```yaml
volume:
  catalog: main           # Databricks catalog name
  schema: guanyu_chen     # Databricks schema name  
  volume_name: sgc        # Databricks volume name
```

This constructs the volume path: `/Volumes/{catalog}/{schema}/{volume_name}`

### Data Paths (relative to volume root)

```yaml
data:
  # Source dataset (HuggingFace format) - used for MDS preparation
  source_dataset: geneformer/data/dataset/genecorpus_30M_2048.dataset
  
  # Streaming dataset location (MDS format) - will be created if not exists
  streaming_dataset: geneformer/data/dataset/streaming/genecorpus_30M_2048.dataset
  
  # Token dictionary file
  token_dictionary: geneformer/data/token_dictionary.pkl
  
  # Test split ratio for train/test split
  test_split_ratio: 0.1
```

### Checkpoint Path

```yaml
checkpoints:
  folder: geneformer/checkpoints
```

## Usage

1. Update `parameters.yaml` with your Databricks volume configuration
2. Ensure the source dataset and token dictionary exist in the volume
3. Run training:

```bash
composer train.py parameters.yaml
```

The script will:
1. Check if streaming dataset (train/test) exists
2. If not, prepare it from the source HuggingFace dataset
3. Train the model and save checkpoints to the volume

## Files

- `train.py` - Main training script
- `cfgutils.py` - Configuration utilities (optimizer, scheduler, callbacks, etc.)
- `parameters.yaml` - Training parameters and volume configuration
- `train.yaml` - SGC CLI job configuration
- `dependencies.yaml` - Python dependencies
- `commands.sh` - Setup commands
