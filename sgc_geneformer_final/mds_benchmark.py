# Databricks notebook source
# MAGIC %md
# MAGIC # MDS Conversion Benchmark
# MAGIC 
# MAGIC This notebook benchmarks different methods to convert HuggingFace datasets to MDS format.
# MAGIC Methods are ordered from expected fastest to slowest.
# MAGIC We use a 10% sample to test speed before running on full data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import os
import shutil
import time
import numpy as np

# Paths
VOLUME_PATH = "/Volumes/main/guanyu_chen/sgc"
SOURCE_DATASET = f"{VOLUME_PATH}/geneformer/data/dataset/genecorpus_30M_2048.dataset"
BENCHMARK_DIR = f"{VOLUME_PATH}/geneformer/benchmark"

# Sample size for benchmarking (10% = ~3M samples)
SAMPLE_FRACTION = 0.1
RANDOM_SEED = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load and Sample Dataset

# COMMAND ----------

from datasets import load_from_disk

print(f"Loading source dataset: {SOURCE_DATASET}")
full_dataset = load_from_disk(SOURCE_DATASET)
print(f"Full dataset: {len(full_dataset):,} samples")

# Take a sample for benchmarking
sample_size = int(len(full_dataset) * SAMPLE_FRACTION)
sample_dataset = full_dataset.select(range(sample_size))
print(f"Sample dataset: {len(sample_dataset):,} samples ({SAMPLE_FRACTION*100:.0f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Helper Functions

# COMMAND ----------

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


def cleanup_dir(path: str):
    """Remove directory if exists."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


results = []

def benchmark(func, name: str, output_dir: str):
    """Run a benchmark and return timing."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*60}")
    
    cleanup_dir(output_dir)
    
    start = time.time()
    try:
        func(output_dir)
        elapsed = time.time() - start
        
        size = get_dir_size(output_dir)
        samples_per_sec = len(sample_dataset) / elapsed
        
        print(f"\n✅ {name} Complete!")
        print(f"   Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"   Speed: {samples_per_sec:,.0f} samples/sec")
        print(f"   Output size: {size}")
        
        # Estimate full dataset time
        full_time_estimate = elapsed / SAMPLE_FRACTION
        print(f"   Estimated full dataset time: {full_time_estimate/60:.0f} min ({full_time_estimate/3600:.1f} hrs)")
        
        result = {
            'name': name,
            'time_sec': elapsed,
            'samples_per_sec': samples_per_sec,
            'size': size,
            'full_estimate_min': full_time_estimate / 60,
            'status': 'success'
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ {name} FAILED: {e}")
        result = {
            'name': name,
            'time_sec': float('inf'),
            'samples_per_sec': 0,
            'size': 'N/A',
            'full_estimate_min': float('inf'),
            'status': f'failed: {e}'
        }
    
    results.append(result)
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 🚀 Method 1: dataframe_to_mds (Expected Fastest)
# MAGIC 
# MAGIC Use Spark's native `dataframe_to_mds` for parallel conversion.

# COMMAND ----------

def method_dataframe_to_mds(output_dir: str, num_partitions: int = 32):
    """Use streaming's dataframe_to_mds for parallel Spark conversion."""
    from streaming.base.converters import dataframe_to_mds
    from pyspark.sql.functions import col
    from pyspark.sql.types import LongType
    
    print("Converting HF dataset to Spark DataFrame...")
    pdf = sample_dataset.to_pandas()
    spark_df = spark.createDataFrame(pdf)
    row_count = spark_df.count()
    print(f"Spark DataFrame: {row_count:,} rows")
    
    # Cast length to LongType to match expected 'int64'
    spark_df = spark_df.withColumn("length", col("length").cast(LongType()))
    
    # Repartition for parallel writes
    spark_df = spark_df.repartition(num_partitions)
    print(f"Repartitioned to {num_partitions} partitions")
    
    # MDS kwargs - dataframe_to_mds requires explicit types
    # 'int64' for integers, 'ndarray:int64' for numpy arrays
    mds_kwargs = {
        'out': output_dir,
        'columns': {'input_ids': 'ndarray:int64', 'length': 'int64'},
        'compression': 'zstd',
        'size_limit': '256mb'
    }
    
    print(f"Writing MDS with dataframe_to_mds (parallel)...")
    dataframe_to_mds(spark_df, merge_index=True, mds_kwargs=mds_kwargs)
    print("Done!")

# COMMAND ----------

benchmark(method_dataframe_to_mds, "dataframe_to_mds (32 partitions)", f"{BENCHMARK_DIR}/method1_df_to_mds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 🚀 Method 2: Parallel Thread Writers

# COMMAND ----------

def method_parallel_threads(output_dir: str, num_workers: int = 8):
    """Write shards in parallel using threads."""
    from streaming import MDSWriter
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    columns = {'input_ids': "ndarray", 'length': 'int'}
    total = len(sample_dataset)
    shard_size = (total + num_workers - 1) // num_workers
    
    # Pre-convert to list for faster access
    print("Converting to pandas...")
    all_data = sample_dataset.to_pandas().to_dict('records')
    print(f"Converted {len(all_data):,} records")
    
    written_counts = []
    
    def write_shard(shard_id):
        start = shard_id * shard_size
        end = min(start + shard_size, total)
        shard_dir = f"{output_dir}/shard_{shard_id}"
        os.makedirs(shard_dir, exist_ok=True)
        
        with MDSWriter(out=shard_dir, columns=columns, compression='zstd') as out:
            for i in range(start, end):
                x = all_data[i]
                out.write({"input_ids": x["input_ids"], "length": x["length"]})
        return end - start
    
    print(f"Writing {num_workers} shards in parallel...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        counts = list(tqdm(executor.map(write_shard, range(num_workers)), total=num_workers, desc="Shards"))
    
    print(f"Wrote {sum(counts):,} samples across {num_workers} shards")

# COMMAND ----------

benchmark(method_parallel_threads, "Parallel Threads (8)", f"{BENCHMARK_DIR}/method2_parallel")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 🚀 Method 3: Large Batch Processing

# COMMAND ----------

def method_large_batch(output_dir: str, batch_size: int = 100000):
    """Process in very large batches for efficiency."""
    from streaming import MDSWriter
    from tqdm import tqdm
    
    columns = {'input_ids': "ndarray", 'length': 'int'}
    total = len(sample_dataset)
    
    print(f"Writing MDS in batches of {batch_size:,}...")
    with MDSWriter(out=output_dir, columns=columns, compression='zstd', size_limit='512mb') as out:
        for start in tqdm(range(0, total, batch_size), desc="Batches"):
            end = min(start + batch_size, total)
            # Get batch as pandas for fast iteration
            batch_df = sample_dataset.select(range(start, end)).to_pandas()
            
            for _, row in batch_df.iterrows():
                out.write({"input_ids": row['input_ids'], "length": row['length']})

# COMMAND ----------

benchmark(method_large_batch, "Large Batch (100k)", f"{BENCHMARK_DIR}/method3_batch")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Method 4: Pandas to_dict (Current)

# COMMAND ----------

def method_pandas_to_dict(output_dir: str):
    """Current method: convert to pandas, then to list of dicts."""
    from streaming import MDSWriter
    from tqdm import tqdm
    
    columns = {'input_ids': "ndarray", 'length': 'int'}
    
    print("Converting to pandas...")
    dataset_list = sample_dataset.to_pandas().to_dict('records')
    print(f"Converted {len(dataset_list):,} records")
    
    print("Writing MDS...")
    with MDSWriter(out=output_dir, columns=columns, compression='zstd') as out:
        for x in tqdm(dataset_list, desc="Writing"):
            out.write({"input_ids": x["input_ids"], "length": x["length"]})

# COMMAND ----------

benchmark(method_pandas_to_dict, "Pandas to_dict (current)", f"{BENCHMARK_DIR}/method4_pandas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Method 5: Arrow Direct Access

# COMMAND ----------

def method_arrow_direct(output_dir: str):
    """Use PyArrow tables for iteration."""
    from streaming import MDSWriter
    from tqdm import tqdm
    
    columns = {'input_ids': "ndarray", 'length': 'int'}
    
    print("Accessing Arrow table...")
    arrow_table = sample_dataset.data.table
    print(f"Arrow table: {arrow_table.num_rows:,} rows")
    
    print("Writing MDS...")
    input_ids_col = arrow_table.column('input_ids')
    length_col = arrow_table.column('length')
    
    with MDSWriter(out=output_dir, columns=columns, compression='zstd') as out:
        for i in tqdm(range(arrow_table.num_rows), desc="Writing"):
            input_ids = np.array(input_ids_col[i].as_py(), dtype=np.int64)
            out.write({"input_ids": input_ids, "length": length_col[i].as_py()})

# COMMAND ----------

benchmark(method_arrow_direct, "Arrow Direct", f"{BENCHMARK_DIR}/method5_arrow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Method 6: HF Direct (Baseline - Slowest)

# COMMAND ----------

def method_hf_direct(output_dir: str):
    """Direct iteration over HuggingFace dataset (slow baseline)."""
    from streaming import MDSWriter
    from tqdm import tqdm
    
    columns = {'input_ids': "ndarray", 'length': 'int'}
    
    print("Writing MDS (direct HF iteration)...")
    with MDSWriter(out=output_dir, columns=columns, compression='zstd') as out:
        for x in tqdm(sample_dataset, desc="Writing", total=len(sample_dataset)):
            input_ids = np.array(x["input_ids"], dtype=np.int64)
            out.write({"input_ids": input_ids, "length": x["length"]})

# COMMAND ----------

benchmark(method_hf_direct, "HF Direct (baseline)", f"{BENCHMARK_DIR}/method6_hf_direct")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Results Summary

# COMMAND ----------

import pandas as pd

# Create results DataFrame (filter out failed)
df = pd.DataFrame([r for r in results if r['status'] == 'success'])
if len(df) > 0:
    df = df.sort_values('time_sec')
    df['speedup'] = df['time_sec'].max() / df['time_sec']

    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"\nSample size: {len(sample_dataset):,} samples ({SAMPLE_FRACTION*100:.0f}% of full dataset)")
    print(f"Full dataset: {len(full_dataset):,} samples")
    print()

    # Display results
    print(f"{'Method':<30} | {'Time':>8} | {'Speed':>15} | {'Speedup':>7} | {'Est. Full':>10}")
    print("-"*80)
    for _, row in df.iterrows():
        print(f"{row['name']:<30} | {row['time_sec']:>7.1f}s | {row['samples_per_sec']:>12,.0f}/s | {row['speedup']:>6.1f}x | {row['full_estimate_min']:>7.0f} min")

    print()
    print("="*80)
    print(f"🏆 FASTEST: {df.iloc[0]['name']}")
    print(f"   Speed: {df.iloc[0]['samples_per_sec']:,.0f} samples/sec")
    print(f"   Estimated time for full dataset: {df.iloc[0]['full_estimate_min']:.0f} minutes ({df.iloc[0]['full_estimate_min']/60:.1f} hours)")
    print("="*80)

# Show failed methods
failed = [r for r in results if r['status'] != 'success']
if failed:
    print("\n❌ Failed methods:")
    for r in failed:
        print(f"   - {r['name']}: {r['status']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Cleanup Benchmark Files

# COMMAND ----------

# Uncomment to cleanup benchmark files
# shutil.rmtree(BENCHMARK_DIR)
# print(f"Cleaned up: {BENCHMARK_DIR}")
