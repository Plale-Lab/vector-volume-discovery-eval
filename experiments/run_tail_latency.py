import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from qdrant_client import models

import config
from services import minio_client, qdrant_client, vlm_encoder


CORPUS_SIZE_STEPS = [1278] 
BENCHMARK_QUERIES = [
    "What is a process?", "What is a thread?", "Explain deadlock", 
    "Virtual memory definition", "CPU scheduling types", "Semaphores vs Mutex",
    "paging vs segmentation", "context switch overhead", "process control block",
    "bankers algorithm", "thrashing causes", "kernel mode vs user mode"
]
RESULTS_FILE = "logs/tail_latency_results.csv"

def run_raw_latency_benchmark(model, processor, q_client) -> list:
    """
    Runs queries and returns A LIST of all latencies (no averaging).
    """
    raw_latencies = []
    
    for query_text in BENCHMARK_QUERIES:

        query_vector = vlm_encoder.encode_query(model, processor, query_text, config.DEVICE)
        
        start_time = time.time()
        qdrant_client.search_qdrant(
            q_client, 
            config.COLLECTION_NAME, 
            query_vector, 
            top_k=3,
            vector_name="initial"
        )
        qdrant_time = time.time() - start_time

        raw_latencies.append(qdrant_time * 1000)
        
    return raw_latencies

def main():
    print("--- Initializing Tail Latency Analysis ---")
    

    try:
        model, processor = vlm_encoder.load_vlm_model(config.MODEL_NAME, config.DEVICE)
        q_client = qdrant_client.get_qdrant_client(config.QDRANT_HOST, config.QDRANT_PORT)
        m_client = minio_client.get_minio_client(
            config.MINIO_HOST, config.MINIO_ACCESS_KEY, config.MINIO_SECRET_KEY, config.MINIO_SECURE
        )
    except Exception as e:
        print(f"Failed to initialize services: {e}")
        return


    current_index_size = q_client.count(config.COLLECTION_NAME, exact=True).count
    print(f"Analyzing tail latency against {current_index_size} documents.")


    print("Collecting latency samples...")
    all_latencies = []

    for _ in tqdm(range(5), desc="Sampling Runs"):
        latencies = run_raw_latency_benchmark(model, processor, q_client)
        all_latencies.extend(latencies)

    avg_latency = np.mean(all_latencies)
    p50_latency = np.percentile(all_latencies, 50) # Median
    p95_latency = np.percentile(all_latencies, 95) # 95% of queries are faster than this
    p99_latency = np.percentile(all_latencies, 99) # 99% of queries are faster than this
    max_latency = np.max(all_latencies)            # The single worst outlier

    print("\n--- Tail Latency Results (ms) ---")
    print(f"Count: {len(all_latencies)} queries")
    print(f"Avg:   {avg_latency:.2f} ms")
    print(f"P50:   {p50_latency:.2f} ms")
    print(f"P95:   {p95_latency:.2f} ms ")
    print(f"P99:   {p99_latency:.2f} ms")
    print(f"Max:   {max_latency:.2f} ms")

    os.makedirs("logs", exist_ok=True)
    df = pd.DataFrame(all_latencies, columns=["latency_ms"])
    df.to_csv(RESULTS_FILE, index=False)
    print(f"Raw data saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()