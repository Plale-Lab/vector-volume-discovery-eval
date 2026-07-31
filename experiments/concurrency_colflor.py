import time
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from services import minio_client, qdrant_client, vlm_encoder


CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32] 

TOTAL_QUERIES_PER_STEP = 100 

BENCHMARK_QUERIES = [
    "What is a process?", "What is a thread?", "Explain the concept of a deadlock",
    "What is virtual memory?", "Describe CPU scheduling", "Semaphores vs Mutex",
    "paging vs segmentation", "context switch overhead", "backpropagation algorithm",
    "bankers algorithm", "reconstruction methods", "kernel mode vs user mode"
]

QUERIES_TO_RUN = random.choices(BENCHMARK_QUERIES, k=TOTAL_QUERIES_PER_STEP)

RESULTS_FILE = "logs/concurrency_results.csv"


def run_single_retrieval(query_text: str, model, processor, q_client):
    """
    Runs the full end-to-end RAG pipeline (VLM + Qdrant) for one query.
    Returns the total latency in milliseconds.
    """
    try:
        start_time = time.perf_counter()

        query_vector = vlm_encoder.encode_query(
            model, processor, query_text, config.DEVICE
        )

        results = qdrant_client.search_qdrant(
            q_client, 
            config.COLLECTION_NAME, 
            query_vector, 
            top_k=3,
            vector_name="initial"
        )

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000
    
    except Exception as e:
        print(f"Error during query '{query_text}': {e}")
        return None

def main():
    """
    Main function to orchestrate the Concurrency Test.
    """
    
    print("--- Initializing RAG Concurrency Test ---")
    print("Loading all services into memory...")
    try:
        model, processor = vlm_encoder.load_vlm_model(config.MODEL_NAME, config.DEVICE)
        
        q_client = qdrant_client.get_qdrant_client(config.QDRANT_HOST, config.QDRANT_PORT)
        
        current_index_size = q_client.count(config.COLLECTION_NAME, exact=True).count
        print(f"Services loaded. Testing against index of {current_index_size} documents.")
        
    except Exception as e:
        print(f"Failed to initialize services. Exiting. Error: {e}")
        return

    results_log = []
    
    for num_workers in CONCURRENCY_LEVELS:
        print(f"\n--- Testing Concurrency Level: {num_workers} Workers ---")
        
        latencies = []
        
        total_start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_single_retrieval, 
                    query, model, processor, q_client
                ) for query in QUERIES_TO_RUN
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Workers={num_workers}"):
                result_ms = future.result()
                if result_ms is not None:
                    latencies.append(result_ms)

        total_end_time = time.perf_counter()
        total_time_taken = total_end_time - total_start_time
        
        if not latencies:
            print("No queries were successfully processed.")
            continue
            
        throughput_qps = len(latencies) / total_time_taken 
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"--- Results (Workers={num_workers}) ---")
        print(f"  Throughput: {throughput_qps:.2f} QPS")
        print(f"  Avg Latency: {avg_latency:.2f} ms")
        print(f"  P95 Latency: {p95_latency:.2f} ms")
        print(f"  P99 Latency: {p99_latency:.2f} ms")
        
        results_log.append({
            "num_workers": num_workers,
            "throughput_qps": throughput_qps,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency
        })

    print("\n--- Concurrency Test Complete ---")
    os.makedirs("logs", exist_ok=True)
    results_df = pd.DataFrame(results_log)
    results_df.to_csv(RESULTS_FILE, index=False)
    
    print(f"Results saved to {RESULTS_FILE}")
    print(results_df)

if __name__ == "__main__":
    main()