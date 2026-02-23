import time
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from services import qdrant_client, vlm_encoder

CONCURRENCY_LEVELS = [1, 10, 50, 100, 200] 
TOTAL_QUERIES_PER_STEP = 100 
BENCHMARK_QUERIES = [
    "how to roast duck",
]
RESULTS_FILE = "logs/qdrant_concurrency_results.csv" 


def run_single_qdrant_search(query_vector: np.ndarray, q_client):
    """
    Runs only the Qdrant search for a pre-encoded vector.
    Returns the search latency in milliseconds.
    """
    try:
        start_time = time.perf_counter()

        results = qdrant_client.search_qdrant(
            q_client, 
            config.COLLECTION_NAME, 
            query_vector, 
            top_k=3,
            vector_name="mean_pooling"
        )

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000
    
    except Exception as e:
        print(f"Error during Qdrant search: {e}")
        return None

def main():
    
    print("--- Initializing Qdrant Concurrency Test ---")
    print("Loading all services and pre-encoding single query...")
    try:
        model, processor = vlm_encoder.load_vlm_model(config.MODEL_NAME, config.DEVICE)
        q_client = qdrant_client.get_qdrant_client(config.QDRANT_HOST, config.QDRANT_PORT)
        
        current_index_size = q_client.count(config.COLLECTION_NAME, exact=True).count
        print(f"Services loaded. Testing against index of {current_index_size} documents.")
        

        test_query_text = BENCHMARK_QUERIES[0]
        vectors_dict = vlm_encoder.encode_query(
            model, processor, test_query_text, config.DEVICE
        )
        pre_encoded_vector = vectors_dict["initial"]
        print(f"Pre-encoded vector for query: '{test_query_text}'")
        
    except Exception as e:
        print(f"Failed to initialize services or pre-encode query. Exiting. Error: {e}")
        return

    results_log = []
    
    
    for num_workers in CONCURRENCY_LEVELS:
        print(f"\n--- Testing Concurrency Level: {num_workers} Workers ---")
        
        latencies = []
        
        total_start_time = time.perf_counter()
        queries_needed = max(TOTAL_QUERIES_PER_STEP, num_workers) 
        VECTORS_TO_RUN = [pre_encoded_vector] * queries_needed
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_single_qdrant_search, 
                    vector, q_client
                ) for vector in VECTORS_TO_RUN
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

    print("\n--- Qdrant Concurrency Test Complete ---")
    os.makedirs("logs", exist_ok=True)
    results_df = pd.DataFrame(results_log)
    results_df.to_csv(RESULTS_FILE, index=False)
    
    print(f"Results saved to {RESULTS_FILE}")
    print(results_df)

if __name__ == "__main__":
    main()