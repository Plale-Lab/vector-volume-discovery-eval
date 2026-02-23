import time
import pandas as pd
from tqdm import tqdm
import os

import config
from services import minio_client, qdrant_client, vlm_encoder


VECTOR_FIELDS_TO_TEST = ["initial", "max_pooling", "mean_pooling"]

BENCHMARK_QUERIES = [
    "What is a process?",
    "Describe CPU scheduling",
    "Explain the concept of a deadlock",
    "What is fourier optics?",
    "Difference between thread and process"
]

RESULTS_FILE = "logs/ablation_results.csv"

def run_latency_test_for_field(model, processor, q_client, field_name):
    """
    Runs the benchmark queries against a SPECIFIC vector field.
    Returns the average latency in ms.
    """
    latencies = []
    
    for query_text in BENCHMARK_QUERIES:
        query_vector = vlm_encoder.encode_query(model, processor, query_text, config.DEVICE)
        
        start_time = time.time()
        qdrant_client.search_qdrant(
            q_client, 
            config.COLLECTION_NAME, 
            query_vector, 
            top_k=3, 
            vector_name=field_name
        )
        search_time = time.time() - start_time
        
        latencies.append(search_time * 1000)
        
    return sum(latencies) / len(latencies)

def main():
    print("--- Initializing Vector Field Ablation Study ---")
    
    try:
        model, processor = vlm_encoder.load_vlm_model(config.MODEL_NAME, config.DEVICE)
        q_client = qdrant_client.get_qdrant_client(config.QDRANT_HOST, config.QDRANT_PORT)
    except Exception as e:
        print(f"Failed to initialize services: {e}")
        return

    count = q_client.count(config.COLLECTION_NAME, exact=True).count
    print(f"Benchmarking against {count} indexed documents.")
    
    results = []


    print("\n--- Starting Benchmark Loops ---")
    for field in VECTOR_FIELDS_TO_TEST:
        print(f"\nEvaluating Field: {field}")

        run_times = []
        for i in range(3):
            avg_ms = run_latency_test_for_field(model, processor, q_client, field)
            run_times.append(avg_ms)
            print(f"  Run {i+1}: {avg_ms:.2f} ms")
        final_avg = sum(run_times) / len(run_times)
        best_time = min(run_times)
        
        print(f"  >> Result: {final_avg:.2f} ms avg")
        
        results.append({
            "vector_field": field,
            "avg_latency_ms": final_avg,
            "best_latency_ms": best_time,
        })

    print("\n--- Ablation Study Complete ---")
    df = pd.DataFrame(results)
    
    baseline_latency = df.loc[df['vector_field'] == 'initial', 'avg_latency_ms'].values[0]
    df['speedup_factor'] = baseline_latency / df['avg_latency_ms']
    
    print(df)
    os.makedirs("logs", exist_ok=True)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()