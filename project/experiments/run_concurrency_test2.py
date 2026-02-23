import asyncio
import time
import os
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm 
from qdrant_client import AsyncQdrantClient
import config
from services import vlm_encoder


CONCURRENCY_LEVELS = [10000] 
TOTAL_QUERIES_PER_STEP = 1000 
BENCHMARK_QUERIES = ["What is a kernel?"]
RESULTS_FILE = "logs/qdrant_concurrency_results.csv"

async def run_async_search(client, vector, semaphore):
    """Performs a single search using a semaphore to throttle active requests."""
    async with semaphore:
        try:
            start = time.perf_counter()

            await client.query_points(
                collection_name=config.COLLECTION_NAME,
                query=vector,
                using="initial",
                limit=3,
                timeout=300
            )
            return (time.perf_counter() - start) * 1000
        except Exception as e:
            print(f"DEBUG: {e}")
            return None

async def run_benchmark_step(level, pre_encoded_vector):
    """Sets up the async client and runs a batch of queries for a specific level."""
    client = AsyncQdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    semaphore = asyncio.Semaphore(level)
    
    
    queries_to_run = max(TOTAL_QUERIES_PER_STEP, level)
    tasks = [run_async_search(client, pre_encoded_vector, semaphore) for _ in range(queries_to_run)]
    
    start_time = time.perf_counter()
    
    latencies = await tqdm.gather(*tasks, desc=f"Workers={level}")
    total_time = time.perf_counter() - start_time
    
    await client.close()
    
    valid_latencies = [l for l in latencies if l is not None]
    if not valid_latencies:
        return None

    return {
        "num_workers": level,
        "throughput_qps": len(valid_latencies) / total_time,
        "avg_latency_ms": np.mean(valid_latencies),
        "p95_latency_ms": np.percentile(valid_latencies, 95),
        "p99_latency_ms": np.percentile(valid_latencies, 99)
    }

def main():
    print("--- Initializing Qdrant Concurrency Test ---")
    try:
        
        model, processor = vlm_encoder.load_vlm_model(config.MODEL_NAME, config.DEVICE)
        
        
        test_query_text = BENCHMARK_QUERIES[0]
        vectors_dict = vlm_encoder.encode_query(model, processor, test_query_text, config.DEVICE)
        
        pre_encoded_vector = vectors_dict["mean_pooling"] 
        
        print(f"Pre-encoded vector ready for: '{test_query_text}'")
    except Exception as e:
        print(f"Failed setup: {e}")
        return

    results_log = []

    
    for level in CONCURRENCY_LEVELS:
        print(f"\n--- Testing Concurrency Level: {level} ---")
        
        step_result = asyncio.run(run_benchmark_step(level, pre_encoded_vector))
        
        if step_result:
            results_log.append(step_result)
            print(f"Throughput: {step_result['throughput_qps']:.2f} QPS | Avg: {step_result['avg_latency_ms']:.2f}ms")
        else:
            print(f"Level {level} failed to return results.")

    
    os.makedirs("logs", exist_ok=True)
    results_df = pd.DataFrame(results_log)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()