import time
import os
import random
from tqdm import tqdm
import pandas as pd
from qdrant_client import models

import config
from services import minio_client, qdrant_client, vlm_encoder

BENCHMARK_QUERIES = [
    "What is a process?",
    "What is a thread?",
    "Explain the concept of a deadlock",
    "What is virtual memory?",
    "Describe CPU scheduling"
]
RESULTS_FILE = "logs/scalability_results_full_library.csv"


def run_retrieval_benchmark(model, processor, q_client) -> float:
    latencies = []

    
    for query_text in BENCHMARK_QUERIES:
        query_vector = vlm_encoder.encode_query(model, processor, query_text, config.DEVICE)
        

        start_time = time.time()
        results = qdrant_client.search_qdrant(
            q_client, 
            config.COLLECTION_NAME, 
            query_vector, 
            top_k=3,
            vector_name="initial"
        )
        qdrant_time = time.time() - start_time
        
        if results:
            latencies.append(qdrant_time)
        
    if not latencies:
        return 0.0
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    return avg_latency_ms


def main():
    """
    Main function to orchestrate the scalability test across ALL folders.
    """
    
    print("--- Initializing RAG Scalability Test (Full Library) ---")
    try:
        model, processor = vlm_encoder.load_vlm_model(config.MODEL_NAME, config.DEVICE)
        q_client = qdrant_client.get_qdrant_client(config.QDRANT_HOST, config.QDRANT_PORT)
        m_client = minio_client.get_minio_client(
            config.MINIO_HOST, 
            config.MINIO_ACCESS_KEY, 
            config.MINIO_SECRET_KEY, 
            config.MINIO_SECURE
        )
    except Exception as e:
        print(f"Failed to initialize services. Exiting. Error: {e}")
        return

    print(f"--- Force-recreating Qdrant Collection: {config.COLLECTION_NAME} ---")
    qdrant_client.create_qdrant_collection_if_not_exists(
        q_client, 
        config.COLLECTION_NAME, 
        config.DIM,
        force_recreate=True
    )
    
    objects_list = minio_client.list_images_in_bucket(m_client, config.MINIO_BUCKET)
    if not objects_list:
        print("No images found in MinIO bucket. Exiting.")
        return

    objects_list.sort(key=lambda x: x.object_name)
    
    total_docs = len(objects_list)
    print(f"Found {total_docs} total pages in library.")


    desired_steps = [100, 500, 1000, 2000, 3000]
    corpus_size_steps = [s for s in desired_steps if s < total_docs]
    corpus_size_steps.append(total_docs) # Always include the final total
    
    print(f"Evaluation Checkpoints: {corpus_size_steps}")

    results_log = []
    point_counter = 0
    
    for step_size in corpus_size_steps:
        
        print(f"\n--- Processing Step: Indexing up to {step_size} documents ---")

        objects_to_index_now = objects_list[point_counter:step_size]
        
        if not objects_to_index_now:
            print(f"No new documents to index for this step.")
        else:
            image_batch = []
            payload_batch = []
            point_ids = []

            with tqdm(total=len(objects_to_index_now), desc=f"Indexing batch") as pbar:
                for obj in objects_to_index_now:
                    try:
                        full_object_name = obj.object_name 
                        
                        filename = os.path.basename(full_object_name) 
                        book_name = os.path.dirname(full_object_name) 
                        
                        try:
                            page_number = int(filename.split('_')[1].split('.')[0])
                        except (IndexError, ValueError):
                            page_number = 0

                        image_pil = minio_client.download_image_to_pil(m_client, config.MINIO_BUCKET, full_object_name)
                        if image_pil is None: continue

                        image_url = f"http://{config.MINIO_HOST}/{config.MINIO_BUCKET}/{full_object_name}"
                        
                        image_batch.append(image_pil)
                        payload_batch.append({
                            "page_url": image_url, 
                            "page_number": page_number,
                            "book_name": book_name
                        })
                        point_ids.append(point_counter)
                        point_counter += 1

                        if len(image_batch) == config.BATCH_SIZE:
                            vectors_dict = vlm_encoder.encode_batch(model, processor, image_batch, config.DEVICE, config.IMAGE_SEQ_LENGTH, config.DIM)
                            qdrant_client.upsert_batch_to_qdrant(q_client, config.COLLECTION_NAME, point_ids, payload_batch, vectors_dict)
                            image_batch, payload_batch, point_ids = [], [], []

                    except Exception as e:
                        print(f"Error processing {full_object_name}: {e}")
                    
                    pbar.update(1)

                if image_batch:
                    vectors_dict = vlm_encoder.encode_batch(model, processor, image_batch, config.DEVICE, config.IMAGE_SEQ_LENGTH, config.DIM)
                    qdrant_client.upsert_batch_to_qdrant(q_client, config.COLLECTION_NAME, point_ids, payload_batch, vectors_dict)

        print(f"Indexing for step {step_size} complete. Stabilizing...")
        time.sleep(2)
        current_index_size = q_client.count(config.COLLECTION_NAME, exact=True).count
        print(f"Current Index Size: {current_index_size} vectors")
 
        print("Running benchmark 3 times to stabilize cache...")
        latencies = [
            run_retrieval_benchmark(model, processor, q_client),
            run_retrieval_benchmark(model, processor, q_client),
            run_retrieval_benchmark(model, processor, q_client)
        ]
        avg_latency = sum(latencies) / len(latencies)
        
        print(f"--- Result: {current_index_size} docs = {avg_latency:.2f} ms avg retrieval ---")
        
        results_log.append({
            "corpus_size": current_index_size,
            "avg_retrieval_latency_ms": avg_latency
        })

    print("\n--- Full Library Scalability Test Complete ---")
    os.makedirs("logs", exist_ok=True)
    results_df = pd.DataFrame(results_log)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")
    print(results_df)

if __name__ == "__main__":
    main()