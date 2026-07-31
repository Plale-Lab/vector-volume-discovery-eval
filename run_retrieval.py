import time
from PIL import Image
import os

import config as config
from services import minio_client, qdrant_client, vlm_encoder

USER_QUERY = "What is a kernel?"
TOP_K_RESULTS = 3

def main():
    """
    Main function to orchestrate the RETRIEVAL pipeline:
    1. Load VLM and connect to Qdrant & MinIO
    2. Encode the text query with the VLM
    3. Search Qdrant to get top results
    4. Fetch the result images from MinIO and display them
    5. Print latency metrics
    """
    
    print("--- Initializing RAG Pipeline ---")
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

    print("-" * 40)

    # --- 1. Encode Query ---
    start_time = time.time()
    vectors_dict = vlm_encoder.encode_query(model, processor, USER_QUERY, config.DEVICE)
    query_vector = vectors_dict["initial"]
    vlm_time = time.time() - start_time
    print(f"  VLM Encoding took: {vlm_time*1000:.2f} ms")

    # --- 2. Search Qdrant ---
    start_time = time.time()
    results = qdrant_client.search_qdrant(
        q_client, 
        config.COLLECTION_NAME, 
        query_vector, 
        TOP_K_RESULTS,
        vector_name="initial" # Trying max_pooling for better semantic matching
    )

    qdrant_time = time.time() - start_time
    print(f"  Qdrant Search took: {qdrant_time*1000:.2f} ms")
    print("-" * 40)
    print(f"  Total Retrieval Latency: {(vlm_time + qdrant_time)*1000:.2f} ms")
    print("-" * 40)

    # --- 3. Fetch & Display Results ---
    if not results:
        print("No results found.")
    else:
        print(f"Top {len(results)} results for '{USER_QUERY}':\n")
        
        for i, point in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(f"  Score: {point.score:.4f}")
            
            # Get metadata from payload
            page_num = point.payload.get('page_number')
            book_name = point.payload.get('book_name') # This was added in your new indexing script
            page_url = point.payload.get('page_url')
            
            print(f"  Book: {book_name}")
            print(f"  Page Number: {page_num}")
            print(f"  MinIO URL: {page_url}")

            try:
                
                if book_name:
                    object_name = f"{book_name}/page_{pacudoge_num}.png"
                else:
                    parts = page_url.split('/')
                    object_name = f"{parts[-2]}/{parts[-1]}"
                
                # Download from MinIO
                image = minio_client.download_image_to_pil(
                    m_client, 
                    config.MINIO_BUCKET, 
                    object_name
                )
                
                if image:
                    # On a headless server, we print confirmation instead of .show()
                    print(f"  Image for page {page_num} retrieved successfully.")
                    # image.show() # Uncomment this if running on a local machine with a screen
                else:
                    print(f"  Failed to download image: {object_name}")
                    
                print("-" * 40)
            except Exception as e:
                print(f"  Could not retrieve or display image: {e}")

if __name__ == "__main__":
    main()