# import time
# import os
# from tqdm import tqdm
# from typing import List
# import io
# from PIL import Image

# # Import all our project modules
# import config
# from services import minio_client, qdrant_client, vlm_encoder

# def main():
#     """
#     Indexes ALL textbooks from the MinIO bucket into a single Qdrant collection.
#     Stores the *folder name* (e.g., 'textbook2') as the 'book_name' metadata.
#     """
    
#     # --- 1. Setup Services ---
#     try:
#         print("--- Initializing Service Clients ---")
#         model, processor = vlm_encoder.load_vlm_model(config.MODEL_NAME, config.DEVICE)
#         q_client = qdrant_client.get_qdrant_client(config.QDRANT_HOST, config.QDRANT_PORT)
#         m_client = minio_client.get_minio_client(
#             config.MINIO_HOST, 
#             config.MINIO_ACCESS_KEY, 
#             config.MINIO_SECRET_KEY, 
#             config.MINIO_SECURE
#         )
#     except Exception as e:
#         print(f"Failed to initialize services. Exiting. Error: {e}")
#         return

#     # --- 2. Create/Recreate Qdrant Collection ---
#     # We start fresh to ensure the index is clean and contains all books
#     qdrant_client.create_qdrant_collection_if_not_exists(
#         q_client, 
#         config.COLLECTION_NAME, 
#         config.DIM,
#         force_recreate=True # Start with a clean slate for the full library
#     )
    
#     # --- 3. Get List of ALL Images (Recursive) ---
#     objects_iterator = m_client.list_objects(config.MINIO_BUCKET, recursive=True)
#     objects_list = list(objects_iterator)
    
#     if not objects_list:
#         print("No images found in MinIO bucket. Exiting.")
#         return

#     print(f"Found {len(objects_list)} pages across all textbooks.")

#     # --- 4. Run Indexing Pipeline ---
#     image_batch = []
#     payload_batch = []
#     point_ids = []
#     point_counter = 0

#     with tqdm(total=len(objects_list), desc="Indexing Library") as pbar:
#         for obj in objects_list:
#             try:
#                 # object_name looks like: "ai_modern_approach/page_1.png"
#                 full_object_name = obj.object_name
                
#                 # --- Robust Path Parsing ---
#                 filename = os.path.basename(full_object_name)
                
#                 # **MODIFIED:** We just get the folder name and use it directly
#                 if "/" in full_object_name:
#                     book_name = os.path.dirname(full_object_name) # e.g., "textbook2"
#                 else:
#                     book_name = "uncategorized"

#                 try:
#                     page_number = int(filename.split('_')[1].split('.')[0])
#                 except (IndexError, ValueError):
#                     page_number = 0 

#                 # --- Download & Process ---
#                 image_pil = minio_client.download_image_to_pil(m_client, config.MINIO_BUCKET, full_object_name)
#                 if image_pil is None:
#                     pbar.update(1)
#                     continue

#                 image_url = f"http://{config.MINIO_HOST}/{config.MINIO_BUCKET}/{full_object_name}"
                
#                 image_batch.append(image_pil)
#                 payload_batch.append({
#                     "page_url": image_url,
#                     "page_number": page_number,
#                     "book_name": book_name  # <-- Stores the raw folder name
#                 })
#                 point_ids.append(point_counter)
#                 point_counter += 1

#                 # --- Batch Upload ---
#                 if len(image_batch) == config.BATCH_SIZE:
#                     vectors_dict = vlm_encoder.encode_batch(
#                         model, processor, image_batch, config.DEVICE, 
#                         config.IMAGE_SEQ_LENGTH, config.DIM
#                     )
                    
#                     qdrant_client.upsert_batch_to_qdrant(
#                         q_client, config.COLLECTION_NAME, point_ids, 
#                         payload_batch, vectors_dict
#                     )
                    
#                     # Clear batches
#                     image_batch, payload_batch, point_ids = [], [], []

#             except Exception as e:
#                 print(f"Error processing {full_object_name}: {e}")
            
#             pbar.update(1)
            
#         # Process the final remaining batch
#         if image_batch:
#             vectors_dict = vlm_encoder.encode_batch(
#                 model, processor, image_batch, config.DEVICE, 
#                 config.IMAGE_SEQ_LENGTH, config.DIM
#             )
#             qdrant_client.upsert_batch_to_qdrant(
#                 q_client, config.COLLECTION_NAME, point_ids, 
#                 payload_batch, vectors_dict
#             )

#     print("\n--- Library Indexing Complete ---")
#     print(f"Total pages indexed: {point_counter}")
#     # Wait a moment for Qdrant to commit
#     time.sleep(2)
#     final_count = q_client.count(config.COLLECTION_NAME, exact=True).count
#     print(f"Qdrant collection count: {final_count}")

# if __name__ == "__main__":
#     main()


import time
import os
from tqdm import tqdm
from typing import List
import io
from PIL import Image

# Import all our project modules
import config
from services import minio_client, qdrant_client, vlm_encoder
# NOTE: Assume you have a helper function in minio_client 
# to fetch text/OCR data associated with the image.

def get_mock_text(book_name, page_number):
    """
    MOCK FUNCTION: In a real project, this would fetch the OCR text 
    from a dedicated text file in MinIO or an OCR service.
    """
    if 'textbook6' in book_name.lower() and page_number == 20:
        return "Why and How to Roast Duck: This page contains detailed instructions on preparation, stuffing, and the 180°C slow roasting technique for rendering fat to achieve crispy skin."
    if 'textbook' in book_name.lower():
        # Generic content for CS textbooks
        return f"This page from {book_name} discusses topics related to data processing, system architecture, and computational models. The page number is {page_number}."
    return "No extracted text available for this page."


def main():
    """
    Indexes ALL textbooks from the MinIO bucket into a single Qdrant collection.
    Includes the page's OCR text (page_text) in the payload for RAG Generation.
    """
    
    # --- 1. Setup Services ---
    try:
        print("--- Initializing Service Clients ---")
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

    # --- 2. Create/Recreate Qdrant Collection ---
    qdrant_client.create_qdrant_collection_if_not_exists(
        q_client, 
        config.COLLECTION_NAME, 
        config.DIM,
        force_recreate=True 
    )
    
    # --- 3. Get List of ALL Images (Recursive) ---
    objects_iterator = m_client.list_objects(config.MINIO_BUCKET, recursive=True)
    objects_list = list(objects_iterator)
    
    if not objects_list:
        print("No images found in MinIO bucket. Exiting.")
        return

    print(f"Found {len(objects_list)} pages across all textbooks.")

    # --- 4. Run Indexing Pipeline ---
    image_batch = []
    payload_batch = []
    point_ids = []
    point_counter = 0

    with tqdm(total=len(objects_list), desc="Indexing Library") as pbar:
        for obj in objects_list:
            try:
                full_object_name = obj.object_name
                filename = os.path.basename(full_object_name)
                
                if "/" in full_object_name:
                    book_name = os.path.dirname(full_object_name)
                else:
                    book_name = "uncategorized"

                try:
                    # Assuming filename is formatted like: bookX/page_N.png
                    page_number = int(filename.split('_')[1].split('.')[0])
                except (IndexError, ValueError):
                    page_number = 0 

                # --- Download & Process ---
                image_pil = minio_client.download_image_to_pil(m_client, config.MINIO_BUCKET, full_object_name)
                if image_pil is None:
                    pbar.update(1)
                    continue

                image_url = f"http://{config.MINIO_HOST}/{config.MINIO_BUCKET}/{full_object_name}"
                
                # --- NEW FIX: Retrieve and Store OCR Text for RAG ---
                page_text = get_mock_text(book_name, page_number)
                print("---", page_text,"----")

                image_batch.append(image_pil)
                payload_batch.append({
                    "page_url": image_url,
                    "page_number": page_number,
                    "book_name": book_name,
                    "page_text": page_text
                })
                point_ids.append(point_counter)
                point_counter += 1

                # --- Batch Upload ---
                if len(image_batch) == config.BATCH_SIZE:
                    vectors_dict = vlm_encoder.encode_batch(
                        model, processor, image_batch, config.DEVICE, 
                        config.IMAGE_SEQ_LENGTH, config.DIM
                    )
                    
                    qdrant_client.upsert_batch_to_qdrant(
                        q_client, config.COLLECTION_NAME, point_ids, 
                        payload_batch, vectors_dict
                    )
                    
                    # Clear batches
                    image_batch, payload_batch, point_ids = [], [], []

            except Exception as e:
                print(f"Error processing {full_object_name}: {e}")
            
            pbar.update(1)
            
        # Process the final remaining batch
        if image_batch:
            vectors_dict = vlm_encoder.encode_batch(
                model, processor, image_batch, config.DEVICE, 
                config.IMAGE_SEQ_LENGTH, config.DIM
            )
            qdrant_client.upsert_batch_to_qdrant(
                q_client, config.COLLECTION_NAME, point_ids, 
                payload_batch, vectors_dict
            )

    print("\n--- Library Indexing Complete ---")
    print(f"Total pages indexed: {point_counter}")
    time.sleep(2)
    final_count = q_client.count(config.COLLECTION_NAME, exact=True).count
    print(f"Qdrant collection count: {final_count}")

if __name__ == "__main__":
    main()