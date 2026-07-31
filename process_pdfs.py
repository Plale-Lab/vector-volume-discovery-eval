import os
import sys
from pdf2image import convert_from_path, pdfinfo_from_path
from tqdm import tqdm

# Import your project configuration and MinIO service
import config
from services import minio_client

# --- CONFIGURATION ---
PDF_PATH = 'textbook_data/dataset/textbook9.pdf'
# This prefix acts as the "folder" in your MinIO bucket
MINIO_PREFIX = "textbook9" 
OUTPUT_DIR = f"textbook_data/images/{MINIO_PREFIX}"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_and_upload_book(pdf_path, output_dir, minio_prefix):
    """
    Converts PDF to images, saves them locally, AND uploads them to MinIO.
    """
    
    # 1. Initialize MinIO Client
    try:
        m_client = minio_client.get_minio_client(
            config.MINIO_HOST,
            config.MINIO_ACCESS_KEY,
            config.MINIO_SECRET_KEY,
            config.MINIO_SECURE
        )
    except Exception as e:
        print(f"Failed to connect to MinIO: {e}")
        return

    # 2. Get PDF Info
    try:
        info = pdfinfo_from_path(pdf_path)
        total_pages = info["Pages"]
        print(f"Found {total_pages} pages in '{pdf_path}'. Starting processing...")
    except Exception as e:
        print(f"Error reading PDF info: {e}")
        return

    success_count = 0

    # 3. Loop: Convert -> Save -> Upload
    for i in tqdm(range(1, total_pages + 1), desc="Processing Pages"):
        try:
            # A. Convert Page
            pages = convert_from_path(
                pdf_path,
                dpi=300, # High quality for VLM
                first_page=i,
                last_page=i
            )
            
            if not pages: continue
            image_pil = pages[0]
            
            # B. Save Locally (Optional backup)
            local_filename = f'page_{i}.png'
            local_path = os.path.join(output_dir, local_filename)
            image_pil.save(local_path, 'PNG')
            
            # C. Upload to MinIO
            # Object name will be: "textbook2/page_1.png"
            minio_object_name = f"{minio_prefix}/{local_filename}"
            
            upload_success = minio_client.upload_image_bytes(
                m_client,
                config.MINIO_BUCKET,
                minio_object_name,
                image_pil
            )
            
            if upload_success:
                success_count += 1
            else:
                print(f"Failed to upload page {i} to MinIO.")

        except Exception as e:
            print(f"Error on page {i}: {e}")
            continue
            
    print(f"\nProcessing complete!")
    print(f"Successfully converted and uploaded {success_count}/{total_pages} pages.")
    print(f"Local images: {output_dir}")
    print(f"MinIO path:   {config.MINIO_BUCKET}/{minio_prefix}/")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        sys.exit(1)
        
    process_and_upload_book(PDF_PATH, OUTPUT_DIR, MINIO_PREFIX)