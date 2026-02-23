import os
import torch
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# --- Qdrant Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "colpali_qdrant_os_textbook"

# --- MinIO Configuration ---
MINIO_HOST = "localhost:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio-password")
MINIO_BUCKET = "textbooks"
MINIO_SECURE = False

# --- VLM Model Configuration ---
MODEL_NAME = "vidore/colpali-v1.3"
IMAGE_SEQ_LENGTH = 1024
DIM = 128
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
