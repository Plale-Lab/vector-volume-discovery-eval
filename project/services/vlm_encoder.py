import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from typing import List
from PIL import Image

def load_vlm_model(model_name: str, device: str):
    """Loads the ColPali model and processor with memory optimizations."""
    print(f"Loading VLM Model: {model_name} on {device}...")
    try:
        model = ColPali.from_pretrained(
            model_name,
            dtype=torch.bfloat16,  
            device_map={"": device},     
        ).eval()
        processor = ColPaliProcessor.from_pretrained(model_name)
        return model, processor
    except Exception as e:
        print(f"\nFailed to initialize VLM model ({model_name}).")
        raise e


def encode_batch(
    model: ColPali, 
    processor: ColPaliProcessor, 
    image_batch: List[Image.Image], 
    device: str, 
    image_seq_length: int,
    dim: int
) -> dict:
    """
    Encodes a batch of PIL images and returns a dictionary of
    multi-vector embeddings.
    """
    batch_size_current = len(image_batch)
    
    with torch.no_grad():
        batch_images_processed = processor.process_images(image_batch).to(device)
        image_embeddings = model(**batch_images_processed)

    special_tokens = image_embeddings[:, image_seq_length:, :]
    reshaped_embeddings = image_embeddings[:, :image_seq_length, :].reshape((batch_size_current, 32, 32, dim))
    
    max_pool = torch.cat((torch.max(reshaped_embeddings, dim=2).values, special_tokens), dim=1)
    mean_pool = torch.cat((torch.mean(reshaped_embeddings, dim=2), special_tokens), dim=1)


    return {
        "max_pooling": max_pool.cpu().float().numpy().tolist(), 
        "initial": image_embeddings.cpu().float().numpy().tolist(),
        "mean_pooling": mean_pool.cpu().float().numpy().tolist()
    }

def encode_query(
    model: ColPali, 
    processor: ColPaliProcessor, 
    query_text: str, 
    device: str
) -> dict:
    """
    Encodes a single text query into its multi-vector representation.
    """
    print(f"Encoding query: '{query_text}'...")

    # 1. Process on CPU
    batch_query = processor.process_queries([query_text])
    
    # 2. Move inputs to the device. 
    # Use the 'device' variable directly here since we will fix the model loading next.
    batch_query = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch_query.items()}

    with torch.no_grad():
        # batch_query = processor.process_queries([query_text])
        # batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
        model.to(device) 
        query_embeddings = model(**batch_query)
    
    full_vector_list = query_embeddings[0].cpu().float().numpy().tolist()

    mean_pooled_tensor = torch.mean(query_embeddings, dim=1) 
    mean_pooled_list = mean_pooled_tensor[0].cpu().float().numpy().tolist()

    return {
        "initial": full_vector_list,
        "mean_pooling": mean_pooled_list 
    }