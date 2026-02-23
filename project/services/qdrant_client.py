from typing import List
from qdrant_client import QdrantClient, models

def get_qdrant_client(host: str, port: int) -> QdrantClient:
    """Initializes and returns the Qdrant client."""
    print(f"Connecting to Qdrant at {host}:{port}...")
    try:
        client = QdrantClient(host=host, port=port, timeout=60)
        client.get_collections()
        print("Qdrant connection successful.")
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Please ensure Qdrant Docker container is running.")
        raise

def create_qdrant_collection_if_not_exists(client: QdrantClient, collection_name: str, size: int, force_recreate: bool = False):
    """Creates a scalable Qdrant collection if it doesn't already exist."""
    
    try:
        if force_recreate:
                print(f"--- Force-recreating Qdrant Collection: {collection_name} ---")
        elif client.get_collection(collection_name):
            print(f"Collection '{collection_name}' already exists. Skipping creation.")
            return
        else:
            print(f"--- Creating new Qdrant Collection: {collection_name} ---")
    except Exception:
        print(f"--- Creating Qdrant Collection: {collection_name} ---")
    
    vector_params_initial = models.VectorParams(
        size=size,
        distance=models.Distance.COSINE,
        on_disk=True, 
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True),
        ),
    )
    vector_params_pooled = models.VectorParams(
        size=size,
        distance=models.Distance.COSINE,
        on_disk=True, 
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        )
    )

    client.recreate_collection(
        collection_name=collection_name,
        shard_number=4,
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
        on_disk_payload=True,
        vectors_config={
            "initial": vector_params_initial,
            "max_pooling": vector_params_pooled,
            "mean_pooling": vector_params_pooled,
        }
    )
    print("Scalable Qdrant collection created successfully.")

def upsert_batch_to_qdrant(
    client: QdrantClient, 
    collection_name: str, 
    point_ids: List[int], 
    payloads: List[dict], 
    vectors: dict
):
    """Upserts a batch of points to Qdrant."""
    try:
        client.upsert(
            collection_name=collection_name,
            points=models.Batch( 
                ids=point_ids, 
                payloads=payloads,
                vectors=vectors
            ),
            wait=False
        )
    except Exception as e:
        print(f"Error during Qdrant upsert: {e}")


def search_qdrant(
    client: QdrantClient, 
    collection_name: str, 
    query_vector: List[List[float]], 
    top_k: int,
    vector_name: str = "mean_pooling"
) -> List[models.ScoredPoint]:
    """
    Searches Qdrant using the ColPali multi-vector query.
    """
    print("Searching Qdrant for top matches...")
    try:
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector, 
            limit=top_k, 
            using="initial",
            with_payload=True
        )
        return search_results.points
    except Exception as e:
        print(f"Error during diagnostic search: {e}")
        return []