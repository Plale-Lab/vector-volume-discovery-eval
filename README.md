# Vector Volume Discovery Eval

Reference implementation for the retrieval and system-performance evaluations described in *"Towards High-Performance Multimodal Discovery: Evaluating Retrieval Efficacy and Operational Scalability in Digital Libraries."* The pipeline embeds digitized textbook pages (as images) with [ColPali](https://github.com/illuin-tech/colpali) and indexes them in [Qdrant](https://qdrant.tech/) for multi-vector, late-interaction retrieval, so that a natural-language query can be matched against both the textual and visual content of a page without OCR.

An anonymized mirror of this repository for double-blind review is available at: https://anonymous.4open.science/r/vector-volume-discovery-eval-22FC/

## Architecture

- **Page ingestion** (`process_pdfs.py`): converts a textbook PDF into per-page PNG images (300 DPI) and uploads them to MinIO.
- **Embedding + indexing** (`run_indexing.py`): loads each page image through ColPali/PaliGemma-3B, producing a `1030 × 128` multi-vector patch embedding, and upserts it into a Qdrant collection (HNSW index, on-disk Memmap storage, binary quantization).
- **Retrieval** (`run_retrieval.py`): encodes a natural-language query with the same ColPali backbone and searches Qdrant using late-interaction (MaxSim) scoring.
- **App** (`run_app.py`): a Streamlit front end that layers retrieval-augmented generation (via `services/llm_service.py`) on top of retrieval, so answers are synthesized from the retrieved pages rather than just listed.
- **Experiments** (`experiments/`): the scripts behind the paper's efficacy benchmark (`run_eval.py`), scalability/tail-latency measurements (`run_tail_latency.py`), concurrency stress tests (`run_concurrency_test*.py`), an ablation study, and plotting utilities.

## Prerequisites

- Python 3.10+
- Docker (for MinIO and Qdrant)
- A CUDA-capable GPU is strongly recommended — `config.py` will fall back to CPU (`DEVICE = "cpu"`), but ColPali/PaliGemma-3B and the Llama-2-7B generator are both large models.
- A Hugging Face account with access granted to the gated `meta-llama/Llama-2-7b-chat-hf` model, if you intend to run `run_app.py`'s generation stage.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with real MinIO credentials and (if needed) your HF token

# start MinIO
docker compose -f docker-compose-minio.yml up -d

# start Qdrant (not included in this repo's compose file — run separately)
docker run -p 6333:6333 -p 6334:6334 -v ~/qdrant_data:/qdrant/storage qdrant/qdrant
```

`config.py` currently hardcodes `QDRANT_HOST = "localhost"` and `QDRANT_PORT = 6333`; edit that file directly if you need to point at a different Qdrant deployment.

## Data

The evaluation corpus (five computer-science textbooks plus a recipe book, ~3,600+ pages total) is **not included** in this repository — the source PDFs are commercially copyrighted, which is part of the motivation for the paper's page-as-vector approach in the first place (see the paper's Introduction). To reproduce the pipeline end to end, supply your own PDF(s) and point `process_pdfs.py`'s `PDF_PATH`/`MINIO_PREFIX` constants at them.

## Usage

```bash
# 1. Convert a PDF into page images and upload to MinIO
python process_pdfs.py

# 2. Embed and index all pages currently in the MinIO bucket
python run_indexing.py

# 3. Query the index from the command line
python run_retrieval.py

# 4. Or launch the interactive RAG app
streamlit run run_app.py
```

The benchmark and performance experiments in `experiments/` are standalone scripts; run them directly with `python experiments/<script>.py` once the index above is populated. `run_eval.py` expects the 75-query benchmark referenced in the paper (not included here — see Data, above, on why the underlying source pages aren't distributed; the query/ground-truth set can be shared on request).

## Known gaps

- `services/minio_client.py` is imported by nearly every entry point (`process_pdfs.py`, `run_indexing.py`, `run_retrieval.py`, and everything in `experiments/`) but is not present in this repository yet. Add it (a thin wrapper around the `minio` Python SDK exposing `get_minio_client`, `upload_image_bytes`, `download_image_to_pil`, and `list_images_in_bucket`) before running anything.
- `run_indexing.py`'s `get_mock_text()` is a stand-in for real OCR/text extraction — it returns placeholder strings rather than actual page text, so RAG answers generated via `run_app.py` will cite mocked context until this is replaced.
- Qdrant connection settings live in `config.py`, not `.env` — see Setup above.

## License

Code in this repository is released under the MIT License (see `LICENSE`). This does not extend to any textbook or dataset content you supply yourself, which remains under its original copyright.
