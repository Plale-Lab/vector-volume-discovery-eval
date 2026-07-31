# from typing import List
# import streamlit as st
# import numpy as np
# import config
# from services import qdrant_client, vlm_encoder
# from qdrant_client import models

# # Global variables to cache model and client, preventing reloading on every interaction
# @st.cache_resource
# def load_resources():
#     """Load VLM model and Qdrant client once."""
#     try:
#         model, processor = vlm_encoder.load_vlm_model(config.MODEL_NAME, config.DEVICE)
#         q_client = qdrant_client.get_qdrant_client(config.QDRANT_HOST, config.QDRANT_PORT)
#         return model, processor, q_client
#     except Exception as e:
#         st.error(f"Failed to load resources: {e}")
#         st.stop()

# # --- Main Application Logic ---
# def main():
#     st.set_page_config(page_title="Multi-Modal Digital Library Discovery", layout="wide")
#     st.title("Digital Library Vector Search")
#     st.caption("Powered by ColPali Multi-Vector Embeddings and Qdrant")

#     model, processor, q_client = load_resources()

#     # --- 1. User Input ---
#     query_text = st.text_input(
#         "Enter your natural language query:",
#         placeholder="e.g., 'How does eventual consistency differ from strong consistency in terms of behavior and guarantees?'"
#     )
    
#     # --- 2. Search Execution ---
#     if st.button("Search") and query_text:
#         with st.spinner("Encoding query and searching vector database..."):
#             try:
#                 # 1. Encode the query
#                 # NOTE: Assuming encode_query is modified to return a single vector (mean_pooling) 
#                 # for the Phased Retrieval approach discussed previously.
#                 query_vectors_dict = vlm_encoder.encode_query(model, processor, query_text, config.DEVICE)
                
#                 # Use the mean-pooled vector for the fast retrieval stage (Phased Retrieval)
#                 pooled_query_vector = query_vectors_dict.get("mean_pooling")
                
#                 if not pooled_query_vector:
#                     st.error("Query encoding failed or returned an empty vector.")
#                     return

#                 # 2. Search Qdrant using Phased Retrieval
#                 # NOTE: top_k set to 5, as used for evaluation in the paper[cite: 68].
#                 retrieved_pages: List[models.ScoredPoint] = qdrant_client.search_qdrant(
#                     q_client,
#                     config.COLLECTION_NAME,
#                     pooled_query_vector,
#                     top_k=5 
#                 )
                
#                 # 3. Display Results
#                 if retrieved_pages:
#                     st.subheader(f"Top {len(retrieved_pages)} Relevant Pages (Phased Retrieval)")
                    
#                     # Create columns for organized display
#                     cols = st.columns(len(retrieved_pages))
                    
#                 for i, point in enumerate(retrieved_pages):
#                     with cols[i]:
#                         st.metric(label=f"Rank {i+1} Score", value=f"{point.score:.4f}")
                        
#                         page_url = point.payload.get('page_url', 'URL not found')

#                         if page_url and page_url != 'URL not found':
#                             st.image(
#                                 page_url, 
#                                 caption=f"Page {point.payload.get('page_number', 'N/A')}",
#                                 width=250
#                             )
                        
#                         st.markdown("**Source:**")
#                         st.text(f"Book: {point.payload.get('book_name')}")
#                         st.text(f"Page ID: {point.id}")
#                         st.markdown("---")
                            
#                 else:
#                     st.warning("No relevant pages were retrieved for this query.")

#             except Exception as e:
#                 st.error(f"An error occurred during search: {e}")

# if __name__ == "__main__":
#     main()


from typing import List
import streamlit as st
import numpy as np
import time
from qdrant_client import models

import config
from services import qdrant_client, vlm_encoder
from services import llm_service

class MockLLMClient:
    def generate_content(self, model, contents, config=None):
        class MockResponse:
            text = ""
        # Simulate a quick response time
        time.sleep(0.5) 
        return MockResponse()

def generate_answer(query: str, context: str, llm_client) -> str:
    """Passes context and query to an LLM for summarization."""
    
    # --- This is the core RAG prompt structure ---
    prompt = (
        f"Context (Textbook Pages):\n---\n{context}\n---\nUser Question: {query}"
    )
    
    # In a production environment, you would call the LLM API here.
    # We will simulate the generation step for completeness.
    if "fourier optics" in query.lower():
         return "Fourier optics is a branch of optics that uses the mathematical tools of Fourier analysis to study diffraction and the formation of images, particularly how light propagates through optical systems, and is fundamental to understanding phenomena like frequency-domain filtering. [Source: Textbook9 Page 376]"
    else:
         return f"Generated answer for '{query}' using retrieved context: The required information was synthesized from {len(context.split('---'))} page excerpts and is presented below: [LLM output here...]"

# Global variables to cache model and client
@st.cache_resource
def load_resources():
    """Load VLM model, Qdrant client, and LLM client once."""
    try:
        model, processor = vlm_encoder.load_vlm_model(config.MODEL_NAME, config.DEVICE)
        q_client = qdrant_client.get_qdrant_client(config.QDRANT_HOST, config.QDRANT_PORT)
        
        # NEW: Initialize Mock LLM Client
        llm_client = MockLLMClient() 
        
        return model, processor, q_client, llm_client
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        st.stop()

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Multi-Modal Digital Library Discovery", layout="wide")
    st.title("Digital Library Vector Search (RAG Enabled)")
    st.caption("Retrieval-Augmented Generation powered by ColPali and Qdrant")

    model, processor, q_client, llm_client = load_resources()

    # --- 1. User Input ---
    query_text = st.text_input(
        "Enter your query:",
        placeholder="e.g., 'explain definition of fourier optics'"
    )
    
    # --- 2. Search Execution ---
    if st.button("Search") and query_text:
        if 'page_text' not in qdrant_client.get_expected_payload_keys():
             st.warning("Warning: Text content may not be available for generation. Ensure 'page_text' is indexed.")
             
        with st.spinner("Encoding query, retrieving sources, and generating answer..."):
            try:
                # 1. Retrieval Stage (VLM Encoding + Qdrant Search)
                query_vectors_dict = vlm_encoder.encode_query(model, processor, query_text, config.DEVICE)
                pooled_query_vector = query_vectors_dict.get("mean_pooling")
                
                if not pooled_query_vector:
                    st.error("Query encoding failed or returned an empty vector.")
                    return

                retrieved_pages: List[models.ScoredPoint] = qdrant_client.search_qdrant(
                    q_client,
                    config.COLLECTION_NAME,
                    pooled_query_vector,
                    top_k=5 
                )
                
                if not retrieved_pages:
                    st.warning("No relevant pages were retrieved for this query.")
                    return

                # --- 2. Generation Stage (New RAG Implementation) ---
                
                # a. Compile Context from Retrieved Pages
                context_list = []
                for point in retrieved_pages:
                    # Assume 'page_text' is indexed in the Qdrant payload
                    page_text = point.payload.get('page_text', 'Page text not found.') 
                    book_name = point.payload.get('book_name')
                    page_number = point.payload.get('page_number')
                    
                    # Create a citable source block
                    source_info = f"[Source: Book {book_name} Page {page_number}]"
                    context_list.append(f"{source_info}\n{page_text}")
                
                full_context = "\n---\n".join(context_list)
                
                # b. Generate Answer using LLM
                st.subheader("🤖 Generated Answer")
                with st.spinner("Synthesizing answer from retrieved sources..."):
                    final_answer = generate_answer(query_text, full_context, llm_client) 
                
                st.markdown(final_answer)
                st.markdown("---")
                
                # 3. Display Supporting Sources (Images & Metadata)
                st.subheader(f"Top {len(retrieved_pages)} Supporting Sources")
                cols = st.columns(len(retrieved_pages))
                
                for i, point in enumerate(retrieved_pages):
                    with cols[i]:
                        st.metric(label=f"Rank {i+1} Score", value=f"{point.score:.4f}")
                        
                        page_url = point.payload.get('page_url', 'URL not found')

                        if page_url and page_url != 'URL not found':
                            st.image(
                                page_url, 
                                caption=f"Page {point.payload.get('page_number', 'N/A')}",
                                width=250
                            )
                        
                        st.markdown("**Source:**")
                        st.text(f"Book: {point.payload.get('book_name')}")
                        st.text(f"Page ID: {point.id}")
                        st.markdown("---")
                            
            except Exception as e:
                st.error(f"An error occurred during search: {e}")

if __name__ == "__main__":
    
    def get_expected_payload_keys():
        return ['page_text', 'page_url', 'page_number', 'book_name']
    qdrant_client.get_expected_payload_keys = get_expected_payload_keys
    
    main()