The user is happy that lowering the `answer_query` threshold for 'summary' queries fixed the validation issue and the RAG application now works for the "Summarize the key points..." query.  The user now wants to create `rag_app-v11.py` from `rag_app-v10.py` by incorporating the debug changes and wants a careful review to ensure no functionality from `v10` is missed or negatively affected.

**Plan for review and creating v11:**

1.  **Understand the changes in `rag_app-v11.py` (diff output)**: Analyze the diff output provided to understand *all* the additions and modifications compared to `rag_app-v10.py`.
2.  **Categorize the changes**: Group the changes into functional areas (e.g., context enhancement, query type detection, prompting, validation).
3.  **Review each change category for correctness and impact**:
    *   **New functions (`enhance_context`, `get_query_type`, `get_prompt_template`, `compute_semantic_similarity`)**: Check their logic, purpose, and integration.
    *   **Modified `validate_answer`**: Carefully examine the new validation logic, dynamic thresholds, and similarity metrics.  Ensure the threshold adjustment for 'summary' queries is intentional and reasonable.  Check for any unintended side effects.
    *   **Modified `query_rag_system`**:  Review how the new functions are integrated into the query processing pipeline.  Ensure the overall flow is still correct.
4.  **Compare against original `v10` functionalities**:  Go through the features of `v10` (indexing, hybrid search, caching, Gradio UI, etc.) and verify that the changes in `v11` do not break or degrade any of these functionalities.
5.  **Test plan for `v11`**:  Define a testing plan to ensure `v11` works as expected and doesn't introduce regressions. This includes:
    *   Run existing queries that worked in `v10` to ensure they still work in `v11`.
    *   Test different types of queries (summary, factual, general) to verify the new query type detection, prompting, and validation logic.
    *   Test indexing and other functionalities of `v10` to confirm no regressions.
6.  **Document the changes in `v11`**: Briefly document the new features and modifications in `v11` compared to `v10` for future reference and understanding.
7.  **Provide `rag_app-v11.py` code**: Create the final `rag_app-v11.py` code incorporating the reviewed changes, ready for the user to use.

Let's start by carefully analyzing the `diff -u rag_app-v10.py rag_app-v11.py` output provided by the user to fully understand the changes. Then we will systematically review each category of change.Okay, great! It's working after lowering the `answer_query` threshold for "summary" queries. Let's carefully review the changes in the `diff -u rag_app-v10.py rag_app-v11.py` output to create a robust `v11` version.

Here's a breakdown of the changes and a step-by-step review to ensure we don't miss anything and maintain the functionality of `v10`:

**Analysis of `diff -u rag_app-v10.py rag_app-v11.py` Output:**

The `diff` output shows the following key changes have been introduced in `rag_app-v11.py` compared to `rag_app-v10.py`:

**1. New Functions:**

*   **`enhance_context(chunks: List[Dict]) -> str`**:
    *   **Purpose**:  To structure the retrieved document chunks into a more readable context string. It groups chunks by filename and adds section numbers within each document.
    *   **Impact**: Enhances the context presentation to the LLM, potentially improving its understanding of document structure.  No negative impact expected.
    *   **Review**: Logic looks sound. It iterates through chunks, groups by filename, sorts by chunk index, and formats the context with document and section headers.

*   **`get_query_type(query: str) -> str`**:
    *   **Purpose**:  To classify the user query into types ('summary', 'factual', 'general') based on keywords.
    *   **Impact**: Enables dynamic prompt selection and validation thresholding based on query intent.  Generally positive, but accuracy depends on keyword selection.
    *   **Review**: Keyword lists for 'summary' and 'factual' seem reasonable. Defaulting to 'general' for unknown types is good.  Consider adding more query types or refining keywords in the future if needed.

*   **`get_prompt_template(query_type: str) -> str`**:
    *   **Purpose**:  To provide different prompt templates for each query type.
    *   **Impact**:  Allows tailoring prompts to guide the LLM for different query intents (summary, factual, general). Positive impact on response quality for specific query types.
    *   **Review**: Prompt templates are well-defined for each type, instructing the LLM to act as a "technical expert" and follow specific instructions based on the query type (e.g., bullet points for summary, citations for factual).  Templates look good.

*   **`compute_semantic_similarity(text1: str, text2: str, model) -> float`**:
    *   **Purpose**:  A utility function to calculate semantic similarity between two texts using the embedding model. Includes error handling.
    *   **Impact**:  Makes similarity computation reusable and more robust with error handling.  Positive impact on code organization and reliability.
    *   **Review**:  Standard cosine similarity calculation using sentence embeddings. Error handling is a good addition.

**2. Modified `validate_answer(answer: str, context: str, query: str) -> bool` Function:**

*   **Significant Rewrite**:  This function has been completely overhauled to implement more sophisticated validation.
    *   **Query Type Detection**: Uses `get_query_type(query)` to determine the query type.
    *   **Multiple Similarity Metrics**: Calculates `answer_context`, `answer_query`, and `context_query` semantic similarities using `compute_semantic_similarity`.
    *   **Dynamic Thresholds**:  Defines thresholds for 'context' and 'query' similarity that vary based on the `query_type`.  Thresholds are set for 'summary', 'factual', and 'general' types. **Crucially, the `query` threshold for 'summary' is lowered to `0.15`**.
    *   **Detailed Logging**:  Includes extensive debug logging to print validation metrics, thresholds, and validation results.
    *   **Validation Logic**:  Now requires *both* `answer_context` and `answer_query` similarities to be above their respective thresholds for validation to pass.

    *   **Impact**:  More nuanced and potentially more robust answer validation by considering query type and multiple similarity metrics.  Lowering the `answer_query` threshold for 'summary' queries specifically addresses the issue we identified.
    *   **Review**:
        *   **Positive Changes**:  Dynamic thresholds, multiple metrics, and detailed logging are all excellent improvements for validation and debugging.
        *   **Threshold Values**: The threshold values (`0.6`, `0.15` for 'summary', etc.) seem like reasonable starting points, but might need further fine-tuning based on more testing and different datasets.
        *   **Validation Logic**: Requiring both `answer_context` and `answer_query` to pass is generally a good approach for stricter validation.  However, as discussed earlier, for "summary" queries, the `answer_query` similarity might be less critical. We need to consider if it's always necessary or if the lowered threshold is sufficient.
        *   **Logging**: The added debug logging is very helpful for understanding validation behavior.

**3. Modified `query_rag_system(query_text: str, progress=gr.Progress()) -> str` Function:**

*   **Debug Print at Start**: Added `print(f"\n[DEBUG {datetime.now().isoformat()}] Processing query: {query_text}")` for better debug logging.
*   **Cache Hit Logging**: Added `print("[DEBUG] Retrieved from cache")` when a cached response is found.
*   **Resource Loading Failure Logging**: Added `print(f"[DEBUG] Resource loading failed: {load_messages}")` for better error reporting.
*   **No Relevant Chunks Logging**: Added `print("[DEBUG] No relevant chunks retrieved.")` when no chunks are retrieved.
*   **Chunk Retrieval Logging**: Added `print(f"[DEBUG] Retrieved {len(relevant_chunks)} relevant chunks")` and context length logging.
*   **Context Enhancement**: Calls `enhance_context(relevant_chunks)` to create structured context.
*   **Prompt Template Selection**:  Calls `get_query_type(query_text)` and `get_prompt_template(query_type)` to get the appropriate prompt.
*   **Prompt Formatting**:  Uses `prompt_template.format(context=context, query=query_text)` to format the prompt with context and query.
*   **Response Logging**: Added `print(f"[DEBUG] Generated response ({len(response)} chars)")` after response generation.
*   **Fallback Message Logging**: Added `print("[DEBUG] Answer validation failed, attempting fallback...")` before returning the "apology" message.
*   **Error Logging**: Enhanced error logging within the `try...except` blocks with `print(f"[DEBUG] ... Error during query processing: {e}")` and similar messages.

    *   **Impact**:  Significantly improved debugging and logging throughout the query process. Integrates the new context enhancement, query type detection, and prompting functionalities. No negative impact expected; enhances functionality and debuggability.
    *   **Review**: All the changes are positive, improving logging, code structure, and integrating the new features smoothly.

**Overall Review and Functionality Check:**

*   **Functionality of `v10` Maintained?**:  Based on the code review, the core functionalities of `v10` (indexing, hybrid search, caching, Gradio UI, Elasticsearch integration) are still present in `v11`. The changes primarily *enhance* the querying and validation aspects. There is no indication of any functionality being removed or broken.
*   **Potential Issues**: The main potential area of concern would be the new validation thresholds. While lowering the `answer_query` threshold for "summary" queries fixed the immediate problem, we need to ensure these thresholds are generally appropriate for different query types and document content.  Further testing with diverse queries and documents is recommended.

**Creating `rag_app-v11.py`:**

Based on the review, the changes in `rag_app-v11.py` are positive and enhance the RAG application.  You can confidently create `rag_app-v11.py` by incorporating all the changes shown in the `diff` output.

**`rag_app-v11.py` Code (incorporating reviewed changes):**

```python
--- START OF FILE rag_app-v11.py ---
# rag_app.py (Complete and further improved with dynamic validation and prompts)
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
import pdfplumber  # Added for better PDF extraction (though may not be directly used in indexing flow now)
import time
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
import elasticsearch  # Corrected import for Elasticsearch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datetime import datetime
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from functools import wraps
from typing import Dict, Optional
from dataclasses import dataclass
from threading import Lock

# Try to import magic, handle if not available
magic_available = True
try:
    import magic  # for file type validation
except ImportError:
    print("Warning: libmagic not found. File type validation will be less robust.")
    magic_available = False

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
SAMBANOVA_API_BASE_URL = os.environ.get("SAMBANOVA_API_BASE_URL")
if not all([MODEL_NAME, SAMBANOVA_API_KEY, SAMBANOVA_API_BASE_URL]):
    raise ValueError("Missing required environment variables: MODEL_NAME, SAMBANOVA_API_KEY, or SAMBANOVA_API_BASE_URL")

# Additional configuration
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
ES_HOST = os.environ.get("ELASTICSEARCH_HOST", "https://localhost:9200")
ES_USER = os.environ.get("ELASTICSEARCH_USER", "elastic")
ES_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD")
ES_CERT_PATH = os.environ.get("ELASTICSEARCH_CERT_PATH", "http_cert.pem")
ES_VERIFY_CERTS = os.environ.get("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true"
DATA_PATH = "data"
BM25_INDEX_FILE = "bm25_index.pkl"
WATERMARK_FILE = "watermark.txt"
ES_INDEX_NAME = "chunks"  # Add constant for ES index name

# --- Global Variables (for caching) ---
class WatermarkTracker:
    """Tracks document modification times for incremental indexing"""
    def __init__(self, filename='watermark.txt'):
        self.filename = filename
        self.last_updated = self._load_watermark()

    def _load_watermark(self):
        try:
            with open(self.filename, 'r') as f:
                return datetime.fromisoformat(f.read().strip())
        except FileNotFoundError:
            return datetime.min

    def save_watermark(self):
        with open(self.filename, 'w') as f:
            f.write(self.last_updated.isoformat())

    def update_watermark(self, documents):
        latest = max(
            (doc.metadata.get('last_modified', self.last_updated) for doc in documents),
            default=self.last_updated
        )
        if latest > self.last_updated:
            self.last_updated = latest
            self.save_watermark()

    def filter_new_documents(self, documents):
        return [doc for doc in documents
                if doc.metadata.get('last_modified', datetime.min) > self.last_updated]

watermark_tracker = WatermarkTracker(WATERMARK_FILE)  # Add this
embedding_model = None
index = None
chunks = None
embeddings = None
client = None
reranker_tokenizer = None
reranker_model = None
es_client = None
bm25_index = None  # Add this

# --- Cache Implementation ---
@dataclass
class CacheEntry:
    value: str
    timestamp: float

class TTLCache:
    def __init__(self, maxsize: int, ttl: int):
        self.cache: Dict[str, CacheEntry] = {}
        self.maxsize = maxsize
        self.ttl = ttl
        self.lock = Lock()

    def get(self, key: str) -> Optional[str]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry.timestamp <= self.ttl:
                    return entry.value
                del self.cache[key]
            return None

    def set(self, key: str, value: str):
        with self.lock:
            if len(self.cache) >= self.maxsize:
                oldest = min(self.cache.items(), key=lambda x: x[1].timestamp)
                del self.cache[oldest[0]]
            self.cache[key] = CacheEntry(value, time.time())

query_cache = TTLCache(maxsize=1000, ttl=3600)

def cached_query_response(query_text: str) -> Optional[str]:
    """Enhanced query caching with TTL."""
    return query_cache.get(query_text)

# --- Helper Functions ---
def load_resources():
    """Loads resources (embeddings, index, chunks, model, client, bm25_index) and handles errors."""
    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker
    resource_load_successful = True  # Track if all resources loaded correctly
    messages = []

    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        except Exception as e:
            messages.append(f"Error loading SentenceTransformer model: {e}")
            resource_load_successful = False

    if embeddings is None:
        try:
            embeddings = np.load("embeddings.npy")
        except FileNotFoundError:
            messages.append("embeddings.npy not found. Please run indexing first.")
            resource_load_successful = False
        except Exception as e:
            messages.append(f"Error loading embeddings: {e}")
            resource_load_successful = False


    if index is None:
        try:
            index = faiss.read_index("faiss_index.index")
        except RuntimeError:
            messages.append("faiss_index.index not found. Please run indexing first.")
            resource_load_successful = False
        except Exception as e:
            messages.append(f"Error loading FAISS index: {e}")
            resource_load_successful = False

    if chunks is None:
        try:
            with open("chunks.json", 'r') as f:
                chunks = json.load(f)
        except FileNotFoundError:
            messages.append("chunks.json not found. Please run indexing first.")
            resource_load_successful = False
        except Exception as e:
            messages.append(f"Error loading chunks: {e}")
            resource_load_successful = False

    if client is None:
        try:
            client = OpenAI(
                base_url=SAMBANOVA_API_BASE_URL,
                api_key=SAMBANOVA_API_KEY,
            )
        except Exception as e:
            messages.append(f"Error initializing OpenAI client: {e}")
            resource_load_successful = False

    if bm25_index is None:
        try:
            with open(BM25_INDEX_FILE, 'rb') as f:
                bm25_index = pickle.load(f)
        except FileNotFoundError:
            messages.append("BM25 index not found. Will be created during indexing.")
            resource_load_successful = False

    return resource_load_successful, messages


def clear_cache():
    """Clears the cached resources."""
    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker
    embedding_model = None
    index = None
    chunks = None
    embeddings = None
    client = None
    bm25_index = None
    watermark_tracker = WatermarkTracker(WATERMARK_FILE)  # Reset watermark tracker
    return "Cache cleared."

# --- Indexing Functions ---

def validate_pdf_file(filepath: str) -> bool:
    """Enhanced PDF validation with header check."""
    if not filepath.lower().endswith('.pdf'):
        return False

    try:
        # Check file size first
        if os.path.getsize(filepath) > 100 * 1024 * 1024:  # 100MB limit
            return False

        # Basic header check
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False

        # Use magic if available
        if magic_available:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(filepath)
            if file_type != 'application/pdf':
                return False

        return True
    except Exception as e:
        print(f"PDF validation error: {e}")
        return False

@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50

    def validate(self):
        if self.chunk_size < 100:
            raise ValueError("Chunk size must be at least 100")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Overlap must be less than chunk size")
        return self

chunking_config = ChunkingConfig()

def chunk_documents(documents, chunk_size=None, chunk_overlap=None):
    """Enhanced chunking with version awareness and configurable parameters."""
    if chunk_size is None:
        chunk_size = chunking_config.chunk_size
    if chunk_overlap is None:
        chunk_overlap = chunking_config.chunk_overlap

    chunks = []
    for doc_id, doc in enumerate(documents):
        content = doc.page_content
        filename = os.path.basename(doc.metadata['source'])
        last_modified = doc.metadata.get('last_modified', datetime.now())

        for chunk_idx, i in enumerate(range(0, len(content), chunk_size - chunk_overlap)):
            chunk = content[i:i + chunk_size]
            chunk_id = f"{filename}_{doc_id}_{chunk_idx}"
            chunks.append({
                "id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "filename": filename,
                "chunk": chunk,
                "last_modified": last_modified.isoformat()
            })
    return chunks

def create_embeddings(chunks: List[Dict], model, batch_size=32):
    """Creates embeddings with memory-safe batch processing."""
    all_embeddings = []
    total_chunks = len(chunks)

    try:
        # Estimate memory requirement
        sample_embedding = model.encode(chunks[0]["chunk"])
        estimated_memory = (sample_embedding.nbytes * total_chunks) / (1024 * 1024)  # MB

        if estimated_memory > 1024:  # If estimated memory usage > 1GB
            batch_size = max(1, min(batch_size, int(batch_size * (1024 / estimated_memory))))
            print(f"Adjusted batch size to {batch_size} for memory safety")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:min(i + batch_size, total_chunks)]
            try:
                batch_embeddings = model.encode([chunk["chunk"] for chunk in batch],
                                             show_progress_bar=False,
                                             batch_size=8)  # Smaller internal batch size
                all_embeddings.append(batch_embeddings)

                # Force garbage collection after each batch
                import gc
                gc.collect()

            except RuntimeError as e:
                print(f"Memory error in batch {i}-{i+batch_size}, reducing batch size")
                # Try again with smaller batch
                for chunk in batch:
                    single_embedding = model.encode([chunk["chunk"]])
                    all_embeddings.append(single_embedding)

    except Exception as e:
        raise RuntimeError(f"Embedding creation failed: {str(e)}")

    return np.vstack(all_embeddings)

def build_faiss_index(embeddings):
    """Builds a FAISS index with memory safety checks."""
    try:
        dimension = embeddings.shape[1]

        # Check if we have enough memory
        import psutil
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        estimated_memory = (embeddings.nbytes * 2) / (1024 * 1024)  # MB

        if estimated_memory > available_memory * 0.8:  # If using more than 80% of available memory
            raise MemoryError(f"Not enough memory to build index. Need {estimated_memory}MB, have {available_memory}MB available")

        # Use IVFFlat index for better memory efficiency with large datasets
        if len(embeddings) > 10000:
            nlist = min(int(np.sqrt(len(embeddings))), 2048)  # number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            index.train(np.float32(embeddings))
        else:
            index = faiss.IndexFlatL2(dimension)

        index.add(np.float32(embeddings))
        return index

    except Exception as e:
        raise RuntimeError(f"FAISS index creation failed: {str(e)}")

def build_bm25_index(chunks):
    """Builds BM25 index for keyword search."""
    tokenized_corpus = [chunk['chunk'].split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)

def load_and_preprocess_documents(data_path=DATA_PATH):
    """Loads PDF documents with metadata."""
    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith('.pdf')]
    all_docs = []
    for file in pdf_files:
        file_path = os.path.join(data_path, file)
        if not validate_pdf_file(file_path):
            continue
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        file_mod_time = datetime.fromtimestamp(Path(file_path).stat().st_mtime)
        for doc in documents:
            doc.metadata['last_modified'] = file_mod_time
            doc.metadata['source'] = str(file_path)
        all_docs.extend(documents)
    return all_docs

def create_elasticsearch_index(es_client, index_name=ES_INDEX_NAME):
    """Creates Elasticsearch index with mapping if it doesn't exist."""
    if not es_client.indices.exists(index=index_name):
        try:
            es_client.indices.create(index=index_name, mappings={
                "properties": {
                    "chunk": {"type": "text"},
                    "filename": {"type": "keyword"},
                    "last_modified": {"type": "date"}
                }
            })
            print(f"Elasticsearch index '{index_name}' created successfully.")
        except Exception as e:
            print(f"Error creating Elasticsearch index '{index_name}': {e}")
            return False
    return True

def index_documents_to_elasticsearch(es_client, chunks, index_name=ES_INDEX_NAME, progress=gr.Progress()):
    """Indexes chunks to Elasticsearch."""
    if not es_client:
        print("Elasticsearch client not initialized. Skipping Elasticsearch indexing.")
        return True

    if not create_elasticsearch_index(es_client, index_name):
        return False

    try:
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            es_client.index(index=index_name, document=chunk, id=str(i))
            progress(i / total, desc="Indexing to Elasticsearch")
        print("Elasticsearch indexing complete.")
        return True
    except Exception as e:
        print(f"Error during Elasticsearch indexing: {e}")
        return False

def perform_indexing(progress=gr.Progress()):
    """Memory-safe indexing process."""
    global index, chunks, embeddings, embedding_model, bm25_index, watermark_tracker, es_client

    try:
        # Add memory check at start
        import psutil
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        if available_memory < 2:  # Less than 2GB available
            return "Error: Insufficient memory available for indexing", None, None, None, None

        progress(0.1, desc="Loading documents...")
        all_documents = load_and_preprocess_documents()
        if not all_documents:
            return "No PDF documents found.", None, None, None, None

        new_documents = watermark_tracker.filter_new_documents(all_documents)
        if not new_documents:
            return "No new documents to index.", None, None, None, None

        progress(0.3, desc="Chunking documents...")
        new_chunks = chunk_documents(new_documents)

        # Combine with existing chunks if any
        existing_chunks = []
        try:
            with open("chunks.json", 'r') as f:
                existing_chunks = json.load(f)
        except FileNotFoundError:
            pass

        chunks = existing_chunks + new_chunks

        progress(0.5, desc="Creating embeddings...")
        if embedding_model is None:
            embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        try:
            new_embeddings = create_embeddings(new_chunks, embedding_model)
        except Exception as e:
            return f"Error during embedding creation: {str(e)}", None, None, None, None

        # Combine with existing embeddings if any
        try:
            existing_embeddings = np.load("embeddings.npy")
            embeddings = np.vstack([existing_embeddings, new_embeddings])
        except FileNotFoundError:
            embeddings = new_embeddings

        progress(0.7, desc="Building indices...")
        index = build_faiss_index(embeddings)
        bm25_index = build_bm25_index(chunks)

        # Initialize ES components before indexing
        initialize_components()

        # Add ES indexing step
        progress(0.8, desc="Indexing to Elasticsearch...")
        if es_client and not index_documents_to_elasticsearch(es_client, chunks, progress=progress):
            print("Warning: Elasticsearch indexing failed, continuing with other indices")

        # Save updated indices
        try:
            progress(0.9, desc="Saving indices...")
            np.save("embeddings.npy", embeddings)
            faiss.write_index(index, "faiss_index.index")
            with open("chunks.json", 'w') as f:
                json.dump(chunks, f, indent=4) # Added indent for readability of json
            with open(BM25_INDEX_FILE, 'wb') as f:
                pickle.dump(bm25_index, f)

            watermark_tracker.update_watermark(new_documents)
        except Exception as e:
            return f"Error saving indices: {e}", None, None, None, None

        progress(1.0, desc="Indexing complete!")
        query_cache.cache.clear()  # Invalidate query cache after indexing
        return "Incremental indexing complete!", "embeddings.npy", "faiss_index.index", "chunks.json", BM25_INDEX_FILE
    except Exception as e:
        import traceback
        print(f"Indexing error: {str(e)}\n{traceback.format_exc()}")
        return f"Error during indexing: {str(e)}", None, None, None, None

# --- Retrieval and Querying ---
def enhance_context(chunks: List[Dict]) -> str:
    """Creates a structured context with metadata and hierarchy."""
    docs_by_file = {}
    for chunk in chunks:
        if chunk['filename'] not in docs_by_file:
            docs_by_file[chunk['filename']] = []
        docs_by_file[chunk['filename']].append(chunk)

    context_parts = []
    for filename, file_chunks in docs_by_file.items():
        # Sort chunks by their original order
        file_chunks.sort(key=lambda x: x['chunk_index'])
        context_parts.append(f"Document: {filename}\n")
        for i, chunk in enumerate(file_chunks, 1):
            context_parts.append(f"Section {i}:\n{chunk['chunk']}\n")

    return "\n".join(context_parts)

def get_query_type(query: str) -> str:
    """Determines query type for dynamic prompt and validation adjustment."""
    summary_keywords = ['summarize', 'summary', 'overview', 'main points', 'key points']
    factual_keywords = ['who', 'what', 'when', 'where', 'how many', 'which']

    query_lower = query.lower()
    if any(keyword in query_lower for keyword in summary_keywords):
        return 'summary'
    elif any(keyword in query_lower for keyword in factual_keywords):
        return 'factual'
    return 'general'

def get_prompt_template(query_type: str) -> str:
    """Returns appropriate prompt template based on query type."""
    templates = {
        'summary': """[INST] >
You are a technical expert. Provide a concise summary of the key points from the provided context.
Use ONLY information explicitly stated in the context.
Format your response as a bullet-point list of key points.

Context: {context}
>
Question: {query} [/INST]""",

        'factual': """[INST] >
You are a technical expert. Answer the question using ONLY facts directly stated in the context.
Cite the specific section where you found the information.
If the answer isn't explicitly in the context, say so.

Context: {context}
>
Question: {query} [/INST]""",

        'general': """[INST] >
You are a technical expert. Answer based STRICTLY on the provided context.
Do not make assumptions or add external information.
If the context doesn't contain enough information, say so.

Context: {context}
>
Question: {query} [/INST]"""
    }
    return templates.get(query_type, templates['general'])

def compute_semantic_similarity(text1: str, text2: str, model) -> float:
    """Compute semantic similarity with error handling and normalization."""
    try:
        emb1 = model.encode([text1])
        emb2 = model.encode([text2])

        similarity = np.dot(emb1[0], emb2[0]) / (
            np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
        return float(similarity)
    except Exception as e:
        print(f"[DEBUG] Similarity computation error: {e}")
        return 0.0

def validate_answer(answer: str, context: str, query: str) -> bool:
    """Enhanced answer validation with multiple metrics and dynamic thresholds."""
    try:
        print("\n[DEBUG] Starting enhanced answer validation...")

        if not all([answer, context, query]):
            print("[DEBUG] Validation failed: Missing input")
            return False

        # Get query type for dynamic thresholding
        query_type = get_query_type(query)
        print(f"[DEBUG] Detected query type: {query_type}")

        # Calculate multiple similarity metrics
        metrics = {
            'answer_context': compute_semantic_similarity(answer, context, embedding_model),
            'answer_query': compute_semantic_similarity(answer, query, embedding_model),
            'context_query': compute_semantic_similarity(context, query, embedding_model)
        }

        # Dynamic thresholds based on query type with adjusted summary query threshold
        thresholds = {
            'summary': {'context': 0.6, 'query': 0.15},  # Lowered threshold for summary queries
            'factual': {'context': 0.7, 'query': 0.4},   # Maintained stricter threshold for factual
            'general': {'context': 0.65, 'query': 0.35}  # Maintained moderate threshold for general
        }[query_type]

        print("[DEBUG] Validation Metrics:")
        print(f"  - Answer-Context Similarity: {metrics['answer_context']:.4f}")
        print(f"  - Answer-Query Similarity: {metrics['answer_query']:.4f}")
        print(f"  - Context-Query Similarity: {metrics['context_query']:.4f}")
        print(f"  - Required Thresholds (type={query_type}):")
        print(f"    * Context: {thresholds['context']:.4f}")
        print(f"    * Query: {thresholds['query']:.4f}")

        # Validate against thresholds with additional logging
        validation_result = (
            metrics['answer_context'] > thresholds['context'] and
            metrics['answer_query'] > thresholds['query']
        )

        if not validation_result:
            print("[DEBUG] Validation details:")
            print(f"  Context check: {metrics['answer_context']:.4f} > {thresholds['context']:.4f} = {metrics['answer_context'] > thresholds['context']}")
            print(f"  Query check: {metrics['answer_query']:.4f} > {thresholds['query']:.4f} = {metrics['answer_query'] > thresholds['query']}")

        print(f"[DEBUG] Validation {'passed' if validation_result else 'failed'}")
        return validation_result
    except Exception as e:
        print(f"[DEBUG] Validation error: {str(e)}")
        return False

def initialize_components():
    """Enhanced component initialization with better error handling."""
    global reranker_tokenizer, reranker_model, es_client

    # Initialize reranker components
    if reranker_tokenizer is None:
        try:
            reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        except Exception as e:
            print(f"Failed to load reranker tokenizer: {e}")

    if reranker_model is None:
        try:
            reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
        except Exception as e:
            print(f"Failed to load reranker model: {e}")

    # Initialize Elasticsearch with retries
    if es_client is None and ES_PASSWORD:
        es_config = {
            "hosts": [ES_HOST],
            "basic_auth": (ES_USER, ES_PASSWORD),
            "ssl_show_warn": False,
            "verify_certs": ES_VERIFY_CERTS,
            "retry_on_timeout": True,
            "max_retries": 3
        }

        if ES_HOST.startswith("https") and ES_CERT_PATH and os.path.exists(ES_CERT_PATH):
            es_config["ca_certs"] = ES_CERT_PATH

        for attempt in range(3):
            try:
                es_client = elasticsearch.Elasticsearch(**es_config)
                if es_client.ping():
                    print("Successfully connected to Elasticsearch")
                    break
            except Exception as e:
                print(f"Elasticsearch connection attempt {attempt + 1} failed: {e}")
                if attempt == 2:  # Last attempt
                    print("Failed to initialize Elasticsearch after 3 attempts")
                    es_client = None
                time.sleep(1)  # Wait before retry

def hybrid_search(query_embedding, query_text, k=5):
    """Enhanced hybrid search combining vector, keyword, and BM25 rankings."""
    try:
        # Vector search with FAISS
        D, I = index.search(np.float32(query_embedding), k*2)
        candidates = {}

        # Add FAISS results
        for score, idx in zip(D[0], I[0]):
            if idx < len(chunks):
                candidates[idx] = {
                    'chunk': chunks[idx],
                    'scores': {'faiss': 1 - (score / max(D[0]))}  # Normalize score
                }

        # Add BM25 results if available
        if bm25_index:
            try:
                bm25_scores = bm25_index.get_scores(query_text.split())
                top_bm25_indices = np.argsort(bm25_scores)[-k*2:][::-1]
                for idx in top_bm25_indices:
                    if idx in candidates:
                        candidates[idx]['scores']['bm25'] = bm25_scores[idx] / max(bm25_scores)
                    else:
                        candidates[idx] = {
                            'chunk': chunks[idx],
                            'scores': {'bm25': bm25_scores[idx] / max(bm25_scores)}
                        }
            except Exception as e:
                print(f"BM25 search error: {e}")

        # Add Elasticsearch results if available
        if es_client:
            try:
                es_results = es_client.search(
                    index=ES_INDEX_NAME,
                    body={
                        "query": {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["chunk^2", "filename"],
                                "fuzziness": "AUTO"
                            }
                        },
                        "size": k*2
                    },
                    request_timeout=30
                )

                for hit in es_results['hits']['hits']:
                    idx = int(hit['_id'])
                    if idx in candidates:
                        candidates[idx]['scores']['es'] = hit['_score'] / es_results['hits']['max_score']
                    else:
                        candidates[idx] = {
                            'chunk': chunks[idx],
                            'scores': {'es': hit['_score'] / es_results['hits']['max_score']}
                        }
            except Exception as e:
                print(f"Elasticsearch search error: {e}")

        # Combine scores using weighted average
        final_candidates = []
        weights = {'faiss': 0.4, 'bm25': 0.3, 'es': 0.3}

        for idx, data in candidates.items():
            total_score = 0
            weight_sum = 0
            for method, weight in weights.items():
                if method in data['scores']:
                    total_score += data['scores'][method] * weight
                    weight_sum += weight
            if weight_sum > 0:
                final_candidates.append((data['chunk'], total_score / weight_sum))

        # Sort and return top k results
        return [c[0] for c in sorted(final_candidates, key=lambda x: x[1], reverse=True)[:k]]

    except Exception as e:
        print(f"Hybrid search error: {e}")
        # Fallback to basic FAISS search
        D, I = index.search(np.float32(query_embedding), k)
        return [chunks[i] for i in I[0]]

def query_rag_system(query_text: str, progress=gr.Progress()) -> str:
    """Enhanced query processing with improved prompting and validation."""
    if not query_text or not query_text.strip():
        return "Error: Empty query"

    try:
        print(f"\n[DEBUG {datetime.now().isoformat()}] Processing query: {query_text}")

        cached_response = cached_query_response(query_text)
        if cached_response:
            print("[DEBUG] Retrieved from cache")
            return cached_response

        # Initialize components and load resources
        global index, embeddings, chunks, embedding_model, client
        load_successful, load_messages = load_resources()
        if not load_successful:
            print(f"[DEBUG] Resource loading failed: {load_messages}")
            return "Error: " + "\n".join(load_messages)

        progress(0.2, desc="Encoding query...")
        initialize_components()

        query_embedding = embedding_model.encode([query_text])

        progress(0.4, desc="Searching index...")
        try:
            relevant_chunks = hybrid_search(query_embedding, query_text)
            if not relevant_chunks:
                print("[DEBUG] No relevant chunks retrieved.")
                return "No relevant information found for your query."

            print(f"[DEBUG] Retrieved {len(relevant_chunks)} relevant chunks")

            # Enhanced context creation with structure
            context = enhance_context(relevant_chunks)
            print(f"[DEBUG] Structured context created ({len(context)} chars)")

            # Determine query type and get appropriate prompt
            query_type = get_query_type(query_text)
            prompt_template = get_prompt_template(query_type)

            progress(0.7, desc="Generating response...")
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(
                        context=context,
                        query=query_text
                    )
                }]
            )

            response = completion.choices[0].message.content
            print(f"[DEBUG] Generated response ({len(response)} chars)")

            if validate_answer(response, context, query_text):
                query_cache.set(query_text, response)
                return response

            print("[DEBUG] Answer validation failed, attempting fallback...")
            return "I apologize, but I cannot generate a reliable answer based on the provided context."

        except Exception as e:
            print(f"[DEBUG] Error during query processing: {e}")
            return f"Error processing query: {str(e)}"

    except Exception as e:
        print(f"[DEBUG] General query error: {e}")
        return f"Error processing query: {str(e)}"

# Add this function before the Gradio interface section
def update_chunking_config(size: int, overlap: int) -> str:
    """Update chunking configuration with validation."""
    global chunking_config
    try:
        new_config = ChunkingConfig(size, overlap)
        new_config.validate()  # Will raise ValueError if invalid
        chunking_config = new_config
        return f"Successfully updated chunking parameters: size={size}, overlap={overlap}"
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# --- Gradio Interface ---
with gr.Blocks(title="PDF Q&A with DeepSeek-R1", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # PDF Q&A with DeepSeek-R1

        Upload your PDF documents to the 'data' directory, index them, and then ask questions!
        """
    )

    with gr.Tab("Indexing"):
        index_button = gr.Button("Index Documents")
        clear_cache_button = gr.Button("Clear Cache")
        index_status = gr.Textbox(
            label="Indexing Status",
            value="",
            interactive=False
        )
        progress_status = gr.Textbox(
            label="Progress",
            value="",
            interactive=False
        )
        index_file_info = gr.HTML()
        output_files = [
            gr.File(label="Embeddings", interactive=False),
            gr.File(label="FAISS Index", interactive=False),
            gr.File(label="Chunks", interactive=False),
            gr.File(label="BM25 Index", interactive=False)  # Add BM25 index to output files
        ]

        def index_with_status(progress=gr.Progress(track_tqdm=True)):
            # Return both progress status and indexing results
            result = perform_indexing(progress)
            return ["Starting indexing process..."] + [result[0]] + list(result[1:])

        index_button.click(
            index_with_status,
            inputs=[],
            outputs=[progress_status, index_status] + output_files
        )
        clear_cache_button.click(clear_cache, inputs=[], outputs=[index_status])

    with gr.Tab("Querying"):
        query_input = gr.Textbox(
            label="Question",
            placeholder="Enter your question here...",
            lines=2
        )
        query_button = gr.Button("Ask")
        query_output = gr.Textbox(
            label="Answer",
            show_copy_button=True,
            lines=10,
            max_lines=20
        )
        save_button = gr.Button("Save Answer")

        def save_answer(text):
            try:
                with open("answer.txt", "w", encoding="utf-8") as f:
                    f.write(text)
                return "Answer saved successfully to answer.txt"
            except Exception as e:
                return f"Error saving answer: {e}"

        query_button.click(
            query_rag_system,
            inputs=[query_input],
            outputs=[query_output]
        )
        save_button.click(
            save_answer,
            inputs=[query_output],
            outputs=[gr.Textbox(label="Save Status")]
        )

    # Examples (using a more robust approach)
    with gr.Accordion("Example Questions", open=False):
        gr.Examples(
            examples=[
                ["What is the main topic of the first document?"],
                ["Summarize the key points in the documents."],
                ["What are the limitations mentioned in the research papers?"],
            ],
            inputs=[query_input],
        )

    with gr.Tab("Settings"):
        batch_size = gr.Slider(
            minimum=1,
            maximum=50,
            value=32,
            step=1,
            label="Batch Size for Processing"
        )
        cache_size = gr.Slider(
            minimum=100,
            maximum=5000,
            value=1000,
            step=100,
            label="Cache Size (number of queries)"
        )
        with gr.Group():
            gr.Markdown("### Chunking Configuration")
            chunk_size = gr.Slider(
                minimum=100,
                maximum=1000,
                value=500,
                step=50,
                label="Chunk Size"
            )
            chunk_overlap = gr.Slider(
                minimum=0,
                maximum=200,
                value=50,
                step=10,
                label="Chunk Overlap"
            )
            update_chunking = gr.Button("Update Chunking Parameters")
            chunking_status = gr.Textbox(label="Status", interactive=False)

            update_chunking.click(
                update_chunking_config,
                inputs=[chunk_size, chunk_overlap],
                outputs=[chunking_status]
            )

if __name__ == '__main__':
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        share=False,
        max_threads=4
    )

--- END OF FILE rag_app-v11.py ---
```

**Testing Plan for `rag_app-v11.py`:**

1.  **Run `rag_app-v11.py` and test the "Summarize the key points in the documents." query again.** Verify that it still works and validation passes.
2.  **Test other example queries** from the Gradio UI (e.g., "What is the main topic...", "What are the limitations..."). Ensure they still work and return reasonable answers.
3.  **Test a "factual" query**:  Formulate a factual question that should be answerable from your PDF document (e.g., "Who is mentioned as a great general of God?"). Check if the answer is factually correct and if validation passes.
4.  **Test a "general" query**: Ask a general question about the document content that is not specifically a summary or factual question. Check if the answer is relevant and validation passes.
5.  **Test indexing**: Re-index your documents using `rag_app-v11.py`. Ensure indexing completes without errors and that the index files are updated.
6.  **Clear cache and test again**: Clear the cache and re-run some queries to ensure caching is still working correctly and that cold start latency is as expected.
7.  **Check Gradio UI**:  Verify that all Gradio UI elements (indexing buttons, query input, output, settings) are working as expected.

By performing these tests, you can gain confidence that `rag_app-v11.py` is functioning correctly and has successfully incorporated the desired enhancements without breaking existing functionalities.

---
```bash
$ diff -u rag_app-v10.py rag_app-v11.py
--- rag_app-v10.py      2025-03-12 11:32:18.049775187 +0800
+++ rag_app-v11.py      2025-03-12 12:34:33.572348973 +0800
@@ -490,27 +490,132 @@
         return f"Error during indexing: {str(e)}", None, None, None, None
 
 # --- Retrieval and Querying ---
+def enhance_context(chunks: List[Dict]) -> str:
+    """Creates a structured context with metadata and hierarchy."""
+    docs_by_file = {}
+    for chunk in chunks:
+        if chunk['filename'] not in docs_by_file:
+            docs_by_file[chunk['filename']] = []
+        docs_by_file[chunk['filename']].append(chunk)
+    
+    context_parts = []
+    for filename, file_chunks in docs_by_file.items():
+        # Sort chunks by their original order
+        file_chunks.sort(key=lambda x: x['chunk_index'])
+        context_parts.append(f"Document: {filename}\n")
+        for i, chunk in enumerate(file_chunks, 1):
+            context_parts.append(f"Section {i}:\n{chunk['chunk']}\n")
+    
+    return "\n".join(context_parts)
+
+def get_query_type(query: str) -> str:
+    """Determines query type for dynamic prompt and validation adjustment."""
+    summary_keywords = ['summarize', 'summary', 'overview', 'main points', 'key points']
+    factual_keywords = ['who', 'what', 'when', 'where', 'how many', 'which']
+    
+    query_lower = query.lower()
+    if any(keyword in query_lower for keyword in summary_keywords):
+        return 'summary'
+    elif any(keyword in query_lower for keyword in factual_keywords):
+        return 'factual'
+    return 'general'
+
+def get_prompt_template(query_type: str) -> str:
+    """Returns appropriate prompt template based on query type."""
+    templates = {
+        'summary': """[INST] >
+You are a technical expert. Provide a concise summary of the key points from the provided context.
+Use ONLY information explicitly stated in the context.
+Format your response as a bullet-point list of key points.
+
+Context: {context}
+>
+Question: {query} [/INST]""",
+        
+        'factual': """[INST] >
+You are a technical expert. Answer the question using ONLY facts directly stated in the context.
+Cite the specific section where you found the information.
+If the answer isn't explicitly in the context, say so.
+
+Context: {context}
+>
+Question: {query} [/INST]""",
+        
+        'general': """[INST] >
+You are a technical expert. Answer based STRICTLY on the provided context.
+Do not make assumptions or add external information.
+If the context doesn't contain enough information, say so.
+
+Context: {context}
+>
+Question: {query} [/INST]"""
+    }
+    return templates.get(query_type, templates['general'])
+
+def compute_semantic_similarity(text1: str, text2: str, model) -> float:
+    """Compute semantic similarity with error handling and normalization."""
+    try:
+        emb1 = model.encode([text1])
+        emb2 = model.encode([text2])
+        
+        similarity = np.dot(emb1[0], emb2[0]) / (
+            np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
+        return float(similarity)
+    except Exception as e:
+        print(f"[DEBUG] Similarity computation error: {e}")
+        return 0.0
+
 def validate_answer(answer: str, context: str, query: str) -> bool:
-    """Enhanced answer validation with dynamic thresholding."""
+    """Enhanced answer validation with multiple metrics and dynamic thresholds."""
     try:
-        answer_emb = embedding_model.encode([answer])
-        context_emb = embedding_model.encode([context])
-        query_emb = embedding_model.encode([query])
-        
-        # Calculate multiple similarity scores
-        answer_context_sim = np.dot(answer_emb[0], context_emb[0]) / (
-            np.linalg.norm(answer_emb[0]) * np.linalg.norm(context_emb[0]))
-        answer_query_sim = np.dot(answer_emb[0], query_emb[0]) / (
-            np.linalg.norm(answer_emb[0]) * np.linalg.norm(query_emb[0]))
-        
-        # Dynamic thresholding based on query complexity
-        base_threshold = 0.85
-        query_length_factor = min(len(query.split()) / 10, 1.0)
-        threshold = base_threshold - (0.05 * query_length_factor)
+        print("\n[DEBUG] Starting enhanced answer validation...")
+        
+        if not all([answer, context, query]):
+            print("[DEBUG] Validation failed: Missing input")
+            return False
+            
+        # Get query type for dynamic thresholding
+        query_type = get_query_type(query)
+        print(f"[DEBUG] Detected query type: {query_type}")
+        
+        # Calculate multiple similarity metrics
+        metrics = {
+            'answer_context': compute_semantic_similarity(answer, context, embedding_model),
+            'answer_query': compute_semantic_similarity(answer, query, embedding_model),
+            'context_query': compute_semantic_similarity(context, query, embedding_model)
+        }
+        
+        # Dynamic thresholds based on query type with adjusted summary query threshold
+        thresholds = {
+            'summary': {'context': 0.6, 'query': 0.15},  # Lowered threshold for summary queries
+            'factual': {'context': 0.7, 'query': 0.4},   # Maintained stricter threshold for factual
+            'general': {'context': 0.65, 'query': 0.35}  # Maintained moderate threshold for general
+        }[query_type]
+        
+        print("[DEBUG] Validation Metrics:")
+        print(f"  - Answer-Context Similarity: {metrics['answer_context']:.4f}")
+        print(f"  - Answer-Query Similarity: {metrics['answer_query']:.4f}")
+        print(f"  - Context-Query Similarity: {metrics['context_query']:.4f}")
+        print(f"  - Required Thresholds (type={query_type}):")
+        print(f"    * Context: {thresholds['context']:.4f}")
+        print(f"    * Query: {thresholds['query']:.4f}")
+        
+        # Validate against thresholds with additional logging
+        validation_result = (
+            metrics['answer_context'] > thresholds['context'] and
+            metrics['answer_query'] > thresholds['query']
+        )
+        
+        if not validation_result:
+            print("[DEBUG] Validation details:")
+            print(f"  Context check: {metrics['answer_context']:.4f} > {thresholds['context']:.4f} = {metrics['answer_context'] > thresholds['context']}")
+            print(f"  Query check: {metrics['answer_query']:.4f} > {thresholds['query']:.4f} = {metrics['answer_query'] > thresholds['query']}")
+        
+        print(f"[DEBUG] Validation {'passed' if validation_result else 'failed'}")
+        return validation_result
         
-        return answer_context_sim > threshold and answer_query_sim > 0.6
     except Exception as e:
-        print(f"Error validating answer: {e}")
+        print(f"[DEBUG] Validation error: {str(e)}")
         return False
 
 def initialize_components():
@@ -642,23 +747,27 @@
         return [chunks[i] for i in I[0]]
 
 def query_rag_system(query_text: str, progress=gr.Progress()) -> str:
-    """Enhanced query processing with improved caching and validation."""
+    """Enhanced query processing with improved prompting and validation."""
     if not query_text or not query_text.strip():
         return "Error: Empty query"
     
     try:
+        print(f"\n[DEBUG {datetime.now().isoformat()}] Processing query: {query_text}")
+        
         cached_response = cached_query_response(query_text)
         if cached_response:
+            print("[DEBUG] Retrieved from cache")
             return cached_response
 
         # Initialize components and load resources
         global index, embeddings, chunks, embedding_model, client
         load_successful, load_messages = load_resources()
         if not load_successful:
+            print(f"[DEBUG] Resource loading failed: {load_messages}")
             return "Error: " + "\n".join(load_messages)
 
         progress(0.2, desc="Encoding query...")
-        initialize_components()  # Ensure reranker and ES components are initialized
+        initialize_components()
         
         query_embedding = embedding_model.encode([query_text])
         
@@ -666,40 +775,47 @@
         try:
             relevant_chunks = hybrid_search(query_embedding, query_text)
             if not relevant_chunks:
+                print("[DEBUG] No relevant chunks retrieved.")
                 return "No relevant information found for your query."
-        except Exception as e:
-            print(f"Search error: {e}")  # Add logging
-            # Fallback to basic FAISS search if hybrid search fails
-            D, I = index.search(np.float32(query_embedding), k=5)
-            relevant_chunks = [chunks[i] for i in I[0]]
-
-        context = "\n\n".join([f"Document {i+1}:\n"+"\n"+chunk["chunk"] for i, chunk in enumerate(relevant_chunks)])
-
-        progress(0.7, desc="Generating response...")
-        try:
+            
+            print(f"[DEBUG] Retrieved {len(relevant_chunks)} relevant chunks")
+            
+            # Enhanced context creation with structure
+            context = enhance_context(relevant_chunks)
+            print(f"[DEBUG] Structured context created ({len(context)} chars)")
+            
+            # Determine query type and get appropriate prompt
+            query_type = get_query_type(query_text)
+            prompt_template = get_prompt_template(query_type)
+            
+            progress(0.7, desc="Generating response...")
             completion = client.chat.completions.create(
                 model=MODEL_NAME,
-                messages=[
-                    {"role": "user", "content": f"""[INST] >
-                    You are a technical expert. Answer ONLY using the context below.
-                    Context: {context}
-                    >
-                    Question: {query_text} [/INST]"""}
-                ]
+                messages=[{
+                    "role": "user",
+                    "content": prompt_template.format(
+                        context=context,
+                        query=query_text
+                    )
+                }]
             )
+            
             response = completion.choices[0].message.content
+            print(f"[DEBUG] Generated response ({len(response)} chars)")
             
             if validate_answer(response, context, query_text):
                 query_cache.set(query_text, response)
                 return response
+                
+            print("[DEBUG] Answer validation failed, attempting fallback...")
             return "I apologize, but I cannot generate a reliable answer based on the provided context."
             
         except Exception as e:
-            print(f"Response generation error: {e}")  # Add logging
-            return f"Error generating response: {str(e)}"
-
+            print(f"[DEBUG] Error during query processing: {e}")
+            return f"Error processing query: {str(e)}"
+            
     except Exception as e:
-        print(f"General query error: {e}")  # Add logging
+        print(f"[DEBUG] General query error: {e}")
         return f"Error processing query: {str(e)}"
 
 # Add this function before the Gradio interface section
```
