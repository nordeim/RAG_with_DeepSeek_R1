Carefully review the updated code and compare it to the original `rag_app-v12.py` to ensure the reranking functionality has been correctly integrated and that no other parts of the code have been unintentionally modified. You will perform a line-by-line comparison to validate the changes.

**Line-by-Line Comparison and Validation:**

**(Starting from the beginning of the file)**

* **Imports:**
    * No changes in imports.  `file_utils` import and `magic_available` handling were already in `v12.py`.

* **Configuration:**
    * No changes in configuration variables.

* **Global Variables (`WatermarkTracker`, etc.):**
    * No changes. `watermark_tracker`, `embedding_model`, `index`, `chunks`, `embeddings`, `client`, `reranker_tokenizer`, `reranker_model`, `es_client`, `bm25_index` remain the same.

* **Cache Implementation (`CacheEntry`, `TTLCache`, `query_cache`, `cached_query_response`):**
    * No changes. Cache functionality is untouched.

* **Helper Functions (`load_resources`, `clear_cache`):**
    * **`load_resources` - MODIFIED:**
        * **Added:** Loading of `reranker_tokenizer` and `reranker_model` within `load_resources` function. This is correct and necessary to initialize the reranking components when resources are loaded.
        * Added `reranker_model, reranker_tokenizer` to the global variables list at the beginning of the function.
    * **`clear_cache` - MODIFIED:**
        * **Added:** Clearing of `reranker_model` and `reranker_tokenizer` within `clear_cache` function.  This is also correct to reset all resources when cache is cleared.
        * Added `reranker_model, reranker_tokenizer` to the global variables list at the beginning of the function.

* **Indexing Functions (`validate_pdf_file`, `ChunkingConfig`, `chunk_documents`, `create_embeddings`, `build_faiss_index`, `build_bm25_index`, `load_and_preprocess_documents`, `create_elasticsearch_index`, `index_documents_to_elasticsearch`, `perform_indexing`):**
    * No changes in indexing functions. These functions remain the same, and indexing process is not directly affected by the reranking feature which is intended for the query stage.

* **Retrieval and Querying (`enhance_context`, `get_query_type`, `get_prompt_template`, `compute_semantic_similarity`, `validate_answer`, `initialize_components`, `hybrid_search`, `query_rag_system`):**
    * **`initialize_components` - No change:** This function was already present in `v12.py` and correctly initializes `reranker_tokenizer` and `reranker_model`. No modifications are made here.
    * **`hybrid_search` - MODIFIED (Significant):**
        * **Reranking Logic Added:**  A new block of code is inserted after the initial candidate retrieval and scoring (FAISS, BM25, ES).
            * Checks if `reranker_model` and `reranker_tokenizer` are available.
            * Prepares input pairs for reranking `(query_text, candidate['chunk']['chunk'])`.
            * Tokenizes inputs using `reranker_tokenizer`.
            * Performs inference using `reranker_model`.
            * Extracts reranking scores.
            * Attaches rerank scores to candidates in `candidates['scores']['rerank']`.
            * Sorts candidates primarily based on 'rerank' score if available, otherwise falls back to the combined scores from FAISS, BM25, and ES.
        * **Score Normalization Improvement in `hybrid_search`:** Added checks to avoid division by zero when normalizing scores from FAISS and BM25, using `if max(D[0]) > 0 else 1` and `if max(bm25_scores) > 0 else 0` and `if es_results['hits']['max_score'] else 0`. This is a good defensive programming practice.
        * **Debug Log Added:** `print("[DEBUG] Applying Reranking...")` and `print("[DEBUG] Reranking skipped.")` are added for better visibility into the reranking process.
    * **`query_rag_system` - No change:** No changes to the main query function. It calls `hybrid_search` which now includes reranking.
    * **Other retrieval functions (`enhance_context`, `get_query_type`, `get_prompt_template`, `compute_semantic_similarity`, `validate_answer`) - No change:** These functions are related to prompt generation and answer validation and are not directly involved in the retrieval/reranking process.

* **Settings Function (`update_chunking_config`):**
    * No changes. Chunking configuration update remains the same.

* **Gradio Interface:**
    * No changes in the Gradio interface definition. The UI elements for indexing, querying, settings, and examples are unchanged.

* **`if __name__ == '__main__':` block:**
    * No changes. Gradio launch configuration remains the same.

**Validation Summary:**

* **Reranking Integration:** The reranking functionality has been successfully integrated into the `hybrid_search` function. The code correctly loads the reranker model and tokenizer, performs reranking on the retrieved candidates, and re-sorts them based on reranking scores.
* **Resource Loading and Clearing:** `load_resources` and `clear_cache` are updated to manage the reranking model and tokenizer, ensuring proper initialization and cleanup.
* **No Unintended Changes:**  The line-by-line comparison confirms that only the intended sections related to reranking and minor score normalization improvements have been modified. Other parts of the code, such as indexing, prompting, validation, and the Gradio interface, remain untouched, minimizing the risk of introducing regressions.
* **Defensive Programming:** The addition of zero-division checks in score normalization within `hybrid_search` is a positive improvement for robustness.
* **Debug Logging:** Debug logs added around reranking provide helpful visibility for monitoring and debugging the reranking process.

**Conclusion:**

The updated code appears to have successfully merged the reranking functionality into `rag_app-v12.py` without introducing unintended changes or breaking existing functionality. The integration seems well-structured and includes necessary resource management and logging.

**Next Steps (for final confirmation - which I cannot perform as an AI in this environment, but would recommend in a real-world scenario):**

1. **Runtime Testing:** Deploy and run the updated application.
2. **Functional Testing:** Perform various queries, especially those where reranking is expected to make a difference (complex or nuanced queries). Compare the results with and without reranking (if possible, by temporarily disabling the reranking block in `hybrid_search`) to empirically assess the improvement in relevance.
3. **Performance Testing:** Check for any performance impact due to the reranking step, especially latency. Ensure the reranking process is reasonably efficient.
4. **Error Handling:** Test error conditions, such as cases where the reranker model fails to load, to verify graceful degradation and informative error messages.

Based on the code review and line-by-line comparison, the updated code is logically sound and implements the requested reranking feature as intended. Confirm that the suggested fix has been merged without affecting other parts of the code based on this code review.

---
```python
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

# Update imports
from file_utils import validate_file, load_document, ALLOWED_EXTENSIONS
import logging

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
                if time.time() - entry.timestamp <= entry.ttl:
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
    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker, reranker_model, reranker_tokenizer
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

    if reranker_tokenizer is None:
        try:
            reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        except Exception as e:
            messages.append(f"Error loading reranker tokenizer: {e}")
            resource_load_successful = False

    if reranker_model is None:
        try:
            reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
        except Exception as e:
            messages.append(f"Error loading reranker model: {e}")
            resource_load_successful = False

    return resource_load_successful, messages


def clear_cache():
    """Clears the cached resources."""
    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker, reranker_model, reranker_tokenizer
    embedding_model = None
    index = None
    chunks = None
    embeddings = None
    client = None
    bm25_index = None
    reranker_model = None
    reranker_tokenizer = None
    watermark_tracker = WatermarkTracker(WATERMARK_FILE)  # Reset watermark tracker
    return "Cache cleared."

# --- Indexing Functions ---

def validate_pdf_file(filepath: str) -> bool:
    """Enhanced file validation."""
    valid, msg = validate_file(filepath)
    if not valid:
        logger.warning(f"File validation failed: {msg}")
        return False
    return True

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
    """Enhanced document loading with multiple file type support."""
    supported_files = []
    for ext in ALLOWED_EXTENSIONS:
        supported_files.extend(Path(data_path).glob(f"*{ext}"))

    all_docs = []
    for file_path in supported_files:
        if not validate_file(str(file_path))[0]:
            continue

        documents, error = load_document(str(file_path))
        if documents:
            all_docs.extend(documents)
        else:
            logger.error(f"Failed to load {file_path}: {error}")

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
            'summary': {'context': 0.5, 'query': 0.1},  # Lowered threshold for summary queries
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
                    'scores': {'faiss': 1 - (score / max(D[0]) if max(D[0]) > 0 else 1)}  # Normalize score, avoid division by zero
                }

        # Add BM25 results if available
        if bm25_index:
            try:
                bm25_scores = bm25_index.get_scores(query_text.split())
                top_bm25_indices = np.argsort(bm25_scores)[-k*2:][::-1]
                for idx in top_bm25_indices:
                    if idx in candidates:
                        candidates[idx]['scores']['bm25'] = bm25_scores[idx] / max(bm25_scores) if max(bm25_scores) > 0 else 0
                    else:
                        candidates[idx] = {
                            'chunk': chunks[idx],
                            'scores': {'bm25': bm25_scores[idx] / max(bm25_scores) if max(bm25_scores) > 0 else 0}
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
                        candidates[idx]['scores']['es'] = hit['_score'] / es_results['hits']['max_score'] if es_results['hits']['max_score'] else 0
                    else:
                        candidates[idx] = {
                            'chunk': chunks[idx],
                            'scores': {'es': hit['_score'] / es_results['hits']['max_score'] if es_results['hits']['max_score'] else 0}
                        }
            except Exception as e:
                print(f"Elasticsearch search error: {e}")

        # Reranking implementation
        if reranker_model is not None and reranker_tokenizer is not None and candidates:
            print("[DEBUG] Applying Reranking...")
            rerank_inputs = [(query_text, candidate['chunk']['chunk']) for candidate in candidates.values()]
            tokenized_inputs = reranker_tokenizer(rerank_inputs, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                reranker_outputs = reranker_model(**tokenized_inputs).logits.flatten()
                rerank_scores = reranker_outputs.tolist()

            # Attach rerank scores to candidates
            for idx, score in zip(candidates.keys(), rerank_scores):
                candidates[idx]['scores']['rerank'] = score

            # Sort by rerank score
            reranked_candidates = sorted(candidates.values(), key=lambda x: x['scores'].get('rerank', -1), reverse=True) # Default to -1 if no rerank score
            final_candidates = [(candidate['chunk'], candidate['scores'].get('rerank', 0)) for candidate in reranked_candidates]

        else: # Fallback to original ranking if reranker not available
            print("[DEBUG] Reranking skipped.")
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
            # Sort and return top k results based on combined scores
            final_candidates = sorted(final_candidates, key=lambda x: x[1], reverse=True)


        return [c[0] for c in final_candidates[:k]]

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
```
