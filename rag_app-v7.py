# rag_app.py (Complete and further improved)
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
ES_HOST = os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")
DATA_PATH = "data"
BM25_INDEX_FILE = "bm25_index.pkl"
WATERMARK_FILE = "watermark.txt"

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
    """Validates if file is a valid PDF."""
    global magic_available
    if not magic_available: # Fallback validation if magic is not available
        return filepath.lower().endswith('.pdf') and os.path.getsize(filepath) <= 100 * 1024 * 1024

    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(filepath)
        if file_type != 'application/pdf':
            return False
        if os.path.getsize(filepath) > 100 * 1024 * 1024:  # 100MB limit
            return False
        return True
    except Exception:
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
    """Creates embeddings with batch processing to manage memory."""
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = model.encode([chunk["chunk"] for chunk in batch])
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

def build_faiss_index(embeddings):
    """Builds a FAISS index for efficient similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.float32(embeddings))
    return index

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

def perform_indexing(progress=gr.Progress()):
    """Enhanced indexing process with incremental indexing."""
    global index, chunks, embeddings, embedding_model, bm25_index, watermark_tracker

    try:
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
        
        new_embeddings = create_embeddings(new_chunks, embedding_model)
        
        # Combine with existing embeddings if any
        try:
            existing_embeddings = np.load("embeddings.npy")
            embeddings = np.vstack([existing_embeddings, new_embeddings])
        except FileNotFoundError:
            embeddings = new_embeddings

        progress(0.7, desc="Building indices...")
        index = build_faiss_index(embeddings)
        bm25_index = build_bm25_index(chunks)

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
        return "Incremental indexing complete!", "embeddings.npy", "faiss_index.index", "chunks.json", BM25_INDEX_FILE
    except Exception as e:
        return f"Error during indexing: {e}", None, None, None, None

# --- Retrieval and Querying ---
def validate_answer(answer: str, context: str, query: str) -> bool:
    """Enhanced answer validation with dynamic thresholding."""
    try:
        answer_emb = embedding_model.encode([answer])
        context_emb = embedding_model.encode([context])
        query_emb = embedding_model.encode([query])
        
        # Calculate multiple similarity scores
        answer_context_sim = np.dot(answer_emb[0], context_emb[0]) / (
            np.linalg.norm(answer_emb[0]) * np.linalg.norm(context_emb[0]))
        answer_query_sim = np.dot(answer_emb[0], query_emb[0]) / (
            np.linalg.norm(answer_emb[0]) * np.linalg.norm(query_emb[0]))
        
        # Dynamic thresholding based on query complexity
        base_threshold = 0.85
        query_length_factor = min(len(query.split()) / 10, 1.0)
        threshold = base_threshold - (0.05 * query_length_factor)
        
        return answer_context_sim > threshold and answer_query_sim > 0.6
    except Exception as e:
        print(f"Error validating answer: {e}")
        return False

def initialize_components():
    """Initialize reranker and Elasticsearch components."""
    global reranker_tokenizer, reranker_model, es_client
    if reranker_tokenizer is None:
        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    if reranker_model is None:
        reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
    if es_client is None:
        try: # Add try-except for Elasticsearch client initialization
            es_client = elasticsearch.Elasticsearch(ES_HOST)
        except Exception as e:
            print(f"Warning: Could not initialize Elasticsearch client. Hybrid search might be degraded. Error: {e}")
            es_client = None # Set to None if initialization fails

def hybrid_search(query_embedding, query_text, k=5):
    """Perform hybrid search using FAISS and Elasticsearch."""
    # Vector search with FAISS
    D, I = index.search(np.float32(query_embedding), k*2)

    candidates = []
    seen_indices_faiss = set() # To track indices from FAISS

    for idx in I[0]:
        if idx not in seen_indices_faiss:
            candidates.append(chunks[idx])
            seen_indices_faiss.add(idx)

    if es_client: # Only use Elasticsearch if client is initialized
        try:
            es_results = es_client.search(
                index="chunks",
                body={
                    "query": {
                        "match": {
                            "chunk": query_text
                        }
                    },
                    "size": k*2
                }
            )

            seen_filenames_es = set()

            for hit in es_results['hits']['hits']:
                chunk_index = int(hit['_id'])
                if chunk_index < len(chunks) and chunks[chunk_index]['filename'] not in seen_filenames_es:
                    candidates.append(chunks[chunk_index])
                    seen_filenames_es.add(chunks[chunk_index]['filename'])
        except Exception as e:
            print(f"Elasticsearch query failed: {e}")

    return rerank_results(query_text, candidates)[:k]

def rerank_results(query, candidates):
    """Rerank results using the reranker model."""
    inputs = reranker_tokenizer(
        [[query, c["chunk"]] for c in candidates],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)
    
    ranked_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in ranked_pairs]

def query_rag_system(query_text: str, progress=gr.Progress()) -> str:
    """Enhanced query processing with improved caching and validation."""
    if not query_text or not query_text.strip():
        return "Error: Empty query"
    
    try:
        cached_response = cached_query_response(query_text)
        if cached_response:
            return cached_response

        # Initialize components and load resources
        global index, embeddings, chunks, embedding_model, client
        load_successful, load_messages = load_resources()
        if not load_successful:
            return "Error: " + "\n".join(load_messages)

        progress(0.2, desc="Encoding query...")
        initialize_components()  # Ensure reranker and ES components are initialized
        
        query_embedding = embedding_model.encode([query_text])
        
        progress(0.4, desc="Searching index...")
        try:
            relevant_chunks = hybrid_search(query_embedding, query_text)
            if not relevant_chunks:
                return "No relevant information found for your query."
        except Exception as e:
            print(f"Search error: {e}")  # Add logging
            # Fallback to basic FAISS search if hybrid search fails
            D, I = index.search(np.float32(query_embedding), k=5)
            relevant_chunks = [chunks[i] for i in I[0]]

        context = "\n\n".join([f"Document {i+1}:\n"+"\n"+chunk["chunk"] for i, chunk in enumerate(relevant_chunks)])

        progress(0.7, desc="Generating response...")
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": f"""[INST] >
                    You are a technical expert. Answer ONLY using the context below.
                    Context: {context}
                    >
                    Question: {query_text} [/INST]"""}
                ]
            )
            response = completion.choices[0].message.content
            
            if validate_answer(response, context, query_text):
                query_cache.set(query_text, response)
                return response
            return "I apologize, but I cannot generate a reliable answer based on the provided context."
            
        except Exception as e:
            print(f"Response generation error: {e}")  # Add logging
            return f"Error generating response: {str(e)}"

    except Exception as e:
        print(f"General query error: {e}")  # Add logging
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
            gr.File(label="Chunks", interactive=False)
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

