*Your task:* carefully review the attached Python code `rag_app-v9.py` and also do a deep dive research to create a research paper of at least 3000 words about it.

Use the following methodology and flow to guide you to systematically complete your task:

1. **Code Review of `rag_app-v9.py`**:
    - Identify any coding or logic errors.
    - Understand the code logic and flow, especially around indexing and querying.
    - Analyze the improvements from `rag_app-v5.py` to `rag_app-v9.py` based on the provided `diff` output.
    - Search the web for best practices and potential improvements for RAG systems, specifically related to the techniques used in the code (hybrid search, incremental indexing, validation, caching, BM25, FAISS, reranking, Elasticsearch).

2. **Research Paper Review and Update**:
    - Review the existing research paper "Research Paper: Analysis of Retrieval-Augmented Generation (RAG) Implementation in rag_app-v9.py".
    - Update the paper based on your code review findings, web searches, and analysis of v5 to v9 changes.
    - Focus on the effectiveness of indexing and querying methodologies in `rag_app-v9.py`.
    - Incorporate findings from web searches to strengthen the analysis and recommendations in the paper.
    - Pay attention to the "Identified Issues" and "Critical Improvements" sections of the paper and update them based on your careful review and deep analysis.

**Suggested Execution Plan:**

1. **Analyze the `diff` output from v5 to v9**: Understand the changes made from v5 to v9. This will give you a good starting point for code review.

2. **Code Review of `rag_app-v9.py`**:
    - Read through the code section by section, focusing on the changed parts and the core logic.
    - Look for potential errors, inefficiencies, and areas for improvement.
    - Pay attention to the identified issues in the research paper and see if they are valid.

3. **Web Search**:
    - Search for information on RAG best practices, hybrid search, incremental indexing, BM25, FAISS, reranking, Elasticsearch, and answer validation in RAG systems.
    - Look for evidence to support or refute the claims and issues identified in the research paper and during code review.

4. **Create the Research Paper (at least 3000 words)**:
    - Based on the code review and web search findings, update each section of the research paper.
    - Specifically address the "Identified Issues" and "Critical Improvements" sections, providing more detailed analysis and concrete recommendations.
    - Enhance the "Performance Evaluation" section with realistic metrics and improvement strategies.
    - Ensure the paper is well-structured, clear, and provides valuable insights into the RAG system.
    - each section should have a detailed explanation of the logic, rationale of the code, with code snippets as examples

5. **Final Review**: Reread the updated research paper and code to ensure consistency and accuracy.

Now start by analyzing the `diff` output below to understand the changes from v5 to v9. Then deep dive into the code review.

```bash
$ diff -u rag_app-v5.py rag_app-v9.py
--- rag_app-v5.py       2025-03-06 12:29:14.024055496 +0800
+++ rag_app-v9.py       2025-03-10 20:03:26.450697308 +0800
@@ -7,14 +7,30 @@
 from openai import OpenAI
 from dotenv import load_dotenv
 import gradio as gr
-import pdfplumber  # Added for better PDF extraction
+import pdfplumber  # Added for better PDF extraction (though may not be directly used in indexing flow now)
 import time
 from typing import List, Tuple, Dict, Optional
 from functools import lru_cache
-import magic  # for file type validation
-from elasticsearch import Elasticsearch  # Added for hybrid search
-from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Added for reranking
-import torch  # Add this import at the top with other imports
+import elasticsearch  # Corrected import for Elasticsearch
+from transformers import AutoModelForSequenceClassification, AutoTokenizer
+import torch
+from datetime import datetime
+from pathlib import Path
+import pickle
+from rank_bm25 import BM25Okapi
+from langchain_community.document_loaders import PyPDFLoader  # Updated import
+from functools import wraps
+from typing import Dict, Optional
+from dataclasses import dataclass
+from threading import Lock
+
+# Try to import magic, handle if not available
+magic_available = True
+try:
+    import magic  # for file type validation
+except ImportError:
+    print("Warning: libmagic not found. File type validation will be less robust.")
+    magic_available = False
 
 # Load environment variables
 load_dotenv()
@@ -28,9 +44,48 @@
 
 # Additional configuration
 RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
-ES_HOST = os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")
+ES_HOST = os.environ.get("ELASTICSEARCH_HOST", "https://localhost:9200")
+ES_USER = os.environ.get("ELASTICSEARCH_USER", "elastic")
+ES_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD")
+ES_CERT_PATH = os.environ.get("ELASTICSEARCH_CERT_PATH", "http_cert.pem")
+ES_VERIFY_CERTS = os.environ.get("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true"
+DATA_PATH = "data"
+BM25_INDEX_FILE = "bm25_index.pkl"
+WATERMARK_FILE = "watermark.txt"
+ES_INDEX_NAME = "chunks"  # Add constant for ES index name
 
 # --- Global Variables (for caching) ---
+class WatermarkTracker:
+    """Tracks document modification times for incremental indexing"""
+    def __init__(self, filename='watermark.txt'):
+        self.filename = filename
+        self.last_updated = self._load_watermark()
+
+    def _load_watermark(self):
+        try:
+            with open(self.filename, 'r') as f:
+                return datetime.fromisoformat(f.read().strip())
+        except FileNotFoundError:
+            return datetime.min
+
+    def save_watermark(self):
+        with open(self.filename, 'w') as f:
+            f.write(self.last_updated.isoformat())
+
+    def update_watermark(self, documents):
+        latest = max(
+            (doc.metadata.get('last_modified', self.last_updated) for doc in documents),
+            default=self.last_updated
+        )
+        if latest > self.last_updated:
+            self.last_updated = latest
+            self.save_watermark()
+
+    def filter_new_documents(self, documents):
+        return [doc for doc in documents
+                if doc.metadata.get('last_modified', datetime.min) > self.last_updated]
+
+watermark_tracker = WatermarkTracker(WATERMARK_FILE)  # Add this
 embedding_model = None
 index = None
 chunks = None
@@ -39,11 +94,47 @@
 reranker_tokenizer = None
 reranker_model = None
 es_client = None
+bm25_index = None  # Add this
+
+# --- Cache Implementation ---
+@dataclass
+class CacheEntry:
+    value: str
+    timestamp: float
+
+class TTLCache:
+    def __init__(self, maxsize: int, ttl: int):
+        self.cache: Dict[str, CacheEntry] = {}
+        self.maxsize = maxsize
+        self.ttl = ttl
+        self.lock = Lock()
+
+    def get(self, key: str) -> Optional[str]:
+        with self.lock:
+            if key in self.cache:
+                entry = self.cache[key]
+                if time.time() - entry.timestamp <= self.ttl:
+                    return entry.value
+                del self.cache[key]
+            return None
+
+    def set(self, key: str, value: str):
+        with self.lock:
+            if len(self.cache) >= self.maxsize:
+                oldest = min(self.cache.items(), key=lambda x: x[1].timestamp)
+                del self.cache[oldest[0]]
+            self.cache[key] = CacheEntry(value, time.time())
+
+query_cache = TTLCache(maxsize=1000, ttl=3600)
+
+def cached_query_response(query_text: str) -> Optional[str]:
+    """Enhanced query caching with TTL."""
+    return query_cache.get(query_text)
 
 # --- Helper Functions ---
 def load_resources():
-    """Loads resources (embeddings, index, chunks, model, client) and handles errors."""
-    global embedding_model, index, chunks, embeddings, client
+    """Loads resources (embeddings, index, chunks, model, client, bm25_index) and handles errors."""
+    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker
     resource_load_successful = True  # Track if all resources loaded correctly
     messages = []
 
@@ -96,23 +187,37 @@
             messages.append(f"Error initializing OpenAI client: {e}")
             resource_load_successful = False
 
+    if bm25_index is None:
+        try:
+            with open(BM25_INDEX_FILE, 'rb') as f:
+                bm25_index = pickle.load(f)
+        except FileNotFoundError:
+            messages.append("BM25 index not found. Will be created during indexing.")
+            resource_load_successful = False
+
     return resource_load_successful, messages
 
 
 def clear_cache():
     """Clears the cached resources."""
-    global embedding_model, index, chunks, embeddings, client
+    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker
     embedding_model = None
     index = None
     chunks = None
     embeddings = None
     client = None
+    bm25_index = None
+    watermark_tracker = WatermarkTracker(WATERMARK_FILE)  # Reset watermark tracker
     return "Cache cleared."
 
 # --- Indexing Functions ---
 
 def validate_pdf_file(filepath: str) -> bool:
     """Validates if file is a valid PDF."""
+    global magic_available
+    if not magic_available: # Fallback validation if magic is not available
+        return filepath.lower().endswith('.pdf') and os.path.getsize(filepath) <= 100 * 1024 * 1024
+
     try:
         mime = magic.Magic(mime=True)
         file_type = mime.from_file(filepath)
@@ -124,53 +229,44 @@
     except Exception:
         return False
 
-def load_pdf_documents(data_dir='data', batch_size=5):
-    """Enhanced PDF document loading with pdfplumber."""
-    documents = []
-    valid_files = []
+@dataclass
+class ChunkingConfig:
+    chunk_size: int = 500
+    chunk_overlap: int = 50
     
-    # First validate all files
-    for filename in os.listdir(data_dir):
-        if not filename.endswith('.pdf'):
-            continue
-        filepath = os.path.join(data_dir, filename)
-        if not validate_pdf_file(filepath):
-            print(f"Skipping invalid or too large PDF: {filename}")
-            continue
-        valid_files.append(filename)
-    
-    # Process in batches with pdfplumber
-    for i in range(0, len(valid_files), batch_size):
-        batch = valid_files[i:i + batch_size]
-        for filename in batch:
-            filepath = os.path.join(data_dir, filename)
-            try:
-                with pdfplumber.open(filepath) as pdf:
-                    text = ""
-                    for page in pdf.pages:
-                        text += page.extract_text() or ""
-                    if text.strip():
-                        documents.append({
-                            "filename": filename,
-                            "content": text,
-                            "metadata": {
-                                "timestamp": time.time(),
-                                "page_count": len(pdf.pages)
-                            }
-                        })
-            except Exception as e:
-                print(f"Error reading {filename}: {e}")
-    return documents
-
-def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
-    """Chunks the documents into smaller pieces."""
+    def validate(self):
+        if self.chunk_size < 100:
+            raise ValueError("Chunk size must be at least 100")
+        if self.chunk_overlap >= self.chunk_size:
+            raise ValueError("Overlap must be less than chunk size")
+        return self
+
+chunking_config = ChunkingConfig()
+
+def chunk_documents(documents, chunk_size=None, chunk_overlap=None):
+    """Enhanced chunking with version awareness and configurable parameters."""
+    if chunk_size is None:
+        chunk_size = chunking_config.chunk_size
+    if chunk_overlap is None:
+        chunk_overlap = chunking_config.chunk_overlap
+        
     chunks = []
-    for doc in documents:
-        content = doc["content"]
-        filename = doc["filename"]
-        for i in range(0, len(content), chunk_size - chunk_overlap):
+    for doc_id, doc in enumerate(documents):
+        content = doc.page_content
+        filename = os.path.basename(doc.metadata['source'])
+        last_modified = doc.metadata.get('last_modified', datetime.now())
+        
+        for chunk_idx, i in enumerate(range(0, len(content), chunk_size - chunk_overlap)):
             chunk = content[i:i + chunk_size]
-            chunks.append({"filename": filename, "chunk": chunk})
+            chunk_id = f"{filename}_{doc_id}_{chunk_idx}"
+            chunks.append({
+                "id": chunk_id,
+                "doc_id": doc_id,
+                "chunk_index": chunk_idx,
+                "filename": filename,
+                "chunk": chunk,
+                "last_modified": last_modified.isoformat()
+            })
     return chunks
 
 def create_embeddings(chunks: List[Dict], model, batch_size=32):
@@ -189,56 +285,156 @@
     index.add(np.float32(embeddings))
     return index
 
+def build_bm25_index(chunks):
+    """Builds BM25 index for keyword search."""
+    tokenized_corpus = [chunk['chunk'].split() for chunk in chunks]
+    return BM25Okapi(tokenized_corpus)
+
+def load_and_preprocess_documents(data_path=DATA_PATH):
+    """Loads PDF documents with metadata."""
+    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith('.pdf')]
+    all_docs = []
+    for file in pdf_files:
+        file_path = os.path.join(data_path, file)
+        if not validate_pdf_file(file_path):
+            continue
+        loader = PyPDFLoader(file_path)
+        documents = loader.load()
+        
+        file_mod_time = datetime.fromtimestamp(Path(file_path).stat().st_mtime)
+        for doc in documents:
+            doc.metadata['last_modified'] = file_mod_time
+            doc.metadata['source'] = str(file_path)
+        all_docs.extend(documents)
+    return all_docs
+
+def create_elasticsearch_index(es_client, index_name=ES_INDEX_NAME):
+    """Creates Elasticsearch index with mapping if it doesn't exist."""
+    if not es_client.indices.exists(index=index_name):
+        try:
+            es_client.indices.create(index=index_name, mappings={
+                "properties": {
+                    "chunk": {"type": "text"},
+                    "filename": {"type": "keyword"},
+                    "last_modified": {"type": "date"}
+                }
+            })
+            print(f"Elasticsearch index '{index_name}' created successfully.")
+        except Exception as e:
+            print(f"Error creating Elasticsearch index '{index_name}': {e}")
+            return False
+    return True
+
+def index_documents_to_elasticsearch(es_client, chunks, index_name=ES_INDEX_NAME, progress=gr.Progress()):
+    """Indexes chunks to Elasticsearch."""
+    if not es_client:
+        print("Elasticsearch client not initialized. Skipping Elasticsearch indexing.")
+        return True
+
+    if not create_elasticsearch_index(es_client, index_name):
+        return False
+
+    progress_es = gr.Progress(track_tqdm=True)
+    try:
+        for i, chunk in progress_es(enumerate(chunks), desc="Indexing to Elasticsearch"):
+            es_client.index(index=index_name, document=chunk, id=str(i))
+        print("Elasticsearch indexing complete.")
+        return True
+    except Exception as e:
+        print(f"Error during Elasticsearch indexing: {e}")
+        return False
 
 def perform_indexing(progress=gr.Progress()):
-    """Enhanced indexing process with proper progress tracking and validation."""
-    global index, chunks, embeddings, embedding_model
+    """Enhanced indexing process with incremental indexing."""
+    global index, chunks, embeddings, embedding_model, bm25_index, watermark_tracker, es_client
 
     try:
         progress(0.1, desc="Loading documents...")
-        documents = load_pdf_documents()
-        if not documents:
-            return "No PDF documents found or all documents were invalid.", None, None, None
+        all_documents = load_and_preprocess_documents()
+        if not all_documents:
+            return "No PDF documents found.", None, None, None, None
+
+        new_documents = watermark_tracker.filter_new_documents(all_documents)
+        if not new_documents:
+            return "No new documents to index.", None, None, None, None
 
         progress(0.3, desc="Chunking documents...")
-        chunks = chunk_documents(documents)
+        new_chunks = chunk_documents(new_documents)
+        
+        # Combine with existing chunks if any
+        existing_chunks = []
+        try:
+            with open("chunks.json", 'r') as f:
+                existing_chunks = json.load(f)
+        except FileNotFoundError:
+            pass
+        
+        chunks = existing_chunks + new_chunks
 
         progress(0.5, desc="Creating embeddings...")
         if embedding_model is None:
             embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
-        embeddings = create_embeddings(chunks, embedding_model)
+        
+        new_embeddings = create_embeddings(new_chunks, embedding_model)
+        
+        # Combine with existing embeddings if any
+        try:
+            existing_embeddings = np.load("embeddings.npy")
+            embeddings = np.vstack([existing_embeddings, new_embeddings])
+        except FileNotFoundError:
+            embeddings = new_embeddings
 
-        progress(0.8, desc="Building FAISS index...")
+        progress(0.7, desc="Building indices...")
         index = build_faiss_index(embeddings)
+        bm25_index = build_bm25_index(chunks)
+
+        # Initialize ES components before indexing
+        initialize_components()
+        
+        # Add ES indexing step
+        progress(0.8, desc="Indexing to Elasticsearch...")
+        if es_client and not index_documents_to_elasticsearch(es_client, chunks, progress=progress):
+            print("Warning: Elasticsearch indexing failed, continuing with other indices")
 
-        # Save chunks, embeddings, and index
+        # Save updated indices
         try:
-            progress(0.9, desc="Saving index files...")
+            progress(0.9, desc="Saving indices...")
             np.save("embeddings.npy", embeddings)
             faiss.write_index(index, "faiss_index.index")
             with open("chunks.json", 'w') as f:
-                json.dump(chunks, f)
+                json.dump(chunks, f, indent=4) # Added indent for readability of json
+            with open(BM25_INDEX_FILE, 'wb') as f:
+                pickle.dump(bm25_index, f)
+            
+            watermark_tracker.update_watermark(new_documents)
         except Exception as e:
-            return f"Error saving index files: {e}", None, None, None
+            return f"Error saving indices: {e}", None, None, None, None
 
         progress(1.0, desc="Indexing complete!")
-        return "Indexing complete! All files saved successfully.", "embeddings.npy", "faiss_index.index", "chunks.json"
+        return "Incremental indexing complete!", "embeddings.npy", "faiss_index.index", "chunks.json", BM25_INDEX_FILE
     except Exception as e:
-        return f"Error during indexing: {e}", None, None, None
+        return f"Error during indexing: {e}", None, None, None, None
 
 # --- Retrieval and Querying ---
-@lru_cache(maxsize=1000)
-def cached_query_response(query_text: str) -> str:
-    """Caches frequent query responses."""
-    return None  # Just return None to allow main query logic to proceed
-
-def validate_answer(answer: str, context: str) -> bool:
-    """Validates the generated answer against the context using semantic similarity."""
+def validate_answer(answer: str, context: str, query: str) -> bool:
+    """Enhanced answer validation with dynamic thresholding."""
     try:
         answer_emb = embedding_model.encode([answer])
         context_emb = embedding_model.encode([context])
-        similarity = np.dot(answer_emb[0], context_emb[0]) / (np.linalg.norm(answer_emb[0]) * np.linalg.norm(context_emb[0]))
-        return similarity > 0.85
+        query_emb = embedding_model.encode([query])
+        
+        # Calculate multiple similarity scores
+        answer_context_sim = np.dot(answer_emb[0], context_emb[0]) / (
+            np.linalg.norm(answer_emb[0]) * np.linalg.norm(context_emb[0]))
+        answer_query_sim = np.dot(answer_emb[0], query_emb[0]) / (
+            np.linalg.norm(answer_emb[0]) * np.linalg.norm(query_emb[0]))
+        
+        # Dynamic thresholding based on query complexity
+        base_threshold = 0.85
+        query_length_factor = min(len(query.split()) / 10, 1.0)
+        threshold = base_threshold - (0.05 * query_length_factor)
+        
+        return answer_context_sim > threshold and answer_query_sim > 0.6
     except Exception as e:
         print(f"Error validating answer: {e}")
         return False
@@ -251,41 +447,70 @@
     if reranker_model is None:
         reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
     if es_client is None:
-        es_client = Elasticsearch(ES_HOST)
+        try:
+            es_config = {
+                "hosts": [ES_HOST],
+            }
+            
+            # Add authentication if password is provided
+            if ES_PASSWORD:
+                es_config["basic_auth"] = (ES_USER, ES_PASSWORD)
+            
+            # Add SSL certificate verification settings if using HTTPS
+            if ES_HOST.startswith("https"):
+                if not ES_VERIFY_CERTS:
+                    es_config["verify_certs"] = False
+                elif os.path.exists(ES_CERT_PATH):
+                    es_config["ca_certs"] = ES_CERT_PATH
+            
+            es_client = elasticsearch.Elasticsearch(**es_config)
+            
+            # Test connection
+            if not es_client.ping():
+                raise elasticsearch.ConnectionError("Could not connect to Elasticsearch")
+                
+        except Exception as e:
+            print(f"Warning: Could not initialize Elasticsearch client: {e}")
+            print("Hybrid search will operate in degraded mode (vector search only)")
+            es_client = None
 
 def hybrid_search(query_embedding, query_text, k=5):
     """Perform hybrid search using FAISS and Elasticsearch."""
     # Vector search with FAISS
     D, I = index.search(np.float32(query_embedding), k*2)
-    
-    # Text search with Elasticsearch
-    es_results = es_client.search(
-        index="chunks",
-        body={
-            "query": {
-                "match": {
-                    "content": query_text
-                }
-            },
-            "size": k*2
-        }
-    )
-    
-    # Combine and rerank results
+
     candidates = []
-    seen = set()
-    
+    seen_indices_faiss = set() # To track indices from FAISS
+
     for idx in I[0]:
-        if idx not in seen:
+        if idx not in seen_indices_faiss:
             candidates.append(chunks[idx])
-            seen.add(idx)
-            
-    for hit in es_results['hits']['hits']:
-        chunk_id = hit['_source']['chunk_id']
-        if chunk_id not in seen:
-            candidates.append(chunks[chunk_id])
-            seen.add(chunk_id)
-    
+            seen_indices_faiss.add(idx)
+
+    if es_client: # Only use Elasticsearch if client is initialized
+        try:
+            es_results = es_client.search(
+                index=ES_INDEX_NAME,  # Use constant instead of hardcoded string
+                body={
+                    "query": {
+                        "match": {
+                            "chunk": query_text
+                        }
+                    },
+                    "size": k*2
+                }
+            )
+
+            seen_filenames_es = set()
+
+            for hit in es_results['hits']['hits']:
+                chunk_index = int(hit['_id'])
+                if chunk_index < len(chunks) and chunks[chunk_index]['filename'] not in seen_filenames_es:
+                    candidates.append(chunks[chunk_index])
+                    seen_filenames_es.add(chunks[chunk_index]['filename'])
+        except Exception as e:
+            print(f"Elasticsearch query failed: {e}")
+
     return rerank_results(query_text, candidates)[:k]
 
 def rerank_results(query, candidates):
@@ -303,12 +528,11 @@
     return [pair[0] for pair in ranked_pairs]
 
 def query_rag_system(query_text: str, progress=gr.Progress()) -> str:
-    """Enhanced query processing with validation and error handling."""
+    """Enhanced query processing with improved caching and validation."""
     if not query_text or not query_text.strip():
         return "Error: Empty query"
     
     try:
-        # Try to get cached response
         cached_response = cached_query_response(query_text)
         if cached_response:
             return cached_response
@@ -351,7 +575,8 @@
             )
             response = completion.choices[0].message.content
             
-            if validate_answer(response, context):
+            if validate_answer(response, context, query_text):
+                query_cache.set(query_text, response)
                 return response
             return "I apologize, but I cannot generate a reliable answer based on the provided context."
             
@@ -363,6 +588,20 @@
         print(f"General query error: {e}")  # Add logging
         return f"Error processing query: {str(e)}"
 
+# Add this function before the Gradio interface section
+def update_chunking_config(size: int, overlap: int) -> str:
+    """Update chunking configuration with validation."""
+    global chunking_config
+    try:
+        new_config = ChunkingConfig(size, overlap)
+        new_config.validate()  # Will raise ValueError if invalid
+        chunking_config = new_config
+        return f"Successfully updated chunking parameters: size={size}, overlap={overlap}"
+    except ValueError as e:
+        return f"Error: {str(e)}"
+    except Exception as e:
+        return f"Unexpected error: {str(e)}"
+
 # --- Gradio Interface ---
 with gr.Blocks(title="PDF Q&A with DeepSeek-R1", theme=gr.themes.Soft()) as demo:
     gr.Markdown(
@@ -465,6 +704,30 @@
             step=100,
             label="Cache Size (number of queries)"
         )
+        with gr.Group():
+            gr.Markdown("### Chunking Configuration")
+            chunk_size = gr.Slider(
+                minimum=100,
+                maximum=1000,
+                value=500,
+                step=50,
+                label="Chunk Size"
+            )
+            chunk_overlap = gr.Slider(
+                minimum=0,
+                maximum=200,
+                value=50,
+                step=10,
+                label="Chunk Overlap"
+            )
+            update_chunking = gr.Button("Update Chunking Parameters")
+            chunking_status = gr.Textbox(label="Status", interactive=False)
+            
+            update_chunking.click(
+                update_chunking_config,
+                inputs=[chunk_size, chunk_overlap],
+                outputs=[chunking_status]
+            )
 
 if __name__ == '__main__':
     demo.queue(max_size=10).launch(
```
