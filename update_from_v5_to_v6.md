```
$ diff -u rag_app-v5.py rag_app-v6.py
--- rag_app-v5.py       2025-03-06 12:29:14.024055496 +0800
+++ rag_app-v6.py       2025-03-07 19:00:49.050044627 +0800
@@ -7,14 +7,26 @@
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
+from langchain.document_loaders import PyPDFLoader
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
@@ -29,8 +41,42 @@
 # Additional configuration
 RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
 ES_HOST = os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")
+DATA_PATH = "data"
+BM25_INDEX_FILE = "bm25_index.pkl"
+WATERMARK_FILE = "watermark.txt"
 
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
@@ -39,11 +85,12 @@
 reranker_tokenizer = None
 reranker_model = None
 es_client = None
+bm25_index = None  # Add this
 
 # --- Helper Functions ---
 def load_resources():
-    """Loads resources (embeddings, index, chunks, model, client) and handles errors."""
-    global embedding_model, index, chunks, embeddings, client
+    """Loads resources (embeddings, index, chunks, model, client, bm25_index) and handles errors."""
+    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker
     resource_load_successful = True  # Track if all resources loaded correctly
     messages = []
 
@@ -96,23 +143,37 @@
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
@@ -124,50 +185,12 @@
     except Exception:
         return False
 
-def load_pdf_documents(data_dir='data', batch_size=5):
-    """Enhanced PDF document loading with pdfplumber."""
-    documents = []
-    valid_files = []
-    
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
 def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
     """Chunks the documents into smaller pieces."""
     chunks = []
     for doc in documents:
-        content = doc["content"]
-        filename = doc["filename"]
+        content = doc.page_content # Use page_content from Langchain document
+        filename = os.path.basename(doc.metadata['source']) # Extract filename from metadata
         for i in range(0, len(content), chunk_size - chunk_overlap):
             chunk = content[i:i + chunk_size]
             chunks.append({"filename": filename, "chunk": chunk})
@@ -189,42 +212,91 @@
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
 
 def perform_indexing(progress=gr.Progress()):
-    """Enhanced indexing process with proper progress tracking and validation."""
-    global index, chunks, embeddings, embedding_model
+    """Enhanced indexing process with incremental indexing."""
+    global index, chunks, embeddings, embedding_model, bm25_index, watermark_tracker
 
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
 @lru_cache(maxsize=1000)
@@ -251,41 +323,49 @@
     if reranker_model is None:
         reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
     if es_client is None:
-        es_client = Elasticsearch(ES_HOST)
+        try: # Add try-except for Elasticsearch client initialization
+            es_client = elasticsearch.Elasticsearch(ES_HOST)
+        except Exception as e:
+            print(f"Warning: Could not initialize Elasticsearch client. Hybrid search might be degraded. Error: {e}")
+            es_client = None # Set to None if initialization fails
 
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
+                index="chunks",
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
@@ -310,7 +390,7 @@
     try:
         # Try to get cached response
         cached_response = cached_query_response(query_text)
-        if cached_response:
+        if (cached_response):
             return cached_response
 
         # Initialize components and load resources
```
