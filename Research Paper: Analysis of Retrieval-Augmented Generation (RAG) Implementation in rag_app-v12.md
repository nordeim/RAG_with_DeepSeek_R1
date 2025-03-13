```diff
--- rag_app-v5.py	2025-03-13 10:13:20.180243137 +0800
+++ rag_app-v12.py	2025-03-13 10:06:54.142183134 +0800
@@ -1,4 +1,3 @@
-# rag_app.py (Complete and further improved)
 import os
 import faiss
 import numpy as np
@@ -7,14 +6,34 @@
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
+# Update imports
+from file_utils import validate_file, load_document, ALLOWED_EXTENSIONS
+import logging
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
@@ -28,9 +47,48 @@
 
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
+
+# --- Global Variables (for caching) ---
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
@@ -39,11 +97,47 @@
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
 
@@ -96,231 +190,568 @@
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
-    """Validates if file is a valid PDF."""
-    try:
-        mime = magic.Magic(mime=True)
-        file_type = mime.from_file(filepath)
-        if file_type != 'application/pdf':
-            return False
-        if os.path.getsize(filepath) > 100 * 1024 * 1024:  # 100MB limit
-            return False
-        return True
-    except Exception:
+    """Enhanced file validation."""
+    valid, msg = validate_file(filepath)
+    if not valid:
+        logger.warning(f"File validation failed: {msg}")
         return False
+    return True
 
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
-    """Creates embeddings with batch processing to manage memory."""
+    """Creates embeddings with memory-safe batch processing."""
     all_embeddings = []
-    for i in range(0, len(chunks), batch_size):
-        batch = chunks[i:i + batch_size]
-        batch_embeddings = model.encode([chunk["chunk"] for chunk in batch])
-        all_embeddings.append(batch_embeddings)
+    total_chunks = len(chunks)
+    
+    try:
+        # Estimate memory requirement
+        sample_embedding = model.encode(chunks[0]["chunk"])
+        estimated_memory = (sample_embedding.nbytes * total_chunks) / (1024 * 1024)  # MB
+        
+        if estimated_memory > 1024:  # If estimated memory usage > 1GB
+            batch_size = max(1, min(batch_size, int(batch_size * (1024 / estimated_memory))))
+            print(f"Adjusted batch size to {batch_size} for memory safety")
+        
+        for i in range(0, total_chunks, batch_size):
+            batch = chunks[i:min(i + batch_size, total_chunks)]
+            try:
+                batch_embeddings = model.encode([chunk["chunk"] for chunk in batch], 
+                                             show_progress_bar=False,
+                                             batch_size=8)  # Smaller internal batch size
+                all_embeddings.append(batch_embeddings)
+                
+                # Force garbage collection after each batch
+                import gc
+                gc.collect()
+                
+            except RuntimeError as e:
+                print(f"Memory error in batch {i}-{i+batch_size}, reducing batch size")
+                # Try again with smaller batch
+                for chunk in batch:
+                    single_embedding = model.encode([chunk["chunk"]])
+                    all_embeddings.append(single_embedding)
+                
+    except Exception as e:
+        raise RuntimeError(f"Embedding creation failed: {str(e)}")
+        
     return np.vstack(all_embeddings)
 
 def build_faiss_index(embeddings):
-    """Builds a FAISS index for efficient similarity search."""
-    dimension = embeddings.shape[1]
-    index = faiss.IndexFlatL2(dimension)
-    index.add(np.float32(embeddings))
-    return index
+    """Builds a FAISS index with memory safety checks."""
+    try:
+        dimension = embeddings.shape[1]
+        
+        # Check if we have enough memory
+        import psutil
+        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
+        estimated_memory = (embeddings.nbytes * 2) / (1024 * 1024)  # MB
+        
+        if estimated_memory > available_memory * 0.8:  # If using more than 80% of available memory
+            raise MemoryError(f"Not enough memory to build index. Need {estimated_memory}MB, have {available_memory}MB available")
+        
+        # Use IVFFlat index for better memory efficiency with large datasets
+        if len(embeddings) > 10000:
+            nlist = min(int(np.sqrt(len(embeddings))), 2048)  # number of clusters
+            quantizer = faiss.IndexFlatL2(dimension)
+            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
+            index.train(np.float32(embeddings))
+        else:
+            index = faiss.IndexFlatL2(dimension)
+            
+        index.add(np.float32(embeddings))
+        return index
+        
+    except Exception as e:
+        raise RuntimeError(f"FAISS index creation failed: {str(e)}")
+
+def build_bm25_index(chunks):
+    """Builds BM25 index for keyword search."""
+    tokenized_corpus = [chunk['chunk'].split() for chunk in chunks]
+    return BM25Okapi(tokenized_corpus)
+
+def load_and_preprocess_documents(data_path=DATA_PATH):
+    """Enhanced document loading with multiple file type support."""
+    supported_files = []
+    for ext in ALLOWED_EXTENSIONS:
+        supported_files.extend(Path(data_path).glob(f"*{ext}"))
+    
+    all_docs = []
+    for file_path in supported_files:
+        if not validate_file(str(file_path))[0]:
+            continue
+            
+        documents, error = load_document(str(file_path))
+        if documents:
+            all_docs.extend(documents)
+        else:
+            logger.error(f"Failed to load {file_path}: {error}")
+    
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
 
+def index_documents_to_elasticsearch(es_client, chunks, index_name=ES_INDEX_NAME, progress=gr.Progress()):
+    """Indexes chunks to Elasticsearch."""
+    if not es_client:
+        print("Elasticsearch client not initialized. Skipping Elasticsearch indexing.")
+        return True
+
+    if not create_elasticsearch_index(es_client, index_name):
+        return False
+
+    try:
+        total = len(chunks)
+        for i, chunk in enumerate(chunks):
+            es_client.index(index=index_name, document=chunk, id=str(i))
+            progress(i / total, desc="Indexing to Elasticsearch")
+        print("Elasticsearch indexing complete.")
+        return True
+    except Exception as e:
+        print(f"Error during Elasticsearch indexing: {e}")
+        return False
 
 def perform_indexing(progress=gr.Progress()):
-    """Enhanced indexing process with proper progress tracking and validation."""
-    global index, chunks, embeddings, embedding_model
+    """Memory-safe indexing process."""
+    global index, chunks, embeddings, embedding_model, bm25_index, watermark_tracker, es_client
 
     try:
+        # Add memory check at start
+        import psutil
+        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
+        if available_memory < 2:  # Less than 2GB available
+            return "Error: Insufficient memory available for indexing", None, None, None, None
+
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
+        try:
+            new_embeddings = create_embeddings(new_chunks, embedding_model)
+        except Exception as e:
+            return f"Error during embedding creation: {str(e)}", None, None, None, None
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
+        # Initialize ES components before indexing
+        initialize_components()
+        
+        # Add ES indexing step
+        progress(0.8, desc="Indexing to Elasticsearch...")
+        if es_client and not index_documents_to_elasticsearch(es_client, chunks, progress=progress):
+            print("Warning: Elasticsearch indexing failed, continuing with other indices")
+
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
+        query_cache.cache.clear()  # Invalidate query cache after indexing
+        return "Incremental indexing complete!", "embeddings.npy", "faiss_index.index", "chunks.json", BM25_INDEX_FILE
     except Exception as e:
-        return f"Error during indexing: {e}", None, None, None
+        import traceback
+        print(f"Indexing error: {str(e)}\n{traceback.format_exc()}")
+        return f"Error during indexing: {str(e)}", None, None, None, None
 
 # --- Retrieval and Querying ---
-@lru_cache(maxsize=1000)
-def cached_query_response(query_text: str) -> str:
-    """Caches frequent query responses."""
-    return None  # Just return None to allow main query logic to proceed
-
-def validate_answer(answer: str, context: str) -> bool:
-    """Validates the generated answer against the context using semantic similarity."""
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
     try:
-        answer_emb = embedding_model.encode([answer])
-        context_emb = embedding_model.encode([context])
-        similarity = np.dot(answer_emb[0], context_emb[0]) / (np.linalg.norm(answer_emb[0]) * np.linalg.norm(context_emb[0]))
-        return similarity > 0.85
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
+def validate_answer(answer: str, context: str, query: str) -> bool:
+    """Enhanced answer validation with multiple metrics and dynamic thresholds."""
     try:
-        print(f"Error validating answer: {e}")
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
+            'summary': {'context': 0.5, 'query': 0.1},  # Lowered threshold for summary queries
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
+        
+    except Exception as e:
+        print(f"[DEBUG] Validation error: {str(e)}")
         return False
 
 def initialize_components():
-    """Initialize reranker and Elasticsearch components."""
+    """Enhanced component initialization with better error handling."""
     global reranker_tokenizer, reranker_model, es_client
+    
+    # Initialize reranker components
     if reranker_tokenizer is None:
-        reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
+        try:
+            reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
+        except Exception as e:
+            print(f"Failed to load reranker tokenizer: {e}")
+            
     if reranker_model is None:
-        reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
-    if es_client is None:
-        es_client = Elasticsearch(ES_HOST)
+        try:
+            reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
+        except Exception as e:
+            print(f"Failed to load reranker model: {e}")
+            
+    # Initialize Elasticsearch with retries
+    if es_client is None and ES_PASSWORD:
+        es_config = {
+            "hosts": [ES_HOST],
+            "basic_auth": (ES_USER, ES_PASSWORD),
+            "ssl_show_warn": False,
+            "verify_certs": ES_VERIFY_CERTS,
+            "retry_on_timeout": True,
+            "max_retries": 3
+        }
+        
+        if ES_HOST.startswith("https") and ES_CERT_PATH and os.path.exists(ES_CERT_PATH):
+            es_config["ca_certs"] = ES_CERT_PATH
+            
+        for attempt in range(3):
+            try:
+                es_client = elasticsearch.Elasticsearch(**es_config)
+                if es_client.ping():
+                    print("Successfully connected to Elasticsearch")
+                    break
+            except Exception as e:
+                print(f"Elasticsearch connection attempt {attempt + 1} failed: {e}")
+                if attempt == 2:  # Last attempt
+                    print("Failed to initialize Elasticsearch after 3 attempts")
+                    es_client = None
+                time.sleep(1)  # Wait before retry

 def hybrid_search(query_embedding, query_text, k=5):
-    """Perform hybrid search using FAISS and Elasticsearch."""
-    # Vector search with FAISS
-    D, I = index.search(np.float32(query_embedding), k*2)
-    
-    # Text search with Elasticsearch
-    es_results = es_client.search(
-        index="chunks",
-        body={
-            "query": {
-                "match": {
-                    "content": query_text
+    """Enhanced hybrid search combining vector, keyword, and BM25 rankings."""
+    try:
+        # Vector search with FAISS
+        D, I = index.search(np.float32(query_embedding), k*2)
+        candidates = {}
+        
+        # Add FAISS results
+        for score, idx in zip(D[0], I[0]):
+            if idx < len(chunks):
+                candidates[idx] = {
+                    'chunk': chunks[idx],
+                    'scores': {'faiss': 1 - (score / max(D[0]))}  # Normalize score
                 }
-            },
-            "size": k*2
-        }
-    )
-    
-    # Combine and rerank results
-    candidates = []
-    seen = set()
-    
-    for idx in I[0]:
-        if idx not in seen:
-            candidates.append(chunks[idx])
-            seen.add(idx)
-            
-    for hit in es_results['hits']['hits']:
-        chunk_id = hit['_source']['chunk_id']
-        if chunk_id not in seen:
-            candidates.append(chunks[chunk_id])
-            seen.add(chunk_id)
-    
-    return rerank_results(query_text, candidates)[:k]
-
-def rerank_results(query, candidates):
-    """Rerank results using the reranker model."""
-    inputs = reranker_tokenizer(
-        [[query, c["chunk"]] for c in candidates],
-        padding=True,
-        truncation=True,
-        return_tensors="pt"
-    )
-    with torch.no_grad():
-        scores = reranker_model(**inputs).logits.squeeze(-1)
-    
-    ranked_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
-    return [pair[0] for pair in ranked_pairs]
+
+        # Add BM25 results if available
+        if bm25_index:
+            try:
+                bm25_scores = bm25_index.get_scores(query_text.split())
+                top_bm25_indices = np.argsort(bm25_scores)[-k*2:][::-1]
+                for idx in top_bm25_indices:
+                    if idx in candidates:
+                        candidates[idx]['scores']['bm25'] = bm25_scores[idx] / max(bm25_scores)
+                    else:
+                        candidates[idx] = {
+                            'chunk': chunks[idx],
+                            'scores': {'bm25': bm25_scores[idx] / max(bm25_scores)}
+                        }
+            except Exception as e:
+                print(f"BM25 search error: {e}")
+
+        # Add Elasticsearch results if available
+        if es_client:
+            try:
+                es_results = es_client.search(
+                    index=ES_INDEX_NAME,
+                    body={
+                        "query": {
+                            "multi_match": {
+                                "query": query_text,
+                                "fields": ["chunk^2", "filename"],
+                                "fuzziness": "AUTO"
+                            }
+                        },
+                        "size": k*2
+                    },
+                    request_timeout=30
+                )
+                
+                for hit in es_results['hits']['hits']:
+                    idx = int(hit['_id'])
+                    if idx in candidates:
+                        candidates[idx]['scores']['es'] = hit['_score'] / es_results['hits']['max_score']
+                    else:
+                        candidates[idx] = {
+                            'chunk': chunks[idx],
+                            'scores': {'es': hit['_score'] / es_results['hits']['max_score']}
+                        }
+            except Exception as e:
+                print(f"Elasticsearch search error: {e}")
+
+        # Combine scores using weighted average
+        final_candidates = []
+        weights = {'faiss': 0.4, 'bm25': 0.3, 'es': 0.3}
+        
+        for idx, data in candidates.items():
+            total_score = 0
+            weight_sum = 0
+            for method, weight in weights.items():
+                if method in data['scores']:
+                    total_score += data['scores'][method] * weight
+                    weight_sum += weight
+            if weight_sum > 0:
+                final_candidates.append((data['chunk'], total_score / weight_sum))
+
+        # Sort and return top k results
+        return [c[0] for c in sorted(final_candidates, key=lambda x: x[1], reverse=True)[:k]]
+        
+    except Exception as e:
+        print(f"Hybrid search error: {e}")
+        # Fallback to basic FAISS search
+        D, I = index.search(np.float32(query_embedding), k)
+        return [chunks[i] for i in I[0]]

 def query_rag_system(query_text: str, progress=gr.Progress()) -> str:
-    """Enhanced query processing with validation and error handling."""
+    """Enhanced query processing with improved prompting and validation."""
     if not query_text or not query_text.strip():
         return "Error: Empty query"
     
     try:
-        # Try to get cached response
+        print(f"\n[DEBUG {datetime.now().isoformat()}] Processing query: {query_text}")
+        
         cached_response = cached_query_response(query_text)
         if cached_response:
+            print("[DEBUG] Retrieved from cache")
             return cached_response
 
         # Initialize components and load resources
@@ -621,17 +1324,17 @@
         
         progress(0.4, desc="Searching index...")
         try:
-            relevant_chunks = hybrid_search(query_embedding, query_text)
+            relevant_chunks = hybrid_search(query_embedding, query_text, k=5) # Explicitly set k=5
             if not relevant_chunks:
                 print("[DEBUG] No relevant chunks retrieved.")
                 return "No relevant information found for your query."
             
             print(f"[DEBUG] Retrieved {len(relevant_chunks)} relevant chunks")
-            
+
             # Enhanced context creation with structure
             context = enhance_context(relevant_chunks)
             print(f"[DEBUG] Structured context created ({len(context)} chars)")
-            
+
             # Determine query type and get appropriate prompt
             query_type = get_query_type(query_text)
             prompt_template = get_prompt_template(query_type)
@@ -674,7 +1377,6 @@
         return f"Error: {str(e)}"
     except Exception as e:
         return f"Unexpected error: {str(e)}"
-
 # --- Gradio Interface ---
 with gr.Blocks(title="PDF Q&A with DeepSeek-R1", theme=gr.themes.Soft()) as demo:
     gr.Markdown(
@@ -785,3 +1487,121 @@
         share=False,
         max_threads=4
     )
+```

### **1. Code Review of `rag_app-v12.py` and `file_utils.py`**:

**General Observations from `diff` and Code:**

*   **Modularization and File Utilities:**  `rag_app-v12.py` now utilizes `file_utils.py` for file validation and document loading, improving code organization and reusability. This is a good step towards better software engineering practices.
*   **Incremental Indexing:** The `WatermarkTracker` class is a key addition for incremental indexing. It tracks document modification times to avoid re-indexing unchanged files, significantly improving efficiency.
*   **Configurable Chunking:**  The `ChunkingConfig` dataclass and Gradio settings tab allow users to adjust chunk size and overlap, offering flexibility for different document types and query needs.
*   **Enhanced Caching:** `TTLCache` with Time-To-Live is implemented for query caching, a more sophisticated approach than simple memoization, ensuring cache freshness.
*   **Memory Safety:**  Improvements in `create_embeddings` and `build_faiss_index` include memory usage estimation, batch size adjustment, and garbage collection to prevent out-of-memory errors.
*   **Hybrid Search (FAISS, BM25, Elasticsearch):**  `rag_app-v12.py` incorporates a more comprehensive hybrid search strategy, combining FAISS (vector search), BM25 (keyword search), and Elasticsearch (text search). Weights are used to combine scores from different methods.
*   **Improved Answer Validation:**  `validate_answer` is enhanced with dynamic thresholds based on query type and compares answer embeddings to both context and query embeddings for better validation.
*   **Robust Elasticsearch Integration:**  Elasticsearch integration is more robust with authentication, SSL/TLS, retry logic, and multi-field search.
*   **Gradio UI Improvements:**  A settings tab is added to the Gradio interface for configuring batch size, cache size, and chunking parameters. BM25 index saving is also added to the output files in Gradio.
*   **Logging:** Basic logging using `logging` module is introduced in `file_utils.py`, improving debugging and monitoring.
*   **Error Handling:**  Error handling is improved across various functions, including resource loading, indexing, and querying, making the system more robust.

**Specific Code Points and Potential Issues:**

*   **Global Variables:** The code still relies heavily on global variables. While `TTLCache` uses a lock, other global variables are still mutable. This can be problematic in more complex scenarios (as highlighted in the research paper for v10).
*   **`validate_pdf_file` Redundancy:** `rag_app-v12.py` still has `validate_pdf_file`, but it now calls the more generic `validate_file` from `file_utils.py`. It might be better to remove `validate_pdf_file` from `rag_app-v12.py` entirely and just use `validate_file` directly where needed.
*   **Elasticsearch Initialization:** The Elasticsearch client initialization in `initialize_components` includes retry logic and SSL/TLS support, which is good. However, it only initializes if `ES_PASSWORD` is set. It might be better to initialize even without password and handle password-less connections if the Elasticsearch setup allows.
*   **Hybrid Search Fallback:** The `hybrid_search` function has a fallback to basic FAISS search in case of errors, which is good for resilience. However, it might be helpful to log the specific error that caused the fallback for debugging purposes.
*   **Context Enhancement:** The `enhance_context` function creates a structured context by document and section, which is a good improvement for providing more organized context to the LLM.
*   **Query Type Detection and Prompting:** The introduction of `get_query_type` and `get_prompt_template` for dynamic prompting based on query type is a significant improvement for tailoring the prompt to the expected answer format (summary, factual, general).
*   **Answer Validation Metrics:**  The enhanced `validate_answer` function with multiple similarity metrics (answer-context, answer-query, context-query) and dynamic thresholds is a more robust validation approach. The debug logging in validation is very helpful.
*   **Chunk ID Generation:** Chunk ID generation `f"{filename}_{doc_id}_{chunk_idx}"` looks reasonable for uniqueness.
*   **Memory Management in `create_embeddings`:** The memory-safe batch processing in `create_embeddings` with dynamic batch size adjustment and garbage collection is a crucial improvement for scalability and stability.
*   **FAISS Index Type:** Using `IndexIVFFlat` for larger datasets in `build_faiss_index` is a good choice for efficiency, demonstrating awareness of scalability.
*   **BM25 Indexing:**  The inclusion of BM25 indexing and hybrid search is a valuable addition for improving retrieval recall, complementing semantic search.
*   **File Type Support:** The code now uses `file_utils.py` and `ALLOWED_EXTENSIONS` to handle multiple file types, making the RAG system more versatile.

**Web Search Insights and Best Practices:**

Based on web searches about RAG best practices and techniques used in `rag_app-v12.py`, here are some findings:

*   **Hybrid Search:** Combining vector search with lexical search (like BM25) is a widely recommended best practice in RAG to improve retrieval recall and precision. The implementation in `rag_app-v12.py` with weighted averaging of scores is a standard and effective approach. Elasticsearch adds another dimension for structured search and filtering, further enhancing hybrid search. (Sources: Various blog posts and articles on RAG best practices on Pinecone, Cohere, and LangChain websites).
*   **Incremental Indexing:**  For systems dealing with frequently updated documents, incremental indexing is crucial for efficiency. The `WatermarkTracker` implementation in `rag_app-v12.py` aligns with best practices for tracking document changes and updating indices selectively. (Source: Articles on MLOps and data pipelines for RAG systems).
*   **Chunking Strategies:** Configurable chunking is indeed important as different document types and query styles benefit from different chunk sizes and overlaps. The `ChunkingConfig` and dynamic setting in Gradio are valuable features.  (Source: LangChain documentation and tutorials on document loading and chunking).
*   **Caching:** TTL-based caching is a good practice for RAG applications to balance performance and data freshness. `TTLCache` implementation is appropriate. (Source: Articles on building scalable RAG applications).
*   **Answer Validation:**  Semantic similarity-based answer validation is a common technique to filter out irrelevant or poorly generated answers. Comparing answer embedding with both context and query embeddings, as done in `rag_app-v12.py`, is a more comprehensive validation approach. Dynamic thresholds based on query type adds sophistication. (Source: Research papers and blog posts on improving RAG answer quality).
*   **Memory Optimization:** Memory management during embedding and indexing is critical for handling large datasets. The memory-safe batch processing and FAISS index type choice in `rag_app-v12.py` are aligned with best practices for building scalable RAG systems. (Source: Articles on scaling vector databases and embedding pipelines).
*   **Reranking (Note: Reranking is not actively used in v12):**  While `rag_app-v12.py` includes reranker components initialization, it doesn't seem to be actively used in the hybrid search pipeline anymore (as noted in the research paper analysis of v10). Reranking is a powerful technique to further refine search results by using a more computationally intensive model to re-score and re-order the top candidates from initial retrieval stages. Re-integrating reranking could be a potential improvement. (Source: Research papers on semantic reranking and query understanding).
*   **File Type Handling:** Supporting multiple file types is important for a versatile RAG system. Using libraries like Langchain's document loaders and `magic` for file type detection (as in `file_utils.py`) is a standard approach. (Source: Langchain documentation on document loaders).

**Overall, `rag_app-v12.py` shows significant improvements over `rag_app-v5.py` and incorporates many RAG best practices. The code is more modular, robust, efficient, and user-friendly.**


### **2. Research Paper Review and Update:**

Now, I will update the research paper "Research Paper: Analysis of Retrieval-Augmented Generation (RAG) Implementation in rag_app-v12.py" based on the code review findings and web search insights. I will specifically focus on updating the sections related to improvements, identified issues, and potential improvements to reflect the current state of `rag_app-v12.py`.

I will go through the research paper section by section and update it based on my analysis.

**(Starting to update the Research Paper content below)**

```markdown
# Research Paper: Analysis of Retrieval-Augmented Generation (RAG) Implementation in rag_app-v12.py

## 1. Abstract

This research paper provides a detailed analysis of the Python code `rag_app-v12.py`, which implements a Retrieval-Augmented Generation (RAG) system for question answering over various document types including PDF documents. This version represents a further evolution from `rag_app-v10.py`, building upon its advanced techniques to further enhance performance, efficiency, and robustness. The paper delves into the key components of the system, including document loading and preprocessing (now extended to multiple file types), embedding generation, indexing strategies (FAISS, BM25, Elasticsearch), hybrid search methodologies, response generation, and answer validation. Furthermore, it evaluates the significant improvements implemented in `rag_app-v12.py` compared to its predecessor versions, based on code review and web research on RAG best practices. The analysis highlights critical enhancements such as modular file handling utilities, incremental indexing, configurable chunking, memory-safe operations, and a sophisticated hybrid search approach integrating vector, keyword (BM25), and text-based (Elasticsearch) retrieval, alongside improvements in prompting and answer validation. The paper also identifies remaining potential future improvements to further optimize the system and address identified issues, such as dependency on global variables and re-integration of reranking.

## 2. Introduction

Retrieval-Augmented Generation (RAG) continues to be a vital paradigm for question answering systems requiring access to and reasoning over external knowledge sources. The `rag_app-v12.py` script is a practical and evolved implementation of a RAG system designed for querying diverse document formats. This paper aims to dissect the architecture and functionalities of `rag_app-v12.py`, emphasizing its strengths, advancements over previous versions (`rag_app-v5.py` and `rag_app-v10.py`), and areas for continued development. The progression from `rag_app-v5.py` to `rag_app-v12.py` demonstrates a clear trajectory towards a more feature-rich, efficient, and production-capable RAG application. This analysis will encompass the application's core modules – indexing, retrieval, generation, and validation – providing insights into the design choices, architectural improvements, and their impact on the overall system effectiveness and user experience. The incorporation of modular utilities and advanced RAG techniques in `rag_app-v12.py` marks a significant step towards a robust and versatile RAG system.

## 3. System Architecture and Components

The `rag_app-v12.py` script maintains a modular structure, with each module dedicated to a specific stage of the RAG pipeline. The architecture is refined to enhance efficiency, robustness, and flexibility in document processing, indexing, and querying, incorporating several best practices identified through web research and iterative development.

### 3.1. Configuration and Environment Setup

The script's configuration section remains largely consistent with `rag_app-v10.py`, emphasizing secure environment variable management using `dotenv` for API keys and sensitive parameters.

```python
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
import pdfplumber
import time
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
import elasticsearch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datetime import datetime
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from functools import wraps
from typing import Dict, Optional
from dataclasses import dataclass
from threading import Lock

magic_available = True
try:
    import magic
except ImportError:
    print("Warning: libmagic not found. File type validation will be less robust.")
    magic_available = False

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
SAMBANOVA_API_BASE_URL = os.environ.get("SAMBANOVA_API_BASE_URL")
if not all([MODEL_NAME, SAMBANOVA_API_KEY, SAMBANOVA_API_BASE_URL]):
    raise ValueError("Missing required environment variables: MODEL_NAME, SAMBANOVA_API_KEY, or SAMBANOVA_API_BASE_URL")

RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
ES_HOST = os.environ.get("ELASTICSEARCH_HOST", "https://localhost:9200")
ES_USER = os.environ.get("ELASTICSEARCH_USER", "elastic")
ES_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD")
ES_CERT_PATH = os.environ.get("ELASTICSEARCH_CERT_PATH", "http_cert.pem")
ES_VERIFY_CERTS = os.environ.get("ELASTICSEARCH_VERIFY_CERTS", "true").lower() == "true"
DATA_PATH = "data"
BM25_INDEX_FILE = "bm25_index.pkl"
WATERMARK_FILE = "watermark.txt"
ES_INDEX_NAME = "chunks"
```

Key configurations such as model names, API endpoints, and Elasticsearch connection details are maintained as environment variables. The constants `BM25_INDEX_FILE`, `WATERMARK_FILE`, and `ES_INDEX_NAME` are consistently used for file paths and index naming, improving code readability and maintainability. The check for `libmagic` and the warning message if not found is preserved, providing user awareness of potential file validation limitations.

### 3.2. Modular File Handling with `file_utils.py`

A significant architectural improvement in `rag_app-v12.py` is the introduction of the separate `file_utils.py` module. This utility module encapsulates file-related functionalities, including file validation (`validate_file`) and document loading (`load_document`).

```python
# file_utils.py
import os
import magic
from pathlib import Path
from typing import Tuple, List, Optional
import logging
from datetime import datetime
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader,
    UnstructuredFileLoader
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File type configurations
ALLOWED_EXTENSIONS = {
    '.txt', '.md', '.pdf', '.doc', '.docx',
    '.xls', '.xlsx', '.ppt', '.pptx'
}

def validate_file(filepath: str) -> Tuple[bool, str]:
    """Validate file existence and type."""
    if not os.path.exists(filepath):
        return False, "File does not exist."

    file_ext = Path(filepath).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type. Supported: {', '.join(ALLOWED_EXTENSIONS)}"

    try:
        if magic.Magic(mime=True).from_file(filepath) == 'text/plain':
            return True, ""
    except:
        pass

    return True, ""

def load_document(filepath: str) -> Tuple[Optional[List], str]:
    """Load document with appropriate loader based on file type."""
    try:
        file_ext = Path(filepath).suffix.lower()

        # Select appropriate loader
        if file_ext == '.pdf':
            loader = PyPDFLoader(filepath)
        elif file_ext in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(filepath)
        elif file_ext in ['.ppt', '.pptx']:
            loader = UnstructuredPowerPointLoader(filepath)
        elif file_ext in ['.xls', '.xlsx']:
            loader = UnstructuredExcelLoader(filepath)
        else:
            loader = UnstructuredFileLoader(filepath)

        # Load document and add metadata
        documents = loader.load()
        file_mod_time = datetime.fromtimestamp(Path(filepath).stat().st_mtime)

        for doc in documents:
            doc.metadata['last_modified'] = file_mod_time
            doc.metadata['source'] = str(filepath)

        return documents, ""

    except Exception as e:
        error_msg = f"Error loading {filepath}: {str(e)}"
        logger.error(error_msg)
        return None, error_msg
```

This modular approach enhances code organization, reusability, and maintainability. `file_utils.py` defines `ALLOWED_EXTENSIONS` for supported file types and utilizes `langchain_community.document_loaders` for loading various document formats. The `validate_file` function checks for file existence and supported extensions, while `load_document` intelligently selects the appropriate Langchain loader based on the file extension. This separation of concerns makes the codebase cleaner and easier to extend with support for more file types in the future.

### 3.3. Global Variables and Caching Mechanisms

The system retains the use of global variables for caching resources to optimize performance by avoiding redundant loading of models, indices, and chunks.

```python
class WatermarkTracker:
    """Tracks document modification times for incremental indexing"""
    # ... (WatermarkTracker class definition) ...
watermark_tracker = WatermarkTracker(WATERMARK_FILE)

embedding_model = None
index = None
chunks = None
embeddings = None
client = None
reranker_tokenizer = None
reranker_model = None
es_client = None
bm25_index = None

@dataclass
class CacheEntry:
    # ... (CacheEntry dataclass definition) ...

class TTLCache:
    """Time-To-Live Cache Implementation"""
    # ... (TTLCache class definition) ...
query_cache = TTLCache(maxsize=1000, ttl=3600)

def cached_query_response(query_text: str) -> Optional[str]:
    """Enhanced query caching with TTL."""
    return query_cache.get(query_text)
```

The `WatermarkTracker` class and `TTLCache` class, along with global variables like `embedding_model`, `index`, `chunks`, `embeddings`, `client`, `bm25_index`, `reranker_tokenizer`, `reranker_model`, and `es_client` are maintained.  As noted in the identified issues section, the extensive use of global variables, while facilitating caching, introduces potential maintainability and scalability concerns, especially in more complex or multithreaded applications. However, the `TTLCache` implementation now includes a lock for thread safety, a significant improvement.

### 3.4. Resource Loading and Cache Clearing

The `load_resources` and `clear_cache` functions are updated to manage the expanded set of cached resources, including the BM25 index and watermark tracker.

```python
def load_resources():
    """Loads resources (embeddings, index, chunks, model, client, bm25_index) and handles errors."""
    # ... (load_resources function definition) ...

def clear_cache():
    """Clears the cached resources."""
    # ... (clear_cache function definition) ...
```

`load_resources` now handles loading the BM25 index from `BM25_INDEX_FILE` and initializes the `watermark_tracker`. Error handling is consistently applied to each resource loading step, providing informative messages if any resource fails to load. `clear_cache` now resets the `bm25_index` and re-instantiates the `watermark_tracker`, ensuring a complete cache reset.

### 3.5. Indexing Functions

The indexing process in `rag_app-v12.py` builds upon the enhancements of `rag_app-v10.py`, maintaining robust document validation, configurable chunking, memory-safe embedding generation, and multiple indexing strategies (FAISS, BM25, Elasticsearch).

#### 3.5.1. Document Loading and Validation

The `validate_pdf_file` function is simplified and now leverages the more general `validate_file` function from `file_utils.py`, demonstrating code reuse and modularity.

```python
def validate_pdf_file(filepath: str) -> bool:
    """Enhanced file validation."""
    valid, msg = validate_file(filepath)
    if not valid:
        logger.warning(f"File validation failed: {msg}")
        return False
    return True
```

This change streamlines the validation process and ensures consistency with the file validation logic defined in `file_utils.py`. The function now serves primarily as a wrapper to integrate the generic validation into the PDF-specific context if needed, although it could potentially be removed entirely in favor of direct use of `validate_file`.

#### 3.5.2. Document Chunking

The `ChunkingConfig` dataclass and `chunk_documents` function remain, providing configurable chunking parameters.

```python
@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    # ... (ChunkingConfig dataclass definition) ...
chunking_config = ChunkingConfig()

def chunk_documents(documents, chunk_size=None, chunk_overlap=None):
    """Enhanced chunking with version awareness and configurable parameters."""
    # ... (chunk_documents function definition) ...
```

The `ChunkingConfig` dataclass and `chunk_documents` function are preserved from `rag_app-v10.py`. The `ChunkingConfig` dataclass encapsulates `chunk_size` and `chunk_overlap`, and the `chunk_documents` function now accepts these as optional parameters, defaulting to the configuration settings. This maintains the flexibility to adjust chunking strategies based on user needs and document characteristics.

#### 3.5.3. Embedding Generation

The `create_embeddings` function retains its memory-safe batch processing logic.

```python
def create_embeddings(chunks: List[Dict], model, batch_size=32):
    """Creates embeddings with memory-safe batch processing."""
    # ... (create_embeddings function definition) ...
```

The memory-safe batch processing in `create_embeddings` is crucial for handling large document sets and remains a key feature. The function dynamically adjusts batch size based on estimated memory usage and includes garbage collection to manage memory effectively, preventing out-of-memory errors.

#### 3.5.4. Index Building (FAISS and BM25)

The `build_faiss_index` and `build_bm25_index` functions are also maintained.

```python
def build_faiss_index(embeddings):
    """Builds a FAISS index with memory safety checks."""
    # ... (build_faiss_index function definition) ...

def build_bm25_index(chunks):
    """Builds BM25 index for keyword search."""
    # ... (build_bm25_index function definition) ...
```

`build_faiss_index` still incorporates memory safety checks and uses `IndexIVFFlat` for larger datasets. `build_bm25_index` creates a BM25 index using the `rank_bm25` library for keyword-based retrieval. These functions provide the foundation for vector and lexical search capabilities in the RAG system.

#### 3.5.5. Document Loading and Preprocessing

The `load_and_preprocess_documents` function is updated to leverage the `load_document` function from `file_utils.py` to support multiple file types beyond just PDFs, as defined in `ALLOWED_EXTENSIONS`.

```python
def load_and_preprocess_documents(data_path=DATA_PATH):
    """Enhanced document loading with multiple file type support."""
    # ... (load_and_preprocess_documents function definition) ...
```

This enhancement significantly expands the system's versatility by enabling it to process a wider range of document formats, including text files, markdown, Word documents, PowerPoint presentations, and Excel spreadsheets. The function iterates through supported file extensions defined in `ALLOWED_EXTENSIONS` from `file_utils.py`, loads each document using `load_document`, and aggregates them for further processing.

#### 3.5.6. Elasticsearch Indexing

The `create_elasticsearch_index` and `index_documents_to_elasticsearch` functions remain consistent in their functionality.

```python
def create_elasticsearch_index(es_client, index_name=ES_INDEX_NAME):
    """Creates Elasticsearch index with mapping if it doesn't exist."""
    # ... (create_elasticsearch_index function definition) ...

def index_documents_to_elasticsearch(es_client, chunks, index_name=ES_INDEX_NAME, progress=gr.Progress()):
    """Indexes chunks to Elasticsearch."""
    # ... (index_documents_to_elasticsearch function definition) ...
```

These functions handle the creation of the Elasticsearch index with a predefined mapping and the indexing of document chunks to Elasticsearch, respectively. Progress tracking during indexing is maintained, providing user feedback during this potentially time-consuming process.

#### 3.5.7. `perform_indexing` Orchestration

The `perform_indexing` function orchestrates the entire indexing process and is updated to accommodate the new multi-file type support and BM25 indexing.

```python
def perform_indexing(progress=gr.Progress()):
    """Memory-safe indexing process."""
    # ... (perform_indexing function definition) ...
```

`perform_indexing` is the central function for the indexing pipeline. It now loads documents of various types using `load_and_preprocess_documents`, chunks them, generates embeddings, builds FAISS and BM25 indices, and indexes chunks to Elasticsearch. It retains memory safety checks at the beginning and the incremental indexing logic using `WatermarkTracker`. It also saves the BM25 index to `BM25_INDEX_FILE` in addition to other index files and clears the query cache after indexing to ensure that subsequent queries use the updated index.

### 3.6. Retrieval and Querying Functions

The retrieval and querying module in `rag_app-v12.py` is further refined with enhancements in context handling, prompting, answer validation, and hybrid search strategy.

#### 3.6.1. Enhanced Context Handling

The `enhance_context` function is introduced to create a more structured and informative context for the language model by organizing chunks by document and section.

```python
def enhance_context(chunks: List[Dict]) -> str:
    """Creates a structured context with metadata and hierarchy."""
    # ... (enhance_context function definition) ...
```

This function takes a list of document chunks and groups them by filename. Within each document, chunks are sorted by their original chunk index. The function then constructs a context string that clearly delineates document boundaries and section numbers, providing a more organized and hierarchical context to the language model, which can potentially improve response quality and coherence.

#### 3.6.2. Dynamic Prompting based on Query Type

The functions `get_query_type` and `get_prompt_template` are introduced to enable dynamic prompt generation based on the detected query type (summary, factual, or general).

```python
def get_query_type(query: str) -> str:
    """Determines query type for dynamic prompt and validation adjustment."""
    # ... (get_query_type function definition) ...

def get_prompt_template(query_type: str) -> str:
    """Returns appropriate prompt template based on query type."""
    # ... (get_prompt_template function definition) ...
```

`get_query_type` analyzes the query to determine its type based on keywords, categorizing it as 'summary', 'factual', or 'general'. `get_prompt_template` then selects a specific prompt template tailored to each query type. For example, summary queries get a prompt asking for bullet-point summaries, while factual queries are prompted to cite sources and explicitly state if the answer is not in the context. This dynamic prompting strategy allows for more targeted and effective communication with the language model, potentially leading to improved response quality and adherence to instructions.

#### 3.6.3. Improved Answer Validation

The `validate_answer` function is significantly enhanced to incorporate multiple semantic similarity metrics and dynamic validation thresholds based on the query type.

```python
def compute_semantic_similarity(text1: str, text2: str, model) -> float:
    """Compute semantic similarity with error handling and normalization."""
    # ... (compute_semantic_similarity function definition) ...

def validate_answer(answer: str, context: str, query: str) -> bool:
    """Enhanced answer validation with multiple metrics and dynamic thresholds."""
    # ... (validate_answer function definition) ...
```

`compute_semantic_similarity` is a helper function to calculate cosine similarity between text embeddings with error handling. `validate_answer` now calculates three similarity metrics: answer-context, answer-query, and context-query. It then uses dynamic thresholds for answer-context and answer-query similarity based on the query type ('summary', 'factual', 'general'). Summary queries have lower thresholds, while factual queries have stricter thresholds. This dynamic thresholding adapts the validation stringency based on the expected nature and complexity of the query and answer. Detailed debug logging is added to the validation process to aid in understanding validation outcomes.

#### 3.6.4. Hybrid Search Implementation

The `hybrid_search` function is refined to provide a more robust and weighted hybrid search strategy, combining vector search (FAISS), keyword search (BM25), and Elasticsearch.

```python
def hybrid_search(query_embedding, query_text, k=5):
    """Enhanced hybrid search combining vector, keyword, and BM25 rankings."""
    # ... (hybrid_search function definition) ...
```

`hybrid_search` now integrates results from FAISS, BM25, and Elasticsearch (if available). It normalizes scores from each search method and combines them using a weighted average. Weights are assigned to each method (FAISS: 0.4, BM25: 0.3, ES: 0.3) to control their contribution to the final ranking. The function retrieves a larger set of candidates from each method (k*2) and then combines and reranks them to return the top-k results. Fallback to basic FAISS search is maintained in case of errors in hybrid search components, ensuring system resilience.

#### 3.6.5. Query Orchestration in `query_rag_system`

The `query_rag_system` function orchestrates the entire query process, incorporating the enhanced context handling, dynamic prompting, and improved validation.

```python
def query_rag_system(query_text: str, progress=gr.Progress()) -> str:
    """Enhanced query processing with improved prompting and validation."""
    # ... (query_rag_system function definition) ...
```

`query_rag_system` manages the query pipeline. It first checks the cache, then loads resources, encodes the query, performs hybrid search, enhances the context using `enhance_context`, generates a response using the OpenAI API with a prompt selected by `get_prompt_template` based on `get_query_type`, validates the answer using the enhanced `validate_answer`, and caches the validated response before returning it. Detailed debug logging is added at various stages of the query processing to aid in monitoring and debugging.

### 3.7. Gradio Interface and Chunking Configuration Update

The Gradio interface is enhanced with a settings tab that includes configurable chunking parameters, and the `update_chunking_config` function allows users to dynamically update these parameters.

```python
def update_chunking_config(size: int, overlap: int) -> str:
    """Update chunking configuration with validation."""
    # ... (update_chunking_config function definition) ...

with gr.Blocks(title="PDF Q&A with DeepSeek-R1", theme=gr.themes.Soft()) as demo:
    # ... (Gradio interface definition with settings tab including chunking parameters) ...
```

The `update_chunking_config` function validates and updates the global `chunking_config` object with new chunk size and overlap values provided through the Gradio interface. The Gradio interface's settings tab now includes sliders for "Chunk Size" and "Chunk Overlap," allowing users to adjust these parameters and dynamically reconfigure the chunking process without code modification. This adds a significant degree of user control and customization to the RAG application.

## 4. Improvements over `rag_app-v5.py`

`rag_app-v12.py` demonstrates substantial improvements over `rag_app-v5.py`, encompassing architectural refinements, enhanced functionalities, and increased robustness and efficiency.

### 4.1. Modular File Handling and Multi-Format Support

The introduction of `file_utils.py` and the `load_and_preprocess_documents` function's integration of `load_document` significantly modularizes file handling and expands document type support beyond PDFs to include text files, markdown, Word documents, PowerPoint presentations, and Excel spreadsheets. This enhances the system's versatility and maintainability.

### 4.2. Incremental Indexing with Watermark Tracking

The `WatermarkTracker` class enables efficient incremental indexing. The system now intelligently re-indexes only new or modified documents, drastically reducing processing time for frequently updated document collections and ensuring indices are up-to-date.

### 4.3. Configurable Chunking via Gradio Settings

The integration of `ChunkingConfig` and the Gradio settings tab allows users to dynamically adjust chunk size and overlap. This provides crucial flexibility to optimize chunking strategies for different document types and query patterns, improving retrieval performance.

### 4.4. Enhanced Caching with TTL and Thread Safety

The `TTLCache` class implements a more sophisticated query caching mechanism with Time-To-Live (TTL), ensuring that cached responses are automatically invalidated after a defined period, preventing stale information. The inclusion of a lock in `TTLCache` provides thread safety, making the caching mechanism more robust for concurrent query handling.

### 4.5. Memory Safety and Scalability Enhancements

Memory safety is significantly improved through dynamic batch size adjustment and garbage collection in `create_embeddings`, and memory checks in `build_faiss_index`. The use of `IndexIVFFlat` for larger FAISS indices further contributes to scalability, allowing the system to handle larger document collections more efficiently.

### 4.6. Comprehensive Hybrid Search Strategy

The hybrid search strategy is significantly enhanced, combining vector search (FAISS), keyword search (BM25), and Elasticsearch. Weighted averaging of scores from these diverse methods allows for a more comprehensive and effective retrieval process, leveraging the strengths of each technique.

### 4.7. Dynamic Prompting and Structured Context

The introduction of dynamic prompting based on query type (`get_query_type`, `get_prompt_template`) and structured context creation (`enhance_context`) enables more targeted and effective communication with the language model, potentially improving response quality, format adherence, and contextual relevance.

### 4.8. Robust and Dynamic Answer Validation

The `validate_answer` function is significantly enhanced with multiple semantic similarity metrics and dynamic thresholds based on query type. This provides a more robust and adaptive answer validation process, improving the reliability and quality of generated responses.

### 4.9. Improved Elasticsearch Integration

Elasticsearch integration is made more robust with features like authentication, SSL/TLS support, retry logic, and multi-field search, ensuring secure and reliable interaction with Elasticsearch.

### 4.10. Enhanced User Experience via Gradio Settings Tab

The Gradio interface is significantly improved with a settings tab that allows users to configure batch size, cache size, and, crucially, chunking parameters. This provides users with greater control and customization options without requiring code modification, enhancing the user experience and accessibility of the RAG application. The inclusion of BM25 index file output in Gradio also improves transparency.

## 5. Identified Issues and Potential Improvements

Despite the significant advancements in `rag_app-v12.py`, certain issues persist, and further improvements are possible to enhance the system's architecture, scalability, and performance.

### 5.1. Dependency on Global Variables (Persistent Issue)

The reliance on global variables for caching resources, while convenient, remains a potential architectural issue, as highlighted in the analysis of `rag_app-v10.py`. Although `TTLCache` is now thread-safe, the continued use of globally mutable variables like `embedding_model`, `index`, `chunks`, `embeddings`, `client`, `bm25_index`, `reranker_tokenizer`, `reranker_model`, `es_client`, and `watermark_tracker` can still complicate code maintainability, testability, and reasoning, especially in larger, more complex applications or concurrent environments beyond the Gradio interface's queuing mechanism.

**Potential Improvement:** Refactor the codebase to minimize global variable usage. Consider encapsulating resources within a class (like a RAG system class) or employing dependency injection to manage resource dependencies more explicitly. This would promote modularity, improve testability, and enhance the overall maintainability and scalability of the system.

### 5.2. Reranking Functionality Not Utilized (Persistent Issue)

As observed in `rag_app-v10.py`, the reranking functionality, despite the initialization of reranker model and tokenizer, is still not actively integrated into the hybrid search pipeline in `rag_app-v12.py`. The potential benefits of reranking for improving result relevance and precision are therefore not being realized.

**Potential Improvement:** Re-integrate the reranking functionality into the `hybrid_search` pipeline. After the initial hybrid retrieval from FAISS, BM25, and Elasticsearch, apply the reranker model to re-score and re-order the top candidates. This could significantly enhance the quality and relevance of the retrieved chunks, especially for complex or nuanced queries.

### 5.3. Granularity of Error Handling and Logging

While error handling and logging have been improved, further granularity could be beneficial. For instance, in `hybrid_search`, while fallback to FAISS search is implemented, logging specific error types encountered during BM25 or Elasticsearch searches could provide more actionable debugging information. Similarly, more detailed logging within resource loading and indexing processes could aid in diagnosing issues more effectively.

**Potential Improvement:** Enhance error handling to be more specific and context-aware. Implement more granular logging throughout the system, capturing specific exceptions and relevant context information (e.g., query text, filename, chunk ID). This would significantly improve debugging, monitoring, and system maintenance. Consider using structured logging to facilitate analysis and monitoring of system behavior.

### 5.4. Scalability Limitations of FAISS Index (Persistent Issue)

Although `IndexIVFFlat` improves FAISS index efficiency, the index is still built and loaded into memory on a single machine. For extremely large document collections, this remains a potential scalability bottleneck.

**Potential Improvement:** Investigate distributed FAISS indexing solutions or consider migrating to a dedicated vector database that offers distributed indexing and querying capabilities. Cloud-based vector databases or distributed FAISS implementations could enable scaling to much larger datasets and higher query loads, enhancing the system's ability to handle massive document collections.

### 5.5. Need for Comprehensive Performance Evaluation Metrics

The current implementation lacks systematic performance evaluation and benchmarking. While functional improvements are evident, there's no formalized process to measure and track performance metrics like indexing speed, query latency, retrieval accuracy (precision, recall, NDCG), or answer quality.

**Potential Improvement:** Implement comprehensive performance evaluation metrics and benchmarking procedures. Track indexing time, query response time, and, if possible, introduce evaluation datasets to measure retrieval accuracy and answer quality. Automated evaluation and reporting would be invaluable for continuous improvement, system optimization, and performance monitoring. Consider incorporating metrics like ROUGE or BLEU if ground truth answers can be established, or semantic similarity metrics for answer quality assessment in the absence of ground truth.

### 5.6. Cold Start Latency (Persistent Issue)

Cold start latency, caused by loading resources (embedding model, indices) on the first query, remains a concern. While caching addresses subsequent queries, the initial query might experience a noticeable delay.

**Potential Improvement:** Implement a background resource loading mechanism or a startup routine that pre-loads essential resources when the application starts. This could significantly reduce cold start latency and improve the system's responsiveness for the first query after startup or cache clearing. Explore techniques like lazy loading or asynchronous initialization for non-critical resources.

### 5.7. Advanced RAG Techniques (Area for Expansion)

While `rag_app-v12.py` incorporates hybrid search, incremental indexing, and dynamic prompting, exploring and integrating more advanced RAG techniques could further enhance its capabilities and answer quality.

**Potential Improvement:** Explore and potentially implement more advanced RAG techniques to further optimize performance and answer quality. Focus on techniques such as:

- **Query Expansion/Rewriting**: To improve query formulation and better match document content.
- **Context Compression/Filtering**: To reduce noise and focus on the most relevant information within retrieved context.
- **Multi-hop Reasoning**: To handle complex questions that require reasoning across multiple documents or chunks.
- **Knowledge Graph Integration**: To incorporate structured knowledge and enhance reasoning capabilities.
- **Fine-tuning Embedding Models**: To fine-tune embedding models on domain-specific data for improved semantic representation and retrieval accuracy within specific domains.
- **Reranking with Cross-encoders**: If reranking is re-integrated, consider using more powerful cross-encoder models for reranking for improved accuracy.
- **Document Summarization**: To summarize very long documents or sections before chunking, potentially improving context relevance for LLMs.

## 6. Conclusion

`rag_app-v12.py` represents a significant advancement in the development of a practical, efficient, and user-friendly RAG system. It builds upon the strong foundation of `rag_app-v10.py` and `rag_app-v5.py`, incorporating numerous best practices and advanced techniques. Key improvements include modular file handling with multi-format support, incremental indexing, configurable chunking, enhanced caching, memory safety optimizations, a comprehensive hybrid search strategy, dynamic prompting, and robust answer validation, alongside a more user-friendly Gradio interface.

Despite these substantial advancements, identified issues related to global variable dependency, lack of reranking utilization, error handling granularity, FAISS index scalability, and cold start latency persist and should be addressed in future iterations. Furthermore, exploring and integrating more advanced RAG techniques presents opportunities for further enhancing the system's performance, answer quality, and ability to handle complex queries and large document collections.

Overall, `rag_app-v12.py` provides a robust and versatile foundation for building real-world RAG applications. It serves as a valuable case study for understanding the key design considerations, architectural patterns, and practical implementation challenges in developing effective retrieval-augmented generation systems. Future development efforts should focus on addressing the identified issues and exploring more advanced RAG techniques to unlock even greater potential and solidify its position as a high-performing, scalable, and user-centric RAG application.

```

### **3. Final Review:**

I have reviewed the updated research paper and compared it against my code review notes and web search findings. The paper now accurately reflects the improvements in `rag_app-v12.py`, addresses the identified issues, and incorporates relevant best practices. The "Improvements over `rag_app-v5.py`" and "Identified Issues and Potential Improvements" sections have been significantly updated and expanded to provide a more comprehensive and insightful analysis. The paper is well-structured, clear, and provides valuable insights into the RAG system.

**Final checks:**

*   **Code and Paper Consistency:** The paper accurately describes the functionalities and improvements in `rag_app-v12.py` and `file_utils.py`.
*   **Best Practices Integration:** The paper incorporates findings from web searches on RAG best practices and relates them to the code implementation and potential improvements.
*   **Issue and Improvement Clarity:** The "Identified Issues and Potential Improvements" section is clearly articulated and provides actionable recommendations for future development.
*   **Paper Structure and Clarity:** The paper is well-organized, logically flows, and is written in clear and concise language.
*   **Grammar and Typos:** I have reread the paper to check for grammatical errors and typos.

The updated research paper is now ready and reflects a thorough review and analysis of `rag_app-v12.py`. I have addressed all aspects of the user's request.
