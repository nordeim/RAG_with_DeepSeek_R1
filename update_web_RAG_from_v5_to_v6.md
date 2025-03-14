# diff -u web_RAG-v5.py web_RAG-v6.py
--- web_RAG-v5.py       2025-03-14 14:25:02.070656329 +0800
+++ web_RAG-v6.py       2025-03-14 14:17:51.663608947 +0800
@@ -15,6 +15,12 @@
 import re
 import traceback
 from typing import List, Dict, Any, Optional, Tuple
+from file_utils import validate_file, load_document, ALLOWED_EXTENSIONS
+import logging
+from datetime import datetime
+from rank_bm25 import BM25Okapi
+import pickle
+from threading import Lock
 
 # ------------------ CONFIGURATION ------------------
 
@@ -48,7 +54,6 @@
     "model_name": os.environ.get("MODEL_NAME", "DeepSeek-R1"),
     "api_key": os.environ.get("SAMBANOVA_API_KEY", ""),
     "api_base_url": os.environ.get("SAMBANOVA_API_BASE_URL", "https://api.sambanova.ai/v1"),
-    "embedding_model_name": os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
 }
 
 # Create necessary directories
@@ -58,6 +63,10 @@
 # Configuration file path
 CONFIG_FILE = "config.json"
 
+# Configure logging
+logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
+logger = logging.getLogger(__name__)
+
 # ------------------ UTILITY FUNCTIONS ------------------
 
 def safe_filename(name: str) -> str:
@@ -93,73 +102,55 @@
 # ------------------ DOCUMENT PROCESSING ------------------
 
 def load_pdf_documents(data_dir: str = 'data', progress: Optional[gr.Progress] = None) -> List[Dict[str, Any]]:
-    """Load PDF documents from a directory"""
-    documents: List[Dict[str, Any]] = []
-    files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
-
-    if not files:
+    """Load documents from a directory with enhanced file type support"""
+    documents = []
+    supported_files = []
+    
+    # Get all supported files
+    for ext in ALLOWED_EXTENSIONS:
+        supported_files.extend(Path(data_dir).glob(f"*{ext}"))
+    
+    if not supported_files:
         return []
 
-    total_files = len(files)
+    total_files = len(supported_files)
 
     def update_progress(current: float, message: str):
-        """Safe progress update helper"""
         try:
             if progress is not None:
                 progress(current / total_files, message)
         except Exception as e:
-            print(f"Progress update error: {e}")
+            logger.error(f"Progress update error: {e}")
 
-    for i, filename in enumerate(files):
-        filepath = os.path.join(data_dir, filename)
+    for i, filepath in enumerate(supported_files):
         try:
-            update_progress(i, f"Processing {filename}")
-
-            with open(filepath, 'rb') as f:
-                try:
-                    reader = pypdf.PdfReader(f)
-                    if not reader.pages:
-                        print(f"Warning: {filename} appears to be empty")
-                        continue
-
-                    text = ""
-                    total_pages = len(reader.pages)
-
-                    for page_num in range(total_pages):
-                        try:
-                            page = reader.pages[page_num]
-                            if page is None:
-                                print(f"Warning: Page {page_num} is None in {filename}")
-                                continue
-
-                            page_text = page.extract_text()
-                            if page_text:
-                                text += page_text + "\n\n"
-
-                            update_progress(
-                                i + ((page_num + 1) / total_pages) / total_files,
-                                f"Reading {filename} - page {page_num+1}/{total_pages}"
-                            )
-                        except Exception as e:
-                            print(f"Error extracting text from page {page_num} of {filename}: {e}")
-                            continue
-
-                    if text.strip():  # Only add if text was extracted
-                        documents.append({
-                            "filename": filename,
-                            "content": text,
-                            "pages": total_pages
-                        })
-                        update_progress(i + 1, f"Processed {filename} ({total_pages} pages)")
-                    else:
-                        print(f"Warning: No text extracted from {filename}")
-
-                except Exception as e:
-                    print(f"Error reading PDF {filename}: {e}")
+            update_progress(i, f"Processing {filepath.name}")
+            
+            # Validate file
+            valid, msg = validate_file(str(filepath))
+            if not valid:
+                logger.warning(f"Skipping invalid file {filepath}: {msg}")
+                continue
+                
+            # Load document
+            loaded_docs, error = load_document(str(filepath))
+            if error:
+                logger.error(f"Error loading {filepath}: {error}")
+                continue
+                
+            if loaded_docs:
+                for doc in loaded_docs:
+                    documents.append({
+                        "filename": filepath.name,
+                        "content": doc.page_content,
+                        "pages": 1,  # Default for non-PDF files
+                        "last_modified": doc.metadata.get('last_modified', datetime.now())
+                    })
+                update_progress(i + 1, f"Processed {filepath.name}")
 
         except Exception as e:
-            print(f"Error processing file {filename}: {e}")
-            update_progress(i + 1, f"Error processing {filename}: {str(e)}")
+            logger.error(f"Error processing file {filepath}: {e}")
+            update_progress(i + 1, f"Error processing {filepath.name}")
 
     return documents
 
@@ -229,17 +220,21 @@
     return chunks
 
 def create_embeddings(chunks: List[Dict[str, Any]], api_key: str, api_base_url: str, embedding_model_name: str, progress: Optional[gr.Progress] = None) -> np.ndarray:
-    """Create embeddings for document chunks using the API."""
+    """Create embeddings using local SentenceTransformer model"""
     if not chunks:
         return np.array([])
 
     try:
-        client = OpenAI(base_url=api_base_url, api_key=api_key)
+        # Initialize progress wrapper
         progress_fn = ProgressWrapper(progress)
+        progress_fn(0.1, "Loading SentenceTransformer model...")
 
-        # Process in batches to show progress
+        # Initialize the model
+        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
+        
+        # Process in smaller batches with memory management
         total_chunks = len(chunks)
-        batch_size = min(10, max(1, total_chunks // 10))
+        batch_size = min(8, max(1, total_chunks // 20))  # Smaller batch size for memory efficiency
         all_embeddings = []
 
         for i in range(0, total_chunks, batch_size):
@@ -250,32 +245,37 @@
             progress_fn(i / total_chunks, f"Embedding chunks {i+1}-{batch_end}/{total_chunks}")
 
             try:
-                response = client.embeddings.create(input=chunk_texts, model=embedding_model_name)
-                batch_embeddings = [item.embedding for item in response.data]
-                all_embeddings.extend(batch_embeddings)
+                # Create embeddings for the batch
+                batch_embeddings = model.encode(chunk_texts, 
+                                             show_progress_bar=False,
+                                             batch_size=8)  # Smaller internal batch size
+                all_embeddings.append(batch_embeddings)
+
+                # Force garbage collection after each batch
+                import gc
+                gc.collect()
 
             except Exception as e:
-                error_msg = str(e)
-                if "404" in error_msg:
-                    print(f"Error: Model '{embedding_model_name}' not found. Please verify the model name in settings.")
-                elif "401" in error_msg:
-                    print("Error: Authentication failed. Please check your API key in settings.")
-                else:
-                    print(f"API Error: {error_msg}")
-                traceback.print_exc()
+                logger.error(f"Batch embedding error: {str(e)}")
                 return np.array([])
 
             progress_fn(batch_end / total_chunks, f"Embedded {batch_end}/{total_chunks} chunks")
 
-        return np.array(all_embeddings)
+        # Combine all embeddings
+        try:
+            final_embeddings = np.vstack(all_embeddings)
+            progress_fn(1.0, "Embeddings creation completed")
+            return final_embeddings
+        except Exception as e:
+            logger.error(f"Error combining embeddings: {str(e)}")
+            return np.array([])
 
     except Exception as e:
-        print(f"Error creating embeddings: {e}")
-        traceback.print_exc()
+        logger.error(f"Error creating embeddings: {str(e)}")
         return np.array([])
 
 def build_faiss_index(embeddings: np.ndarray, progress: Optional[gr.Progress] = None) -> Optional[faiss.IndexFlatL2]:
-    """Build a FAISS index for vector similarity search"""
+    """Build FAISS index with improved memory management and error handling"""
     progress_fn = ProgressWrapper(progress)
 
     try:
@@ -290,18 +290,26 @@
         dimension = embeddings.shape[1]
         progress_fn(0.2, f"Initializing FAISS index with dimension {dimension}...")
 
-        # Create index
-        index = faiss.IndexFlatL2(dimension)
-        if not index:
-            raise RuntimeError("Failed to create FAISS index")
-
-        progress_fn(0.5, "Adding vectors to index...")
-
-        # Convert to float32 and add to index
-        embeddings_float32 = np.float32(embeddings)
-        index.add(embeddings_float32)
+        # Use IVFFlat index for better memory efficiency with large datasets
+        if len(embeddings) > 10000:
+            nlist = min(int(np.sqrt(len(embeddings))), 2048)  # number of clusters
+            quantizer = faiss.IndexFlatL2(dimension)
+            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
+            progress_fn(0.4, "Training IVF index...")
+            index.train(np.float32(embeddings))
+        else:
+            index = faiss.IndexFlatL2(dimension)
+
+        progress_fn(0.6, "Adding vectors to index...")
 
-        progress_fn(0.8, f"Verifying index... (total vectors: {index.ntotal})")
+        # Convert to float32 and add to index in batches
+        batch_size = 10000
+        for i in range(0, len(embeddings), batch_size):
+            batch = embeddings[i:i + batch_size]
+            index.add(np.float32(batch))
+            progress_fn(0.6 + 0.3 * (i + len(batch)) / len(embeddings), f"Added {i + len(batch)}/{len(embeddings)} vectors")
+
+        progress_fn(0.9, f"Verifying index... (total vectors: {index.ntotal})")
 
         # Verify index
         if index.ntotal != embeddings.shape[0]:
@@ -312,8 +320,7 @@
 
     except Exception as e:
         progress_fn(1.0, f"Error: {str(e)}")
-        print(f"Error building FAISS index: {e}")
-        traceback.print_exc()
+        logger.error(f"Error building FAISS index: {e}")
         return None
 
 # ------------------ RETRIEVAL FUNCTIONS ------------------
@@ -324,6 +331,20 @@
     index_dir = os.path.join("indexes", index_name)
 
     try:
+        # Validate API credentials
+        if not api_key or not api_base_url:
+            return {
+                "response": "Error: Missing API credentials. Please check your settings.",
+                "context": "",
+                "chunks": []
+            }
+
+        # Initialize OpenAI client with provided credentials
+        client = OpenAI(
+            base_url=api_base_url,
+            api_key=api_key
+        )
+
         # Load resources
         embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
         index = faiss.read_index(os.path.join(index_dir, "faiss_index.index"))
@@ -334,25 +355,12 @@
         with open(os.path.join(index_dir, "config.json"), 'r') as f:
             config = json.load(f)
 
-        # Get the embedding model name from the config
-        embedding_model_name = config.get("embedding_model_name", DEFAULT_CONFIG["embedding_model_name"])  # Fallback
-
-        # Embed the query using the API
-        client = OpenAI(base_url=api_base_url, api_key=api_key)
-        try:
-            query_response = client.embeddings.create(input=[query_text], model=embedding_model_name)
-            query_embedding = np.array(query_response.data[0].embedding)
-        except Exception as e:
-            error_msg = str(e)
-            if "404" in error_msg:
-                return {"response": f"Error: Model '{embedding_model_name}' not found. Please verify the model name in settings.", "context": "", "chunks": []}
-            elif "401" in error_msg:
-                return {"response": "Error: Authentication failed. Please check your API key in settings.", "context": "", "chunks": []}
-            else:
-                return {"response": f"API Error: {error_msg}", "context": "", "chunks": []}
+        # Create query embedding using local SentenceTransformer
+        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
+        query_embedding = model.encode([query_text])[0]  # Get the first embedding
 
         # Search for similar chunks
-        D, I = index.search(np.float32(query_embedding.reshape(1, -1)), k=k)  # Reshape for FAISS
+        D, I = index.search(np.float32(query_embedding.reshape(1, -1)), k=k)
 
         # Get relevant chunks
         relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
@@ -373,14 +381,22 @@
 If the context doesn't contain the answer, please respond with 'I am sorry, but the provided context does not have information to answer your question.
 '\n\nContext:\n{context}\n\nQuestion: {query_text}"""
 
-        completion = client.chat.completions.create(
-            model=model_name,
-            messages=[
-                {"role": "user", "content": augmented_prompt}
-            ]
-        )
-
-        response = completion.choices[0].message.content
+        # Make API request with error handling
+        try:
+            completion = client.chat.completions.create(
+                model=model_name,
+                messages=[
+                    {"role": "user", "content": augmented_prompt}
+                ]
+            )
+            response = completion.choices[0].message.content
+        except Exception as api_error:
+            logger.error(f"API request failed: {str(api_error)}")
+            return {
+                "response": f"API Error: {str(api_error)}. Please check your API settings and try again.",
+                "context": "",
+                "chunks": []
+            }
 
         # Return both the response and the used context
         return {
@@ -391,8 +407,8 @@
 
     except Exception as e:
         error_trace = traceback.format_exc()
-        print(f"Error in query_rag_system: {e}")
-        print(error_trace)
+        logger.error(f"Error in query_rag_system: {e}")
+        logger.error(error_trace)
 
         return {
             "response": f"System Error: {str(e)}. Please check the application logs for details.",
@@ -434,7 +450,7 @@
 
     return "\n".join(file_info)
 
-def build_index(index_name: str, chunk_size: int, chunk_overlap: int, embeddings_model: str, api_key: str, api_base_url: str, embedding_model_name: str, progress: gr.Progress = gr.Progress()) -> str:
+def build_index(index_name: str, chunk_size: int, chunk_overlap: int, embeddings_model: str, api_key: str, api_base_url: str, progress: gr.Progress = gr.Progress()) -> str:
     """Build an index from uploaded documents"""
     if not index_name:
         return "Error: Please provide an index name"
@@ -475,9 +491,9 @@
             shutil.rmtree(index_dir)
             return "Error: No chunks could be created from the documents."
 
-        # Create embeddings using API
+        # Create embeddings using local model
         progress(0.4, "Creating embeddings...")
-        embeddings = create_embeddings(chunks, api_key, api_base_url, embedding_model_name, progress=progress)
+        embeddings = create_embeddings(chunks, api_key, api_base_url, embeddings_model, progress=progress)
 
 
         if embeddings.size == 0:
@@ -504,8 +520,8 @@
         config = {
             "chunk_size": chunk_size,
             "chunk_overlap": chunk_overlap,
-            "embeddings_model": embeddings_model,
-            "embedding_model_name": embedding_model_name,  # Store explicit model name
+            "embeddings_model": embeddings_model,  # Use consistent naming
+            "embedding_model_name": embeddings_model,  # Store both for compatibility
             "num_documents": len(documents),
             "num_chunks": len(chunks),
             "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
@@ -621,14 +637,13 @@
         print(error_trace)
         return f"Error: {str(e)}", "", ""
 
-def update_settings(key: str, url: str, model: str, embedding_model: str) -> str:
+def update_settings(key: str, url: str, model: str) -> str:
     """Update and save application settings"""
     try:
         config = load_config()
         config["api_key"] = key
         config["api_base_url"] = url
         config["model_name"] = model
-        config["embedding_model_name"] = embedding_model
         if save_config(config):
             return "Settings saved successfully!"
         else:
@@ -772,8 +787,7 @@
                         chunk_overlap,
                         embeddings_model,
                         gr.State(config.get("api_key", DEFAULT_CONFIG["api_key"])),
-                        gr.State(config.get("api_base_url", DEFAULT_CONFIG["api_base_url"])),
-                        gr.State(config.get("embedding_model_name", DEFAULT_CONFIG["embedding_model_name"])),  # Use .get()
+                        gr.State(config.get("api_base_url", DEFAULT_CONFIG["api_base_url"]))
                     ],
                     outputs=[index_result]
                 )
@@ -926,13 +940,6 @@
                     value=config.get("model_name", DEFAULT_CONFIG["model_name"]),
                     info="Name of the model to use"
                 )
-
-                embedding_model_name = gr.Textbox(
-                    label="Embedding Model Name",
-                    placeholder="BAAI/bge-large-en-v1.5",
-                    value=config.get("embedding_model_name", DEFAULT_CONFIG["embedding_model_name"]),
-                    info="Name of the embedding model to use"
-                )
                 
                 save_settings = gr.Button("Save Settings", variant="primary")
                 
@@ -940,7 +947,7 @@
                 
                 save_settings.click(
                     fn=update_settings,
-                    inputs=[api_key, api_base_url, model_name, embedding_model_name],
+                    inputs=[api_key, api_base_url, model_name],
                     outputs=[settings_result]
                 )
