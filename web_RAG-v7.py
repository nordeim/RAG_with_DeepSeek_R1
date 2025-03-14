import os
import sys
import json
import numpy as np
import faiss
import gradio as gr
import pypdf
from sentence_transformers import SentenceTransformer  # Keep for potential fallback, but primary embedding is via API now
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import shutil
import time
from pathlib import Path
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple
from file_utils import validate_file, load_document, ALLOWED_EXTENSIONS
import logging
from datetime import datetime
from rank_bm25 import BM25Okapi
import pickle
from threading import Lock

# ------------------ CONFIGURATION ------------------

# Load environment variables
load_dotenv()

# Configuration Instructions
"""
IMPORTANT CONFIGURATION NOTES:

1. API Setup:
   - Set SAMBANOVA_API_KEY in your environment or .env file
   - Set SAMBANOVA_API_BASE_URL if using a custom endpoint
   
2. Model Configuration:
   - Default embedding model: BAAI/bge-large-en-v1.5
   - Default LLM model: DeepSeek-R1
   
3. Configuration can be modified via:
   - Environment variables
   - .env file
   - Settings tab in the web interface
"""

# Default configuration with enhanced documentation
DEFAULT_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "embeddings_model": "sentence-transformers/all-mpnet-base-v2",  # Legacy config key
    "retrieval_chunks": 20,
    "model_name": os.environ.get("MODEL_NAME", "DeepSeek-R1"),
    "api_key": os.environ.get("SAMBANOVA_API_KEY", ""),
    "api_base_url": os.environ.get("SAMBANOVA_API_BASE_URL", "https://api.sambanova.ai/v1"),
}

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("indexes", exist_ok=True)

# Configuration file path
CONFIG_FILE = "config.json"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------ UTILITY FUNCTIONS ------------------

def safe_filename(name: str) -> str:
    """Convert a string to a safe filename"""
    # Remove invalid characters and replace spaces with underscores
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to disk"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def load_config() -> Dict[str, Any]:
    """Load configuration from disk, merging with defaults."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Update loaded config with any missing keys from DEFAULT_CONFIG
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
    except Exception as e:
        print(f"Error loading config: {e}")
    return DEFAULT_CONFIG.copy()

# ------------------ DOCUMENT PROCESSING ------------------

def load_pdf_documents(data_dir: str = 'data', progress: Optional[gr.Progress] = None) -> List[Dict[str, Any]]:
    """Load documents from a directory with enhanced file type support"""
    documents = []
    supported_files = []
    
    # Get all supported files
    for ext in ALLOWED_EXTENSIONS:
        supported_files.extend(Path(data_dir).glob(f"*{ext}"))
    
    if not supported_files:
        return []

    total_files = len(supported_files)

    def update_progress(current: float, message: str):
        try:
            if progress is not None:
                progress(current / total_files, message)
        except Exception as e:
            logger.error(f"Progress update error: {e}")

    for i, filepath in enumerate(supported_files):
        try:
            update_progress(i, f"Processing {filepath.name}")
            
            # Validate file
            valid, msg = validate_file(str(filepath))
            if not valid:
                logger.warning(f"Skipping invalid file {filepath}: {msg}")
                continue
                
            # Load document
            loaded_docs, error = load_document(str(filepath))
            if error:
                logger.error(f"Error loading {filepath}: {error}")
                continue
                
            if loaded_docs:
                for doc in loaded_docs:
                    documents.append({
                        "filename": filepath.name,
                        "content": doc.page_content,
                        "pages": 1,  # Default for non-PDF files
                        "last_modified": doc.metadata.get('last_modified', datetime.now())
                    })
                update_progress(i + 1, f"Processed {filepath.name}")

        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            update_progress(i + 1, f"Error processing {filepath.name}")

    return documents

class ProgressWrapper:
    """Wrapper for safe progress updates"""
    def __init__(self, progress_obj: Optional[gr.Progress] = None):
        self.progress = progress_obj

    def __call__(self, value: float, desc: str = ""):
        try:
            if self.progress is not None:
                self.progress(value, desc)
        except Exception as e:
            print(f"Progress update error: {str(e)}")

def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50, progress: Optional[gr.Progress] = None) -> List[Dict[str, Any]]:
    """Split documents into overlapping chunks"""
    chunks: List[Dict[str, Any]] = []
    if not documents:
        return []

    # Create progress wrapper
    progress_fn = ProgressWrapper(progress)
    total_docs = len(documents)

    for i, doc in enumerate(documents):
        try:
            content = doc["content"]
            filename = doc["filename"]

            # Update progress at start of each document
            progress_fn(i / total_docs, f"Processing {filename}")

            # Simple chunking by sliding window
            doc_chunks: List[Dict[str, Any]] = []

            # For very short documents, just create one chunk
            if len(content) <= chunk_size:
                doc_chunks.append({
                    "filename": filename,
                    "chunk": content,
                    "chunk_id": len(chunks)
                })
            else:
                # For longer documents, use sliding window
                for j in range(0, len(content), chunk_size - chunk_overlap):
                    chunk_text = content[j:j + chunk_size]
                    if len(chunk_text.strip()) > 0:  # Only add non-empty chunks
                        doc_chunks.append({
                            "filename": filename,
                            "chunk": chunk_text,
                            "chunk_id": len(chunks) + len(doc_chunks)
                        })

            chunks.extend(doc_chunks)

            # Update progress after processing each document
            progress_fn((i + 1) / total_docs, f"Created {len(doc_chunks)} chunks from {filename}")

        except Exception as e:
            print(f"Error chunking document {filename}: {e}")
            continue

    # Final progress update
    progress_fn(1.0, f"Completed chunking {len(documents)} documents into {len(chunks)} chunks")

    return chunks

def create_embeddings(chunks: List[Dict[str, Any]], api_key: str, api_base_url: str, embedding_model_name: str, progress: Optional[gr.Progress] = None) -> np.ndarray:
    """Create embeddings using local SentenceTransformer model"""
    if not chunks:
        return np.array([])

    try:
        # Initialize progress wrapper
        progress_fn = ProgressWrapper(progress)
        progress_fn(0.1, "Loading SentenceTransformer model...")

        # Initialize the model
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Process in smaller batches with memory management
        total_chunks = len(chunks)
        batch_size = min(8, max(1, total_chunks // 20))  # Smaller batch size for memory efficiency
        all_embeddings = []

        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch = chunks[i:batch_end]
            chunk_texts = [chunk["chunk"] for chunk in batch]

            progress_fn(i / total_chunks, f"Embedding chunks {i+1}-{batch_end}/{total_chunks}")

            try:
                # Create embeddings for the batch
                batch_embeddings = model.encode(chunk_texts, 
                                             show_progress_bar=False,
                                             batch_size=8)  # Smaller internal batch size
                all_embeddings.append(batch_embeddings)

                # Force garbage collection after each batch
                import gc
                gc.collect()

            except Exception as e:
                logger.error(f"Batch embedding error: {str(e)}")
                return np.array([])

            progress_fn(batch_end / total_chunks, f"Embedded {batch_end}/{total_chunks} chunks")

        # Combine all embeddings
        try:
            final_embeddings = np.vstack(all_embeddings)
            progress_fn(1.0, "Embeddings creation completed")
            return final_embeddings
        except Exception as e:
            logger.error(f"Error combining embeddings: {str(e)}")
            return np.array([])

    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        return np.array([])

def build_faiss_index(embeddings: np.ndarray, progress: Optional[gr.Progress] = None) -> Optional[faiss.IndexFlatL2]:
    """Build FAISS index with improved memory management and error handling"""
    progress_fn = ProgressWrapper(progress)

    try:
        # Validate embeddings
        if embeddings.size == 0:
            raise ValueError("Empty embeddings array provided")

        if len(embeddings.shape) != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {embeddings.shape}")

        # Get embedding dimension
        dimension = embeddings.shape[1]
        progress_fn(0.2, f"Initializing FAISS index with dimension {dimension}...")

        # Use IVFFlat index for better memory efficiency with large datasets
        if len(embeddings) > 10000:
            nlist = min(int(np.sqrt(len(embeddings))), 2048)  # number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            progress_fn(0.4, "Training IVF index...")
            index.train(np.float32(embeddings))
        else:
            index = faiss.IndexFlatL2(dimension)

        progress_fn(0.6, "Adding vectors to index...")

        # Convert to float32 and add to index in batches
        batch_size = 10000
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            index.add(np.float32(batch))
            progress_fn(0.6 + 0.3 * (i + len(batch)) / len(embeddings), f"Added {i + len(batch)}/{len(embeddings)} vectors")

        progress_fn(0.9, f"Verifying index... (total vectors: {index.ntotal})")

        # Verify index
        if index.ntotal != embeddings.shape[0]:
            raise RuntimeError(f"Index verification failed: expected {embeddings.shape[0]} vectors, got {index.ntotal}")

        progress_fn(1.0, "FAISS index built successfully")
        return index

    except Exception as e:
        progress_fn(1.0, f"Error: {str(e)}")
        logger.error(f"Error building FAISS index: {e}")
        return None

# ------------------ RETRIEVAL FUNCTIONS ------------------

class HybridSearchHelper:
    """Helper class for hybrid search implementation"""
    def __init__(self, faiss_index, chunks: List[Dict[str, Any]], bm25_index: Optional[BM25Okapi] = None):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.bm25_index = bm25_index

    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range"""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def get_hybrid_scores(self, query_embedding: np.ndarray, query_text: str, k: int = 20, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Get combined scores from FAISS and BM25"""
        # Get FAISS scores
        dense_scores, dense_indices = self.faiss_index.search(query_embedding.reshape(1, -1), k=k)
        dense_scores = self.normalize_scores(dense_scores[0])
        
        # Get BM25 scores if available
        if self.bm25_index is not None:
            tokenized_query = query_text.lower().split()
            bm25_scores = np.array(self.bm25_index.get_scores(tokenized_query))
            bm25_scores = self.normalize_scores(bm25_scores)
            
            # Combine scores
            all_scores = alpha * dense_scores + (1 - alpha) * bm25_scores[dense_indices]
            return all_scores, dense_indices
        
        return dense_scores, dense_indices

def query_rag_system(query_text: str, index_name: str, api_key: str, api_base_url: str, model_name: str, k: int = 20) -> Dict[str, Any]:
    """Query the RAG system with a user question"""
    # Load the index
    index_dir = os.path.join("indexes", index_name)

    try:
        # Validate API credentials
        if not api_key or not api_base_url:
            return {
                "response": "Error: Missing API credentials. Please check your settings.",
                "context": "",
                "chunks": []
            }

        # Initialize OpenAI client with provided credentials
        client = OpenAI(
            base_url=api_base_url,
            api_key=api_key
        )

        # Load resources
        embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        faiss_index = faiss.read_index(os.path.join(index_dir, "faiss_index.index"))

        with open(os.path.join(index_dir, "chunks.json"), 'r') as f:
            chunks = json.load(f)

        # Load BM25 index if available
        bm25_index = None
        bm25_path = os.path.join(index_dir, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                bm25_index = pickle.load(f)

        # Create query embedding using local SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        query_embedding = model.encode([query_text])[0]  # Get the first embedding

        # Initialize hybrid search
        hybrid_search = HybridSearchHelper(faiss_index, chunks, bm25_index)
        
        # Get hybrid search results
        scores, indices = hybrid_search.get_hybrid_scores(
            np.float32(query_embedding),
            query_text,
            k=k,
            alpha=0.7  # Favor dense retrieval slightly
        )

        # Convert numpy array indices to Python list - fix for nested array
        indices_list = indices[0].tolist()  # Get the inner array and convert to list

        # Get relevant chunks using converted indices
        relevant_chunks = [chunks[i] for i in indices_list if i < len(chunks)]

        # Prepare context
        context = "\n\n".join([
            f"Document {i+1} (from {chunk['filename']}):\n{chunk['chunk']}"
            for i, chunk in enumerate(relevant_chunks)
        ])

        # Prepare augmented prompt
        augmented_prompt = f"""Please answer the following question based on the context provided.

Before answering, analyze each document in the context and identify if it contains the answer to the question.
Assign a score to each document based on its relevance to the question and then use this information to ignore documents that are not relevant to the question.
Also, make sure to list the most relevant documents first and then answer the question based on those documents only.

If the context doesn't contain the answer, please respond with 'I am sorry, but the provided context does not have information to answer your question.
'\n\nContext:\n{context}\n\nQuestion: {query_text}"""

        # Make API request with error handling
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": augmented_prompt}
                ]
            )
            response = completion.choices[0].message.content
        except Exception as api_error:
            logger.error(f"API request failed: {str(api_error)}")
            return {
                "response": f"API Error: {str(api_error)}. Please check your API settings and try again.",
                "context": "",
                "chunks": []
            }

        # Return both the response and the used context
        return {
            "response": response,
            "context": context,
            "chunks": relevant_chunks
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in query_rag_system: {e}")
        logger.error(error_trace)

        return {
            "response": f"System Error: {str(e)}. Please check the application logs for details.",
            "context": "",
            "chunks": []
        }

# ------------------ GRADIO UI FUNCTIONS ------------------

def get_indexes() -> List[str]:
    """Get list of available indexes"""
    if not os.path.exists("indexes"):
        return []
    return [d for d in os.listdir("indexes") if os.path.isdir(os.path.join("indexes", d))]

def handle_upload(files: List[gr.utils.NamedString]) -> str:
    """Handle file uploads to the data directory"""
    if not files:
        return "No files selected for upload."

    # Clear data directory
    if os.path.exists("data"):
        shutil.rmtree("data")
    os.makedirs("data", exist_ok=True)

    # Copy uploaded files to data directory
    file_info = []
    for file in files:
        try:
            if not file.name.lower().endswith('.pdf'):
                file_info.append(f"❌ Skipped {os.path.basename(file.name)} - Not a PDF file")
                continue

            filename = os.path.basename(file.name)
            shutil.copy(file.name, os.path.join("data", filename))
            file_info.append(f"✅ {filename}")
        except Exception as e:
            file_info.append(f"❌ Error with {os.path.basename(file.name)}: {str(e)}")

    return "\n".join(file_info)

def build_index(index_name: str, chunk_size: int, chunk_overlap: int, embeddings_model: str, api_key: str, api_base_url: str, progress: gr.Progress = gr.Progress()) -> str:
    """Build an index from uploaded documents"""
    if not index_name:
        return "Error: Please provide an index name"

    # Sanitize index name
    index_name = safe_filename(index_name)
    if not index_name:
        return "Error: Invalid index name. Please use alphanumeric characters and underscores."

    # Check if files exist in data directory
    if not os.listdir("data"):
        return "Error: No files in data directory. Please upload files first."

    # Create index directory
    index_dir = os.path.join("indexes", index_name)
    if os.path.exists(index_dir):
        return f"Error: An index with the name '{index_name}' already exists. Please choose a different name."

    os.makedirs(index_dir, exist_ok=True)
    
    # Store traceback for debugging
    error_traceback = ""

    try:
        # Load documents with progress tracking
        progress(0, "Loading PDF documents...")
        documents = load_pdf_documents(progress=progress)

        if not documents:
            shutil.rmtree(index_dir)
            return "Error: No PDF documents could be processed. Please check your files and ensure they contain extractable text."

        # Create chunks
        progress(0.2, "Chunking documents...")
        chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap, progress=progress)

        if not chunks:
            shutil.rmtree(index_dir)
            return "Error: No chunks could be created from the documents."

        # Create BM25 index
        progress(0.3, "Building BM25 index...")
        tokenized_chunks = [chunk["chunk"].lower().split() for chunk in chunks]
        bm25_index = BM25Okapi(tokenized_chunks)

        # Create embeddings using local model
        progress(0.4, "Creating embeddings...")
        embeddings = create_embeddings(chunks, api_key, api_base_url, embeddings_model, progress=progress)


        if embeddings.size == 0:
            shutil.rmtree(index_dir)
            return "Error: Failed to create embeddings."

        # Build FAISS index
        progress(0.8, "Building FAISS index...")
        index = build_faiss_index(embeddings, progress=progress)

        if index is None:
            shutil.rmtree(index_dir)
            return "Error: Failed to build FAISS index."

        # Save files
        progress(0.9, "Saving index files...")
        np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)
        faiss.write_index(index, os.path.join(index_dir, "faiss_index.index"))

        # Save BM25 index
        with open(os.path.join(index_dir, "bm25_index.pkl"), 'wb') as f:
            pickle.dump(bm25_index, f)

        with open(os.path.join(index_dir, "chunks.json"), 'w') as f:
            json.dump(chunks, f)

        # Save configuration
        config = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embeddings_model": embeddings_model,  # Use consistent naming
            "embedding_model_name": embeddings_model,  # Store both for compatibility
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(index_dir, "config.json"), 'w') as f:
            json.dump(config, f)

        # Generate summary
        doc_names = [doc["filename"] for doc in documents]
        summary = f"""
### Index Summary: {index_name}

- **Documents Processed**: {len(documents)}
- **Total Chunks**: {len(chunks)}
- **Embedding Dimensions**: {embeddings.shape[1]}
- **Chunk Size**: {chunk_size}
- **Chunk Overlap**: {chunk_overlap}
- **Embedding Model**: {embeddings_model}
- **Created**: {config['created_at']}

#### Documents in this index:
"""
        for doc in sorted(doc_names):
            summary += f"- {doc}\n"

        summary += "\nIndex successfully built and saved!"

        return summary

    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(index_dir, ignore_errors=True)
        error_traceback = traceback.format_exc()  # Capture the full traceback
        print(f"Error building index: {e}\n{error_traceback}")  # Print to console as well
        return f"Error building index: {str(e)}\nDetails:\n{error_traceback}"  # Include in the return message


def load_index_details(index_name: str) -> str:
    """Load and display details about an index"""
    if not index_name:
        return ""

    index_dir = os.path.join("indexes", index_name)

    try:
        with open(os.path.join(index_dir, "config.json"), 'r') as f:
            config = json.load(f)

        with open(os.path.join(index_dir, "chunks.json"), 'r') as f:
            chunks = json.load(f)

        doc_names = set(chunk["filename"] for chunk in chunks)

        details = f"""
### Index: {index_name}

- **Documents**: {config.get('num_documents', len(doc_names))}
- **Chunks**: {config.get('num_chunks', len(chunks))}
- **Created**: {config.get('created_at', 'Unknown')}
- **Chunk Size**: {config.get('chunk_size', 'Unknown')}
- **Chunk Overlap**: {config.get('chunk_overlap', 'Unknown')}
- **Embedding Model**: {config.get('embeddings_model', 'Unknown')}

#### Documents in this index:
"""
        for doc in sorted(doc_names):
            details += f"- {doc}\n"

        return details

    except Exception as e:
        return f"Error loading index details: {str(e)}"

def handle_query(query: str, index_name: str, api_key: str, api_base_url: str, model_name: str, num_chunks: int, show_context: bool) -> Tuple[str, str, str]:
    """Process a query using the RAG system"""
    if not query:
        return "Please enter a query", "", ""

    if not index_name:
        return "Please select an index", "", ""

    if not api_key:
        return "Please provide an API key in the Settings tab", "", ""

    try:
        result = query_rag_system(
            query_text=query,
            index_name=index_name,
            api_key=api_key,
            api_base_url=api_base_url,
            model_name=model_name,
            k=num_chunks
        )

        response = result["response"]
        context = result["context"]

        # Format chunks for display
        chunks_display = ""
        for i, chunk in enumerate(result["chunks"]):
            chunks_display += f"### Document {i+1}: {chunk['filename']}\n\n"
            chunks_display += chunk['chunk'] + "\n\n---\n\n"

        if show_context:
            return response, chunks_display, context
        else:
            return response, chunks_display, ""

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in handle_query: {e}")
        print(error_trace)
        return f"Error: {str(e)}", "", ""

def update_settings(key: str, url: str, model: str) -> str:
    """Update and save application settings"""
    try:
        config = load_config()
        config["api_key"] = key
        config["api_base_url"] = url
        config["model_name"] = model
        if save_config(config):
            return "Settings saved successfully!"
        else:
            return "Error: Could not save settings."
    except Exception as e:
        return f"Error updating settings: {str(e)}"

def delete_index(index_name: str) -> Tuple[str, gr.update]:
    """Delete an index"""
    if not index_name:
        return "No index selected", gr.update()

    index_dir = os.path.join("indexes", index_name)

    try:
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            return f"Index '{index_name}' deleted successfully", gr.update(choices=get_indexes(), value=None)
        else:
            return f"Index '{index_name}' not found", gr.update(choices=get_indexes())
    except Exception as e:
        return f"Error deleting index: {str(e)}", gr.update()

def export_response(response: str) -> str:
    """Export the response to a file"""
    if not response:
        return "No response to export"

    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"response_{timestamp}.txt"

        with open(filename, 'w') as f:
            f.write(response)

        return f"Response exported to {filename}"
    except Exception as e:
        return f"Error exporting response: {str(e)}"

# ------------------ GRADIO UI DEFINITION ------------------

def create_gradio_app() -> gr.Blocks:
    """Create the Gradio web interface"""
    # Load configuration
    config = load_config()

    # Define the interface
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 📚 RAG System with DeepSeek-R1

        This application allows you to:
        1. Upload PDF documents and create searchable indexes
        2. Ask questions about your documents
        3. Get AI-generated answers using DeepSeek-R1
        """)

        with gr.Tabs() as tabs:
            # Indexing Tab
            with gr.TabItem("📋 Document Indexing"):
                with gr.Row():
                    with gr.Column(scale=2):
                        files_upload = gr.File(
                            label="Upload PDF Documents",
                            file_types=[".pdf"],
                            file_count="multiple"
                        )
                        uploaded_files = gr.Textbox(
                            label="Uploaded Files",
                            placeholder="No files uploaded yet",
                            interactive=False,
                            lines=5
                        )
                        upload_button = gr.Button("Upload Files", variant="primary")

                    with gr.Column(scale=3):
                        index_name = gr.Textbox(
                            label="Index Name",
                            placeholder="my_index",
                            info="Give your index a unique name"
                        )
                        
                        with gr.Row():
                            chunk_size = gr.Slider(
                                label="Chunk Size",
                                minimum=100,
                                maximum=2000,
                                value=config["chunk_size"],
                                step=50,
                                info="Number of characters per chunk"
                            )
                            chunk_overlap = gr.Slider(
                                label="Chunk Overlap",
                                minimum=0,
                                maximum=200,
                                value=config["chunk_overlap"],
                                step=10,
                                info="Overlap between consecutive chunks"
                            )
                        
                        embeddings_model = gr.Dropdown(
                            label="Embeddings Model (Display Only)",  # Changed label
                            choices=[
                                "sentence-transformers/all-mpnet-base-v2",
                                "jinaai/jina-embeddings-v3",
                                "sentence-transformers/all-MiniLM-L6-v2",
                                "BAAI/bge-m3",
                                "sentence-transformers/multi-qa-mpnet-base-dot-v1"
                            ],
                            value=config["embeddings_model"],
                            info="Model used for creating embeddings (metadata - actual model set in Settings)",
                            interactive=False  # Make it non-interactive
                        )
                        
                        index_button = gr.Button("Build Index", variant="primary")
                
                index_result = gr.Markdown(label="Indexing Result")
                
                with gr.Accordion("Manage Indexes", open=False):
                    with gr.Row():
                        delete_index_dropdown = gr.Dropdown(
                            label="Select Index to Delete",
                            choices=get_indexes(),
                            info="WARNING: This will permanently delete the index"
                        )
                        refresh_index_list = gr.Button("Refresh List")
                        delete_button = gr.Button("Delete Index", variant="stop")
                    
                    delete_result = gr.Markdown()
                    upload_button.click(
                    fn=handle_upload,
                    inputs=[files_upload],
                    outputs=[uploaded_files]
                )
                
                index_button.click(
                    fn=build_index,
                    inputs=[
                        index_name,
                        chunk_size,
                        chunk_overlap,
                        embeddings_model,
                        gr.State(config.get("api_key", DEFAULT_CONFIG["api_key"])),
                        gr.State(config.get("api_base_url", DEFAULT_CONFIG["api_base_url"]))
                    ],
                    outputs=[index_result]
                )
                
                refresh_index_list.click(
                    fn=lambda: gr.update(choices=get_indexes()),
                    inputs=[],
                    outputs=[delete_index_dropdown]
                )
                
                delete_button.click(
                    fn=delete_index,
                    inputs=[delete_index_dropdown],
                    outputs=[delete_result, delete_index_dropdown]
                )
            
            # Query Tab
            with gr.TabItem("🔍 Query Documents"):
                with gr.Row():
                    with gr.Column(scale=2):
                        index_selector = gr.Dropdown(
                            label="Select Index",
                            choices=get_indexes(),
                            info="Choose an index to query",
                            interactive=True
                        )
                        
                        index_details = gr.Markdown(label="Index Details")
                        
                        refresh_indexes = gr.Button("Refresh Indexes")
                        
                        with gr.Accordion("Advanced Query Settings", open=False):
                            num_chunks = gr.Slider(
                                label="Number of Chunks to Retrieve",
                                minimum=1,
                                maximum=50,
                                value=config["retrieval_chunks"],
                                step=1,
                                info="Number of similar chunks to retrieve"
                            )
                            show_context = gr.Checkbox(
                                label="Show Full Context",
                                value=False,
                                info="Display the full context sent to the model"
                            )
                    
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about your documents...",
                            lines=3
                        )
                        
                        query_button = gr.Button("Submit Query", variant="primary")
                        
                        response_output = gr.Textbox(
                            label="Response",
                            placeholder="Response will appear here",
                            lines=10,
                            interactive=False
                        )
                        
                        with gr.Row():
                            # Replace the copy button implementation
                            copy_button = gr.Button("📋 Copy Response")
                            export_button = gr.Button("💾 Export to File")
                        
                        export_status = gr.Markdown()
                
                with gr.Accordion("Retrieved Document Chunks", open=False):
                    chunks_output = gr.Markdown()
                
                with gr.Accordion("Full Context (Advanced)", open=False, visible=False) as context_accordion:
                    context_output = gr.Textbox(
                        label="Full Context",
                        lines=20,
                        interactive=False
                    )
                
                # Show/hide context accordion based on checkbox
                show_context.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[show_context],
                    outputs=[context_accordion]
                )
                
                # Refresh index list
                refresh_indexes.click(
                    fn=lambda: gr.update(choices=get_indexes()),
                    inputs=[],
                    outputs=[index_selector]
                )
                
                # Show index details when selected
                index_selector.change(
                    fn=load_index_details,
                    inputs=[index_selector],
                    outputs=[index_details]
                )
                
                # Handle query submission
                query_button.click(
                    fn=handle_query,
                    inputs=[
                        query_input,
                        index_selector,
                        gr.State(config.get("api_key", DEFAULT_CONFIG["api_key"])),
                        gr.State(config.get("api_base_url", DEFAULT_CONFIG["api_base_url"])),
                        gr.State(config.get("model_name", DEFAULT_CONFIG["model_name"])),
                        num_chunks,
                        show_context
                    ],
                    outputs=[response_output, chunks_output, context_output]
                )
                
                # Replace JavaScript copy handler with Python function
                copy_button.click(
                    fn=lambda x: f"✅ Response ready to copy!",
                    inputs=[response_output],
                    outputs=[export_status]
                )
                
                # Handle export to file
                export_button.click(
                    fn=export_response,
                    inputs=[response_output],
                    outputs=[export_status]
                )
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                api_key = gr.Textbox(
                    label="SambaNova API Key",
                    placeholder="Enter your API key",
                    type="password",
                    value=config.get("api_key", DEFAULT_CONFIG["api_key"]),
                    info="Your SambaNova API key (kept local and not shared)"
                )
                
                api_base_url = gr.Textbox(
                    label="API Base URL",
                    placeholder="https://api.sambanova.ai/v1",
                    value=config.get("api_base_url", DEFAULT_CONFIG["api_base_url"]),
                    info="Base URL for the SambaNova API"
                )
                
                model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="DeepSeek-R1",
                    value=config.get("model_name", DEFAULT_CONFIG["model_name"]),
                    info="Name of the model to use"
                )
                
                save_settings = gr.Button("Save Settings", variant="primary")
                
                settings_result = gr.Markdown()
                
                save_settings.click(
                    fn=update_settings,
                    inputs=[api_key, api_base_url, model_name],
                    outputs=[settings_result]
                )
                
                with gr.Accordion("About", open=False):
                    gr.Markdown("""
                    ## RAG System with DeepSeek-R1
                    
                    This application implements a Retrieval-Augmented Generation (RAG) system that:
                    
                    1. Indexes PDF documents by extracting text, chunking it, and creating vector embeddings
                    2. Retrieves relevant document chunks based on semantic similarity to user queries
                    3. Augments prompts with the retrieved context and uses DeepSeek-R1 to generate accurate answers
                    
                    ### How to use:
                    
                    1. **Document Indexing**: Upload PDF documents and create an index
                    2. **Query Documents**: Ask questions about your indexed documents
                    3. **Settings**: Configure API settings for the LLM
                    
                    ### Technologies:
                    
                    - **PyPDF**: For PDF processing
                    - **SentenceTransformers**: For text embeddings
                    - **FAISS**: For vector similarity search
                    - **OpenAI Python client**: For API communication
                    - **Gradio**: For the web interface
                    """)
    
        # Move the load event registration inside the Blocks context
        app.load(
            fn=lambda: [
                gr.update(choices=get_indexes()),
                gr.update(choices=get_indexes())
            ],
            inputs=None,
            outputs=[index_selector, delete_index_dropdown],
            show_progress=False
        )
    
    return app

# ------------------ MAIN FUNCTION ------------------

def main() -> None:
    # Load or create config
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
    
    # Create and launch the app
    app = create_gradio_app()
    app.queue().launch(share=False, show_error=True)

if __name__ == "__main__":
    main()

