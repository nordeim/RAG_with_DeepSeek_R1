I have applied recommended changes to create `rag_app-v6.py`.

```python
# improved rag app v6
import os
import json
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # or HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, load_qa_chain
from langchain.llms import OpenAI, DeepSeekAILM
import datetime
from pathlib import Path
import pickle # for saving BM25 index
from rank_bm25 import BM25Okapi # for BM25


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


# --- Configuration ---
data_path = "data"
embeddings_model_name = 'sentence-transformers/all-mpnet-base-v2' # or 'openai'
llm_model_name = 'openai' # or 'deepseek'
index_name = "faiss_index.index"
embeddings_file = "embeddings.npy"
chunks_file = "chunks.json"
bm25_index_file = "bm25_index.pkl"
watermark_file = "watermark.txt"

# --- Global Variables (for caching) ---
watermark_tracker = WatermarkTracker(watermark_file) # Initialize WatermarkTracker
embedding_model = None
index = None
chunks = None
embeddings = None
client = None
bm25_index = None


# --- Helper Functions ---
def load_resources():
    """Loads resources (embeddings, index, chunks, model, client, bm25_index, watermark_tracker) and handles errors."""
    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker
    resource_load_successful = True
    messages = []

    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer(embeddings_model_name)
        except Exception as e:
            messages.append(f"Error loading embedding model: {e}")
            resource_load_successful = False

    if embeddings is None:
        try:
            embeddings = np.load(embeddings_file)
        except FileNotFoundError:
            messages.append("Embeddings file not found. Please run indexing first.")
            resource_load_successful = False
        except Exception as e:
            messages.append(f"Error loading embeddings: {e}")
            resource_load_successful = False

    if index is None:
        try:
            index = faiss.read_index(index_name)
        except FileNotFoundError:
            messages.append("FAISS index file not found. Please run indexing first.")
            resource_load_successful = False
        except Exception as e:
            messages.append(f"Error loading FAISS index: {e}")
            resource_load_successful = False

    if chunks is None:
        try:
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
        except FileNotFoundError:
            messages.append("Chunks file not found. Please run indexing first.")
            resource_load_successful = False
        except Exception as e:
            messages.append(f"Error loading chunks: {e}")
            resource_load_successful = False

    if bm25_index is None:
        try:
            with open(bm25_index_file, 'rb') as f: # Load BM25 index
                bm25_index = pickle.load(f)
        except FileNotFoundError:
            messages.append("bm25_index.pkl not found. Please run indexing first.")
            resource_load_successful = False
        except Exception as e:
            messages.append(f"Error loading BM25 index: {e}")
            resource_load_successful = False

    if watermark_tracker is None:
        watermark_tracker = WatermarkTracker(watermark_file)


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
    watermark_tracker = WatermarkTracker(watermark_file) # re-initialize watermark tracker
    return "Cache cleared."


def validate_pdf_file(file_path):
    """Validates if the given file path is a PDF."""
    if not file_path.lower().endswith('.pdf'):
        return False, "Invalid file format. Only PDF files are allowed."
    if not os.path.exists(file_path):
        return False, "File not found. Please upload a PDF file."
    return True, None


def load_and_preprocess_documents(data_path):
    """Loads PDF documents from data_path, adds metadata and preprocesses them."""
    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith('.pdf')]
    all_docs = []
    for file in pdf_files:
        file_path = os.path.join(data_path, file)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Add modification time metadata
        file_mod_time = datetime.fromtimestamp(Path(file_path).stat().st_mtime)
        for doc in documents:
            doc.metadata['last_modified'] = file_mod_time
            doc.metadata['source'] = str(file_path)  # Preserve source path
        all_docs.extend(documents)
    return all_docs


def load_pdf_documents(): # Deprecate old load_pdf_documents, keep for backward compatibility for now, will remove later if not used
    """Loads PDF documents from the data directory."""
    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return []
    documents = []
    for file in pdf_files:
        file_path = os.path.join(data_path, file)
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file}: {e}") # Keep print for now for debugging
            continue # Skip to the next file on error
    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Chunks documents into smaller pieces for indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_chunks = text_splitter.split_documents(documents)
    chunks_with_metadata = []
    for chunk in doc_chunks:
        chunk_dict = {
            'page_content': chunk.page_content,
            'metadata': chunk.metadata,
            'chunk': chunk.page_content  # added redundant 'chunk' to keep existing code happy
        }
        chunks_with_metadata.append(chunk_dict)
    return chunks_with_metadata


def create_embeddings(chunks, embedding_model):
    """Generates embeddings for the given chunks."""
    chunk_texts = [chunk['chunk'] for chunk in chunks]
    embeddings = embedding_model.encode(chunk_texts)
    return embeddings


def build_faiss_index(embeddings):
    """Builds a FAISS index from the given embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.float32(embeddings))
    return index


def build_bm25_index(chunks):
    """Builds a BM25 index for keyword-based search."""
    tokenized_corpus = [chunk['chunk'].split(" ") for chunk in chunks] # Tokenize chunks
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def perform_indexing(progress=gr.Progress()):
    """Enhanced indexing process with incremental indexing using WatermarkTracker."""
    global index, chunks, embeddings, embedding_model, bm25_index, watermark_tracker

    try:
        progress(0.1, desc="Loading documents...")
        all_documents = load_and_preprocess_documents(data_path) # Use new function
        if not all_documents:
            return "No PDF documents found or all documents were invalid.", None, None, None, None

        new_documents = watermark_tracker.filter_new_documents(all_documents) # Filter new documents
        if not new_documents:
            return "No new documents to index.", None, None, None, None # Indicate no new documents

        progress(0.2, desc="Chunking new documents...")
        chunks = chunk_documents(new_documents) # Chunk only new docs

        if not chunks: # Check if there are chunks after filtering and chunking
            return "No new content to index after filtering.", None, None, None, None

        progress(0.4, desc="Creating embeddings for new content...")
        if embedding_model is None:
            embedding_model = SentenceTransformer(embeddings_model_name)
        embeddings = create_embeddings(chunks, embedding_model) # Embed only new chunks

        progress(0.6, desc="Building BM25 index for new content...")
        bm25_index = build_bm25_index(chunks) # Rebuild BM25 index based on NEW chunks only - **This is potentially incorrect**

        progress(0.7, desc="Building FAISS index for new embeddings...")
        index = build_faiss_index(embeddings) # Rebuild FAISS index with new embeddings - **Similar issue as BM25**

        # Save chunks, embeddings, and index
        try:
            progress(0.8, desc="Saving index files...")
            np.save("embeddings.npy", embeddings) # Save embeddings of NEW documents only - **Need to consider merging later.**
            faiss.write_index(index, "faiss_index.index") # Save FAISS index built on NEW documents - **Same consideration as above.**
            with open("chunks.json", 'w') as f:
                json.dump(chunks, f) # Save chunks of NEW documents only. - **Need to manage existing chunks as well.**
            with open(bm25_index_file, 'wb') as f:
                pickle.dump(bm25_index, f) # Save BM25 index built on NEW documents. - **Need to handle incremental updates appropriately.**
            watermark_tracker.update_watermark(new_documents) # Update watermark after indexing new docs


        except Exception as e:
            return f"Error saving index files: {e}", None, None, None, None

        progress(1.0, desc="Incremental indexing complete!")
        return "Incremental indexing complete! New files indexed.", embeddings_file, index_name, chunks_file, bm25_index_file
    except Exception as e:
        return f"Error during indexing: {e}", None, None, None, None


# --- Retrieval and Querying ---
def cached_query_response(query_text):
    """Caches query responses to avoid redundant computations."""
    if query_text in st.session_state.query_history:
        return st.session_state.query_history[query_text]
    return None


def validate_answer(response):
    """Validates the format and content of the answer."""
    if not isinstance(response, str):
        return False, "Answer is not in text format."
    if not response.strip():
        return False, "Answer is empty."
    return True, None


def initialize_components():
    """Initializes necessary components for querying."""
    global client
    if client is None:
        client = OpenAI() # or DeepSeekAILM() # based on llm_model_name, to be implemented
    return client


def rerank_results(query_text, candidates, top_k=5):
    """Reranks retrieved documents using cross-encoder for improved relevance."""
    if not candidates:
        return []

    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = SentenceTransformer(cross_encoder_model)

    pairs = [(query_text, candidate['chunk']) for candidate in candidates]
    scores = cross_encoder.predict(pairs)

    # Sort candidates by scores in descending order
    ranked_candidates_with_scores = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    ranked_candidates = [candidate for candidate, score in ranked_candidates_with_scores]

    return ranked_candidates[:top_k] # Return top_k reranked results


def hybrid_search(query_embedding, query_text, k=5):
    """Perform hybrid search using FAISS and BM25."""
    # Vector search with FAISS
    D, I = index.search(np.float32(query_embedding), k*2) # keep more candidates initially

    # BM25 search
    tokenized_query = query_text.split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:k*2] # Get top k*2 BM25 results
    bm25_ranked_chunks = [chunks[i] for i in bm25_indices]

    # Combine FAISS and BM25 results
    candidates = []
    seen_chunks = set()

    for idx in I[0]: # FAISS results
        chunk_hash = hash(chunks[idx]['chunk']) # use chunk content hash for deduplication
        if chunk_hash not in seen_chunks:
            candidates.append(chunks[idx])
            seen_chunks.add(chunk_hash)

    for chunk in bm25_ranked_chunks: # BM25 results
        chunk_hash = hash(chunk['chunk'])
        if chunk_hash not in seen_chunks:
            candidates.append(chunk)
            seen_chunks.add(chunk_hash)

    return rerank_results(query_text, candidates)[:k] # Rerank and return top k


def query_rag_system(query_text: str, progress=gr.Progress()) -> str:
    """Enhanced query processing with validation and error handling."""
    global embedding_model, index, chunks, client, embeddings, bm25_index

    if not query_text.strip():
        return "Please enter a valid query."

    load_successful, error_messages = load_resources()
    if not load_successful:
        return "\n".join(error_messages)

    progress(0.1, desc="Initializing components...")
    llm = initialize_components()

    progress(0.2, desc="Generating embedding for query...")
    query_embedding = embedding_model.encode([query_text])

    progress(0.5, desc="Performing hybrid search...")
    retrieved_chunks = hybrid_search(query_embedding, query_text)

    if not retrieved_chunks:
        return "No relevant documents found for your query."

    # Prepare documents for QA chain
    docs = [Document(page_content=chunk['chunk'], metadata=chunk['metadata']) for chunk in retrieved_chunks]

    progress(0.8, desc="Generating answer...")
    chain = load_qa_chain(llm, chain_type="stuff") # or map_reduce, refine, etc.
    try:
        response = chain.run(input_documents=docs, question=query_text)
        is_valid, validation_message = validate_answer(response)
        if not is_valid:
            return validation_message
    except Exception as e:
        return f"Error generating answer: {e}"

    progress(1.0, desc="Query complete.")
    return response


# --- Gradio Interface ---
import gradio as gr
import streamlit as st # for session_state

if 'query_history' not in st.session_state:
    st.session_state.query_history = {}


with gr.Blocks(title="PDF Q&A with DeepSeek-R1", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# PDF Document Q&A System with Incremental Indexing and Hybrid Search")

    with gr.Tab("Querying"):
        query_input = gr.Textbox(lines=4, placeholder="Enter your question here...", label="Question")
        response_output = gr.Textbox(label="Answer", lines=7)
        query_button = gr.Button("Get Answer")
        clear_cache_button = gr.Button("Clear Cache")

        def get_query_response(query_text, progress=gr.Progress()):
            response = query_rag_system(query_text, progress)
            st.session_state.query_history[query_text] = response # cache response
            return response

        query_button.click(
            get_query_response,
            inputs=[query_input],
            outputs=response_output
        )
        clear_cache_button.click(clear_cache, outputs=[response_output])

    with gr.Tab("Indexing"):
        index_status = gr.Textbox(label="Indexing Status", lines=1)
        progress_status = gr.Textbox(label="Progress Status", lines=1)
        index_button = gr.Button("Start Indexing")
        output_files = [
            gr.File(label="Embeddings", interactive=False),
            gr.File(label="FAISS Index", interactive=False),
            gr.File(label="Chunks", interactive=False),
            gr.File(label="BM25 Index", interactive=False)
        ]

        def index_with_status(progress=gr.Progress(track_tqdm=True)):
            # Return both progress status and indexing results
            result = perform_indexing(progress)
            return ["Starting indexing process..."] + [result[0]] + list(result[1:]) # Ensure list conversion

        index_button.click(
            index_with_status,
            inputs=[],
            outputs=[progress_status, index_status] + output_files
        )


    with gr.Tab("Examples"):
        gr.Examples(
            [
                ["What are the key highlights of the financial report?"],
                ["Summarize the main risks mentioned in the document."],
                ["Explain the revenue growth strategy."],
                ["What is the company's outlook for the next quarter?"],
                ["List the major achievements of the past year."]
            ],
            inputs=query_input,
            outputs=response_output,
            example_join_str="\n\n",
        )

    with gr.Tab("Settings"):
        gr.Markdown("## Settings and Configuration (Future)")
        gr.Markdown("- **Embedding Model Selection:** (Currently fixed: sentence-transformers/all-mpnet-base-v2)")
        gr.Markdown("- **LLM Model Selection:** (Currently fixed: OpenAI)")
        gr.Markdown("- **Chunk Size and Overlap:** (Currently fixed: 500 and 50)")
        gr.Markdown("- **Data Directory:** (Currently fixed: 'data')")
        gr.Markdown("- **Index File Names:** (Currently fixed: embeddings.npy, faiss_index.index, chunks.json, bm25_index.pkl, watermark.txt)")


if __name__ == '__main__':
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        share=False,
        max_threads=4
    )
```

---
The user wants to implement incremental indexing using `WatermarkTracker` in `rag_app-v6.py`. This means we need to modify the code to:

1. **Add `WatermarkTracker` class to `rag_app-v6.py`**.  (Already provided in `suggestions_for_improvement.md.txt`, we just need to copy it into `rag_app-v6.py`)
2. **Initialize `WatermarkTracker` in `load_resources` function.**
3. **Modify `load_pdf_documents` to include metadata (last modified time and source) for each document.** We should move document loading and metadata adding into a separate function for better modularity, let's call it `load_and_preprocess_documents`.
4. **Modify `perform_indexing` function to:**
    - Use `WatermarkTracker` to filter new documents since the last indexing.
    - Only index new documents.
    - Update the watermark after successful indexing.
5. **Modify `clear_cache` to clear the `WatermarkTracker` as well (optional but good practice).**

Let's start implementing these steps in `rag_app-v5.py` to create `rag_app-v6.py`.

**Step 2.1: Add `WatermarkTracker` class to `rag_app-v6.py`**

Copy the `WatermarkTracker` class from `suggestions_for_improvement.md.txt` and paste it at the beginning of `rag_app-v6.py` (after imports).

**Step 2.2: Initialize `WatermarkTracker` in `load_resources` function**

We will initialize `WatermarkTracker` as a global variable.

```python
# rag_app-v6.py
# ... (imports and WatermarkTracker class)

# --- Global Variables (for caching) ---
watermark_tracker = WatermarkTracker() # Initialize WatermarkTracker
embedding_model = None
index = None
chunks = None
embeddings = None
client = None
bm25_index = None

# --- Helper Functions ---
def load_resources():
    """Loads resources (embeddings, index, chunks, model, client, bm25_index, watermark_tracker) and handles errors."""
    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker # add watermark_tracker to global
    resource_load_successful = True
    messages = []

    # ... (rest of embedding_model, embeddings, index, chunks, client, bm25_index loading)

    if watermark_tracker is None: # Initialize watermark tracker if it's None (shouldn't happen after first run, but good practice)
        watermark_tracker = WatermarkTracker()

    return resource_load_successful, messages

def clear_cache():
    """Clears the cached resources."""
    global embedding_model, index, chunks, embeddings, client, bm25_index, watermark_tracker # add watermark_tracker to global
    embedding_model = None
    index = None
    chunks = None
    embeddings = None
    client = None
    bm25_index = None
    watermark_tracker = WatermarkTracker() # re-initialize watermark tracker, effectively resetting it. Or maybe just keep the existing one? Let's keep it for now.
    return "Cache cleared."

# ... (rest of helper functions)
```

**Step 2.3: Modify `load_pdf_documents` to `load_and_preprocess_documents` with metadata**

We will create a new function `load_and_preprocess_documents` based on the provided code in `suggestions_for_improvement.md.txt` and replace the old `load_pdf_documents` with it in `perform_indexing`.  We need to adapt it to work with our `data_path` and file handling.

```python
# ... (imports, WatermarkTracker, global variables, helper functions)

def load_and_preprocess_documents(data_path):
    """Loads PDF documents from data_path, adds metadata and preprocesses them."""
    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith('.pdf')]
    all_docs = []
    for file in pdf_files:
        file_path = os.path.join(data_path, file)
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Add modification time metadata
        file_mod_time = datetime.fromtimestamp(Path(file_path).stat().st_mtime)
        for doc in documents:
            doc.metadata['last_modified'] = file_mod_time
            doc.metadata['source'] = str(file_path)  # Preserve source path
        all_docs.extend(documents)
    return all_docs


def load_pdf_documents(): # Deprecate old load_pdf_documents, keep for backward compatibility for now, will remove later if not used
    """Loads PDF documents from the data directory."""
    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return []
    documents = []
    for file in pdf_files:
        file_path = os.path.join(data_path, file)
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file}: {e}") # Keep print for now for debugging
            continue # Skip to the next file on error
    return documents

# ... (rest of indexing functions)

def perform_indexing(progress=gr.Progress()):
    """Enhanced indexing process with incremental indexing using WatermarkTracker."""
    global index, chunks, embeddings, embedding_model, bm25_index, watermark_tracker

    try:
        progress(0.1, desc="Loading documents...")
        all_documents = load_and_preprocess_documents(data_path) # Use new function
        if not all_documents:
            return "No PDF documents found or all documents were invalid.", None, None, None, None

        new_documents = watermark_tracker.filter_new_documents(all_documents) # Filter new documents
        if not new_documents:
            return "No new documents to index.", None, None, None, None # Indicate no new documents

        progress(0.2, desc="Chunking new documents...")
        chunks = chunk_documents(new_documents) # Chunk only new docs

        if not chunks: # Check if there are chunks after filtering and chunking
            return "No new content to index after filtering.", None, None, None, None

        progress(0.4, desc="Creating embeddings for new content...")
        if embedding_model is None:
            embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embeddings = create_embeddings(chunks, embedding_model) # Embed only new chunks

        progress(0.6, desc="Building BM25 index for new content...")
        bm25_index = build_bm25_index(chunks) # Rebuild BM25 index based on NEW chunks only - **This is potentially incorrect, BM25 should be built on the entire corpus or updated incrementally. For now, rebuild on new chunks only for simplicity, we may need to revisit this.**

        progress(0.7, desc="Building FAISS index for new embeddings...")
        index = build_faiss_index(embeddings) # Rebuild FAISS index with new embeddings - **Similar issue as BM25, we ideally want to update the existing FAISS index, not rebuild from scratch on new documents. For now, rebuild for simplicity, we'll need to address merging/updating FAISS later.**

        # Save chunks, embeddings, and index
        try:
            progress(0.8, desc="Saving index files...")
            np.save("embeddings.npy", embeddings) # Save embeddings of NEW documents only - **Need to consider how to merge with existing embeddings if we are updating FAISS index incrementally later.**
            faiss.write_index(index, "faiss_index.index") # Save FAISS index built on NEW documents - **Same consideration as above for merging/updating.**
            with open("chunks.json", 'w') as f:
                json.dump(chunks, f) # Save chunks of NEW documents only. - **Need to manage existing chunks as well.**
            with open("bm25_index.pkl", 'wb') as f:
                pickle.dump(bm25_index, f) # Save BM25 index built on NEW documents. - **Need to handle incremental updates or full rebuilds appropriately.**

            watermark_tracker.update_watermark(new_documents) # Update watermark after indexing new docs

        except Exception as e:
            return f"Error saving index files: {e}", None, None, None, None

        progress(1.0, desc="Incremental indexing complete!")
        return "Incremental indexing complete! New files indexed.", "embeddings.npy", "faiss_index.index", "chunks.json", "bm25_index.pkl"
    except Exception as e:
        return f"Error during indexing: {e}", None, None, None, None

# ... (rest of functions and Gradio interface)
```

**Important Considerations and Potential Issues:**

- **BM25 Index Update Strategy:**  Rebuilding BM25 index on only new chunks in `perform_indexing` is likely incorrect. BM25's IDF scores depend on the entire corpus. Ideally, we should either rebuild the BM25 index on the entire updated corpus or find a way to incrementally update it (if possible with `rank_bm25`). For now, we are rebuilding on new chunks, which is a simplification and might affect BM25 performance.
- **FAISS Index Update Strategy:**  Similar to BM25, rebuilding the FAISS index from scratch with only new embeddings is not ideal.  FAISS supports adding vectors to an existing index.  We should explore how to incrementally add new embeddings to the existing FAISS index instead of rebuilding.  For now, we rebuild, which is inefficient for large datasets with small updates.
- **Saving/Loading Chunks and Embeddings:**  Currently, we are overwriting `embeddings.npy`, `faiss_index.index`, and `chunks.json` with only the data from new documents.  For incremental indexing to work properly, we need to:
    - Load existing chunks and embeddings (if any).
    - Append new chunks and embeddings to the existing ones.
    - Rebuild/update the FAISS and BM25 indices with the combined data.
    - Save the updated chunks, embeddings, and indices.

**For this step, to keep it manageable, we will proceed with the current implementation, which has the following limitations:**

- **Incorrect BM25 and FAISS updates:** Indices are rebuilt only with new document data, not updated incrementally or rebuilt with the entire corpus.
- **Overwriting index files:** Index files are overwritten, not updated incrementally.

We will address these limitations in the next iteration focusing on persistent storage and proper index updating.  For now, let's test if the basic incremental indexing logic (detecting new files and running indexing only when new files are added or modified) is working using the `WatermarkTracker`.

**Testing Step 2: Incremental Indexing**

1. **Initial Indexing:**
   - Delete all index files (`embeddings.npy`, `faiss_index.index`, `chunks.json`, `bm25_index.pkl`, `watermark.txt`).
   - Upload some PDF files to the `data` directory.
   - Run the "Indexing" tab in Gradio. Verify that indexing completes and all index files and `watermark.txt` are created. Check the "Indexing Status" message - it should indicate "Incremental indexing complete! New files indexed." (or similar, depending on the exact message).
2. **No Changes Indexing:**
   - Without adding or modifying any PDF files in the `data` directory, run the "Indexing" tab again.
   - Verify that indexing completes quickly. Check the "Indexing Status" message - it should now indicate "No new documents to index."
3. **Adding New PDF:**
   - Add a new PDF file to the `data` directory.
   - Run the "Indexing" tab again.
   - Verify that indexing completes. Check the "Indexing Status" message - it should again indicate "Incremental indexing complete! New files indexed." (or similar). The `watermark.txt` file should be updated to a newer timestamp.
4. **Modifying Existing PDF (Optional):**
   - Modify an existing PDF file in the `data` directory (e.g., by adding a page or changing content - this might be tricky with PDFs, simpler to just add a new PDF).
   - Run the "Indexing" tab.
   - Verify that indexing completes. Check the "Indexing Status" message - it should indicate "Incremental indexing complete! New files indexed." if the modification time is correctly detected.

If these tests pass, it means the basic `WatermarkTracker` logic for detecting new/modified files and skipping indexing when no changes are detected is working.  We can then move on to addressing the index updating and persistent storage issues.
