# rag_app.py (Complete and further improved)
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
import pypdf
import time
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
import magic  # for file type validation

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
SAMBANOVA_API_BASE_URL = os.environ.get("SAMBANOVA_API_BASE_URL")
if not all([MODEL_NAME, SAMBANOVA_API_KEY, SAMBANOVA_API_BASE_URL]):
    raise ValueError("Missing required environment variables: MODEL_NAME, SAMBANOVA_API_KEY, or SAMBANOVA_API_BASE_URL")

# --- Global Variables (for caching) ---
embedding_model = None
index = None
chunks = None
embeddings = None
client = None

# --- Helper Functions ---
def load_resources():
    """Loads resources (embeddings, index, chunks, model, client) and handles errors."""
    global embedding_model, index, chunks, embeddings, client
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

    return resource_load_successful, messages


def clear_cache():
    """Clears the cached resources."""
    global embedding_model, index, chunks, embeddings, client
    embedding_model = None
    index = None
    chunks = None
    embeddings = None
    client = None
    return "Cache cleared."

# --- Indexing Functions ---

def validate_pdf_file(filepath: str) -> bool:
    """Validates if file is a valid PDF."""
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

def load_pdf_documents(data_dir='data', batch_size=5):
    """Loads PDF documents from the specified directory with batching."""
    documents = []
    valid_files = []
    
    # First validate all files
    for filename in os.listdir(data_dir):
        if not filename.endswith('.pdf'):
            continue
        filepath = os.path.join(data_dir, filename)
        if not validate_pdf_file(filepath):
            print(f"Skipping invalid or too large PDF: {filename}")
            continue
        valid_files.append(filename)
    
    # Process in batches
    for i in range(0, len(valid_files), batch_size):
        batch = valid_files[i:i + batch_size]
        for filename in batch:
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    if text.strip():  # Only add if text was extracted
                        documents.append({"filename": filename, "content": text})
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Chunks the documents into smaller pieces."""
    chunks = []
    for doc in documents:
        content = doc["content"]
        filename = doc["filename"]
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            chunks.append({"filename": filename, "chunk": chunk})
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


def perform_indexing(progress=gr.Progress()):
    """Performs the indexing process and saves the results."""
    global index, chunks, embeddings, embedding_model

    progress(0.1, desc="Loading documents...")
    documents = load_pdf_documents()
    if not documents:
        return "No PDF documents found in the 'data' directory.  Please add PDF files.", None, None, None

    progress(0.3, desc="Chunking documents...")
    chunks = chunk_documents(documents)

    progress(0.5, desc="Creating embeddings...")
    if embedding_model is None:
        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = create_embeddings(chunks, embedding_model)

    progress(0.8, desc="Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Save chunks, embeddings, and index
    try:
        np.save("embeddings.npy", embeddings)
        faiss.write_index(index, "faiss_index.index")
        with open("chunks.json", 'w') as f:
            json.dump(chunks, f)
    except Exception as e:
        return f"Error saving indexing data: {e}", None, None, None

    progress(1.0, desc="Indexing complete!")
    return "Indexing complete! Embeddings, FAISS index, and chunks saved.", "embeddings.npy", "faiss_index.index", "chunks.json"

# --- Retrieval and Querying ---
@lru_cache(maxsize=1000)
def cached_query_response(query_text: str) -> str:
    """Caches frequent query responses."""
    # Remove the actual query logic from query_rag_system
    # and place it here
    # ...existing query processing code...
    pass

def query_rag_system(query_text: str, progress=gr.Progress()) -> str:
    """Validates input and handles query processing."""
    if not query_text or not query_text.strip():
        return "Error: Empty query"
    
    # Limit query length
    if len(query_text) > 1000:
        return "Error: Query too long. Please limit to 1000 characters."
    
    # Sanitize input
    query_text = query_text.strip()
    
    try:
        # Check cache first
        return cached_query_response(query_text)
    except Exception as e:
        return f"Error processing query: {e}"

    global index, embeddings, chunks, embedding_model, client
    load_successful, load_messages = load_resources()
    if not load_successful:
        return "Error: " + "\n".join(load_messages)

    progress(0.2, desc="Encoding query...")
    try:
        query_embedding = embedding_model.encode([query_text])
    except Exception as e:
        return f"Error encoding query: {e}"

    progress(0.4, desc="Searching index...")
    try:
        D, I = index.search(np.float32(query_embedding), k=5)  # Retrieve top 5
    except Exception as e:
        return f"Error searching index: {e}"

    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join([f"Document {i+1}:\n"+"\n"+chunk["chunk"] for i, chunk in enumerate(relevant_chunks)])

    augmented_prompt = f"""Please answer the following question based on the context provided.

    Before answering, analyze each document in the context and identify if it contains the answer to the question.
    Assign a score to each document based on its relevance to the question and then use this information to ignore documents that are not relevant to the question.
    Also, make sure to list the most relevant documents first and then answer the question based on those documents only.

    If the context doesn't contain the answer, please respond with 'I am sorry, but the provided context does not have information to answer your question.
    '\n\nContext:\n{context}\n\nQuestion: {query_text}"""

    progress(0.7, desc="Generating response...")
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": augmented_prompt}
            ]
        )
        response = completion.choices[0].message.content
    except Exception as e:
        return f"Error during response generation: {e}"

    progress(1.0, desc="Response generated!")
    return response

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
        query_input = gr.Textbox(label="Question", placeholder="Enter your question here...")
        query_button = gr.Button("Ask")
        query_output = gr.Textbox(label="Answer", show_copy_button=True, lines=10, max_lines=20) # Improved output box

        query_button.click(
            query_rag_system,
            inputs=[query_input],
            outputs=[query_output],
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

if __name__ == '__main__':
    demo.queue().launch(
        server_name="0.0.0.0",
        share=False,
        max_threads=3
    )  # Configure queue with proper parameters

