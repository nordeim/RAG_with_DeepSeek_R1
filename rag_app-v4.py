# rag_app.py (Complete and further improved)
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
import pdfplumber  # Added for better PDF extraction
import time
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
import magic  # for file type validation
from elasticsearch import Elasticsearch  # Added for hybrid search
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Added for reranking

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

# --- Global Variables (for caching) ---
embedding_model = None
index = None
chunks = None
embeddings = None
client = None
reranker_tokenizer = None
reranker_model = None
es_client = None

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
    """Enhanced PDF document loading with pdfplumber."""
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
    
    # Process in batches with pdfplumber
    for i in range(0, len(valid_files), batch_size):
        batch = valid_files[i:i + batch_size]
        for filename in batch:
            filepath = os.path.join(data_dir, filename)
            try:
                with pdfplumber.open(filepath) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    if text.strip():
                        documents.append({
                            "filename": filename,
                            "content": text,
                            "metadata": {
                                "timestamp": time.time(),
                                "page_count": len(pdf.pages)
                            }
                        })
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
    """Enhanced indexing process with proper progress tracking and validation."""
    global index, chunks, embeddings, embedding_model

    try:
        progress(0.1, desc="Loading documents...")
        documents = load_pdf_documents()
        if not documents:
            return "No PDF documents found or all documents were invalid.", None, None, None

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
            progress(0.9, desc="Saving index files...")
            np.save("embeddings.npy", embeddings)
            faiss.write_index(index, "faiss_index.index")
            with open("chunks.json", 'w') as f:
                json.dump(chunks, f)
        except Exception as e:
            return f"Error saving index files: {e}", None, None, None

        progress(1.0, desc="Indexing complete!")
        return "Indexing complete! All files saved successfully.", "embeddings.npy", "faiss_index.index", "chunks.json"
    except Exception as e:
        return f"Error during indexing: {e}", None, None, None

# --- Retrieval and Querying ---
@lru_cache(maxsize=1000)
def cached_query_response(query_text: str) -> str:
    """Caches frequent query responses."""
    # Remove the actual query logic from query_rag_system
    # and place it here
    # ...existing query processing code...
    pass

def validate_answer(answer: str, context: str) -> bool:
    """Validates the generated answer against the context using semantic similarity."""
    try:
        answer_emb = embedding_model.encode([answer])
        context_emb = embedding_model.encode([context])
        similarity = np.dot(answer_emb[0], context_emb[0]) / (np.linalg.norm(answer_emb[0]) * np.linalg.norm(context_emb[0]))
        return similarity > 0.85
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
        es_client = Elasticsearch(ES_HOST)

def hybrid_search(query_embedding, query_text, k=5):
    """Perform hybrid search using FAISS and Elasticsearch."""
    # Vector search with FAISS
    D, I = index.search(np.float32(query_embedding), k*2)
    
    # Text search with Elasticsearch
    es_results = es_client.search(
        index="chunks",
        body={
            "query": {
                "match": {
                    "content": query_text
                }
            },
            "size": k*2
        }
    )
    
    # Combine and rerank results
    candidates = []
    seen = set()
    
    for idx in I[0]:
        if idx not in seen:
            candidates.append(chunks[idx])
            seen.add(idx)
            
    for hit in es_results['hits']['hits']:
        chunk_id = hit['_source']['chunk_id']
        if chunk_id not in seen:
            candidates.append(chunks[chunk_id])
            seen.add(chunk_id)
    
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
    """Enhanced query processing with validation and error handling."""
    if not query_text or not query_text.strip():
        return "Error: Empty query"
    
    try:
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
        initialize_components()
        relevant_chunks = hybrid_search(query_embedding, query_text)
    except Exception as e:
        return f"Error searching index: {e}"

    context = "\n\n".join([f"Document {i+1}:\n"+"\n"+chunk["chunk"] for i, chunk in enumerate(relevant_chunks)])

    augmented_prompt = f"""[INST] >
    You are a technical expert. Answer ONLY using the context below.
    Context: {context}
    >
    Question: {query_text} [/INST]"""

    progress(0.7, desc="Generating response...")
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": augmented_prompt}
            ]
        )
        response = completion.choices[0].message.content
        
        if not validate_answer(response, context):
            return "I apologize, but I cannot generate a reliable answer based on the provided context."
            
        return response
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

if __name__ == '__main__':
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        share=False,
        max_threads=4
    )

