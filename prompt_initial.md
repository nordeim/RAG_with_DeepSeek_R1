You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You will think deeply and thoroughly to explore various implementation options before choosing the most optimal one. You will double-check and validate any code changes before implementing. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

*Your task:* carefully review the design document for a python based Retrieval-Augmented Generation (RAG) application and the two python code files, indexing.py and retreival.py, below. then think deeply and thoroughly to explore various implementation options to combine the functionalities in the two code files with a gradio based web ui to create an enhanced python app that is modern looking, attractive, user-friendly and intuitive to use. you will explore various implementation options to make the enhanced app more robust, more capable, with a query output display box that support copying and saving the content. after careful exploration, choose the most optimal implementation option to create for me a *complete* fully tested working python code that can work as is without me having to edit it further.


# Design Document Retrieval-Augmented Generation (RAG) system using DeepSeek-R1 LLM via SambaNova API to answer questions based on PDF documents.

## Overview

This project implements a RAG system that:
1. Indexes PDF documents by extracting text, chunking it, and creating vector embeddings
2. Retrieves relevant document chunks based on semantic similarity to user queries
3. Augments prompts with the retrieved context and uses DeepSeek-R1 to generate accurate answers

## Components

- **Indexing System** (`indexing.py`): Processes PDF documents, chunks text, and builds a FAISS vector index
- **Retrieval System** (`retreival.py`): Handles user queries, retrieves relevant context, and generates answers using DeepSeek-R1

## Requirements

- Python 3.8+
- PyPDF (for PDF processing)
- SentenceTransformers (for text embeddings)
- FAISS (for vector similarity search)
- OpenAI Python client (for API communication)
- python-dotenv (for environment variables)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/PromtEngineer/RAG_with_DeepSeek_R1
   cd RAG_with_DeepSeek_R1
   ```

2. Install dependencies:
   ```
   pip install pypdf sentence-transformers faiss-cpu numpy openai python-dotenv
   ```

3. Create a `.env` file with your SambaNova API credentials:
   ```
   SAMBANOVA_API_KEY=your_api_key
   SAMBANOVA_API_BASE_URL=https://api.sambanova.ai/v1
   MODEL_NAME=DeepSeek-R1
   ```

4. Create a `data` directory and add your PDF documents:
   ```
   mkdir data
   # Copy your PDFs to the data directory
   ```

## Usage

1. Index your documents:
   ```
   python indexing.py
   ```
   This creates:
   - `embeddings.npy`: Vector embeddings for document chunks
   - `faiss_index.index`: FAISS similarity search index
   - `chunks.json`: Text chunks with metadata

2. Query the RAG system:
   ```
   python retreival.py
   ```
   Enter your questions about the documents at the prompt.

## How It Works

1. **Indexing Pipeline**:
   - PDF documents are loaded from the `data` directory
   - Documents are split into overlapping chunks
   - SentenceTransformer model creates embeddings for each chunk
   - Embeddings are indexed using FAISS for efficient similarity search

2. **Retrieval Pipeline**:
   - User query is encoded using the same embedding model
   - Similar document chunks are retrieved using FAISS
   - Retrieved context is combined with the query in a prompt
   - DeepSeek-R1 generates a response based on the context

## Customization

- Adjust chunk size and overlap in `indexing.py`
- Modify the number of retrieved chunks in `retreival.py`
- Edit the prompt template in `retreival.py` to change the response format

---
```python
# indexing.py
import os
import pypdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load PDF documents from the 'data' directory
def load_pdf_documents(data_dir='data'):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.pdf'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents.append({"filename": filename, "content": text})
    return documents

# 2. Chunk documents (simple chunking by page for now, can be improved)
def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    chunks = []
    for doc in documents:
        content = doc["content"]
        filename = doc["filename"]
        # Simple chunking by sliding window
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            chunks.append({"filename": filename, "chunk": chunk})
    return chunks

# 3. Create embeddings using sentence-transformers model
def create_embeddings(chunks):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode([chunk["chunk"] for chunk in chunks])
    return embeddings

# 4. Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(np.float32(embeddings))
    return index

if __name__ == '__main__':
    documents = load_pdf_documents()
    if not documents:
        print("No PDF documents found in the 'data' directory. Please add PDF files to the 'data' directory.")
    else:
        chunks = chunk_documents(documents)
        embeddings = create_embeddings(chunks)
        index = build_faiss_index(embeddings)

        # Save chunks and embeddings for later use in querying
        np.save("embeddings.npy", embeddings)
        faiss.write_index(index, "faiss_index.index")
        # Optionally save chunks to a file for easy retrieval during querying
        import json
        with open("chunks.json", 'w') as f:
            json.dump(chunks, f)


        print("PDF documents loaded, chunks created, embeddings generated, and FAISS index built.")
        print("Embeddings saved to 'embeddings.npy'")
        print("FAISS index saved to 'faiss_index.index'")
        print("Chunks saved to 'chunks.json'")
```
---
```python
# retreival.py
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL_NAME = os.environ.get("MODEL_NAME")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
SAMBANOVA_API_BASE_URL = os.environ.get("SAMBANOVA_API_BASE_URL")

# --- Load Resources ---
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.index")
with open("chunks.json", 'r') as f:
    chunks = json.load(f)

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# --- Query Function ---
def query_rag_system(query_text, index, embeddings, chunks, embedding_model): # Removed llm_agent parameter
    query_embedding = embedding_model.encode([query_text])
    D, I = index.search(np.float32(query_embedding), k=20)  # Retrieve top 3 chunks

    # print("Retrieved chunks:", [chunks[i] for i in I[0]])

    relevant_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join([f"Document {i+1}:\n"+"\n"+chunk["chunk"] for i, chunk in enumerate(relevant_chunks)])

    print("___________________________________________________________")
    print("Context:\n", context)
    print("Question:\n", query_text)
    print("___________________________________________________________")

    augmented_prompt = f"""Please answer the following question based on the context provided. 

    Before answering, analyze each document in the context and identify if it contains the answer to the question. 
    Assign a score to each document based on its relevance to the question and then use this information to ignore documents that are not relevant to the question.
    Also, make sure to list the most relevant documents first and then answer the question based on those documents only.
    
    If the context doesn't contain the answer, please respond with 'I am sorry, but the provided context does not have information to answer your question.
    '\n\nContext:\n{context}\n\nQuestion: {query_text}"""

    client = OpenAI(
        base_url=SAMBANOVA_API_BASE_URL, 
        api_key=SAMBANOVA_API_KEY,
    )

    completion = client.chat.completions.create( # Using openai library directly
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": augmented_prompt}
        ]
    )
    response = completion.choices[0].message.content
    return response

if __name__ == '__main__':
    while True:
        user_query = input("Ask a question about the PDF documents (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        response = query_rag_system(user_query, index, embeddings, chunks, embedding_model) # Removed llm_agent parameter
        print(f"\n--- Response from {MODEL_NAME} (via SambaNova) ---")
        print(response)
        print("\n" + "-"*50 + "\n")
```

