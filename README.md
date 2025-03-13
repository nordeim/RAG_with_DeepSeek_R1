**two enhanced web based versions of [RAG (Retrieval-Augmented Generation)](https://github.com/PromtEngineer/RAG_with_DeepSeek_R1)**

[Review of rag_app-v12.py and file_utils.py](https://github.com/nordeim/RAG_with_DeepSeek_R1/blob/main/Research%20Paper%3A%20Analysis%20of%20Retrieval-Augmented%20Generation%20(RAG)%20Implementation%20in%20rag_app-v12.md)
```
pip install pypdf sentence-transformers faiss-cpu numpy openai python-dotenv pdfplumber libmagic
pip install python-magic unstructured python-docx openpyxl
python rag_app-v12.py  # added multiple file types extraction with utility, file_utils.py
```
![image](https://github.com/user-attachments/assets/735832ce-f16d-4929-8c7c-677bef6eaa97)
![image](https://github.com/user-attachments/assets/a0c3b4d0-936b-4d89-9244-5699f824b2f3)

```
$ env | grep ELASTIC
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_PASSWORD=vakWlJO9UFVHL=Cugr3_
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_VERIFY_CERTS=false
ELASTICSEARCH_CERT_PATH=/etc/ssl/certs/ca-certificates.crt

$ python3 rag_app-v12.py
* Running on local URL:  http://0.0.0.0:7860
INFO:httpx:HTTP Request: GET http://localhost:7860/gradio_api/startup-events "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD http://localhost:7860/ "HTTP/1.1 200 OK"

To create a public link, set `share=True` in `launch()`.
INFO:httpx:HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cpu
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: sentence-transformers/all-mpnet-base-v2
INFO:elastic_transport.transport:HEAD http://localhost:9200/ [status:200 duration:0.099s]
Successfully connected to Elasticsearch
INFO:elastic_transport.transport:HEAD http://localhost:9200/chunks [status:200 duration:0.003s]
INFO:elastic_transport.transport:PUT http://localhost:9200/chunks/_doc/0 [status:200 duration:0.105s]
INFO:elastic_transport.transport:PUT http://localhost:9200/chunks/_doc/1 [status:200 duration:0.210s]
INFO:elastic_transport.transport:PUT http://localhost:9200/chunks/_doc/2 [status:200 duration:0.077s]
Elasticsearch indexing complete.

[DEBUG 2025-03-13T09:57:09.219194] Processing query: Summarize the key points in the documents.
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.71steps/s]
/cdrom/venv/mychat/RAG_app/rag_app-v12.py:683: DeprecationWarning: Passing transport options in the API method is deprecated. Use 'Elasticsearch.options()' instead.
  es_results = es_client.search(
INFO:elastic_transport.transport:POST http://localhost:9200/chunks/_search [status:200 duration:0.057s]
[DEBUG] Retrieved 4 relevant chunks
[DEBUG] Structured context created (1220 chars)
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
[DEBUG] Generated response (804 chars)

[DEBUG] Starting enhanced answer validation...
[DEBUG] Detected query type: summary
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.46steps/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.13s/steps]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.91steps/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.89steps/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.14s/steps]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.82steps/s]
[DEBUG] Validation Metrics:
  - Answer-Context Similarity: 0.5985
  - Answer-Query Similarity: 0.1450
  - Context-Query Similarity: 0.1815
  - Required Thresholds (type=summary):
    * Context: 0.6000
    * Query: 0.1500
[DEBUG] Validation details:
  Context check: 0.5985 > 0.6000 = False
  Query check: 0.1450 > 0.1500 = False
[DEBUG] Validation failed
[DEBUG] Answer validation failed, attempting fallback...

$ python3 rag_app-v12.py
* Running on local URL:  http://0.0.0.0:7860
INFO:httpx:HTTP Request: GET http://localhost:7860/gradio_api/startup-events "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: HEAD http://localhost:7860/ "HTTP/1.1 200 OK"

To create a public link, set `share=True` in `launch()`.
INFO:httpx:HTTP Request: GET https://api.gradio.app/pkg-version "HTTP/1.1 200 OK"

[DEBUG 2025-03-13T10:02:37.047342] Processing query: Summarize the key points in the documents.
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cpu
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: sentence-transformers/all-mpnet-base-v2
INFO:elastic_transport.transport:HEAD http://localhost:9200/ [status:200 duration:0.004s]
Successfully connected to Elasticsearch
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.20steps/s]
/cdrom/venv/mychat/RAG_app/rag_app-v12.py:683: DeprecationWarning: Passing transport options in the API method is deprecated. Use 'Elasticsearch.options()' instead.
  es_results = es_client.search(
INFO:elastic_transport.transport:POST http://localhost:9200/chunks/_search [status:200 duration:0.044s]
[DEBUG] Retrieved 4 relevant chunks
[DEBUG] Structured context created (1220 chars)
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
[DEBUG] Generated response (691 chars)

[DEBUG] Starting enhanced answer validation...
[DEBUG] Detected query type: summary
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.69steps/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.15s/steps]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.94steps/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.09steps/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.13s/steps]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.38steps/s]
[DEBUG] Validation Metrics:
  - Answer-Context Similarity: 0.6736
  - Answer-Query Similarity: 0.2545
  - Context-Query Similarity: 0.1815
  - Required Thresholds (type=summary):
    * Context: 0.5000
    * Query: 0.1000
[DEBUG] Validation passed
```

```
pip install pypdf sentence-transformers faiss-cpu numpy openai python-dotenv pdfplumber libmagic
python rag_app-v11.py # working great!

$ python3 rag_app-v11.py
* Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.

[DEBUG 2025-03-12T12:35:58.585122] Processing query: Summarize the key points in the documents.
Successfully connected to Elasticsearch
/cdrom/venv/mychat/RAG_app/rag_app-v11.py:699: DeprecationWarning: Passing transport options in the API method is deprecated. Use 'Elasticsearch.options()' instead.
  es_results = es_client.search(
[DEBUG] Retrieved 4 relevant chunks
[DEBUG] Structured context created (1220 chars)
[DEBUG] Generated response (650 chars)

[DEBUG] Starting enhanced answer validation...
[DEBUG] Detected query type: summary
[DEBUG] Validation Metrics:
  - Answer-Context Similarity: 0.6591
  - Answer-Query Similarity: 0.2085
  - Context-Query Similarity: 0.1815
  - Required Thresholds (type=summary):
    * Context: 0.6000
    * Query: 0.1500
[DEBUG] Validation passed
```

---
```
python rag_app-v5.py
```
![image](https://github.com/user-attachments/assets/7a3e09cb-1cab-480f-a0d2-891e2170f173)

![image](https://github.com/user-attachments/assets/cfc0e2f4-7078-43b1-a2bf-e503081aea92)

```
$ python3 rag_app-v5.py
* Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 443/443 [00:00<00:00, 3.30MB/s]
sentencepiece.bpe.model: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.07M/5.07M [00:01<00:00, 3.27MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17.1M/17.1M [00:00<00:00, 21.3MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 279/279 [00:00<00:00, 2.65MB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 801/801 [00:00<00:00, 6.47MB/s]
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.24G/2.24G [01:58<00:00, 18.9MB/s]
Search error: Connection error caused by: ConnectionError(Connection error caused by: NewConnectionError(<urllib3.connection.HTTPConnection object at 0x7b83161c8950>: Failed to establish a new connection: [Errno 111] Connection refused))
```

---
```
python web_RAG-v2.py
```
![image](https://github.com/user-attachments/assets/a351238c-d9ca-4ef0-9e7a-fb2b51026961)

![image](https://github.com/user-attachments/assets/42e78fb9-c7e1-4996-bea0-348b51811290)

![image](https://github.com/user-attachments/assets/ec4759f6-d90b-4d52-af0f-af32141c0712)

```
$ python3 web_RAG-v2.py
* Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Skipping data after last boundary
Illegal instruction (core dumped)
```
