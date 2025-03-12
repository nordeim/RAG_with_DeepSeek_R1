## two enhanced web based versions of https://github.com/PromtEngineer/RAG_with_DeepSeek_R1

```
pip install pypdf sentence-transformers faiss-cpu numpy openai python-dotenv pdfplumber libmagic
python rag_app-v11.py # working great!
```
![image](https://github.com/user-attachments/assets/281d6c08-7e7c-45e9-adbf-82c044fea85c)
![image](https://github.com/user-attachments/assets/54004d8a-4fc4-4c2a-a0a0-bf0d30cd3f92)

```
$ env | grep ELASTIC
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_PASSWORD=vakWlJO9UFVHL=Cugr3_
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_VERIFY_CERTS=false
ELASTICSEARCH_CERT_PATH=/etc/ssl/certs/ca-certificates.crt

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
