## two enhanced web based versions of https://github.com/PromtEngineer/RAG_with_DeepSeek_R1

```
pip install pypdf sentence-transformers faiss-cpu numpy openai python-dotenv pdfplumber libmagic
python rag_app-v9.py
```
![image](https://github.com/user-attachments/assets/f52e2275-4303-4b5b-8720-14fe44bdc3ec)
![image](https://github.com/user-attachments/assets/a55f7af0-6195-4eab-a8a1-18b2e12cbf2e)

- currect error: *I apologize, but I cannot generate a reliable answer based on the provided context.*
- [suggested fix](https://github.com/nordeim/RAG_with_DeepSeek_R1/blob/main/fix_Elasticsearch_connection_error.md)
```
$ env | grep ELASTIC
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_PASSWORD=vakWlJO9UFVHL=Cugr3_
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_VERIFY_CERTS=false
ELASTICSEARCH_CERT_PATH=/etc/ssl/certs/ca-certificates.crt
(mychat) pete@pop-os:/cdrom/venv/mychat/RAG_app
$ python3 rag_app-v9.py
* Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
Elasticsearch index 'chunks' created successfully.
Error during Elasticsearch indexing: 1 validation error for ProgressUnit
progress
  Input should be a valid number [type=float_type, input_value=<enumerate object at 0x7c24cebf2bb0>, input_type=enumerate]
    For further information visit https://errors.pydantic.dev/2.10/v/float_type
Warning: Elasticsearch indexing failed, continuing with other indices
/cdrom/venv/mychat/lib/python3.12/site-packages/gradio/blocks.py:1829: UserWarning: A function (index_with_status) returned too many output values (needed: 5, returned: 6). Ignoring extra values.
    Output components:
        [textbox, textbox, file, file, file]
    Output values returned:
        ["Starting indexing process...", "Incremental indexing complete!", "embeddings.npy", "faiss_index.index", "chunks.json", "bm25_index.pkl"]
  warnings.warn(
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
