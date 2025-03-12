## Key Issues & Solutions in the RAG Implementation

### 1. **Incomplete Hybrid Search Implementation**
**Problem**: The `hybrid_search` function lacks Elasticsearch integration and BM25 utilization.

**Solution**:
```python
def hybrid_search(query_embedding, query_text, k=5):
    """Perform true hybrid search combining vector + keyword + BM25 results."""
    # Vector search (FAISS)
    D, I = index.search(np.float32(query_embedding), k*2)
    faiss_results = [chunks[idx] for idx in I[0] if idx  bool:
    if not filepath.lower().endswith('.pdf'):
        return False
    
    try:
        # Basic header check
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False
    except:
        return False
    
    return os.path.getsize(filepath)  chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            current_chunk.append(sent)
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    return chunks
```

### 5. **Robust Elasticsearch Connection**
**Enhanced Initialization**:
```python
def initialize_components():
    # ... existing code ...
    if es_client is None and ES_PASSWORD:
        es_config = {
            "hosts": [ES_HOST],
            "basic_auth": (ES_USER, ES_PASSWORD),
            "ssl_show_warn": False,
            "verify_certs": ES_VERIFY_CERTS
        }
        
        if ES_CERT_PATH and os.path.exists(ES_CERT_PATH):
            es_config["ca_certs"] = ES_CERT_PATH
            
        try:
            es_client = elasticsearch.Elasticsearch(**es_config,
                retry_on_timeout=True,
                max_retries=3
            )
            if not es_client.ping():
                raise ConnectionError
        except Exception as e:
            print(f"ES Connection Failed: {e}")
```

### 6. **Cache Invalidation**
**Add to Indexing Completion**:
```python
# After saving indices
query_cache.cache.clear()
```

---

## Implementation Validation Checklist
1. **Hybrid Search Test**
```python
def test_hybrid_search():
    test_query = "machine learning"
    query_embedding = embedding_model.encode([test_query])
    results = hybrid_search(query_embedding, test_query)
    assert len(results) == 5, "Should return 5 results"
    assert any('machine' in res['chunk'] for res in results)
```

2. **Index Update Test**
```python
def test_incremental_indexing():
    initial_count = len(chunks)
    # Modify test.pdf
    perform_indexing()
    assert len(chunks) > initial_count, "Should add new chunks"
    assert any('updated_content' in c['chunk'] for c in chunks)
```

---

This comprehensive approach addresses the core architectural issues while maintaining backward compatibility. The solutions prioritize:
1. True hybrid search capabilities
2. Data freshness through proper update handling
3. Robust document validation
4. Context-aware processing
5. Enterprise-grade error handling

---
https://www.perplexity.ai/search/you-are-a-deep-thinking-ai-you-btYkAtXjSLy9iv0gczwZvQ
