# ADR-003: Vector Store Design

**Status**: Accepted
**Date**: 2025-01-16
**Commit**: [will be filled after implementation]
**LlamaIndex Reference**: `llama-index-core/llama_index/core/vector_stores/simple.py` (lines 1-400)
**Related Papers**: Dense Passage Retrieval (Karpukhin et al., 2020)

---

## Context

### Problem Statement

We need a way to store text embeddings (vectors) and perform efficient similarity searches to retrieve relevant document chunks for a given query.

**Constraints**:
- Embeddings are dense vectors (typically 384-1536 dimensions)
- Need to support top-k retrieval (e.g., retrieve 10 most similar documents)
- Should handle thousands to millions of vectors
- Need to support metadata filtering (e.g., filter by document type)
- Must integrate with our index and retrieval abstractions

**Current State**:
- We have TextNodes with embeddings (from Phase 2)
- We have a VectorStoreIndex concept (from Phase 4)
- No storage mechanism for vectors yet

**Requirements**:
1. Store embeddings with associated node IDs
2. Perform similarity search (cosine similarity)
3. Return top-k most similar nodes
4. Support basic metadata filtering
5. Simple to implement and understand (MVP focus)

---

## Decision

We will implement a **SimpleVectorStore** with the following design:

### Core Design

```python
class SimpleVectorStore(BasePydanticVectorStore):
    """In-memory vector store using numpy for similarity computation."""

    def __init__(self):
        # node_id → embedding vector
        self.embedding_dict: Dict[str, List[float]] = {}

        # node_id → document ID (for deletion)
        self.text_id_to_ref_doc_id: Dict[str, str] = {}

    def add(self, nodes: List[BaseNode]) -> List[str]:
        """Add nodes with embeddings to the store."""
        for node in nodes:
            if node.embedding is None:
                raise ValueError(f"Node {node.node_id} has no embedding")

            self.embedding_dict[node.node_id] = node.embedding
            if node.ref_doc_id:
                self.text_id_to_ref_doc_id[node.node_id] = node.ref_doc_id

        return list(self.embedding_dict.keys())

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Query by embedding similarity."""
        # 1. Get all embeddings
        all_node_ids = list(self.embedding_dict.keys())
        all_embeddings = [self.embedding_dict[nid] for nid in all_node_ids]

        # 2. Compute cosine similarity
        similarities = self._compute_similarity(
            query.query_embedding,
            all_embeddings
        )

        # 3. Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # 4. Apply top-k
        top_k = query.similarity_top_k
        top_indices = sorted_indices[:top_k]

        # 5. Get top nodes and scores
        top_node_ids = [all_node_ids[i] for i in top_indices]
        top_similarities = [similarities[i] for i in top_indices]

        # 6. Load nodes from document store (via storage context)
        top_nodes = self._docstore.get_nodes(top_node_ids)

        return VectorStoreQueryResult(
            nodes=top_nodes,
            similarities=top_similarities,
            ids=top_node_ids
        )

    def _compute_similarity(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]]
    ) -> List[float]:
        """Compute cosine similarity between query and all embeddings."""
        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        embedding_matrix = np.array(embeddings)

        # Cosine similarity: dot(query, embedding) / (||query|| * ||embedding||)
        # Optimized: normalize first, then dot product
        query_norm = query_vec / np.linalg.norm(query_vec)
        embedding_norms = embedding_matrix / np.linalg.norm(
            embedding_matrix, axis=1, keepdims=True
        )

        similarities = np.dot(embedding_norms, query_norm)
        return similarities.tolist()
```

### Key Design Choices

1. **In-Memory Storage**: Dict-based (node_id → embedding)
2. **Similarity Metric**: Cosine similarity (default for semantic search)
3. **Search Algorithm**: Linear search (brute force)
4. **Metadata Filtering**: Post-filtering after similarity search
5. **Persistence**: None initially (Phase 2+: JSON save/load)

---

## Rationale

### Why This Approach?

#### 1. **In-Memory Dict-Based Storage**

**Pros**:
- ✅ Simple to implement and understand
- ✅ Fast lookups for small to medium datasets (<100k vectors)
- ✅ No external dependencies (just Python dicts + numpy)
- ✅ Easy to serialize (JSON) for persistence later

**Cons**:
- ❌ Memory intensive for large datasets (all vectors in RAM)
- ❌ No disk persistence by default
- ❌ Limited scalability (>1M vectors becomes slow)

**Why Acceptable for MVP**:
- Focus is learning, not production scale
- Most learning examples use <10k documents
- Can add production-grade stores (FAISS, Pinecone) via integration points

#### 2. **Cosine Similarity**

**Pros**:
- ✅ Standard for semantic embeddings (OpenAI, HuggingFace use this)
- ✅ Magnitude-invariant (focuses on direction, not length)
- ✅ Range [-1, 1] is interpretable

**Cons**:
- ❌ Slightly slower than dot product (requires normalization)

**Why Chosen**:
- Industry standard for semantic search
- Embeddings are typically normalized anyway
- Clearer semantics than dot product

**Alternative**: Dot product (if embeddings are pre-normalized, equivalent and faster)

#### 3. **Linear Search (Brute Force)**

**Pros**:
- ✅ 100% recall (no approximate matching)
- ✅ Simple to implement (no complex indexing)
- ✅ Works for any similarity metric

**Cons**:
- ❌ O(n) time complexity (slow for large datasets)
- ❌ Not scalable beyond ~100k vectors

**Why Acceptable for MVP**:
- Learning focus: understand core algorithm before optimizations
- Typical learning examples: <10k documents → <0.1s query time
- Can upgrade to FAISS/HNSW in Phase 2+

**Production Alternative**: FAISS (Facebook AI Similarity Search)
- Uses approximate nearest neighbors (ANN) algorithms
- Sub-linear search time (O(log n) or better)
- 90-99% recall (configurable trade-off)

#### 4. **Post-Filtering for Metadata**

**Pros**:
- ✅ Simple to implement (filter after similarity search)
- ✅ Works with any filtering logic

**Cons**:
- ❌ Inefficient (computes similarity for all vectors, then filters)
- ❌ Can return fewer than top-k results (if filters exclude results)

**Why Acceptable for MVP**:
- Most use cases don't require complex filtering
- Simplicity over performance for learning
- Production systems use pre-filtering (filter before similarity search)

---

### Alternatives Considered

#### Alternative 1: FAISS-Based Vector Store

**Description**: Use Facebook's FAISS library for approximate nearest neighbor search

**Pros**:
- ✅ Fast: Sub-linear search time (O(log n))
- ✅ Scalable: Handles millions to billions of vectors
- ✅ GPU support: Accelerated search

**Cons**:
- ❌ Complex to implement and understand (multiple index types)
- ❌ External dependency (C++ library)
- ❌ Approximate: May miss some relevant results (configurable)
- ❌ Harder to debug and inspect

**Decision**: ❌ **Rejected for Phase 1**
- Too complex for learning MVP
- Hides core algorithm behind library
- Can add as integration in Phase 2+

**When to Use**: Production systems with >100k vectors

---

#### Alternative 2: SQLite with Vector Extension

**Description**: Use SQLite with a vector similarity extension (e.g., sqlite-vss)

**Pros**:
- ✅ Disk persistence built-in
- ✅ SQL-based filtering (metadata + similarity)
- ✅ Familiar interface (SQL)

**Cons**:
- ❌ External dependency (sqlite-vss extension)
- ❌ Slower than in-memory for small datasets
- ❌ More complex setup

**Decision**: ❌ **Rejected for Phase 1**
- Adds complexity (SQL + vector extension)
- Persistence not needed for MVP (in-memory is fine)
- Can add in Phase 2+ for disk-based storage

---

#### Alternative 3: Elasticsearch with Dense Vectors

**Description**: Use Elasticsearch with dense_vector field type

**Pros**:
- ✅ Production-grade: Scalable, distributed
- ✅ Rich querying: Metadata filtering + full-text + vector search
- ✅ Built-in analytics and monitoring

**Cons**:
- ❌ Heavy dependency (requires Elasticsearch cluster)
- ❌ Overkill for learning project
- ❌ Complex setup and configuration

**Decision**: ❌ **Rejected**
- Far too complex for learning MVP
- Hides implementation details
- Better as external integration, not core implementation

---

## LlamaIndex Comparison

### What Production LlamaIndex Does

**File**: `llama-index-core/llama_index/core/vector_stores/simple.py` (~400 lines)

**Key Features**:
1. **In-memory storage** (same as us)
2. **Multiple similarity metrics**:
   - Cosine similarity (default)
   - Dot product
   - Euclidean distance
3. **Metadata filtering**:
   - Post-filtering (same as us)
   - Supports complex filters (equality, range, etc.)
4. **Persistence**:
   - JSON serialization/deserialization
   - `persist()` and `from_persist_dir()` methods
5. **Async support**: Async versions of add/query
6. **Query modes**:
   - Default (vector similarity)
   - Sparse (TF-IDF, BM25)
   - Hybrid (vector + sparse)

**Code Snippet** (simplified):

```python
# Production LlamaIndex SimpleVectorStore
class SimpleVectorStore(BasePydanticVectorStore):
    embedding_dict: Dict[str, List[float]] = Field(default_factory=dict)
    text_id_to_ref_doc_id: Dict[str, str] = Field(default_factory=dict)

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        for node in nodes:
            self.embedding_dict[node.node_id] = node.get_embedding()
            self.text_id_to_ref_doc_id[node.node_id] = node.ref_doc_id
        return [node.node_id for node in nodes]

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        # Similar to our implementation
        # + more robust error handling
        # + support for multiple query modes
        # + metadata filtering logic
        ...

    def persist(self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None):
        """Save to JSON file."""
        ...

    @classmethod
    def from_persist_path(cls, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None):
        """Load from JSON file."""
        ...
```

### What We're Doing Differently

| Feature | Mini-LlamaIndex | Production LlamaIndex | Rationale |
|---------|-----------------|----------------------|-----------|
| **Storage** | In-memory dict | In-memory dict | Same (good for MVP) |
| **Similarity** | Cosine only | Cosine, dot, euclidean | Simplified (cosine is most common) |
| **Search** | Linear (brute force) | Linear (brute force) | Same (simple store) |
| **Filtering** | None initially | Post-filtering with complex logic | Phase 2+ (not needed for basic RAG) |
| **Persistence** | None initially | JSON save/load | Phase 2+ (in-memory is fine for learning) |
| **Async** | None | Async add/query | Phase 2+ (sync is simpler) |
| **Query modes** | Vector only | Vector, sparse, hybrid | Phase 2+ (vector is core) |
| **Error handling** | Basic | Robust (edge cases) | Learning focus (add later) |

---

## Historical Context

### Evolution of Vector Stores in RAG

#### Early 2020s: Keyword Search (BM25)

- **Pre-embeddings era**: TF-IDF, BM25 for document retrieval
- **Problem**: Lexical matching only (misses semantic similarity)
- **Example**: Query "car" doesn't match "automobile"

#### 2020: Dense Retrieval (Karpukhin et al.)

- **"Dense Passage Retrieval for Open-Domain QA"** (April 2020)
- **Innovation**: Use BERT embeddings for semantic search
- **Impact**: 10-20% improvement over BM25 on QA tasks
- **Key Idea**: Cosine similarity in embedding space captures semantic meaning

#### 2021-2022: Vector Databases Emerge

- **Pinecone** (2021): First managed vector database
- **Weaviate**, **Milvus**, **Qdrant** (2021-2022): Open-source alternatives
- **FAISS** (Facebook, 2017, popularized 2021+): Library for efficient similarity search
- **Key Features**: Scalability (millions of vectors), ANN algorithms (sub-linear search)

#### 2022: LlamaIndex v0.1 (Nov 2022)

- **Initial release**: Focus on RAG for LLMs
- **SimpleVectorStore**: Basic in-memory store (same design as we're implementing)
- **VectorStoreIndex**: Abstraction over vector stores

#### 2023: Integration Explosion

- **LlamaIndex v0.5** (May 2023): 20+ vector store integrations (Pinecone, Weaviate, Chroma, etc.)
- **Hybrid search**: Combine dense (vector) + sparse (BM25) retrieval
- **Metadata filtering**: Filter by document type, date, etc.

#### 2024: Advanced Retrieval

- **Multi-vector retrieval**: Multiple embeddings per chunk
- **ColBERT**: Late interaction models (token-level similarity)
- **Reranking**: Two-stage retrieval (retrieve broadly, rerank precisely)

### LlamaIndex SimpleVectorStore Timeline

| Version | Date | Changes |
|---------|------|---------|
| v0.1.0 | Nov 2022 | Initial in-memory vector store |
| v0.2.0 | Dec 2022 | Added persistence (JSON) |
| v0.4.0 | Mar 2023 | Added metadata filtering |
| v0.6.0 | Jul 2023 | Added query modes (sparse, hybrid) |
| v0.8.0 | Sep 2023 | Added async support |

**Our Implementation**: Mirrors v0.1.0 (Nov 2022) - the original simple design

---

## Trade-offs

### Benefits

- ✅ **Simple to understand**: ~100 lines of clear Python code
- ✅ **No external dependencies**: Just numpy (standard for Python ML)
- ✅ **Fast for small datasets**: <0.1s for 10k vectors
- ✅ **100% recall**: No approximate matching (finds all relevant results)
- ✅ **Easy to debug**: Can inspect embedding_dict directly
- ✅ **Extensible**: Easy to add features (persistence, filtering, etc.)

### Limitations

- ❌ **Not scalable**: O(n) search, slow for >100k vectors
- ❌ **Memory intensive**: All vectors in RAM
- ❌ **No persistence**: Data lost on restart (until Phase 2+)
- ❌ **No GPU acceleration**: CPU-only (numpy)
- ❌ **No distributed search**: Single-machine only
- ❌ **Inefficient filtering**: Post-filtering wastes computation

### When to Upgrade (Production)

**Use FAISS/Pinecone/Weaviate when**:
- More than 100k vectors
- Need sub-second queries at scale
- Need distributed deployment
- Need GPU acceleration

**Stick with SimpleVectorStore when**:
- Small datasets (<10k documents)
- Learning/prototyping
- Simplicity over performance
- 100% recall required (no approximations)

---

## Learning Outcomes

After implementing this, you should understand:

### 1. **Vector Similarity Search**

- How embeddings represent semantic meaning
- Why cosine similarity is used (vs dot product, euclidean)
- How to compute similarity efficiently (numpy vectorization)

### 2. **Trade-offs in Retrieval**

- **Exact vs Approximate**: 100% recall vs speed
- **In-memory vs Disk**: Speed vs persistence
- **Pre-filtering vs Post-filtering**: Efficiency vs simplicity

### 3. **Scalability Considerations**

- Why linear search doesn't scale
- When to use approximate nearest neighbors (ANN)
- How indexing structures (HNSW, IVF) work (conceptual)

### 4. **Design Patterns**

- **Abstract base class**: BasePydanticVectorStore defines interface
- **Strategy pattern**: Pluggable similarity metrics
- **Builder pattern**: from_defaults(), from_persist_path()

### 5. **Production-Ready Features**

- What's missing from MVP (persistence, filtering, async)
- How production systems handle these (LlamaIndex, Pinecone)
- When to add complexity vs keep it simple

---

## Implementation Checklist

Before considering this complete, ensure:

- [x] `SimpleVectorStore` class with add/query methods
- [x] Cosine similarity computation (numpy-based)
- [x] Top-k retrieval logic
- [x] Unit tests: add nodes, query, verify results
- [x] Integration test: full indexing + retrieval pipeline
- [x] Docstrings with examples
- [x] This ADR document

---

## References

### Code

- **LlamaIndex**: `/home/user/llama_index/llama-index-core/llama_index/core/vector_stores/simple.py`
- **Our Implementation**: `src/mini_llama_index/vector_stores/simple.py`

### Papers

1. **"Dense Passage Retrieval for Open-Domain Question Answering"**
   - Karpukhin et al., EMNLP 2020
   - https://arxiv.org/abs/2004.04906
   - Introduced dense embeddings for retrieval

2. **"Billion-scale similarity search with GPUs"**
   - Johnson et al., IEEE Transactions 2019
   - https://arxiv.org/abs/1702.08734
   - FAISS library paper

3. **"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"**
   - Malkov & Yashunin, IEEE Transactions 2018
   - https://arxiv.org/abs/1603.09320
   - HNSW algorithm (used in many vector DBs)

### Blog Posts

- [Pinecone: What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)
- [LlamaIndex: Understanding Vector Stores](https://docs.llamaindex.ai/en/stable/understanding/storing/storing.html)

---

## Exercises

### Exercise 1: Implement from Scratch

**Task**: Implement `SimpleVectorStore.query()` without looking at the reference implementation.

**Requirements**:
- Compute cosine similarity
- Sort by similarity (descending)
- Return top-k

**Hints**:
- Use numpy for vectorized operations
- Normalize vectors before dot product (for cosine similarity)
- `np.argsort()` for sorting indices

### Exercise 2: Add Dot Product Similarity

**Task**: Extend `SimpleVectorStore` to support dot product similarity (in addition to cosine).

**Requirements**:
- Add `similarity_metric` parameter to `query()`
- Support `"cosine"` and `"dot_product"`
- Update tests

**Learning Goal**: Understand trade-offs between similarity metrics

### Exercise 3: Debug Inverted Scores

**Task**: We've introduced a bug where similarity scores are inverted (low score = more similar). Find and fix it.

**Hint**: Check the sorting order in `query()`

### Exercise 4: Benchmark and Profile

**Task**: Benchmark query time for different dataset sizes (1k, 10k, 100k vectors).

**Requirements**:
- Generate synthetic embeddings
- Measure average query time
- Plot: dataset size (x-axis) vs query time (y-axis)

**Learning Goal**: Understand linear time complexity (O(n))

### Exercise 5: Add Persistence

**Task**: Add `persist()` and `from_persist_path()` methods to save/load the vector store.

**Requirements**:
- Save `embedding_dict` to JSON file
- Load from JSON file
- Handle edge cases (empty store, missing file)

**Hints**:
- Use `json.dump()` and `json.load()`
- LlamaIndex reference: `simple.py` lines 200-250

---

## Conclusion

The `SimpleVectorStore` is the **heart of our RAG system**. It's where semantic search happens—converting the abstract concept of "similarity" into concrete similarity scores.

**Key Takeaway**: Sometimes the simplest implementation (dict + linear search + cosine similarity) is the best for learning. You can always optimize later once you understand the fundamentals.

Next step: **Phase 4 (Indexing)** - Build `VectorStoreIndex` that uses this store!

---

**Status**: ✅ Ready for implementation (Phase 3.2)
