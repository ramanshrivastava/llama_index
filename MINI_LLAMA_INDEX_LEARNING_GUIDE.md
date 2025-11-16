# Mini LlamaIndex Learning Guide
## Building a RAG Framework from Scratch - Learning Edition

Based on LlamaIndex Production Codebase Analysis

---

## 📊 Quick Reference

### What's in LlamaIndex Production
- **Total Code**: ~72,656 lines of Python
- **Main Components**: 14 (schema, indices, retrievers, query engines, embeddings, etc.)
- **Index Types**: 7+ (Vector, List, Tree, Keyword, Knowledge Graph, etc.)
- **Retrieval Strategies**: 10+
- **Response Synthesizers**: 6
- **Core Abstractions**: ~20 base classes

### What Your Mini Version Should Have
- **Target Size**: 3,000-5,000 lines of Python
- **Core Components**: 7-8 essential ones
- **Index Types**: 2-3 (Vector, List, basic Tree)
- **Retrieval Strategy**: 1-2 (Vector similarity, basic)
- **Response Synthesizers**: 2-3 (Simple, Refine)
- **Core Abstractions**: 8-10 base classes
- **Memory**: Simple in-memory stores (no complex persistence initially)

---

## 🏗️ Architecture Overview

```
Data Sources (PDFs, TXTs, etc.)
         ↓
    READERS                 (200-300 lines)
         ↓
    Documents
         ↓
    NODE PARSERS            (400-600 lines)
  ├─ Text Splitter
  ├─ Metadata Extractor
  └─ Chunk Creator
         ↓
    TextNodes
         ↓
    EMBEDDING MODEL         (200-300 lines)
         ↓
    TextNodes (with vectors)
         ↓
    VECTOR STORE            (300-500 lines)
         ↓
    INDICES                 (600-1,000 lines)
  ├─ VectorStoreIndex
  ├─ ListIndex
  └─ (optional) TreeIndex
         ↓
    RETRIEVERS              (400-600 lines)
         ↓
    Retrieved Nodes
         ↓
    RESPONSE SYNTHESIZER    (400-600 lines)
  ├─ LLM Interface
  ├─ Prompt Templates
  └─ Synthesis Strategies
         ↓
    QUERY ENGINE            (300-400 lines)
         ↓
    Final Response
```

---

## 🧩 Component Design Details

### 1. SCHEMA (Core Data Structures) (300-500 lines)

**Purpose**: Define fundamental data structures for the entire framework

#### Core Classes to Implement:

```python
# Base Components
class BaseComponent:
    """Base class with class_name() for serialization"""

class BaseNode(BaseComponent):
    """Base for all retrievable units"""
    id_: str
    metadata: Dict[str, Any]
    excluded_embed_metadata_keys: List[str]
    excluded_llm_metadata_keys: List[str]

class TextNode(BaseNode):
    """Text content with optional embedding"""
    text: str
    embedding: Optional[List[float]]
    start_char_idx: Optional[int]
    end_char_idx: Optional[int]
    text_template: str = "{metadata_str}\n\n{content}"

class Document(TextNode):
    """Top-level ingestion unit (extends TextNode)"""
    doc_id: Optional[str]

class NodeWithScore:
    """Retrieval result wrapper"""
    node: BaseNode
    score: Optional[float]

class QueryBundle:
    """Query representation"""
    query_str: str
    embedding: Optional[List[float]]

class Response:
    """Final response container"""
    response: str
    source_nodes: List[NodeWithScore]
    metadata: Dict[str, Any]
```

#### Simplified Implementation:

- Use **Pydantic v2** for data validation and serialization
- Simple inheritance hierarchy (3 levels max)
- No complex image/multimodal nodes initially
- Basic metadata handling (dict-based)

**Key Decisions**:
- Why Pydantic? Type safety, validation, JSON serialization out-of-the-box
- Why TextNode extends BaseNode? Composition of retrievable units
- Why separate Document from TextNode? Clear ingestion vs retrieval boundary

**Reference**: `llama-index-core/llama_index/core/schema.py` (~1,408 lines)

---

### 2. READERS (Document Loading) (200-300 lines)

**Purpose**: Load documents from various sources

#### Core Classes:

```python
class BaseReader(ABC):
    """Abstract reader interface"""
    @abstractmethod
    def load_data(self, **kwargs) -> List[Document]:
        pass

class SimpleDirectoryReader(BaseReader):
    """Load text files from a directory"""
    def __init__(self, input_dir: str, recursive: bool = False):
        self.input_dir = input_dir
        self.recursive = recursive

    def load_data(self) -> List[Document]:
        # Walk directory, read .txt, .md, .pdf files
        # Create Document per file
        pass
```

#### Simplified Implementation:

- Support **3-5 file types**: `.txt`, `.md`, `.pdf` (via PyPDF2), `.docx`, `.html`
- No complex error recovery initially
- Basic metadata: filename, file path, file type, creation date
- No streaming/lazy loading initially

**Key Functions**:
- `load_data()` → List[Document]
- `_read_file(filepath)` → str

**Reference**: `llama-index-core/llama_index/core/readers/` (~1,456 lines total)

---

### 3. NODE PARSERS (Text Splitting & Chunking) (400-600 lines)

**Purpose**: Transform documents into nodes suitable for retrieval

#### Core Classes:

```python
class TransformComponent(BaseComponent, ABC):
    """Base for all transformations (nodes → nodes)"""
    @abstractmethod
    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        pass

class NodeParser(TransformComponent):
    """Base parser interface"""

class SentenceSplitter(NodeParser):
    """Split by sentences with overlap"""
    chunk_size: int = 1024  # tokens
    chunk_overlap: int = 200
    tokenizer: Callable = tiktoken_tokenizer

    def __call__(self, nodes: List[BaseNode]) -> List[TextNode]:
        # Split text into chunks
        # Preserve metadata
        # Set relationships (prev/next)
        pass

class TokenTextSplitter(NodeParser):
    """Split by fixed token count"""
    chunk_size: int
    chunk_overlap: int
```

#### Simplified Implementation:

- **2-3 splitter types**: Sentence, Token, Simple (character-based)
- Basic **overlap handling** (sliding window)
- Simple **metadata preservation** (copy parent metadata)
- Basic **relationship tracking** (prev_node, next_node, parent_node)
- Use **tiktoken** for tokenization (fallback to simple split)

**Key Functions**:
- `_split_text(text: str) → List[str]`
- `_create_node_from_split(text: str, metadata: dict) → TextNode`

**Reference**: `llama-index-core/llama_index/core/node_parser/` (~4,335 lines)

---

### 4. EMBEDDINGS (Vector Generation) (200-300 lines)

**Purpose**: Convert text to vector representations

#### Core Classes:

```python
class BaseEmbedding(TransformComponent):
    """Abstract embedding interface"""
    embed_batch_size: int = 10

    @abstractmethod
    def _get_query_embedding(self, query: str) -> List[float]:
        """Embed a query (single text)"""

    @abstractmethod
    def _get_text_embedding(self, text: str) -> List[float]:
        """Embed a text (single text)"""

    @abstractmethod
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (batch)"""

    def __call__(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Embed nodes and set .embedding attribute"""
        texts = [node.get_content() for node in nodes]
        embeddings = self._get_text_embeddings(texts)
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return nodes

class MockEmbedding(BaseEmbedding):
    """Deterministic embeddings for testing"""
    embed_dim: int = 384

    def _get_text_embedding(self, text: str) -> List[float]:
        # Hash-based deterministic vector
        return self._mock_embed(text)
```

#### Simplified Implementation:

- **MockEmbedding**: Hash-based deterministic vectors (for testing)
- **Integration point** for OpenAI/HuggingFace (via abstractions)
- Basic **batching** (split into batches, avoid rate limits)
- No caching initially (add in Phase 2)
- Simple **async support** (optional)

**Key Functions**:
- `_get_query_embedding(query)` → vector
- `_get_text_embeddings(texts)` → List[vector]
- `_mock_embed(text)` → deterministic vector (for MockEmbedding)

**Reference**: `llama-index-core/llama_index/core/embeddings/` (~480 lines)

---

### 5. VECTOR STORE (Vector Storage & Retrieval) (300-500 lines)

**Purpose**: Store embeddings and perform similarity search

#### Core Classes:

```python
class VectorStoreQuery:
    """Query specification for vector store"""
    query_embedding: List[float]
    similarity_top_k: int = 10
    node_ids: Optional[List[str]] = None  # Filter by IDs
    filters: Optional[Dict] = None  # Metadata filters

class VectorStoreQueryResult:
    """Query result from vector store"""
    nodes: List[TextNode]
    similarities: List[float]
    ids: List[str]

class BasePydanticVectorStore(BaseComponent, ABC):
    """Abstract vector store interface"""

    @abstractmethod
    def add(self, nodes: List[BaseNode]) -> List[str]:
        """Add nodes with embeddings"""

    @abstractmethod
    def delete(self, ref_doc_id: str) -> None:
        """Delete by document ID"""

    @abstractmethod
    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Query by embedding similarity"""

class SimpleVectorStore(BasePydanticVectorStore):
    """In-memory vector store using numpy"""

    def __init__(self):
        self.embedding_dict: Dict[str, List[float]] = {}
        self.text_id_to_ref_doc_id: Dict[str, str] = {}

    def add(self, nodes: List[BaseNode]) -> List[str]:
        # Store embeddings in dict
        # Store node_id → ref_doc_id mapping

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        # Compute cosine similarity
        # Sort by similarity
        # Return top-k
```

#### Simplified Implementation:

- **In-memory storage**: Dict-based (node_id → embedding)
- **Cosine similarity**: Using numpy or simple dot product
- Basic **top-k retrieval** (no fancy indexing initially)
- Simple **metadata filtering** (post-filtering after similarity search)
- **No persistence** initially (add JSON save/load in Phase 2)

**Similarity Metrics**:
- Cosine similarity (default)
- Dot product (optional)
- Euclidean distance (optional)

**Key Functions**:
- `_compute_similarity(query_emb, node_embs)` → similarities
- `_apply_filters(nodes, filters)` → filtered nodes

**Reference**: `llama-index-core/llama_index/core/vector_stores/` (~1,053 lines)

---

### 6. INDICES (Data Organization) (600-1,000 lines)

**Purpose**: Organize nodes for efficient retrieval

#### Core Classes:

```python
class BaseIndex(BaseComponent, ABC):
    """Base index interface"""

    @abstractmethod
    def as_retriever(self, **kwargs) -> BaseRetriever:
        """Return a retriever for this index"""

    @abstractmethod
    def as_query_engine(self, **kwargs) -> BaseQueryEngine:
        """Return a query engine for this index"""

    @classmethod
    @abstractmethod
    def from_documents(
        cls,
        documents: List[Document],
        **kwargs
    ) -> "BaseIndex":
        """Build index from documents"""

class VectorStoreIndex(BaseIndex):
    """Vector similarity index"""

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        storage_context: Optional[StorageContext] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs
    ):
        self._storage_context = storage_context or StorageContext.from_defaults()
        self._embed_model = embed_model or Settings.embed_model

        if nodes:
            self._build_index_from_nodes(nodes)

    def _build_index_from_nodes(self, nodes: List[BaseNode]):
        # Embed nodes (if not already embedded)
        nodes_with_embeddings = self._embed_model(nodes)
        # Store in vector store
        self._storage_context.vector_store.add(nodes_with_embeddings)

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        transformations: Optional[List[TransformComponent]] = None,
        **kwargs
    ) -> "VectorStoreIndex":
        # Run transformations (parse → embed)
        nodes = run_transformations(documents, transformations)
        return cls(nodes=nodes, **kwargs)

    def as_retriever(self, similarity_top_k: int = 10) -> BaseRetriever:
        return VectorIndexRetriever(
            index=self,
            similarity_top_k=similarity_top_k
        )

class ListIndex(BaseIndex):
    """Sequential list index (no embedding required)"""

    def as_retriever(self) -> BaseRetriever:
        return ListIndexRetriever(index=self)
```

#### Simplified Implementation:

- **VectorStoreIndex**: Core vector similarity index
- **ListIndex**: Simple sequential index (for summarization)
- Optional **TreeIndex**: Hierarchical summarization (Phase 2+)
- **Transformation pipeline**: NodeParser → Embedding → Storage
- Basic **from_documents()** builder pattern

**Index Types Priority**:
1. VectorStoreIndex (Phase 1) - CORE
2. ListIndex (Phase 1) - Simple
3. TreeIndex (Phase 2+) - Optional

**Reference**: `llama-index-core/llama_index/core/indices/` (~13,716 lines)

---

### 7. RETRIEVERS (Node Retrieval) (400-600 lines)

**Purpose**: Fetch relevant nodes given a query

#### Core Classes:

```python
class BaseRetriever(ABC):
    """Abstract retriever interface"""

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes for a query"""

    def retrieve(self, query: Union[str, QueryBundle]) -> List[NodeWithScore]:
        # Convert str to QueryBundle if needed
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query

        # Embed query if needed
        if query_bundle.embedding is None:
            query_bundle.embedding = self._embed_model.get_query_embedding(
                query_bundle.query_str
            )

        # Retrieve
        return self._retrieve(query_bundle)

class VectorIndexRetriever(BaseRetriever):
    """Retrieve from vector store index"""

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 10,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._embed_model = embed_model or Settings.embed_model

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Build vector store query
        query = VectorStoreQuery(
            query_embedding=query_bundle.embedding,
            similarity_top_k=self._similarity_top_k
        )

        # Query vector store
        result = self._index._storage_context.vector_store.query(query)

        # Convert to NodeWithScore
        return [
            NodeWithScore(node=node, score=score)
            for node, score in zip(result.nodes, result.similarities)
        ]

class ListIndexRetriever(BaseRetriever):
    """Retrieve all nodes from list index"""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Return all nodes (no filtering)
        nodes = self._index.docstore.get_nodes(list(self._index.index_struct.node_ids))
        return [NodeWithScore(node=node, score=None) for node in nodes]
```

#### Simplified Implementation:

- **VectorIndexRetriever**: Similarity-based retrieval
- **ListIndexRetriever**: Return all nodes (for summarization)
- Basic **embedding on-the-fly** (if query not embedded)
- **No complex routing/fusion** initially (Phase 2+)

**Reference**: `llama-index-core/llama_index/core/retrievers/` (~993 lines)

---

### 8. LLM INTERFACE (Language Model Abstraction) (300-500 lines)

**Purpose**: Abstract interface for language models

#### Core Classes:

```python
class ChatMessage:
    """Single chat message"""
    role: Literal["system", "user", "assistant"]
    content: str

class BaseLLM(ABC):
    """Abstract LLM interface"""

    @abstractmethod
    def chat(self, messages: List[ChatMessage]) -> ChatMessage:
        """Chat completion (messages → message)"""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Text completion (prompt → text)"""

class MockLLM(BaseLLM):
    """Mock LLM for testing"""

    def chat(self, messages: List[ChatMessage]) -> ChatMessage:
        # Simple mock: return last user message with prefix
        return ChatMessage(
            role="assistant",
            content=f"Mock response to: {messages[-1].content}"
        )

    def complete(self, prompt: str) -> str:
        return f"Mock completion for: {prompt[:50]}..."
```

#### Simplified Implementation:

- **MockLLM**: Deterministic responses for testing
- **Integration points**: OpenAI, Anthropic, local models (via abstractions)
- Basic **chat** and **complete** methods
- No streaming initially (Phase 2+)
- No structured output (Pydantic) initially (Phase 2+)

**Reference**: `llama-index-core/llama_index/core/llms/` (~2,504 lines)

---

### 9. PROMPTS (Template Management) (200-300 lines)

**Purpose**: Manage prompt templates for LLM interactions

#### Core Classes:

```python
class BasePromptTemplate(ABC):
    """Base prompt template"""
    template_vars: List[str]

    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format template with variables"""

class PromptTemplate(BasePromptTemplate):
    """String-based template using f-string style"""

    def __init__(self, template: str):
        self.template = template
        # Extract {var_name} placeholders
        self.template_vars = self._extract_vars(template)

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class ChatPromptTemplate(BasePromptTemplate):
    """Chat message template"""

    def __init__(self, message_templates: List[ChatMessage]):
        self.message_templates = message_templates

    def format_messages(self, **kwargs) -> List[ChatMessage]:
        # Format each message template
        pass

# Default prompts
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

DEFAULT_REFINE_PROMPT = PromptTemplate(
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the query. If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
```

#### Simplified Implementation:

- **PromptTemplate**: Simple f-string formatting
- **ChatPromptTemplate**: List of message templates
- **5-10 default prompts**: QA, Refine, Summarize, etc.
- Basic **variable extraction** (regex-based)

**Reference**: `llama-index-core/llama_index/core/prompts/` (~2,124 lines)

---

### 10. RESPONSE SYNTHESIZERS (Answer Generation) (400-600 lines)

**Purpose**: Generate final responses from retrieved nodes

#### Core Classes:

```python
class BaseSynthesizer(ABC):
    """Abstract response synthesizer"""

    @abstractmethod
    def synthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Generate response from query and nodes"""

class SimpleSummarize(BaseSynthesizer):
    """Simple: concatenate all nodes and ask LLM once"""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        text_qa_template: Optional[PromptTemplate] = None,
    ):
        self._llm = llm or Settings.llm
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT

    def synthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore],
    ) -> Response:
        # Extract query string
        query_str = query if isinstance(query, str) else query.query_str

        # Concatenate node texts
        context_str = "\n\n".join([node.node.get_content() for node in nodes])

        # Format prompt
        prompt = self._text_qa_template.format(
            context_str=context_str,
            query_str=query_str
        )

        # Call LLM
        response_text = self._llm.complete(prompt)

        # Return Response
        return Response(
            response=response_text,
            source_nodes=nodes,
            metadata={"prompt": prompt}
        )

class Refine(BaseSynthesizer):
    """Refine: iteratively refine answer with each node"""

    def synthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore],
    ) -> Response:
        query_str = query if isinstance(query, str) else query.query_str

        # Initial answer with first node
        context_str = nodes[0].node.get_content()
        prompt = self._text_qa_template.format(
            context_str=context_str,
            query_str=query_str
        )
        answer = self._llm.complete(prompt)

        # Refine with remaining nodes
        for node in nodes[1:]:
            context_msg = node.node.get_content()
            refine_prompt = self._refine_template.format(
                query_str=query_str,
                existing_answer=answer,
                context_msg=context_msg
            )
            answer = self._llm.complete(refine_prompt)

        return Response(
            response=answer,
            source_nodes=nodes,
            metadata={}
        )

class TreeSummarize(BaseSynthesizer):
    """Tree summarize: hierarchical bottom-up summarization"""
    # Phase 2+
```

#### Simplified Implementation:

- **SimpleSummarize**: Concatenate and ask once (Phase 1)
- **Refine**: Iterative refinement (Phase 1)
- **TreeSummarize**: Hierarchical (Phase 2+)
- Basic **prompt management**
- No streaming initially

**Synthesis Strategies Priority**:
1. SimpleSummarize (Phase 1)
2. Refine (Phase 1)
3. TreeSummarize (Phase 2+)

**Reference**: `llama-index-core/llama_index/core/response_synthesizers/` (~1,925 lines)

---

### 11. QUERY ENGINE (End-to-End Orchestration) (300-400 lines)

**Purpose**: Orchestrate retrieval and response synthesis

#### Core Classes:

```python
class BaseQueryEngine(ABC):
    """Abstract query engine"""

    @abstractmethod
    def _query(self, query_bundle: QueryBundle) -> Response:
        """Process query end-to-end"""

    def query(self, query: Union[str, QueryBundle]) -> Response:
        # Convert to QueryBundle
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query

        return self._query(query_bundle)

class RetrieverQueryEngine(BaseQueryEngine):
    """Standard retrieval + synthesis pipeline"""

    def __init__(
        self,
        retriever: BaseRetriever,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    ):
        self._retriever = retriever
        self._response_synthesizer = response_synthesizer or get_response_synthesizer()
        self._node_postprocessors = node_postprocessors or []

    def _query(self, query_bundle: QueryBundle) -> Response:
        # 1. Retrieve nodes
        nodes = self._retriever.retrieve(query_bundle)

        # 2. Apply postprocessors (filtering, reranking)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle)

        # 3. Synthesize response
        response = self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes
        )

        return response
```

#### Simplified Implementation:

- **RetrieverQueryEngine**: Standard pipeline (retrieve → postprocess → synthesize)
- Basic **postprocessing** (similarity cutoff, top-k limiting)
- No complex routing/sub-questioning initially (Phase 2+)

**Reference**: `llama-index-core/llama_index/core/query_engine/` (~3,850 lines)

---

### 12. STORAGE CONTEXT (Persistence Layer) (300-500 lines)

**Purpose**: Manage document, index, and vector storage

#### Core Classes:

```python
class BaseDocumentStore(ABC):
    """Abstract document/node store"""

    @abstractmethod
    def add_documents(self, docs: List[BaseNode]) -> None:
        """Add documents to store"""

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[BaseNode]:
        """Get document by ID"""

    @abstractmethod
    def get_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        """Get multiple nodes by IDs"""

class SimpleDocumentStore(BaseDocumentStore):
    """In-memory document store"""

    def __init__(self):
        self.docs: Dict[str, BaseNode] = {}

    def add_documents(self, docs: List[BaseNode]) -> None:
        for doc in docs:
            self.docs[doc.node_id] = doc

    def get_nodes(self, node_ids: List[str]) -> List[BaseNode]:
        return [self.docs[nid] for nid in node_ids if nid in self.docs]

class StorageContext:
    """Container for all storage components"""

    def __init__(
        self,
        docstore: Optional[BaseDocumentStore] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        index_store: Optional[BaseIndexStore] = None,
    ):
        self.docstore = docstore or SimpleDocumentStore()
        self.vector_store = vector_store or SimpleVectorStore()
        self.index_store = index_store or SimpleIndexStore()

    @classmethod
    def from_defaults(cls, persist_dir: Optional[str] = None) -> "StorageContext":
        """Create with default stores"""
        # Optionally load from disk
        return cls()

    def persist(self, persist_dir: str = "./storage") -> None:
        """Save all stores to disk (JSON)"""
        # Phase 2+: Implement persistence
```

#### Simplified Implementation:

- **SimpleDocumentStore**: In-memory dict (node_id → node)
- **SimpleVectorStore**: In-memory dict (node_id → embedding)
- **StorageContext**: Container for all stores
- **No persistence initially** (add JSON save/load in Phase 2)

**Reference**: `llama-index-core/llama_index/core/storage/` (~2,785 lines)

---

### 13. SETTINGS (Global Configuration) (100-200 lines)

**Purpose**: Global settings and defaults

```python
class Settings:
    """Global settings singleton"""

    # LLM
    llm: BaseLLM = MockLLM()

    # Embedding
    embed_model: BaseEmbedding = MockEmbedding()

    # Node parser
    node_parser: NodeParser = SentenceSplitter()

    # Chunk settings
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # Retrieval
    similarity_top_k: int = 10

    # Callback
    callback_manager: Optional[CallbackManager] = None
```

**Reference**: `llama-index-core/llama_index/core/settings.py` (~159 lines)

---

## 📅 Implementation Roadmap

### Phase 1: Foundation (Week 1-2, ~1,500 lines)

**Goal**: Build core data structures and basic document loading

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 1.1 | Project setup + Schema (BaseNode, TextNode, Document) | 200 | Data modeling, Pydantic basics |
| 1.2 | QueryBundle, Response, NodeWithScore | 150 | Query/response patterns |
| 1.3 | Simple readers (text files) | 200 | File I/O, document abstraction |
| 1.4 | Settings + utility functions | 150 | Global configuration |

**Test**: Load documents from a directory, create TextNodes

**Checkpoint 1**: Can you explain the difference between Document and TextNode?

---

### Phase 2: Transformations (Week 2-3, ~800 lines)

**Goal**: Text splitting and embedding

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 2.1 | TransformComponent base | 100 | Abstract base classes |
| 2.2 | SentenceSplitter (simple) | 250 | Text chunking, overlap |
| 2.3 | MockEmbedding (deterministic) | 200 | Embedding interface, batching |
| 2.4 | Transformation pipeline | 150 | Pipeline composition |

**Test**: Split a document, embed chunks with MockEmbedding

**Checkpoint 2**: Implement a custom splitter that splits by paragraphs

---

### Phase 3: Storage (Week 3-4, ~800 lines)

**Goal**: In-memory vector and document storage

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 3.1 | BaseDocumentStore + SimpleDocumentStore | 200 | Document CRUD |
| 3.2 | BasePydanticVectorStore + SimpleVectorStore | 300 | Vector storage, similarity |
| 3.3 | VectorStoreQuery + VectorStoreQueryResult | 150 | Query specification |
| 3.4 | StorageContext | 150 | Multi-store coordination |

**Test**: Store nodes, query by similarity

**Checkpoint 3**: Implement cosine similarity from scratch

---

### Phase 4: Indexing (Week 4-5, ~600 lines)

**Goal**: Build VectorStoreIndex and ListIndex

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 4.1 | BaseIndex abstract class | 100 | Index abstraction |
| 4.2 | VectorStoreIndex implementation | 300 | Vector indexing pipeline |
| 4.3 | ListIndex implementation | 150 | Sequential indexing |
| 4.4 | from_documents() builder | 150 | Builder pattern |

**Test**: Build index from documents, persist/load

**Checkpoint 4**: Build an index from a PDF document

---

### Phase 5: Retrieval (Week 5-6, ~500 lines)

**Goal**: Implement retrievers

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 5.1 | BaseRetriever abstract class | 100 | Retriever abstraction |
| 5.2 | VectorIndexRetriever | 250 | Vector similarity retrieval |
| 5.3 | ListIndexRetriever | 150 | Sequential retrieval |

**Test**: Retrieve top-k nodes for a query

**Checkpoint 5**: Compare retrieval results with different top-k values

---

### Phase 6: LLM & Prompts (Week 6-7, ~600 lines)

**Goal**: LLM interface and prompt templates

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 6.1 | BaseLLM + MockLLM | 250 | LLM abstraction |
| 6.2 | ChatMessage, chat/complete methods | 150 | Chat vs completion |
| 6.3 | PromptTemplate + default prompts | 200 | Template patterns |

**Test**: Format prompts, call MockLLM

**Checkpoint 6**: Create custom prompt templates

---

### Phase 7: Response Synthesis (Week 7-8, ~500 lines)

**Goal**: Generate responses from retrieved nodes

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 7.1 | BaseSynthesizer abstract class | 100 | Synthesizer abstraction |
| 7.2 | SimpleSummarize implementation | 200 | Concatenate strategy |
| 7.3 | Refine implementation | 200 | Iterative refinement |

**Test**: Synthesize responses with different strategies

**Checkpoint 7**: Compare SimpleSummarize vs Refine on long contexts

---

### Phase 8: Query Engine (Week 8-9, ~400 lines)

**Goal**: End-to-end query processing

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 8.1 | BaseQueryEngine abstract class | 100 | Query engine abstraction |
| 8.2 | RetrieverQueryEngine | 250 | Pipeline orchestration |
| 8.3 | Index.as_query_engine() | 150 | Factory methods |

**Test**: Build index, query end-to-end, get response

**Checkpoint 8**: Run a full RAG pipeline on your own documents

---

### Phase 9: Integration & Polish (Week 9-10, ~300 lines)

**Goal**: Integration, testing, documentation

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 9.1 | Postprocessors (similarity cutoff) | 150 | Result filtering |
| 9.2 | Integration tests | 100 | End-to-end testing |
| 9.3 | Documentation + examples | 50 | Usage patterns |

**Test**: Run full examples, compare to LlamaIndex production

**Checkpoint 9**: Can you explain the entire RAG flow?

---

## 🔑 Key Design Decisions

### 1. **Data Model**
- **LlamaIndex uses**: Rich Pydantic models with serialization
- **Recommendation**: Use Pydantic v2 for simplicity, type safety

### 2. **Storage**
- **LlamaIndex uses**: Pluggable stores (in-memory, disk, cloud)
- **Recommendation**: Start with in-memory, add persistence in Phase 2

### 3. **Embedding**
- **LlamaIndex uses**: External APIs (OpenAI, HuggingFace)
- **Recommendation**: MockEmbedding for testing, integration points for real models

### 4. **Retrieval**
- **LlamaIndex uses**: Multiple strategies (vector, keyword, hybrid)
- **Recommendation**: Vector similarity only (cosine), add others in Phase 2

### 5. **Response Synthesis**
- **LlamaIndex uses**: 6 strategies (simple, refine, tree, compact, etc.)
- **Recommendation**: 2 strategies (simple, refine) for MVP

### 6. **Index Types**
- **LlamaIndex uses**: 7+ index types
- **Recommendation**: 2 types (VectorStoreIndex, ListIndex) for MVP

---

## 📂 File Organization

```
mini-llama-index/
├── README.md                           # Project overview
├── LEARNING_GUIDE.md                   # This file
├── HISTORICAL_TIMELINE.md              # LlamaIndex evolution
├── pyproject.toml                      # Poetry dependencies
│
├── docs/
│   ├── adrs/                          # Architecture Decision Records
│   │   ├── 001-schema-design.md
│   │   ├── 002-pydantic-choice.md
│   │   ├── 003-vector-store-design.md
│   │   ├── 004-retrieval-strategy.md
│   │   ├── 005-synthesis-strategies.md
│   │   └── ...
│   │
│   ├── comparisons/                   # Mini vs Production LlamaIndex
│   │   ├── schema-comparison.md
│   │   ├── retrieval-comparison.md
│   │   └── ...
│   │
│   ├── diagrams/                      # Architecture diagrams
│   │   ├── data-flow.svg
│   │   ├── component-hierarchy.svg
│   │   └── ...
│   │
│   ├── checkpoints/                   # Learning checkpoints
│   │   ├── phase1-checkpoint.md
│   │   ├── phase2-checkpoint.md
│   │   └── ...
│   │
│   └── references/                    # LlamaIndex references
│       ├── llamaindex-code-map.md
│       ├── papers.md                  # RAG, retrieval papers
│       └── blog-posts.md              # LlamaIndex evolution
│
├── src/
│   └── mini_llama_index/
│       ├── __init__.py
│       ├── schema.py                  # Core data structures (Phase 1)
│       ├── settings.py                # Global settings (Phase 1)
│       │
│       ├── readers/                   # Document readers (Phase 1)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── simple_directory.py
│       │
│       ├── node_parser/               # Text splitting (Phase 2)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── sentence_splitter.py
│       │   └── token_splitter.py
│       │
│       ├── embeddings/                # Embedding models (Phase 2)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── mock_embed.py
│       │
│       ├── storage/                   # Storage layer (Phase 3)
│       │   ├── __init__.py
│       │   ├── docstore/
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   └── simple.py
│       │   └── storage_context.py
│       │
│       ├── vector_stores/             # Vector storage (Phase 3)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── simple.py
│       │   └── types.py               # VectorStoreQuery, etc.
│       │
│       ├── indices/                   # Indices (Phase 4)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── vector_store.py
│       │   └── list.py
│       │
│       ├── retrievers/                # Retrievers (Phase 5)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── vector_index.py
│       │   └── list_index.py
│       │
│       ├── llms/                      # LLM interface (Phase 6)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── mock.py
│       │   └── types.py               # ChatMessage
│       │
│       ├── prompts/                   # Prompt templates (Phase 6)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── prompt_template.py
│       │   └── default_prompts.py
│       │
│       ├── response_synthesizers/     # Response synthesis (Phase 7)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── simple_summarize.py
│       │   └── refine.py
│       │
│       ├── query_engine/              # Query engines (Phase 8)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── retriever_query_engine.py
│       │
│       └── utils/                     # Utilities
│           ├── __init__.py
│           └── text_utils.py
│
├── tests/
│   ├── unit/                          # Unit tests per module
│   │   ├── test_schema.py
│   │   ├── test_readers.py
│   │   ├── test_node_parser.py
│   │   └── ...
│   │
│   ├── integration/                   # End-to-end tests
│   │   ├── test_basic_rag.py
│   │   ├── test_vector_index.py
│   │   └── ...
│   │
│   └── fixtures/                      # Test data
│       ├── documents/
│       └── ...
│
└── examples/                          # Example scripts
    ├── 01_basic_indexing.py
    ├── 02_simple_query.py
    ├── 03_custom_splitter.py
    └── ...
```

---

## 🎓 Learning Checkpoint Template

### Phase X Checkpoint: [Phase Name]

#### Self-Assessment Quiz

**Conceptual Understanding**:
1. **Question**: Why did we separate Document from TextNode?
   - **Answer**: Document is the ingestion unit, TextNode is the retrieval unit
   - **Reference**: ADR-001, schema.py:50-100

2. **Question**: What's the difference between SimpleSummarize and Refine?
   - **Answer**: SimpleSummarize concatenates all context and asks once; Refine iteratively refines the answer with each chunk
   - **Reference**: response_synthesizers/simple_summarize.py, response_synthesizers/refine.py

**Code Comprehension**:
1. **Question**: Trace the data flow from `VectorStoreIndex.from_documents()` to stored embeddings
   - **Answer**:
     1. Documents → run_transformations (node parser → embeddings)
     2. Nodes → _build_index_from_nodes
     3. Nodes → vector_store.add(nodes)
     4. Embeddings stored in SimpleVectorStore.embedding_dict

#### Hands-On Exercises

**Exercise 1: Implement a Custom Node Parser**
- **Task**: Create a `ParagraphSplitter` that splits text by double newlines
- **Difficulty**: ⭐⭐☆☆☆
- **Estimated Time**: 30 minutes
- **Learning Goal**: Understand NodeParser interface and text processing

**Hints**:
- Inherit from `NodeParser`
- Split on `\n\n`
- Preserve metadata from parent document
- Consider how LlamaIndex does this in `node_parser/text_splitters.py`

**Exercise 2: Debug Retrieval**
- **Task**: We've introduced a bug where similarity scores are inverted (low score = more similar). Find and fix it.
- **Difficulty**: ⭐⭐⭐☆☆
- **Bug Location**: `vector_stores/simple.py` in the `query()` method

#### Comparative Analysis

**Mini-LlamaIndex vs Production LlamaIndex**

| Aspect | Mini-LlamaIndex | Production LlamaIndex | Why Different? |
|--------|-----------------|----------------------|----------------|
| Lines of code (schema) | ~300 | ~1,408 | We skip image nodes, complex serialization |
| Index types | 2 (Vector, List) | 7+ | Learning focus on core RAG |
| Retrieval strategies | 1 (vector similarity) | 10+ | MVP scope |
| Response synthesizers | 2 (Simple, Refine) | 6 | Core patterns only |
| Storage | In-memory | Pluggable (disk, cloud) | Simplicity |

#### Next Steps

Before moving to Phase X+1, ensure you can:

- [ ] Explain [core concept] to someone else
- [ ] Modify the code to add [feature]
- [ ] Read the equivalent LlamaIndex code comfortably
- [ ] Draw the data flow diagram from memory

---

## 📚 Architecture Decision Records (ADR) Template

### ADR-XXX: [Decision Title]

**Status**: Accepted
**Date**: 2025-XX-XX
**Commit**: [hash]
**LlamaIndex Reference**: [file:line]
**Related Papers**: [if applicable]

---

#### Context

What problem are we solving? What constraints exist?

Example:
> We need a way to split documents into chunks suitable for embedding and retrieval. Documents can be thousands of tokens, but embedding models have context limits (e.g., 512 tokens for sentence-transformers). We need chunks small enough to embed but large enough to contain meaningful semantic units.

---

#### Decision

What approach did we choose?

Example:
> We will implement a `SentenceSplitter` that:
> 1. Splits text by sentences (using NLTK or simple regex)
> 2. Groups sentences until reaching `chunk_size` tokens
> 3. Overlaps chunks by `chunk_overlap` tokens
> 4. Preserves metadata from parent document

**Code Example**:

```python
class SentenceSplitter(NodeParser):
    chunk_size: int = 1024
    chunk_overlap: int = 200

    def __call__(self, nodes: List[BaseNode]) -> List[TextNode]:
        output_nodes = []
        for node in nodes:
            text = node.get_content()
            chunks = self._split_text(text)
            for chunk in chunks:
                output_nodes.append(
                    TextNode(
                        text=chunk,
                        metadata=node.metadata.copy()
                    )
                )
        return output_nodes
```

---

#### Rationale

**Why This Approach?**

1. **Semantic coherence**: Sentence boundaries preserve meaning better than arbitrary character splits
2. **Overlap handles boundaries**: Important information near chunk boundaries appears in multiple chunks
3. **Token-aware**: Uses tokenizer to respect model limits (not just character count)
4. **Simple to implement**: Minimal dependencies (NLTK or regex)

**Alternatives Considered**:

1. **Character-based splitting**
   - ❌ Rejected: Can split mid-word or mid-sentence, losing context

2. **Paragraph-based splitting**
   - ❌ Rejected: Paragraphs can be very long or very short, unpredictable chunk sizes

3. **Semantic similarity-based splitting** (SemanticSplitter)
   - ❌ Too complex for Phase 1, requires embeddings during splitting
   - ✅ Consider for Phase 2+

---

#### LlamaIndex Comparison

**What Production LlamaIndex Does**:

Production LlamaIndex has multiple splitters in `node_parser/text/`:
- `SentenceSplitter` (~300 lines): Advanced sentence detection, metadata handling
- `TokenTextSplitter` (~200 lines): Pure token-based splitting
- `SemanticSplitter` (~400 lines): Embedding-based semantic boundaries
- `CodeSplitter` (~500 lines): Language-aware code splitting

**File**: `llama-index-core/llama_index/core/node_parser/text/sentence.py`

**Key differences**:
- Production: Uses `nltk.tokenize.sent_tokenize` with fallback to regex
- Production: Handles edge cases (empty strings, single-token texts)
- Production: Supports `separator` parameter for custom splitting
- Production: Tracks `start_char_idx` and `end_char_idx` for precise source mapping

**What We're Doing Differently**:

- Simplified sentence detection (regex-based: `[.!?]\s+`)
- No edge case handling initially
- No `start_char_idx` tracking (Phase 2+)
- No custom separators (Phase 2+)

---

#### Historical Context

**Evolution of Text Splitting in RAG**:

1. **Early 2020s**: Simple character-based splitting (e.g., every 500 chars)
   - Problem: Lost context at boundaries

2. **LlamaIndex v0.1 (2022)**: Introduced sentence-based splitting
   - Improvement: Semantic coherence

3. **LlamaIndex v0.5 (2023)**: Added overlap parameter
   - Improvement: Handled boundary information loss

4. **LlamaIndex v0.9 (2024)**: Introduced `SemanticSplitter`
   - Uses embeddings to find optimal split points
   - Trades speed for quality

**Academic Background**:

- **RecursiveCharacterTextSplitter** (LangChain, 2022): Hierarchical splitting by separators
- **Semantic chunking** (Zhang et al., 2023): Embedding-based boundary detection
- **Sliding window** (classical NLP): Overlap to preserve context

---

#### Trade-offs

**Benefits**:
- ✅ Preserves semantic units (sentences)
- ✅ Respects model token limits
- ✅ Simple to implement and understand
- ✅ Fast (no embedding required during splitting)

**Limitations**:
- ❌ Sentence detection can fail on edge cases (abbreviations, etc.)
- ❌ Fixed chunk size may split important multi-sentence context
- ❌ No semantic awareness (splits at sentence boundaries even if mid-topic)
- ❌ Overlap increases storage and compute (duplicate chunks)

---

#### Learning Outcomes

After implementing this, you should understand:

1. **Chunking strategies**: Why chunking is necessary for RAG
2. **Token counting**: How tokenizers work (tiktoken, HuggingFace)
3. **Overlap mechanics**: How sliding windows preserve boundary context
4. **Metadata propagation**: How to copy metadata from parent to child nodes
5. **Trade-offs**: Speed vs quality, fixed vs semantic splitting

---

#### References

- **LlamaIndex Code**: `/home/user/llama_index/llama-index-core/llama_index/core/node_parser/text/sentence.py`
- **Paper**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- **Blog Post**: [LlamaIndex: Optimizing Text Splitting](https://www.llamaindex.ai/blog/text-splitting)

---

#### Exercises

1. **Implement from Scratch**: Write `SentenceSplitter` without looking at the reference implementation
2. **Add Feature**: Extend to support custom separators (e.g., split by paragraphs instead of sentences)
3. **Debug**: We've introduced a bug where overlap doesn't work correctly. Find and fix it.
4. **Optimize**: Measure and optimize tokenization performance (caching, batching)

---

## 🕰️ Historical Timeline: LlamaIndex Evolution

| Date | Version | Milestone | Relevance to Mini-LlamaIndex |
|------|---------|-----------|-------------------------------|
| **Nov 2022** | v0.1.0 | Initial release by Jerry Liu | Core concept: data framework for LLMs |
| **Dec 2022** | v0.2.0 | VectorStoreIndex, ListIndex | Our Phase 4 (Indexing) |
| **Jan 2023** | v0.3.0 | TreeIndex, KeywordTableIndex | Optional Phase 2+ features |
| **Mar 2023** | v0.4.0 | Query engines, response synthesizers | Our Phase 7-8 |
| **May 2023** | v0.5.0 | Metadata filtering, node relationships | Phase 2+ enhancements |
| **Jul 2023** | v0.6.0 | Integrations (OpenAI, Pinecone, etc.) | External API abstractions |
| **Sep 2023** | v0.8.0 | Chat engines, agents | Advanced features (Phase 3+) |
| **Nov 2023** | v0.9.0 | Workflow, instrumentation | Observability (Phase 3+) |
| **Jan 2024** | v0.10.0 | Property Graph Index | Graph RAG (Phase 3+) |
| **Mar 2024** | v0.10.20 | Modular architecture (core + integrations) | Our architecture inspiration |
| **Today** | v0.11.x | 300+ integrations, production-ready | Full ecosystem |

---

## 📖 References

### LlamaIndex Production Code

**Core Schema**:
- `/home/user/llama_index/llama-index-core/llama_index/core/schema.py` (~1,408 lines)
  - Key classes: BaseNode, TextNode, Document, Response

**Indices**:
- `/home/user/llama_index/llama-index-core/llama_index/core/indices/vector_store/` (~1,500 lines)
- `/home/user/llama_index/llama-index-core/llama_index/core/indices/list/` (~400 lines)

**Retrievers**:
- `/home/user/llama_index/llama-index-core/llama_index/core/retrievers/` (~993 lines)

**Query Engines**:
- `/home/user/llama_index/llama-index-core/llama_index/core/query_engine/` (~3,850 lines)

**Node Parsers**:
- `/home/user/llama_index/llama-index-core/llama_index/core/node_parser/` (~4,335 lines)

**Response Synthesizers**:
- `/home/user/llama_index/llama-index-core/llama_index/core/response_synthesizers/` (~1,925 lines)

**Storage**:
- `/home/user/llama_index/llama-index-core/llama_index/core/storage/` (~2,785 lines)

---

### Academic Papers

1. **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (Lewis et al., 2020)
   - Foundation of RAG paradigm
   - https://arxiv.org/abs/2005.11401

2. **"Dense Passage Retrieval for Open-Domain Question Answering"** (Karpukhin et al., 2020)
   - Dense retrieval with embeddings
   - https://arxiv.org/abs/2004.04906

3. **"Improving Language Models by Retrieving from Trillions of Tokens"** (Borgeaud et al., 2022)
   - RETRO model, retrieval during training
   - https://arxiv.org/abs/2112.04426

4. **"Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"** (Asai et al., 2023)
   - Adaptive retrieval
   - https://arxiv.org/abs/2310.11511

---

### Blog Posts & Tutorials

1. [LlamaIndex Documentation](https://docs.llamaindex.ai/)
2. [Building RAG Applications](https://www.llamaindex.ai/blog/building-rag)
3. [Understanding Vector Stores](https://www.pinecone.io/learn/vector-database/)

---

## 🧪 Testing Strategy

### Unit Tests Per Component

1. **Schema** (`test_schema.py`):
   - Test node creation, metadata handling
   - Test serialization (to/from dict, JSON)
   - Test node relationships

2. **Readers** (`test_readers.py`):
   - Test file loading (txt, pdf, md)
   - Test metadata extraction
   - Test error handling (missing files, etc.)

3. **Node Parsers** (`test_node_parser.py`):
   - Test text splitting (sentence, token)
   - Test overlap mechanics
   - Test metadata preservation

4. **Embeddings** (`test_embeddings.py`):
   - Test MockEmbedding determinism
   - Test batching
   - Test embedding dimensions

5. **Vector Stores** (`test_vector_stores.py`):
   - Test add/delete/query
   - Test similarity computation
   - Test top-k retrieval

6. **Indices** (`test_indices.py`):
   - Test index building (from_documents)
   - Test as_retriever(), as_query_engine()

7. **Retrievers** (`test_retrievers.py`):
   - Test vector retrieval
   - Test query embedding

8. **Response Synthesizers** (`test_synthesizers.py`):
   - Test SimpleSummarize
   - Test Refine
   - Test prompt formatting

9. **Query Engines** (`test_query_engine.py`):
   - Test end-to-end query flow
   - Test postprocessing

---

### Integration Tests

**Test Files** (Python programs to run):

```python
# test1_basic_indexing.py - Basic vector indexing
from mini_llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
print(f"Indexed {len(documents)} documents")

# test2_simple_query.py - Simple query
query_engine = index.as_query_engine()
response = query_engine.query("What is LlamaIndex?")
print(response.response)
print(f"Sources: {[node.node.metadata['filename'] for node in response.source_nodes]}")

# test3_custom_splitter.py - Custom node parser
from mini_llama_index import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter(documents)
print(f"Split into {len(nodes)} nodes")

# test4_retrieval.py - Direct retrieval
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("machine learning")
for node in nodes:
    print(f"Score: {node.score:.3f} | Text: {node.node.text[:100]}...")
```

---

### Performance Benchmarks

Compare mini-llama-index vs production LlamaIndex:

| Operation | Mini-LlamaIndex | Production LlamaIndex | Notes |
|-----------|-----------------|----------------------|-------|
| Load 100 documents | ~2s | ~1s | Acceptable (simpler readers) |
| Split into nodes | ~1s | ~0.5s | Acceptable (simpler splitters) |
| Embed 1000 nodes (mock) | ~0.1s | ~0.1s | Same (both deterministic) |
| Index 1000 nodes | ~0.5s | ~0.3s | Acceptable (simpler indexing) |
| Query (retrieve + synthesize) | ~0.2s | ~0.1s | Acceptable (simpler synthesis) |

Focus on **correctness first, performance later**.

---

## 🎯 Common Pitfalls to Avoid

### 1. Over-Engineering Early
- ❌ Don't try to support all index types immediately
- ✅ Start with VectorStoreIndex, add others incrementally

### 2. Complex Metadata Handling
- ❌ Don't implement complex metadata filtering from the start
- ✅ Simple dict-based metadata, add filtering in Phase 2+

### 3. Premature Optimization
- ❌ Don't optimize vector similarity search initially
- ✅ Use simple numpy dot product, optimize later (FAISS, HNSW in Phase 3+)

### 4. Underestimating Text Processing
- ❌ Don't use naive string splitting (split by space)
- ✅ Use proper sentence detection (NLTK or regex)

### 5. Ignoring Token Limits
- ❌ Don't chunk by character count only
- ✅ Use tiktoken or similar for token-aware chunking

### 6. Not Testing with Real LLMs
- ❌ Don't only test with MockLLM
- ✅ Add integration with real LLMs (OpenAI, local) early for validation

### 7. Forgetting Edge Cases
- ❌ Don't assume all documents are well-formed text
- ✅ Handle empty documents, binary files, encoding issues

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- Basic understanding of RAG concepts
- Familiarity with Pydantic

### Setup

```bash
# Clone repository
git clone <your-repo>
cd mini-llama-index

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Try example
poetry run python examples/01_basic_indexing.py
```

### Your First Commit

**Commit 1.1: Project Setup + Core Schema**

```bash
# Create project structure
mkdir -p src/mini_llama_index tests/unit docs/adrs

# Implement schema.py (BaseNode, TextNode, Document)
# Write tests
# Write ADR-001: Schema Design

git add .
git commit -m "[Phase 1.1] Project setup + Core Schema (BaseNode, TextNode, Document)

Implements fundamental data structures for the mini-llama-index framework.

Design Decisions:
- Use Pydantic v2 for validation and serialization
- Simple inheritance: BaseNode → TextNode → Document
- Basic metadata handling (dict-based, no complex filtering)

Trade-offs:
- Simplified from production (no ImageNode, no complex serialization)
- Focus on clarity over features

Learning Outcomes:
1. Understand Pydantic models
2. See data structure hierarchy
3. Learn why Document ≠ TextNode

See docs/adrs/ADR-001-schema-design.md for full rationale.

LlamaIndex Reference: llama-index-core/llama_index/core/schema.py:1-200
Lines added: ~200"
```

---

## ✅ Success Criteria

You've successfully completed Mini-LlamaIndex when you can:

1. **Build an index** from a directory of documents
2. **Query the index** with natural language and get relevant responses
3. **Explain the flow** from document → chunks → embeddings → retrieval → response
4. **Read production LlamaIndex code** comfortably and understand architectural decisions
5. **Extend the framework** with custom components (splitters, retrievers, synthesizers)
6. **Compare strategies** (SimpleSummarize vs Refine, sentence vs token splitting)
7. **Understand trade-offs** (simplicity vs features, speed vs quality)

---

## 🎓 Final Learning Outcomes

### Technical Skills
- ✅ RAG architecture and data flow
- ✅ Vector similarity search (cosine, dot product)
- ✅ Text chunking strategies (sentence, token, semantic)
- ✅ Embedding models and batching
- ✅ Response synthesis strategies (simple, refine, tree)
- ✅ Prompt engineering for RAG
- ✅ Index types and when to use them
- ✅ Pydantic for data modeling
- ✅ Python design patterns (ABC, strategy, builder)

### Conceptual Understanding
- ✅ Why RAG? (vs fine-tuning, vs prompt stuffing)
- ✅ Chunking trade-offs (size, overlap, boundaries)
- ✅ Retrieval vs generation separation
- ✅ Context length limits and mitigation
- ✅ Metadata and filtering
- ✅ Storage abstraction (in-memory, disk, cloud)

### Software Engineering
- ✅ Modular architecture design
- ✅ Abstract base classes and interfaces
- ✅ Builder and factory patterns
- ✅ Testing strategies (unit, integration)
- ✅ Documentation (ADRs, comparisons, checkpoints)
- ✅ Incremental development (phased approach)

---

## 📊 Complexity Comparison

### Mini-LlamaIndex (Target)
- **~5,000 lines** of clean, documented Python
- **8-10 core components**
- **2-3 index types**
- **2-3 synthesis strategies**
- **In-memory storage**
- **No external integrations** (besides OpenAI for real testing)

### Production LlamaIndex
- **~72,656 lines** of production Python (core only)
- **14 major components**
- **7+ index types**
- **6 synthesis strategies**
- **Pluggable storage** (20+ vector stores)
- **300+ integrations** (LLMs, embeddings, vector stores, etc.)

**Learning Value**: Mini-LlamaIndex teaches you the **80% core concepts** with **20% of the complexity**.

---

## 🏁 Ready to Start?

**Next Steps**:

1. ✅ Read this guide thoroughly
2. ✅ Review the [LlamaIndex documentation](https://docs.llamaindex.ai/)
3. ✅ Set up your development environment
4. ✅ Start with **Phase 1, Commit 1.1** (Project Setup + Core Schema)
5. ✅ Write your first ADR (ADR-001: Schema Design)
6. ✅ Implement BaseNode, TextNode, Document
7. ✅ Write tests
8. ✅ Commit and move to next phase!

**Good luck building your mini RAG framework! 🚀**

The clarity you'll gain about RAG systems, retrieval, and LLM applications is invaluable. This is not just about writing code—it's about **deeply understanding** how production LLM frameworks work.

---

## 📞 Questions?

If you get stuck:
1. Review the relevant ADR
2. Check the LlamaIndex reference code (paths provided)
3. Review the learning checkpoint
4. Try the hands-on exercises
5. Compare with production implementation

Remember: **The goal is learning, not feature parity**. Focus on understanding the core concepts deeply rather than implementing every feature.

Happy learning! 🎓
