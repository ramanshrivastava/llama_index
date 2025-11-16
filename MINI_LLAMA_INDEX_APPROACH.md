# Mini LlamaIndex - Reasoning-Based Learning Approach

## 🎯 Project Overview

**Mini LlamaIndex** is a learning-focused, from-scratch implementation of a RAG (Retrieval-Augmented Generation) framework, inspired by the production [LlamaIndex](https://github.com/run-llama/llama_index) codebase.

### Why This Project?

Building a simplified version of LlamaIndex from scratch helps you:

1. **Deeply understand RAG systems** - Not just use them, but understand *how* and *why* they work
2. **Learn production patterns** - See real-world design patterns (Strategy, Builder, ABC, etc.) in action
3. **Master embeddings & retrieval** - Understand vector similarity, chunking strategies, and retrieval mechanisms
4. **Explore LLM orchestration** - See how to compose LLMs, retrievers, and synthesizers
5. **Appreciate production complexity** - Understand what production frameworks handle (edge cases, optimizations, integrations)

---

## 📐 Approach: Learning-First, Reasoning-Based

This project follows a **historically-grounded, reasoning-based learning framework** inspired by the mini-CPython learning approach. Every commit is a learning experience.

### Core Principles

#### 1. **Architecture Decision Records (ADRs)**

Every significant design decision is documented with:
- **Context**: What problem are we solving?
- **Decision**: What approach did we choose?
- **Rationale**: Why this over alternatives?
- **LlamaIndex Reference**: How does production LlamaIndex handle this?
- **Trade-offs**: What did we gain/lose?
- **Learning Outcomes**: What should you understand after this?

**Example**: [ADR-003: Vector Store Design](docs/adrs/003-vector-store-design.md)

#### 2. **Historical Timeline**

Each commit maps to LlamaIndex's evolution:
- **Commit 1.1** (Schema) → LlamaIndex v0.1.0 (Nov 2022) - Initial data structures
- **Commit 4.2** (VectorStoreIndex) → LlamaIndex v0.2.0 (Dec 2022) - Vector indexing
- **Commit 7.2** (SimpleSummarize) → LlamaIndex v0.4.0 (Mar 2023) - Response synthesis

This helps you understand **when** and **why** features were added to the real framework.

#### 3. **Learning Checkpoints**

After each phase, test your understanding with:
- **Conceptual Quizzes**: Can you explain the design decisions?
- **Hands-On Exercises**: Extend the feature (e.g., add a custom splitter)
- **Debugging Challenges**: Fix intentional bugs
- **Code Reading**: Study equivalent LlamaIndex code
- **Performance Analysis**: Understand bottlenecks and optimizations

#### 4. **Comparative Analysis**

Side-by-side comparison of:
- **Mini-LlamaIndex**: Our simplified implementation
- **Production LlamaIndex**: Real production code
- **Why Different?**: Explicit trade-offs (simplicity vs features)

Example:

| Aspect | Mini-LlamaIndex | Production LlamaIndex | Why Different? |
|--------|-----------------|----------------------|----------------|
| Lines (schema) | ~300 | ~1,408 | We skip image nodes, complex serialization |
| Index types | 2 (Vector, List) | 7+ | Learning focus on core RAG |

#### 5. **Phased Implementation**

9 phases, 20+ commits, each commit is a working, tested state:

```
Phase 1: Foundation      (Schema, Readers)
Phase 2: Transformations (Node Parsers, Embeddings)
Phase 3: Storage         (Vector Store, Document Store)
Phase 4: Indexing        (VectorStoreIndex, ListIndex)
Phase 5: Retrieval       (Retrievers)
Phase 6: LLM & Prompts   (LLM Interface, Templates)
Phase 7: Synthesis       (Response Synthesizers)
Phase 8: Query Engine    (End-to-End Orchestration)
Phase 9: Integration     (Polish, Docs, Examples)
```

---

## 📊 Scope: What We Build vs What We Skip

### ✅ What We Build (MVP - 5,000 lines)

**Core RAG Pipeline**:
- ✅ Document loading (txt, pdf, md)
- ✅ Text chunking (sentence, token splitters)
- ✅ Embeddings (mock + integration points for real models)
- ✅ Vector storage (in-memory, cosine similarity)
- ✅ Vector indexing (VectorStoreIndex, ListIndex)
- ✅ Retrieval (vector similarity, top-k)
- ✅ Response synthesis (SimpleSummarize, Refine)
- ✅ Query engine (orchestration)
- ✅ LLM interface (mock + integration points)
- ✅ Prompt templates

**Data Structures**:
- ✅ BaseNode, TextNode, Document
- ✅ QueryBundle, Response, NodeWithScore
- ✅ VectorStoreQuery, VectorStoreQueryResult

**Design Patterns**:
- ✅ Abstract Base Classes (ABC)
- ✅ Strategy Pattern (multiple synthesis strategies)
- ✅ Builder Pattern (from_documents, from_defaults)
- ✅ Composition (QueryEngine = Retriever + Synthesizer)

### ❌ What We Skip (Phase 2+ / Advanced)

**Complex Features**:
- ❌ Advanced index types (KnowledgeGraph, PropertyGraph, Tree)
- ❌ Multi-modal (images, audio)
- ❌ Agents and tool calling
- ❌ Chat engines (stateful conversations)
- ❌ Streaming responses
- ❌ Structured output (Pydantic objects from LLM)
- ❌ Complex postprocessing (reranking, MMR)
- ❌ Metadata filtering (advanced queries)
- ❌ Callbacks and instrumentation (observability)

**Infrastructure**:
- ❌ Disk persistence (initially in-memory only)
- ❌ Cloud integrations (Pinecone, Weaviate, etc.)
- ❌ Production optimizations (FAISS, HNSW)
- ❌ Distributed indexing
- ❌ Async/streaming (initially sync only)

**Edge Cases**:
- ❌ Complex error recovery
- ❌ Unicode/encoding edge cases
- ❌ Large document handling (>100MB)
- ❌ Rate limiting, retries

**Rationale**: Focus on **core concepts** deeply rather than **feature breadth**. You can add advanced features once you understand the fundamentals.

---

## 🧩 Architecture: Component Overview

### Data Flow (End-to-End RAG)

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PHASE                         │
└─────────────────────────────────────────────────────────────┘

   Documents (PDFs, TXTs, etc.)
           ↓
   [SimpleDirectoryReader]  ← Load documents
           ↓
   Document[]
           ↓
   [SentenceSplitter]       ← Split into chunks
           ↓
   TextNode[] (no embeddings yet)
           ↓
   [Embedding Model]        ← Generate embeddings
           ↓
   TextNode[] (with embeddings)
           ↓
   [VectorStore.add()]      ← Store vectors
   [DocumentStore.add()]    ← Store nodes
           ↓
   VectorStoreIndex (persisted)


┌─────────────────────────────────────────────────────────────┐
│                      QUERY PHASE                            │
└─────────────────────────────────────────────────────────────┘

   User Query ("What is RAG?")
           ↓
   [QueryBundle]            ← Wrap query
           ↓
   [Embedding Model]        ← Embed query
           ↓
   QueryBundle (with query embedding)
           ↓
   [VectorIndexRetriever]   ← Similarity search
           ↓
   NodeWithScore[] (top-k nodes)
           ↓
   [Postprocessor]          ← Filter/rerank (optional)
           ↓
   NodeWithScore[] (refined)
           ↓
   [ResponseSynthesizer]    ← Generate response
           ↓ (combines query + nodes → prompt)
   [LLM]                    ← Generate answer
           ↓
   Response (text + sources + metadata)
```

### Component Hierarchy

```
BaseComponent (Pydantic base with serialization)
  │
  ├─ BaseNode (retrievable units)
  │   ├─ TextNode (text + optional embedding)
  │   │   └─ Document (top-level ingestion unit)
  │   └─ IndexNode (references other nodes)
  │
  ├─ TransformComponent (nodes → nodes)
  │   ├─ NodeParser (chunking)
  │   │   ├─ SentenceSplitter
  │   │   └─ TokenTextSplitter
  │   └─ BaseEmbedding (text → vectors)
  │       └─ MockEmbedding
  │
  ├─ BaseRetriever (query → nodes)
  │   ├─ VectorIndexRetriever
  │   └─ ListIndexRetriever
  │
  ├─ BaseQueryEngine (query → response)
  │   └─ RetrieverQueryEngine
  │
  ├─ BaseSynthesizer (query + nodes → response)
  │   ├─ SimpleSummarize
  │   └─ Refine
  │
  ├─ BaseLLM (prompts → text)
  │   └─ MockLLM
  │
  ├─ BaseIndex (organize nodes for retrieval)
  │   ├─ VectorStoreIndex
  │   └─ ListIndex
  │
  ├─ BasePydanticVectorStore (CRUD for vectors)
  │   └─ SimpleVectorStore
  │
  └─ BaseDocumentStore (CRUD for documents/nodes)
      └─ SimpleDocumentStore
```

---

## 📂 Project Structure

```
mini-llama-index/
├── README.md                           # Quick start
├── LEARNING_GUIDE.md                   # Full learning guide (this doc)
├── HISTORICAL_TIMELINE.md              # LlamaIndex evolution
├── pyproject.toml                      # Poetry dependencies
│
├── docs/
│   ├── adrs/                          # Architecture Decision Records
│   │   ├── 001-schema-design.md
│   │   ├── 002-pydantic-choice.md
│   │   ├── 003-vector-store-design.md
│   │   └── ...
│   │
│   ├── comparisons/                   # Mini vs Production
│   │   ├── schema-comparison.md
│   │   ├── retrieval-comparison.md
│   │   └── ...
│   │
│   ├── checkpoints/                   # Learning checkpoints
│   │   ├── phase1-checkpoint.md
│   │   ├── phase2-checkpoint.md
│   │   └── ...
│   │
│   └── references/                    # External references
│       ├── llamaindex-references.md
│       └── papers.md
│
├── src/mini_llama_index/              # Main source code
│   ├── schema.py                      # Core data structures
│   ├── settings.py                    # Global settings
│   ├── readers/                       # Document readers
│   ├── node_parser/                   # Text splitters
│   ├── embeddings/                    # Embedding models
│   ├── vector_stores/                 # Vector storage
│   ├── indices/                       # Indices
│   ├── retrievers/                    # Retrievers
│   ├── llms/                          # LLM interface
│   ├── prompts/                       # Prompt templates
│   ├── response_synthesizers/         # Response synthesis
│   ├── query_engine/                  # Query engines
│   └── storage/                       # Storage context
│
├── tests/
│   ├── unit/                          # Unit tests
│   ├── integration/                   # End-to-end tests
│   └── fixtures/                      # Test data
│
└── examples/                          # Example scripts
    ├── 01_basic_indexing.py
    ├── 02_simple_query.py
    └── ...
```

---

## 🎓 Learning Methodology

### For Each Phase

1. **Read the Phase Description** in the learning guide
2. **Review the ADR(s)** for that phase (understand *why* before *what*)
3. **Implement the code** following TDD (tests first when possible)
4. **Compare with LlamaIndex** production code (see how they did it)
5. **Complete the checkpoint** (quiz, exercises, debugging)
6. **Commit** with a descriptive message (see commit template below)
7. **Move to next phase**

### Commit Message Template

```
[Phase X.Y] Title - Brief description

Detailed description of what this commit implements.

Design Decisions:
- Decision 1: [rationale]
- Decision 2: [rationale]

Trade-offs:
- We simplified [X] because [Y]
- We kept [A] to preserve [B]

Learning Outcomes:
1. Understand [concept]
2. See how [feature] works

LlamaIndex Reference: [file:line]
ADR: docs/adrs/ADR-XXX.md
Lines added: ~XXX
```

### Example Commit

```
[Phase 3.2] SimpleVectorStore - In-memory vector storage with cosine similarity

Implements an in-memory vector store using numpy for similarity computation.

Design Decisions:
- Use dict-based storage (node_id → embedding) for simplicity
- Cosine similarity as default metric (most common for embeddings)
- Post-filtering for metadata (not during similarity search)

Trade-offs:
- Simplified: No FAISS/HNSW indexing (linear search only)
- Kept: Pluggable similarity metric (easy to extend)

Learning Outcomes:
1. Understand vector similarity search (cosine, dot product)
2. See trade-offs: accuracy vs speed (linear vs indexed search)
3. Learn why metadata filtering is expensive

LlamaIndex Reference: llama-index-core/llama_index/core/vector_stores/simple.py
ADR: docs/adrs/ADR-003-vector-store-design.md
Lines added: ~300
```

---

## 📊 Success Metrics

### Technical Mastery

After completing Mini-LlamaIndex, you should be able to:

- [ ] **Explain RAG** to a non-technical person
- [ ] **Diagram the data flow** from document → response from memory
- [ ] **Implement a custom component** (splitter, retriever, synthesizer) without guidance
- [ ] **Debug retrieval issues** (e.g., why are my results bad?)
- [ ] **Choose chunking strategies** for different use cases
- [ ] **Optimize embeddings** (batching, caching)
- [ ] **Compare synthesis strategies** (when to use SimpleSummarize vs Refine)
- [ ] **Read LlamaIndex production code** comfortably

### Conceptual Understanding

- [ ] **Why chunking?** (embedding limits, semantic coherence)
- [ ] **Why overlap?** (preserve boundary context)
- [ ] **Why vector similarity?** (semantic search vs keyword search)
- [ ] **Why RAG vs fine-tuning?** (cost, flexibility, freshness)
- [ ] **Why multiple synthesis strategies?** (trade-offs: speed vs quality)
- [ ] **Why metadata?** (filtering, attribution, debugging)

### Engineering Practices

- [ ] **Modular design** (composition over inheritance)
- [ ] **Abstract base classes** (interfaces, protocols)
- [ ] **Builder pattern** (from_documents, from_defaults)
- [ ] **Strategy pattern** (pluggable components)
- [ ] **Testing** (unit, integration, fixtures)
- [ ] **Documentation** (ADRs, docstrings, examples)

---

## 🔗 References

### LlamaIndex Production

- **Repository**: https://github.com/run-llama/llama_index
- **Documentation**: https://docs.llamaindex.ai/
- **Code Reference**: `/home/user/llama_index/llama-index-core/`

### Papers

1. [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401) - Lewis et al., 2020
2. [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) - Karpukhin et al., 2020
3. [Self-RAG](https://arxiv.org/abs/2310.11511) - Asai et al., 2023

### Additional Resources

- [Pinecone: Vector Databases](https://www.pinecone.io/learn/vector-database/)
- [LlamaIndex Blog](https://www.llamaindex.ai/blog)
- [Anthropic: Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Poetry (or pip with virtualenv)
- Basic knowledge of Python, Pydantic, numpy
- Familiarity with LLMs and embeddings (conceptual)

### Installation

```bash
# Clone this repository
git clone <your-repo>
cd mini-llama-index

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Try first example
poetry run python examples/01_basic_indexing.py
```

### Your First Task

**Implement Phase 1, Commit 1.1: Core Schema**

1. Read `docs/adrs/001-schema-design.md` (you'll write this)
2. Implement `src/mini_llama_index/schema.py`:
   - `BaseComponent`
   - `BaseNode`
   - `TextNode`
   - `Document`
3. Write tests in `tests/unit/test_schema.py`
4. Commit with descriptive message
5. Move to Phase 1, Commit 1.2

---

## 🎯 Final Thoughts

**This is not a typical coding project.** It's a **learning journey** where you:

1. **Build something real** (a working RAG framework)
2. **Understand production systems** (by comparing to LlamaIndex)
3. **Make deliberate decisions** (documented in ADRs)
4. **Learn by doing** (checkpoints, exercises, debugging)

**Focus on depth, not speed.** Take time to:
- Understand *why* before *what*
- Read the production code
- Complete the checkpoints
- Experiment and break things

By the end, you'll have:
- ✅ A working RAG framework (~5,000 lines)
- ✅ Deep understanding of retrieval and generation
- ✅ Production-level design patterns
- ✅ Confidence to build LLM applications

**Good luck, and enjoy the learning process! 🎓🚀**

---

## 📧 Feedback & Contributions

This is a living learning guide. If you:
- Find errors or unclear explanations
- Have suggestions for better learning exercises
- Want to contribute additional ADRs or comparisons

Please open an issue or submit a PR!

---

**Happy Learning! 🦙📚**
