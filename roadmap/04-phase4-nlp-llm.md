# Phase 4: NLP & LLMs

## 4.1 Language Model Prototyping
**Why it matters:** LLMs are central to modern AI products.
**What it is:** tokenization, prompts, fine-tuning/evaluation.
**Goal:** working LLM prototype with prompt engineering and metrics.

### Steps
1. Choose model (e.g., Llama, GPT, Mistral).
2. Implement input pipeline, chunking, and batch inference.
3. Evaluate quality and latency.
4. Document prompts, failures, and mitigation.

### Status
- [ ] planned
- [ ] in progress
- [ ] done

### Resources
- `nlp/llm/prototype.py`
- `nlp/llm/prompts.md`

---

## 4.2 Retrieval-Augmented Generation (RAG)
**Why it matters:** improves relevance and facts in responses.
**What it is:** embeddings + vector store + retriever + LLM prompt.
**Goal:** end-to-end RAG system demo with canned documents.

### Steps
1. Generate embeddings from docs.
2. Store vectors in Pinecone/FAISS/etc.
3. Build retrieval + LLM response composition.
4. Validate with target use cases.

### Status
- [ ] planned
- [ ] in progress
- [ ] done

### Resources
- `nlp/rag/README.md`
