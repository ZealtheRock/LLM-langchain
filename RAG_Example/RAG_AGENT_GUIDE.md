# Agentic RAG demo

This example is a fictional internal-policy assistant for Northstar Demo. It combines
retrieval-augmented generation (RAG) with a LangChain agent that decides when to call
the `search_knowledge_base` tool. The example uses OpenAI for chat and embeddings and
Chroma for a persistent local vector store.

## Run it

From the repository root, create and activate a virtual environment if needed, then
install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Add your API key to the root
`.env` file:

```dotenv
OPENAI_API_KEY=your-key
```

The key is read by the process as a normal environment variable; `.env` is plain text,
so do not commit it. The default models are `text-embedding-3-small` and
`gpt-4o-mini`; override the chat model with `CHAT_MODEL` if needed.

Build or refresh the local vector store, then ask a question:

```bash
python RAG_Example/rag_agent_demo.py ingest
python RAG_Example/rag_agent_demo.py ask "I took a taxi on an approved work trip. What do I need to submit?"
```

The indexed Chroma files are stored under `RAG_Example/.rag_chroma/`. Ingestion
rebuilds this demo's collection from the dataset, so it is safe to rerun after editing
the data.
Embeddings are generated through the OpenAI API; the first ingestion and each later
re-ingestion require network access and incur embedding usage.

## What happens

1. `RAG_Example/knowledge_base/company_policies.json` is the small, synthetic source dataset.
   In a real system, retrieve current, access-controlled records from approved sources
   such as a document repository, database, or internal API. Track source and version
   information; do not ingest content the user is not authorized to see.
2. The loader validates each entry and creates LangChain `Document` objects with
   document ID, title, category, effective date, source, and related document IDs.
3. `RecursiveCharacterTextSplitter` creates overlapping chunks. The example starts
   with 900-character chunks and 150-character overlap. These are tuning starting
   points, not universal settings; measure retrieval quality on representative queries.
4. `OpenAIEmbeddings` encodes chunks with `text-embedding-3-small`. Chroma persists
   vectors and metadata locally, using cosine distance for its HNSW approximate
   nearest-neighbor index.
5. At question time, the agent can call the vector search tool. It uses maximal
   marginal relevance (MMR) to select relevant, less-redundant chunks and follows
   `related_doc_ids` to retrieve a small number of linked policy passages.
6. Retrieved passages and their source fields go back to the agent, which writes an
   answer with citations. The prompt tells it to abstain when the evidence is
   insufficient; this is a useful guard, not a guarantee. Evaluate answers and sources.

## Embeddings: what algorithm and how to choose

An embedding model is not a clustering algorithm. It maps text into vectors; a vector
index then finds nearby vectors. Start with a current, maintained model that supports
your languages and domain. `text-embedding-3-small` is a practical cost-conscious
baseline; compare it with a larger or domain-appropriate model using your own evaluation
set before choosing. Use the same model and dimensions at ingestion and query time.
Changing models or dimensions requires re-embedding the collection.

There is no best universal distance metric. The demo configures Chroma's HNSW index
with cosine distance, a common choice for semantic text search. Match the metric to
the embedding model's guidance and validate it with measured retrieval results. Do not
confuse vector-index algorithms (such as HNSW) with embedding models.

Best practices:

- Keep a representative set of questions with expected source documents. Measure
  Recall@k / Precision@k or MRR for retrieval and answer faithfulness/citation accuracy
  for generation.
- Compare chunk sizes, overlap, embedding models, metadata filters, and `k` using that
  set. Avoid selecting settings only because one example query looks good.
- Use the same embedding model for documents and queries. Keep model name and vector
  dimensions as part of the collection versioning/rebuild process.
- Use metadata filtering for access control, tenant boundaries, document type, and
  effective dates; do not rely on semantic similarity to enforce permissions.
- Re-embed after meaningful source or embedding-model changes; deduplicate and track
  source versions so stale chunks do not remain searchable.

## Choosing a vector database

This demo uses local Chroma because it is easy to run and persist for a small prototype.
For a real deployment, compare options against:

- **Scale and operations:** expected vector count, write/update rate, latency, high
  availability, backups, and whether you want managed service or self-hosting.
- **Retrieval needs:** metadata filters, hybrid keyword + vector search, reranking,
  namespaces/tenants, index tuning, and deletion/update semantics.
- **Security and governance:** encryption, access controls, data residency, audit logs,
  retention/deletion requirements, and your provider's data-processing terms.
- **Cost and fit:** storage, compute, embedding API cost, network egress, and how well
  it integrates with existing databases and operational skills.

Keep the vector database behind an application service in production. A local
`RAG_Example/.rag_chroma/` directory is not a multi-user database, a backup, or an
authorization boundary.

## Storing data and relationships

Store chunk text, its embedding, and compact filterable metadata together. This demo
uses `doc_id`, `title`, `category`, `effective_date`, `source`, `chunk_index`,
`chunk_count`, `start_index`, and `related_doc_ids`. Stable parent document IDs tie
chunks back to a source. The JSON dataset's `related_doc_ids` links policies; retrieval
expands those links after finding a primary chunk.

Use explicit IDs and idempotent ingestion. The demo rebuilds its small collection on
each ingestion to avoid duplicates and obsolete chunks. For production, use source
versioning, transactional or staged rebuilds, update/delete by source, and backups.
Relationships in metadata are references—not a graph database. If you need multi-hop
entity relationships or graph traversal, store canonical entities/edges in a
relational or graph database and use the vector index for semantic discovery.

## Retrieval choices

The example uses MMR: it first considers a larger candidate set (`fetch_k=12`), then
returns up to four relevant but less redundant chunks. It also adds at most two
explicitly linked passages. Other approaches include:

- Similarity search for a simple nearest-neighbor baseline.
- Hybrid retrieval (BM25/keyword plus vectors), combined for example with reciprocal
  rank fusion, when exact names, IDs, or rare terms matter.
- Metadata filters to narrow by authorized tenant, source, category, or date.
- A reranker over a wider candidate set when top-k ordering needs improvement.
- Query rewriting or decomposition for ambiguous and multi-part questions.

Select retrieval strategy with a held-out query set. Rerankers and hybrid search add
latency and complexity; use them when evaluation shows a measurable benefit.

## Processing the LLM output

Pass retrieved passages with source metadata as tool output/context. In this example,
the agent responds with citations and prints the final answer. In a production app,
validate structured output against a schema where downstream systems depend on it,
verify each citation points to a retrieved source, handle refusals/timeouts/rate limits,
and record retrieval IDs and model versions for debugging. Do not treat generated text
as a verified policy decision or perform consequential actions without authorization
and human review.

## References

- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [Chroma collection configuration and HNSW](https://docs.trychroma.com/docs/collections/configure)
- [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)
