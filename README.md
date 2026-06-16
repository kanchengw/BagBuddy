
## Network Requirement

BagBuddy relies on a Cloudflare Workers proxy (`workers.dev`) for Supabase database access, Stripe payments, and Langfuse observability. In some regions (notably mainland China), `workers.dev` may be blocked or unreliable.

- If you experience timeouts or connection errors, **use a VPN** or ensure your network can reach `workers.dev`.

## BagBuddy - AI Shopping Assistant

AI-powered e-commerce agent with product search, RAG, multi-turn tool calling, Stripe payments, and LLM-based evaluation.

Note that this project is built entirely on the self-developed framework CNLLM, instead of OpenAI, LiteLLM, or LangChain, for LLM invocation and workflow orchestration.
Learn more about CNLLM at https://github.com/kanchengw/cnllm.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: fill in LLM_API_KEY (your Qwen DashScope key)
python -X utf8 main.py
```

Open http://localhost:9000. 

## Zero Config: Proxy Architecture

All service credentials (Stripe secret key, Langfuse API keys, Supabase service role key) are stored server-side in a **Cloudflare Worker**, VPN is required to access the proxy.

The worker at `https://bagbuddy-proxy.wangkancheng1122.workers.dev` proxies:
- **Supabase** - REST API (product queries, pgvector search)
- **Stripe** - Checkout session creation, account info, webhooks
- **Langfuse** - Trace ingestion and dashboard queries

Users only need to provide their own **LLM API key** in .env. The .env.public (committed to git) contains just the proxy URL and app config - no secrets.

## Architecture

The agent runs a multi-turn tool loop (up to 5 rounds per user message):

1. **RAG** - user message embedded via CNLLM API (text-embedding-v3), top-3 products retrieved from pgvector
2. **LLM call** (streaming) - model decides to respond or invoke a tool
3. **Tool execution** - search_products, get_product_details, compare_products, text_to_sql, or create_checkout_session
4. **ContextBox** - tool results fed back into message history via CNLLM
5. **Repeat** until LLM produces a natural-language response

Intermediate LLM text is discarded; only tool results and the final response reach the user.

### RAG Pipeline

Every query triggers automatic semantic retrieval before the agent loop:

1. CNLLM embeds the query via text-embedding-v3 API (1024-dim)
2. Top-3 similar product chunks from Supabase product_embeddings_v2 table
3. Matched product data injected as system context into the LLM prompt
4. Agent uses this context alongside its tools

### Tools

| Tool | Description |
|------|-------------|
| search_products(query) | PostgreSQL ILIKE across name, description, category, brand, features |
| get_product_details(name) | Full product info: specs, colors, stock |
| compare_products(names) | Side-by-side comparison table |
| text_to_sql(query) | Natural language to PostgreSQL query for filtering |
| create_checkout_session(...) | Stripe checkout link generation |

## Pages

| Page | Source | Description |
|------|--------|-------------|
| **Chat** | LLM (CNLLM) + tools | Streaming shopping assistant |
| **Dashboard** | Langfuse API | Conversation count, avg latency, tool success rates |
| **Evaluation** | LLM-as-Judge (CNLLM) | Faithfulness, Answer Relevancy, Context Precision, Context Relevancy |

## Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| LLM | Qwen via CNLLM |
| Embedding | text-embedding-v3 via CNLLM |
| Vector DB | Supabase (PostgreSQL + pgvector) |
| Observability | Langfuse |
| Payments | Stripe (test mode) |
| Search | Hybrid: PostgreSQL ILIKE + pgvector semantic similarity |



