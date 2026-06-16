
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

## Why CNLLM? ——对比 OpenAI SDK 与 LangChain

BagBuddy 完全基于 CNLLM 构建，未使用 OpenAI SDK、LiteLLM 或 LangChain。以下从三个核心维度做对比，展示 CNLLM 的实际设计取舍。

### 1. 流式响应字段累积：CNLLM 自动 vs 手动

**OpenAI SDK** — 每个 chunk 只携带增量 delta，需要手动拼装：

```python
content = ""
tool_calls = {}
for chunk in client.chat.completions.create(model="...", messages=messages, stream=True):
    delta = chunk.choices[0].delta
    if delta.content:
        content += delta.content
    if delta.tool_calls:
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in tool_calls:
                tool_calls[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
            if tc.id:           tool_calls[idx]["id"] = tc.id
            if tc.function.name: tool_calls[idx]["function"]["name"] += tc.function.name
            if tc.function.arguments: tool_calls[idx]["function"]["arguments"] += tc.function.arguments
```

**CNLLM** — 内部自动累积增量字段，消费完流后直接取属性：

```python
resp = client.chat.create(prompt="...", stream=True, tools=[...])
for chunk in resp:               # 消费流（无其他样板）
    pass
print(resp.still)                 # 完整文本
print(resp.think)                 # 推理内容
print(resp.tools)                 # 完整 tool_calls 列表（已是 OpenAI 标准格式）
```

**差异**：CNLLM 在 `resp` 对象上做了增量自动累积，省去手动拼装的 ~15 行样板。对于大规模多轮调用，这种差异会累积为可观的代码量。

### 2. 多轮 Tool Calling 编排：同等能力，不同写法

两者都可以用 for 循环手动编排，核心逻辑等价。以下为 BagBuddy 案例中"检查用户邮箱 → 中断 → 发前端事件"的场景：

**手动循环（LangChain 与 CNLLM 都可实现，写法接近）：**

```python
# LangChain 手动循环
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

while True:
    resp = llm.invoke(messages)                       # 单次 LLM 调用
    if isinstance(resp, AIMessage) and resp.tool_calls:
        for tc in resp.tool_calls:
            result = execute_tool(tc)                  # 执行工具
            if result.get("needs_email"):
                yield {"type": "request_email", ...}   # 中断并发事件
                return
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        continue
    messages.append(resp)
    yield resp.content
    break
```

```python
# CNLLM 手动循环
while True:
    resp = client.chat.create(messages=messages, tools=[...])
    if resp.tools:
        for tc_raw in resp.tools:
            result = execute_tool(tc_raw)               # 执行工具
            if result.get("needs_email"):
                yield {"type": "request_email", ...}    # 中断并发事件
                return
            messages += ContextBox("", "", [tc_raw])    # 注入工具结果
        continue
    messages += ContextBox(resp.still, resp.think, [])
    yield resp.still
    break
```

**对比 AgentExecutor（黑盒方案）：**

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5)
result = agent_executor.invoke({"input": user_input})
# result["output"] 拿到最终回复，但中间过程不可干预
```

**差异**：
- LangChain 需手动构造 `AIMessage` / `ToolMessage` 对象；CNLLM 的 `ContextBox` 一行完成 assistant + tool_calls + tool 结果的消息构造
- AgentExecutor 封装了循环，但一旦需要拦截逻辑（如 BagBuddy 中的邮箱收集逻辑），就必须回归手动循环方案。

### 3. 上下文构建：多行 vs 一行

**OpenAI SDK / LangChain 通用做法：**

```python
messages.append({"role": "assistant", "content": text, "tool_calls": raw_tool_calls})
for tc in raw_tool_calls:
    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
```

**CNLLM ContextBox：**

```python
messages += ContextBox(resp.still, resp.think, resp.tools, executor=execute_tool)
```

`ContextBox` 是一个 `list` 子类，构造函数内部自动生成一个 `assistant` 消息（含 tool_calls）+ N 条 `tool` 消息（通过 `executor` 执行工具取结果）。也可不传 `executor`，手动处理后再构建。

### 对比总结

| 维度 | OpenAI / LangChain 方案 | CNLLM 方案 |
|------|------------------------|------------|
| 流式累积 | ~15 行手动拼装 | `resp.still` / `.think` / `.tools` 自动累积 |
| 上下文构建 | 手动构造 assistant + tool_calls + tool 结果 | `ContextBox()` 一行完成 |
| 工具编排（手动循环） | 等价，写法接近 | 等价，写法接近 |
| 工具编排（AgentExecutor） | 黑盒，无法中途拦截 | 无此概念（手动循环天然可控） |
| 中间结果拦截 | 手动循环：`if needs_email: yield`；AgentExecutor：需 LangGraph | `if needs_email: yield`（同手动循环） |
| 兼容性 | 需对齐 OpenAI message 格式 | CNLLM 原生输出已是标准格式，可直接喂给下一轮 |

- CNLLM 的下一步是实现类似 AgentExecutor 的封装，但保留足够的灵活性来支撑类似中途拦截的场景。 


