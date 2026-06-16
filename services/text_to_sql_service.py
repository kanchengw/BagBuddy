"""Text-to-SQL service: NL question -> SQL -> execute -> format results"""
import json
from cnllm import CNLLM
from config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
from services.supabase_service import sql_query, get_table_schema

_sql_client = None

def _get_sql_client():
    global _sql_client
    if _sql_client is None:
        _sql_client = CNLLM(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _sql_client

SQL_PROMPT = """You are a SQL expert for a product database. Convert the question to PostgreSQL SQL.

Schema:
{schema}

Rules:
- Use ILIKE for text matching
- Use price for cost comparisons
- Use @> ARRAY[] for array contains
- Return ONLY the SQL query, no explanation
- Use valid PostgreSQL syntax only

Question: {question}

SQL:"""

def execute_natural_language_query(question: str) -> dict:
    """Convert NL question to SQL, execute it, return {sql, results}."""
    client = _get_sql_client()
    schema = get_table_schema()
    prompt = SQL_PROMPT.format(schema=schema, question=question)

    resp = client.chat.create(
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    sql = resp.still.strip()
    # Clean markdown SQL fences if present
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if sql.upper().startswith("SQL"):
        sql = sql[3:].strip()

    results = sql_query(sql)
    return {"sql": sql, "results": results}

def format_sql_results(results: list, sql: str = "") -> str:
    """Format SQL query results as readable text."""
    parts = []
    if sql:
        parts.append(f"SQL: `{sql}`")
    if not results:
        parts.append("No results found.")
        return "\n".join(parts)
    parts.append(f"Found {len(results)} product(s):\n")
    for i, row in enumerate(results, 1):
        name = row.get("name", row.get("product_name", f"Result {i}"))
        price = row.get("price", "")
        stock = row.get("stock", "")
        rating = row.get("rating", "")
        brand = row.get("brand", "")
        info = f"  {i}. **{name}**"
        if price:
            dollar = chr(36)
            info += f" | {dollar}{float(price):.2f}" if isinstance(price, (int, float)) else f" | {dollar}{price}"
        if rating:
            info += f" | Star {float(rating):.1f}/5"
        if stock:
            info += f" | Stock: {stock}"
        if brand:
            info += f" | Brand: {brand}"
        parts.append(info)
    return "\n".join(parts)
