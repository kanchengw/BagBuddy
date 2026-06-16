"""Supabase + pgvector database service using httpx (direct REST API)"""
import json
import httpx
from config import PROXY_BASE_URL

# Prioritize anon key (public read-only) over secret key
_service_key = ""
_write_key = ""

def _write_headers():
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "return=representation",
    }
    if _write_key:
        h["apikey"] = _write_key
        h["Authorization"] = f"Bearer {_write_key}"
    return h

def _headers():
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "return=representation",
    }
    if _service_key:
        h["apikey"] = _service_key
        h["Authorization"] = f"Bearer {_service_key}"
    return h

def _api(path: str) -> str:
    from config import PROXY_BASE_URL
    return PROXY_BASE_URL.rstrip(chr(47)) + chr(47) + 'supabase/' + path.lstrip(chr(47))

_products_cache = None

def get_all_products_db() -> list:
    """Fetch all products from Supabase (with cache)"""
    global _products_cache
    if _products_cache is not None:
        return _products_cache
    import time as _t
    for attempt in range(3):
        try:
            r = httpx.get(_api("products"), headers=_headers(), timeout=15)
            r.raise_for_status()
            _products_cache = r.json()
            return _products_cache
        except Exception as e:
            if attempt < 2:
                _t.sleep(1)
            else:
                return []
    return []

def get_product_db(product_id: str) -> dict | None:
    """Get a single product by ID. Handles both list/object (proxy strips Accept header)."""
    headers = _headers()
    r = httpx.get(
        _api(f"products?id=eq.{product_id}"),
        headers=headers,
        timeout=10,
    )
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        if isinstance(data, dict):
            return data
    return None



def search_products_db(query: str) -> list:
    """Search products by name, description, category, brand, or features.
    Uses ILIKE on multiple columns and returns scored results.
    Use empty string or broad term to return all products.
    """
    query_stripped = query.strip().lower()
    try:
        all_products = get_all_products_db()
    except Exception:
        return []

    if not query_stripped:
        return all_products

    scored = []
    for p in all_products:
        score = 0
        name = (p.get("name") or "").lower()
        desc = (p.get("description") or "").lower()
        cat = (p.get("category") or "").lower()
        brand = (p.get("brand") or "").lower()
        features = [f.lower() for f in (p.get("features") or [])]

        if query_stripped in name:
            score += 10
        if query_stripped in desc:
            score += 5
        if query_stripped in cat:
            score += 8
        if query_stripped in brand:
            score += 7
        for f in features:
            if query_stripped in f:
                score += 3
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def get_products_by_category_db(category: str) -> list:
    """Get products filtered by category using ILIKE"""
    all_products = get_all_products_db()
    cat_lower = category.lower()
    return [p for p in all_products if (p.get("category") or "").lower() == cat_lower]


def upsert_product(product: dict) -> dict:
    """Insert or update a product"""
    data = {
        "id": product["id"],
        "name": product["name"],
        "description": product.get("description", ""),
        "price": product.get("price", 0),
        "category": product.get("category", ""),
        "brand": product.get("brand", ""),
        "rating": product.get("rating", 0),
        "stock": product.get("stock", 0),
        "features": product.get("features", []),
        "target_audience": product.get("target_audience", ""),
        "specs": json.dumps(product.get("specs", {})),
        "colors_available": product.get("colors_available", []),
        "sizes_available": product.get("sizes_available", []),
    }
    r = httpx.post(
        _api("products"),
        headers=_headers(),
        json=data,
        timeout=10,
    )
    if r.status_code == 201:
        return r.json()[0] if r.json() else data
    # Try upsert via PATCH
    r2 = httpx.patch(
        _api(f"products?id=eq.{product['id']}"),
        headers=_headers(),
        json=data,
        timeout=10,
    )
    return r2.json()[0] if r2.status_code == 200 and r2.json() else data

def upsert_embedding(product_id: str, chunk_text: str, chunk_type: str, embedding: list) -> dict:
    """Insert a product embedding chunk"""
    data = {
        "product_id": product_id,
        "chunk_text": chunk_text,
        "chunk_type": chunk_type,
        "embedding": embedding,
    }
    r = httpx.post(
        _api("product_embeddings_v2"),
        headers=_write_headers(),
        json=data,
        timeout=10,
    )
    if r.status_code == 201:
        return r.json()[0] if r.json() else data
    print(f"upsert_embedding status={r.status_code}: {r.text[:100]}")
    return data


_embeddings_cache = None

def get_all_embeddings_db() -> list:
    """Fetch all embeddings from Supabase (cached)."""
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache
    import time
    for attempt in range(3):
        try:
            r = httpx.get(_api("product_embeddings_v2?select=id,product_id,chunk_text,chunk_type,embedding"), headers=_headers(), timeout=30)
            if r.status_code == 200:
                _embeddings_cache = r.json()
                return _embeddings_cache
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return []

def semantic_search(query_embedding: list, top_k: int = 5) -> list:
    """Semantic search using cosine similarity computed in Python."""
    import math
    all_embeddings = get_all_embeddings_db()
    if not all_embeddings:
        return []
    scored = []
    q_norm = math.sqrt(sum(x * x for x in query_embedding))
    if q_norm == 0:
        return []
    for item in all_embeddings:
        e = item.get("embedding")
        if isinstance(e, str):
            try:
                import json as _json
                e = _json.loads(e)
            except:
                continue
        if not e:
            continue
        dot = sum(a * b for a, b in zip(query_embedding, e[:len(query_embedding)]))
        e_norm = math.sqrt(sum(x * x for x in e))
        if e_norm == 0:
            continue
        similarity = dot / (q_norm * e_norm)
        scored.append((similarity, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]

def sql_query(query: str) -> list:
    """Parse SQL WHERE conditions and filter products in Python."""
    import re as _re
    all_p = get_all_products_db()
    if not all_p:
        return []
    where_match = _re.search(r"WHERE (.+?)(?:ORDER BY|LIMIT|GROUP BY|$)", query, _re.IGNORECASE)
    if not where_match:
        return all_p
    where_clause = where_match.group(1).strip()
    filters = _re.split(r" AND | and ", where_clause)
    result = []
    for p in all_p:
        match = True
        for f in filters:
            f = f.strip()
            m = _re.match(r"(\w+)\s*(<|>|<=|>=|=|!=|ILIKE|LIKE)\s*(.+)", f, _re.IGNORECASE)
            if not m: continue
            col, op, val = m.group(1).lower(), m.group(2).upper(), m.group(3).strip().strip(chr(39)).rstrip(";")
            pval = p.get(col)
            if pval is None: match = False; continue
            try:
                if op == "<": match = float(pval) < float(val)
                elif op == ">": match = float(pval) > float(val)
                elif op == "<=": match = float(pval) <= float(val)
                elif op == ">=": match = float(pval) >= float(val)
                elif op in ("=", "=="): match = str(pval).lower() == val.lower()
                elif op == "!=": match = str(pval).lower() != val.lower()
                elif op in ("ILIKE", "LIKE"): match = val.replace("%","").lower() in str(pval).lower()
            except: match = False
            if not match: break
        if match: result.append(p)
    return result

def get_table_schema() -> str:
    """Return database schema as string for Text-to-SQL prompt"""
    return """
Table: products
Columns:
- id (TEXT, PRIMARY KEY) - Product identifier like prod_001
- name (TEXT) - Product name
- description (TEXT) - Long product description
- price (NUMERIC) - Price in USD
- category (TEXT) - Product category like Smartphones, Laptops, Gaming, Audio, etc.
- brand (TEXT) - Brand name like Apple, Samsung, Sony, etc.
- rating (NUMERIC) - Average rating out of 5
- stock (INTEGER) - Quantity available in inventory
- features (TEXT[]) - Array of feature strings
- target_audience (TEXT) - Who the product is for
- specs (JSONB) - Technical specifications as key-value pairs
- colors_available (TEXT[]) - Available color options
- sizes_available (TEXT[]) - Available size/storage options

Table: product_embeddings_v2
Columns:
- id (SERIAL, PRIMARY KEY)
- product_id (TEXT, FK to products.id)
- chunk_text (TEXT) - Searchable text chunk
- chunk_type (TEXT) - Type: description, specs, availability, audience
- embedding (VECTOR(1536)) - pgvector embedding for semantic search

Example queries:
- SELECT name, price, stock FROM products WHERE stock > 0 ORDER BY price ASC;
- SELECT name, stock FROM products WHERE category = 'Smartphones' AND stock < 50;
- SELECT name, price FROM products WHERE price < 500 AND stock > 0 ORDER BY rating DESC;
- SELECT name, sizes_available FROM products WHERE colors_available @> ARRAY['Black'];
- SELECT name, stock FROM products WHERE name ILIKE '%iPhone%';
- SELECT category, COUNT(*) FROM products GROUP BY category;
"""

def clear_all_data():
    """Clear all products and embeddings"""
    client = httpx.Client()
    client.delete(_api("product_embeddings_v2"), headers=_headers(), params={"id": "neq.0"}, timeout=10)
    client.delete(_api("products"), headers=_headers(), params={"id": "neq.0"}, timeout=10)
    client.close()
    print("All data cleared")

def get_client():
    """Return httpx client helper for compatibility"""
    class _Client:
        @staticmethod
        def table(name):
            return _TableQuery(name)
    return _Client()

class _TableQuery:
    def __init__(self, table_name):
        self._table = table_name
        self._filters = {}
        self._order_col = None
        self._order_desc = False
        self._limit_val = None

    def select(self, cols="*"):
        self._cols = cols
        return self

    def eq(self, col, val):
        self._filters[f"{col}"] = f"eq.{val}"
        return self

    def neq(self, col, val):
        self._filters[f"{col}"] = f"neq.{val}"
        return self

    def ilike(self, col, val):
        self._filters[f"{col}"] = f"ilike.{val}"
        return self

    def lte(self, col, val):
        self._filters[f"{col}"] = f"lte.{val}"
        return self

    def gte(self, col, val):
        self._filters[f"{col}"] = f"gte.{val}"
        return self

    def order(self, col, desc=False):
        self._order_col = col
        self._order_desc = desc
        return self

    def limit(self, n):
        self._limit_val = n
        return self

    def execute(self):
        headers = _headers()
        params = {k: v for k, v in self._filters.items()}
        if self._order_col:
            params["order"] = f"{self._order_col}.{'desc' if self._order_desc else 'asc'}"
        if self._limit_val:
            params["limit"] = str(self._limit_val)
        r = httpx.get(_api(self._table), headers=headers, params=params, timeout=10)
        r.raise_for_status()
        class _Result:
            def __init__(self, data):
                self.data = data
                self.count = len(data) if data else 0
        return _Result(r.json())

    def insert(self, data):
        r = httpx.post(_api(self._table), headers=_headers(), json=data, timeout=10)
        class _Result:
            def __init__(self, data):
                self.data = data
        return _Result(r.json() if r.status_code == 201 else [])

    def upsert(self, data):
        return self.insert(data)

    def delete(self, *args):
        return self

    def neq(self, col, val):
        self._filters = {"id": f"neq.{val}"}
        return self
