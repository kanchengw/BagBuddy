"""Embedding service using CNLLM API (text-embedding-v3)"""
from config import LLM_API_KEY, LLM_BASE_URL, EMBEDDING_MODEL


def create_embedding(text: str) -> list[float]:
    from cnllm import CNLLM
    client = CNLLM(model=EMBEDDING_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    resp = client.embeddings.create(input=text)
    vec = resp.vectors
    if isinstance(vec, list) and vec and isinstance(vec[0], (int, float)):
        return vec
    return vec[0] if isinstance(vec, list) else vec


def create_embeddings_batch(texts: list[str]) -> list[list[float]]:
    return [create_embedding(t) for t in texts]


DOLLAR = chr(36)


def prepare_product_chunks(product: dict) -> list[dict]:
    chunks = []
    desc_text = (
        f"Product: {product['name']}. "
        f"Category: {product['category']}. "
        f"Brand: {product['brand']}. "
        f"Description: {product['description']}. "
        f"Target audience: {product.get('target_audience', 'General')}. "
        f"Features: {'; '.join(product.get('features', []))}. "
        f"Price: {DOLLAR}{product.get('price', 'N/A')}. "
        f"Rating: {product.get('rating', 0)}/5."
    )
    chunks.append({"chunk_text": desc_text, "chunk_type": "description"})

    specs = product.get("specs", {})
    import json as _j
    if isinstance(specs, str):
        try:
            specs = _j.loads(specs)
        except:
            specs = {}
    if specs and isinstance(specs, dict):
        specs_text = f"{product['name']} specifications: "
        specs_text += "; ".join([f"{k}: {v}" for k, v in specs.items()])
        chunks.append({"chunk_text": specs_text, "chunk_type": "specs"})

    colors = product.get("colors_available", [])
    sizes = product.get("sizes_available", [])
    if colors or sizes:
        avail_parts = [f"Colors: {', '.join(colors)}"] if colors else []
        avail_parts += [f"Sizes: {', '.join(sizes)}"] if sizes else []
        avail_text = f"{product['name']} - {'; '.join(avail_parts)}."
        chunks.append({"chunk_text": avail_text, "chunk_type": "availability"})

    audience = product.get("target_audience", "")
    if audience:
        feat_text = "; ".join(product.get("features", [])[:3])
        audience_text = (
            f"Who is {product['name']} for? {audience}. "
            f"Key highlights: {feat_text}. "
            f"Best suited for users looking for {product['category']} products."
        )
        chunks.append({"chunk_text": audience_text, "chunk_type": "audience"})

    return chunks
