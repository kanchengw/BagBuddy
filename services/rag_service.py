"""RAG service - semantic search returning raw product context for main LLM."""
from services.embedding_service import create_embedding
from services.supabase_service import semantic_search


def answer_product_question(question: str) -> dict:
    """Vector search for product info. Returns raw context + sources (no LLM synthesis).
    
    Args:
        question: The customer question about product features or functionality.
    
    Returns:
        dict with "context" (str) and "sources" (list of product names)
    """
    if not question or len(question.strip()) < 3:
        return {"context": "", "sources": []}

    try:
        # 1. Embed the question
        embedding = create_embedding(question)

        # 2. Semantic search for relevant products
        results = semantic_search(embedding, top_k=5)

        if not results:
            return {"context": "", "sources": []}

        # 3. Fetch all products and build lookup by product_id
        from services.supabase_service import get_all_products_db
        all_products = get_all_products_db()
        product_lookup = {p.get("id", ""): p for p in all_products}

        # 4. Build context from retrieved products
        sources = []
        context_parts = []
        for r in results:
            pid = r.get("product_id", "")
            product = product_lookup.get(pid, {})

            name = product.get("name", "Product")
            sources.append(name)
            brand = product.get("brand", "")
            category = product.get("category", "")
            desc = product.get("description", "")
            features = product.get("features", [])
            specs = product.get("specs", {})

            part = f"Product: {name}"
            if brand:
                part += f" (Brand: {brand}, Category: {category})"
            if desc:
                part += f"\n  Description: {desc}"
            if features:
                if isinstance(features, list):
                    features_str = "; ".join(features)
                else:
                    features_str = str(features)
                part += f"\n  Features: {features_str}"
            if specs:
                if isinstance(specs, dict):
                    specs_flat = "; ".join(f"{k}: {v}" for k, v in specs.items())
                else:
                    specs_flat = str(specs)
                part += f"\n  Specs: {specs_flat}"
            context_parts.append(part)

        context = "\n\n".join(context_parts)

        return {"context": context, "sources": sources}

    except Exception as e:
        return {"context": "", "sources": []}