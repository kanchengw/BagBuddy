import json, uuid, re, os

from cnllm import CNLLM, ContextBox


from services.supabase_service import (
    get_product_db as get_product_by_id,
    get_all_products_db as get_all_products,
    search_products_db as search_products,
    get_products_by_category_db as get_products_by_category,
)

from services.stripe_service import create_checkout_session, get_stripe_account_info
from services.text_to_sql_service import execute_natural_language_query, format_sql_results


from config import (
    LLM_API_KEY,
    LLM_MODEL,
    LLM_BASE_URL,
)

_llm_client = None



def get_llm_client():

    global _llm_client

    if _llm_client is None:
        _llm_client = CNLLM(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    return _llm_client


def get_async_llm_client():

    if not LLM_API_KEY:
        raise ValueError("LLM API Key not configured.\n\n1. Copy .env.example to .env\n2. Set LLM_API_KEY=your_llm_api_key\n3. Restart the server")

    from cnllm.entry.async_client import asyncCNLLM

    return asyncCNLLM(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)




def get_product_by_name(name):

    name = name.strip() if name else ""
    name = re.sub(r"[*_~`]", "", name)

    if not name:

        return None

    results = search_products(name)

    if not results:

        return None

    name_lower = name.lower()
    all_product_names = [(p.get("name", "").lower(), p) for p in results]

    # 1) Exact name match
    for n_lower, p in all_product_names:
        if n_lower == name_lower:
            return p

    # 2) Substring match - prefer shortest product name that contains query
    matches = []
    for n_lower, p in all_product_names:
        if name_lower in n_lower:
            matches.append((len(n_lower), p))
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    # 3) Multi-word overlap - count how many query words appear in product name
    words = name_lower.split()
    if len(words) > 1:
        best_match = None
        best_count = 0
        for n_lower, p in all_product_names:
            match_count = sum(1 for w in words if w in n_lower)
            if match_count > best_count:
                best_count = match_count
                best_match = p
        if best_match and best_count >= max(1, len(words) - 1):
            return best_match

    # 4) Fallback: return first result
    return results[0]


SYSTEM_PROMPT = (
    """You are BagBuddy, an AI shopping assistant for a digital/tech products store. You help customers find products, answer questions, and complete purchases.

CRITICAL RULES - Priority order from highest to lowest:

1. If the user asks about a specific product by name (e.g. "iPhone 15 Pro Max", "Sony WH-1000XM5"), call get_product_by_name with the product name. This will return full details with specs, colors and pricing.

2. [FOLLOW-UP] For follow-up questions about a product that was just shown or discussed (e.g. "what color does it come in", "how much storage", "tell me more"), ALWAYS call get_product_by_name with the previous product name. NEVER use text_to_sql for follow-ups about a single product.

3. For general browsing, searching, listing, or showing all products, call search_products with the user's query terms. Use empty string for listing everything.

4. [COMPARISON] When the user asks to compare, or says "compare", "vs", "versus", "difference", or "which is better":
   - Call compare_products(names=[...]) with the exact product names.
   - If the user mentions product names directly: pass the names as a list.
   - If the user says "compare these two", "compare them", "compare these", "compare those": Look at the products listed in the PREVIOUS assistant message, extract their exact product names, and pass them as a list to compare_products.
   - Do NOT call search_products or get_product_by_name for comparisons - use compare_products directly.

5. For purchases: Call create_checkout_session(product_id) with just the product_id. Do NOT ask for email or details in chat - the system handles email collection via popup. Do NOT generate any text about checkout links or payment - the system displays the payment button automatically.

6. [LOW PRIORITY] Use text_to_sql ONLY for complex multi-product queries that span categories or need aggregation (e.g. "what is the average price of all smartphones", "which products under $50"). For questions about a SPECIFIC product's attributes (color, size, storage), use get_product_by_name instead.

7. [CONTEXT AWARENESS] The conversation history is provided to you. When the user uses words like "these", "those", "them", "they", "it" to refer to products mentioned earlier, look at the PREVIOUS assistant messages to find the specific product names, then call the appropriate tool with those names. NEVER say "I can help you search..." when the answer is in your conversation history.

8. Be friendly and conversational. Include emojis for products.


Available Tools:

- search_products(query): Search for products by name, category, brand, or any keyword. Pass empty string to list all.
- get_product_by_name(name): Get detailed info about a specific product by its name.
- get_product_details(product_id): Get full product info including specs, colors, sizes, stock, target audience.
- text_to_sql(natural_language_query): Convert a natural language question into SQL and return results from the database.
- compare_products(names): Compare two or more products side by side. Takes a list of exact product names. Use this for comparison requests (compare, vs, difference, which is better).
- create_checkout_session(product_id, user_email): Create a Stripe checkout link for purchase. user_email is optional - only provide if user already shared it.


Product categories in our store: Smartphones, Laptops, Tablets, Headphones & Audio, Smartwatches & Wearables, Gaming, Cameras, Streaming & TV, Speakers, Smart Home, VR & AR, Accessories, Networking, Drones, Fitness Tech.

"""
)


class ConversationState:

    def __init__(self):

        self.messages = []
        self.current_product = None
        self.pending_purchase = None

    def add_message(self, role, content):

        self.messages.append({"role": role, "content": content})

        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

    def clear(self):

        self.messages = []
        self.current_product = None
        self.pending_purchase = None


sessions = {}


def get_session(session_id):

    if session_id not in sessions:
        sessions[session_id] = ConversationState()

    return sessions[session_id]


def _strip_dsml_tags(text):

    text = re.sub(r"<ds_safety>.*?</ds_safety>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ds_needed>.*?</ds_needed>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()


def format_search_results(products):

    if not products:

        return (
            "I couldn"
            + "'"
            + "t find any products matching your query. Try different keywords or browse our categories: Smartphones, Laptops, Tablets, Headphones & Audio, Smartwatches & Wearables, Gaming, Cameras, Streaming & TV, Speakers, Smart Home, VR & AR, Accessories, Networking, Drones, Fitness Tech."
        )
    lines = []
    lines.append(f"Here are the products I found! {len(products)} total")
    lines.append("")
    lines.append("| Product | Price | Rating | Brand |")
    lines.append("|---------|-------|--------|-------|")

    for p in products:
        name = p.get("name", "Unknown")
        price = p.get("price", 0)
        rating = p.get("rating", 0)
        brand = p.get("brand", "Various")
        price_str = (
            f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
        )
        rating_str = (
            f"\u2b50 {rating}/5" if isinstance(rating, (int, float)) else str(rating)
        )
        lines.append(f"| **{name}** | {price_str} | {rating_str} | {brand} |")
    lines.append("")
    lines.append(
        "Tell me if you" + "'" + "d like more details on any specific product!"
    )

    return chr(10).join(lines)


def format_product_details(product):

    if not product:

        return (
            "I couldn"
            + "'"
            + "t find that product. Please check the product ID and try again."
        )
    name = product.get("name", "Unknown")
    brand = product.get("brand", "Various")
    price = product.get("price", 0)
    rating = product.get("rating", 0)
    stock = product.get("stock", 0)
    category = product.get("category", "General")
    description = product.get("description", "")
    features = product.get("features", [])

    if isinstance(features, str):

        try:
            features = json.loads(features)

        except:
            features = []
    target_audience = product.get("target_audience", "")
    specs = product.get("specs", {})

    if isinstance(specs, str):

        try:
            specs = json.loads(specs)

        except:
            specs = {}
    colors = product.get("colors_available", [])

    if isinstance(colors, str):

        try:
            colors = json.loads(colors)

        except:
            colors = []
    sizes = product.get("sizes_available", [])

    if isinstance(sizes, str):

        try:
            sizes = json.loads(sizes)

        except:
            sizes = []
    price_str = (
        f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
    )
    rating_str = (
        f"\u2b50 {rating}/5" if isinstance(rating, (int, float)) else str(rating)
    )
    stock_str = (
        f"\u2714 {stock} in stock"
        if isinstance(stock, (int, float)) and stock > 0
        else "Out of stock"
    )
    lines = []
    lines.append(f"**{name}**")
    lines.append(f"- **Brand:** {brand}")
    lines.append(f"- **Price:** {price_str}")
    lines.append(f"- **Rating:** {rating_str}")
    lines.append(f"- **Category:** {category}")
    lines.append(f"- **Stock:** {stock_str}")
    lines.append("")
    lines.append("**Description:**")
    lines.append(description)
    lines.append("")

    if features:
        lines.append("**Key Features:**")

        for f in features:
            lines.append(f"- {f}")
        lines.append("")

    if target_audience:
        lines.append(f"**Best for:** {target_audience}")
        lines.append("")

    if specs:
        lines.append("**Specifications:**")

        for key, val in specs.items():
            label = key.replace("_", " ").title()
            lines.append(f"- {label}: {val}")
        lines.append("")

    if colors:
        lines.append(f"**Available Colors:** {', '.join(colors)}")

    if sizes:
        lines.append(f"**Available Options:** {', '.join(sizes)}")

    return chr(10).join(lines)


    return (
        f"To purchase the **{name}** ({price_str}), please provide your email address "
        f"so I can send you the receipt and create a secure checkout link."
    )


def _build_tools():

    return [
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Search for products by name, category, brand, or keyword. Pass empty string to list all products.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query like "
                            + "'"
                            + "wireless headphones"
                            + "'"
                            + ", "
                            + "'"
                            + "iPhones"
                            + "'"
                            + ", or empty string for all products",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product_by_name",
                "description": "Get detailed info about a specific product by its name (e.g. "
                + "'"
                + "iPhone 15 Pro Max"
                + "'"
                + ", "
                + "'"
                + "Sony WH-1000XM5"
                + "'"
                + ").",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Product name"}
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product_details",
                "description": "Get detailed info by product ID. Use if you already know the product_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "Product ID like prod_001",
                        }
                    },
                    "required": ["product_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "text_to_sql",
                "description": "Convert a natural language question into SQL and query the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language question about products",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_checkout_session",
                "description": "Create a Stripe checkout link. Call when user confirms purchase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "Product ID"},
                        "user_email": {
                            "type": "string",
                            "description": "Customer email",
                        },
                    },
                    "required": ["product_id"],
                },
            },
        },

        {
            "type": "function",
            "function": {
                "name": "compare_products",
                "description": "Compare two or more products side by side. Call when the user asks to compare or says compare, vs, versus, difference, which is better. Extract the exact product names from the conversation history and pass them as a list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of exact product names to compare. Extract from conversation history when user says compare them/these/those.",
                        }
                    },
                    "required": ["names"],
                },
            },
        },
    ]


def _robust_json_parse(raw):
    """Parse tool call JSON with repair fallbacks."""

    if not isinstance(raw, str):

        return raw if isinstance(raw, dict) else {}

    raw = raw.strip()

    if not raw:

        return {}

    # Try 1: Standard JSON

    try:
        result = json.loads(raw)

        if isinstance(result, dict):

            return result

        if isinstance(result, str) and result:

            return {"name": result}

    except json.JSONDecodeError:
        pass

    # Try 2: Single quotes -> double quotes

    try:
        result = json.loads(raw.replace("'", '"'))

        if isinstance(result, dict):

            return result

        if isinstance(result, str) and result:

            return {"name": result}

    except json.JSONDecodeError:
        pass

    # Try 3: Extract key-value patterns
    dq = chr(34)  # double quote
    m = re.search(dq + r"(\w+)" + dq + r"\s*:\s*" + dq + r"([^" + dq + r"]*)" + dq, raw)

    if m:

        return {m.group(1): m.group(2).strip()}

    m = re.search(dq + r"(\w+)" + dq + r"\s*:\s*(.+?)\s*[},\]]*$", raw)

    if m:

        return {m.group(1): m.group(2).strip().strip(dq + "'")}

    # Try 4: Bare string
    clean = raw.strip(dq + "' ")

    if clean:

        return {"name": clean}

    return {}


def _execute_tool(tc):

    name = tc.get("function", {}).get("name", "")
    args_raw = tc.get("function", {}).get("arguments", "{}")
    args = _robust_json_parse(args_raw)

    if name == "search_products":
        q = args.get("query", "")
        results = search_products(q)

        return {"name": name, "result": results, "query": q}

    elif name == "get_product_by_name":
        n = (
            args.get("name")
            or args.get("product_name")
            or args.get("product")
            or args.get("query")
            or ""
        )
        n = re.sub(r"[*_~`]", "", n)
        p = get_product_by_name(n)

        return {"name": name, "result": p}

    elif name == "get_product_details":
        pid = args.get("product_id", "")
        p = get_product_by_id(pid)

        return {"name": name, "result": p}

    elif name == "text_to_sql":
        q = args.get("query", "")

        try:
            sql_result = execute_natural_language_query(q)
            formatted = format_sql_results(
                sql_result.get("results", []), sql_result.get("sql", "")
            )

            return {"name": name, "result": formatted, "raw": sql_result}

        except Exception as e:

            return {"name": name, "result": f"Error: {str(e)[:200]}"}

    elif name == "create_checkout_session":
        pid = args.get("product_id", "")
        email = args.get("user_email", "")

        try:
            p = get_product_by_id(pid)

            if not p:

                return {"name": name, "result": "Product not found"}

            pname = p.get("name", "Product")
            pprice = p.get("price", 0)

            if not email:
                return {
                    "name": name,
                    "result": "",
                    "product_name": pname,
                    "product_id": pid,
                    "needs_email": True,
                }
            url = create_checkout_session(pid, pname, pprice, email)

            return {
                "name": name,
                "result": url,
                "product_name": pname,
                "price": pprice,
                "email": email,
            }

        except Exception as e:

            return {"name": name, "result": f"Error: {str(e)[:200]}"}

    elif name == "answer_product_question":
        from services.rag_service import answer_product_question as _answer

        q = args.get("question", "")
        try:
            result = _answer(q)
            return {"name": name, "result": result}
        except Exception as e:
            return {
                "name": name,
                "result": {
                    "answer": f"Sorry, I couldn't answer that: {str(e)[:100]}",
                    "sources": [],
                },
            }

    elif name == "compare_products":
        names = args.get("names", [])
        if isinstance(names, str):
            import json as _js
            try:
                names = _js.loads(names)
            except:
                names = [names]
        if not names:
            return {"name": name, "result": "No product names provided."}
        _details = []
        for _pn in names:
            _p = get_product_by_name(_pn)
            if _p:
                _details.append(_p)
        if len(_details) < 2:
            return {"name": name, "result": "Could not find enough products to compare."}
        import json as _j2
        _cp = "You are a product comparison expert. Compare the following products." + chr(10) + chr(10)
        for ci, cp in enumerate(_details[:5]):
            _keys = ["name","brand","price","rating","category","description","features","specs","colors_available","sizes_available","target_audience"]
            _data = {k: cp.get(k) for k in _keys if k in cp}
            _cp += "Product " + str(ci+1) + ": " + _j2.dumps(_data, ensure_ascii=False) + chr(10)
        _cp += chr(10) + "Create a detailed comparison table. Compare price, rating, key features, target audience. Highlight key differences. Be concise. Use markdown table."
        try:
            _llm = get_llm_client()
            _resp = _llm.chat.create(messages=[{"role": "user", "content": _cp}], stream=False)
            _text = _resp.still.strip()
        except Exception as _ce:
            _text = "I found " + str(len(_details)) + " products to compare. " + str(_ce)[:100]
        return {"name": name, "result": _text}
    
    return {"name": name, "result": None}



def process_user_message(message, session_state):

    session_state.add_message("user", message)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + session_state.messages[-20:]

    tools = _build_tools()
    client = get_llm_client()

    try:
        resp = client.chat.create(
            messages=msgs, tools=tools, tool_choice="auto", stream=True
        )

    except Exception as e:
        error_text = f"Sorry, I encountered an error processing your request. Please try again. (Error: {str(e)[:100]})"
        session_state.add_message("assistant", error_text)

        return {"text": error_text, "message_type": "normal"}

    for chunk in resp:
        pass
    final_text = resp.still if resp.still else ""

    if final_text:
        final_text = _strip_dsml_tags(final_text)

    if resp.tools:

        for tc_raw in resp.tools:
            result = _execute_tool(tc_raw)
            name = result["name"]

            if name == "search_products":
                products = result["result"]
                query = result["query"]

                if products:
                    formatted = format_search_results(products)

                else:
                    formatted = (
                        "I couldn"
                        + "'"
                        + "t find any products matching your query."
                    )
                session_state.add_message("assistant", formatted)
                user_msg_lower = message.lower()
                detail_kw = [
                    "tell me about",
                    "details",
                    "info on",
                    "about the",
                    "what about",
                    "more about",
                    "specs for",
                    "information on",
                    "show me the",
                    "describe",
                    "i want to know about",
                ]
                is_detail = any(kw in user_msg_lower for kw in detail_kw)

                if is_detail and len(products) == 1:
                    p = get_product_by_name(query) or products[0]

                    if p:
                        detailed = format_product_details(p)
                        session_state.current_product = p
                        session_state.add_message("assistant", detailed)

                        return {
                            "text": detailed,
                            "message_type": "product_info",
                            "product_info": p,
                        }

                return {"text": formatted, "message_type": "search_results"}

            elif name == "get_product_by_name":
                p = result["result"]

                if p:
                    formatted = format_product_details(p)
                    session_state.current_product = p

                else:
                    formatted = (
                        "I couldn"
                        + "'"
                        + "t find that product. Please try a different name."
                    )
                session_state.add_message("assistant", formatted)

                return {
                    "text": formatted,
                    "message_type": "product_info",
                    "product_info": p,
                }

            elif name == "get_product_details":
                p = result["result"]

                if p:
                    formatted = format_product_details(p)
                    session_state.current_product = p

                else:
                    formatted = "I couldn" + "'" + "t find that product."
                session_state.add_message("assistant", formatted)

                return {
                    "text": formatted,
                    "message_type": "product_info",
                    "product_info": p,
                }

            elif name == "text_to_sql":
                formatted = (
                    result["result"]
                    or "I looked that up but didn" + "'" + "t find anything."
                )
                session_state.add_message("assistant", formatted)

                return {"text": formatted, "message_type": "normal"}

            elif name == "create_checkout_session":
                url = result["result"]

                if url and url.startswith("http"):
                    pname = result.get("product_name", "Product")
                    email = result.get("email", "")
                    p = get_product_by_name(pname) if pname != "Product" else None
                    price = p.get("price", 0) if p else 0
                    price_str = (
                        f"${price:.2f}"
                        if isinstance(price, (int, float))
                        else str(price)
                    )
                    formatted = (
                        f"Great! Here"
                        + "'"
                        + "s your checkout link for the **{pname}** ({price_str}):\n\n"
                        f"<a href='{url}' target='_blank' rel='noopener' style='display:inline-block;padding:12px 28px;background:#c8553d;color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px;margin:10px 0'>🤑 Pay with Stripe</a>\n\n"
                        f"A receipt will be sent to {email}. Thank you for shopping with BagBuddy! \U0001f389"
                    )
                    session_state.pending_purchase = None

                else:
                    formatted = (
                        f"Sorry, there was an issue creating the checkout: {url}"
                    )
                session_state.add_message("assistant", formatted)

                return {"text": formatted, "message_type": "payment_link"}

    if not final_text:
        final_text = (
            "I"
            + "'"
            + "m not sure how to help with that. I can help you find products, compare items, or make a purchase. What are you looking for?"
        )
    session_state.add_message("assistant", final_text)

    return {"text": final_text, "message_type": "normal"}




def _detect_type_from_content(text: str) -> tuple | None:
    """Analyze response text to determine message_type, product_count, and product_info.
    Returns (type, count, product_info_dict) or None if undetermined."""
    import re as _re
    
    # Check for product detail (single product with specs/features)
    has_specs = "Specifications:" in text
    has_features = "Key Features:" in text
    has_description = "Description:" in text and "Brand:" in text
    if (has_specs and has_features) or (has_description and (has_specs or has_features)):
        return ("product_info", 1, None)
    
    lines = text.split(chr(10))
    
    # Check for product table/listing with | separators and prices
    table_rows = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 3 and "---" not in stripped:
            # Check if it looks like a product row (has $ price or ? rating)
            if "$" in stripped or "?" in stripped or "?" in stripped:
                table_rows.append(line)
    
    # Detect comparison table (has aspect column)
    is_comparison = any("aspect" in line.lower() for line in lines if line.strip().startswith("|"))
    has_vs = bool(_re.search(r'vs?', text, _re.IGNORECASE)) or "comparison" in text.lower()
    
    if len(table_rows) >= 2:
        if is_comparison or has_vs:
            return ("search_results", len(table_rows), None)
        return ("search_results", len(table_rows), None)
    elif len(table_rows) == 1:
        return ("product_info", 1, None)
    
    # Check for bullet list products with prices (e.g. "**Product** - $price" or "Product - $price")
    product_lines = []
    price_pattern = _re.compile(r'[-??]\s*\$\(\d+,\d+\)?[\d,]*\.?\d*', _re.IGNORECASE)
    for line in lines:
        s = line.strip()
        # Skip empty lines, headers
        if not s or s.startswith("#") or s.startswith("---"):
            continue
        # Look for pattern: starts with emoji or ** and has $price
        if _re.search(r'\$[\d,]+\.?\d*', s):
            product_lines.append(line)
        # Also check for lines with both "Price:" and a product context
        elif s.startswith("-") and "Price:" in s and ("$" in s or "?" in s):
            product_lines.append(line)
    
    if len(product_lines) >= 2:
        return ("search_results", len(product_lines), None)
    elif len(product_lines) == 1:
        return ("product_info", 1, None)
    
    # Check for "Price:" and "Rating:" patterns (format_product_details style)
    price_rating_count = 0
    for line in lines:
        s = line.strip()
        if s.startswith("-") and "Price:" in s and "$" in s:
            price_rating_count += 1
    if price_rating_count >= 2:
        return ("search_results", price_rating_count, None)
    
    return None


async def async_stream_agent_response(message, session_state):
    """Multi-turn async generator. Cycles: LLM with tools -> execute -> ContextBox inject -> loop until LLM responds with text."""
    session_state.add_message("user", message)

    _rag_context = ""
    _rag_product_map = {}
    try:
        _query = message.strip()
        if len(_query) > 5:
            from services.embedding_service import create_embedding
            from services.supabase_service import semantic_search
            _emb = create_embedding(_query)
            _results = semantic_search(_emb, top_k=3)
            if _results:
                from services.supabase_service import get_all_products_db
                _all_p = get_all_products_db()
                _lookup = {p.get("id", ""): p for p in _all_p}
                _parts = []
                for _r in _results:
                    _pid = _r.get("product_id", "")
                    _p = _lookup.get(_pid, {})
                    _name = _p.get("name", "")
                    if _name:
                        _parts.append(f"Product: {_name} - {_p.get('description', '')[:150]}")
                if _parts:
                    _rag_context = "Relevant products from catalog:\n" + "\n".join(_parts)
                    _rag_product_map = {}
                    for _r in _results:
                        _pid = _r.get("product_id", "")
                        _p = _lookup.get(_pid, {})
                        if _p.get("name", ""):
                            _rag_product_map[_pid] = _p
    except Exception:
        pass
    import logging
    _rag_debug = logging.getLogger("bagbuddy.rag")
    _rag_product_count = len(_rag_product_map)
    try:
        _rag_debug.info(f"RAG: query_len={len(_query)}, _emb_dim={len(_emb) if '_emb' in dir() else 0}, _results_len={len(_results) if '_results' in dir() else 0}")
    except:
        _rag_debug.info(f"RAG: variables not available - exception may have occurred")
    _rag_debug.info(f"RAG found {_rag_product_count} products, _final_type will be: {"search_results" if _rag_product_count >= 2 else "product_info" if _rag_product_count == 1 else "normal"}")
    if _rag_context:
        session_state.add_message("system", _rag_context)

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + session_state.messages[-20:]
    tools = _build_tools()

    _final_tool_events = []
    _final_type = "search_results" if _rag_product_count >= 2 else "product_info" if _rag_product_count == 1 else "normal"
    _final_product_info = list(_rag_product_map.values())[0] if _rag_product_count == 1 else None
    _final_product_count = _rag_product_count
    MAX_ROUNDS = 5

    for _round in range(MAX_ROUNDS):
        try:
            client = get_async_llm_client()
        except ValueError as e:
            text = str(e).replace("\\n", "\n")
            yield {"type": "error", "content": text}
            yield {"type": "done", "message_type": "error"}
            return
        try:
            resp = await client.chat.create(
                messages=msgs, tools=tools, tool_choice="auto",
                stream=True, thinking=False
            )
            async for chunk in resp:
                pass
        except Exception as e:
            error_text = f"Sorry, I encountered an error: {str(e)[:100]}"
            session_state.add_message("assistant", error_text)
            yield {"type": "error", "content": error_text}
            yield {"type": "done", "message_type": "error"}
            return

        raw_text = resp.still if resp.still else ""
        final_text = _strip_dsml_tags(raw_text) if raw_text else ""


        if not resp.tools:
            if not final_text:
                final_text = "I'm not sure how to help with that. I can help you find products, compare items, or make a purchase. What are you looking for?"
            # Re-determine _final_type from response content if still "normal"
            if _final_type == "normal" and len(final_text) > 20:
                _detected = _detect_type_from_content(final_text)
                if _detected:
                    _final_type = _detected[0]
                    if _detected[0] == "search_results":
                        _final_product_count = _detected[1]
                    elif _detected[0] == "product_info":
                        _final_product_info = _detected[2]
            session_state.add_message("assistant", final_text)
            done_extra = {}
            if _final_type == "product_info" and _final_product_info:
                done_extra["product_info"] = _final_product_info
            elif _final_type == "search_results" and _final_product_count > 0:
                done_extra["product_count"] = _final_product_count
            _rag_debug.info(f"Yielding chunk with message_type={_final_type}, product_count={_final_product_count}, has_product_info={_final_product_info is not None}")
            yield {"type": "chunk", "content": final_text, "message_type": _final_type, **done_extra}
            yield {"type": "done", "message_type": _final_type, **done_extra}
            return

        _cached_results = []
        for tc_raw in resp.tools:
            name = tc_raw.get("function", {}).get("name", "")
            yield {"type": "tool_exec_start", "name": name}
            result = _execute_tool(tc_raw)
            name = result["name"]

            if name == "create_checkout_session":
                needs_email = result.get("needs_email", False)
                url = result["result"]
                if needs_email:
                    yield {"type": "request_email", "product_name": result.get("product_name", "Product"), "product_id": result.get("product_id", "")}
                    return
                if url and isinstance(url, str) and url.startswith("http"):
                    pname = result.get("product_name", "Product")
                    price = result.get("price", 0)
                    email = result.get("email", "")
                    try:
                        acct = get_stripe_account_info()
                        merchant_name = acct.get("business_name", "Stripe Merchant")
                    except:
                        merchant_name = "Stripe Merchant"
                    yield {"type": "checkout_url", "url": url, "product_name": pname, "price": price, "email": email, "merchant_name": merchant_name}
                    formatted = "Great! Here" + "'" + "s your checkout link for the **" + pname + "**:\n\n"
                    formatted += '<a href="' + url + '" target="_blank" rel="noopener" style="display:inline-block;padding:12px 28px;background:#c8553d;color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:15px;margin:10px 0">&#x1f911; Pay with Stripe</a>'
                    session_state.pending_purchase = None
                else:
                    formatted = "Sorry, there was an issue: " + str(url)
                session_state.add_message("assistant", formatted)
                if not needs_email:
                    yield {"type": "tool_result", "name": name, "text": formatted}
                    yield {"type": "done", "message_type": "payment_link"}
                return

            if name == "search_products":
                products = result.get("result", [])
                if products:
                    formatted = format_search_results(products)
                    _final_tool_events.append({"type": "tool_result", "name": name, "text": formatted, "product_count": len(products)})
                    _final_type = "search_results"
                    _final_product_count = len(products)
                else:
                    formatted = "I couldn" + "'" + "t find any products matching your query."
                    _final_tool_events.append({"type": "tool_result", "name": name, "text": formatted})

            elif name in ("get_product_by_name", "get_product_details"):
                p = result["result"]
                if p:
                    formatted = format_product_details(p)
                    session_state.current_product = p
                    _final_tool_events.append({"type": "tool_result", "name": name, "text": formatted, "product_info": p})
                    _final_type = "product_info"
                    _final_product_info = p
                else:
                    formatted = "I couldn" + "'" + "t find that product."
                    _final_tool_events.append({"type": "tool_result", "name": name, "text": formatted})

            elif name == "compare_products":
                formatted = result["result"] or "I couldn" + "'" + "t compare those products."
                _final_tool_events.append({"type": "tool_result", "name": name, "text": formatted})
                _final_type = "comparison"

            elif name == "text_to_sql":
                formatted = str(result.get("result", "")) or "I didn" + "'" + "t find anything."
                _final_tool_events.append({"type": "tool_result", "name": name, "text": formatted})

            else:
                formatted = str(result.get("result", ""))
                _final_tool_events.append({"type": "tool_result", "name": name, "text": formatted})

            _cached_results.append(result)

        msgs += ContextBox(resp.still if resp.still else "", resp.think if resp.think else "", resp.tools if resp.tools else [])
        for _tc_raw, _result in zip(resp.tools, _cached_results):
            _name = _result.get("name", "")
            _r_text = _result.get("result", "")
            # Use formatted text for known tools to prevent LLM hallucination
            if _name == "search_products" and isinstance(_r_text, list):
                _products = _r_text
                if _products:
                    _r_text = "Search results:\n"
                    for _p in _products:
                        _pn = _p.get("name", "Product")
                        _pb = _p.get("brand", "")
                        _pp = _p.get("price", "N/A")
                        _pr = _p.get("rating", "")
                        _r_text += f"  - {_pn} (Brand: {_pb}, Price: ${_pp}, Rating: {_pr}/5)\n"
                else:
                    _r_text = "Search returned no results."
            elif _name == "get_product_by_name" and isinstance(_r_text, dict):
                _p = _r_text
                _pn = _p.get("name", "Product")
                _r_text = f"Product details: {_pn}, Brand: {_p.get('brand','')}, Price: ${_p.get('price','N/A')}, Category: {_p.get('category','')}"
            elif _name == "compare_products" and _r_text:
                _r_text = str(_r_text)[:500]
            elif _name == "get_product_details" and isinstance(_r_text, dict):
                _p = _r_text
                _r_text = f"Product: {_p.get('name','')}, Brand: {_p.get('brand','')}, Price: ${_p.get('price','N/A')}"
            elif isinstance(_r_text, (dict, list)):
                import json as _js2
                _r_text = _js2.dumps(_r_text, ensure_ascii=False)
            elif not isinstance(_r_text, str):
                _r_text = str(_r_text)
            msgs.append({
                "role": "tool",
                "tool_call_id": _tc_raw.get("id", ""),
                "content": _r_text,
            })

    for evt in _final_tool_events:
        yield evt
    fallback = "Sorry, I couldn't process that request completely."
    session_state.add_message("assistant", fallback)
    yield {"type": "chunk", "content": fallback}
    yield {"type": "done", "message_type": _final_type}
