import compat_patch  # Python 3.12+ multiprocess fix
"""
BagBuddy - Main FastAPI Application
API endpoints for chatbot, payments, observability, and evaluation
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import time
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
SERVER_STARTUP_TIME = str(time.time())
import uuid
import os

# In-memory cart storage: {session_id: [{product_id, name, price, quantity}]}
cart_store = {}

from agent.commerce_agent import process_user_message, get_session, ConversationState
from services.stripe_service import create_checkout_session, create_line_items_checkout, get_stripe_account_info, get_session as get_stripe_session
from services.observability import track_conversation, track_purchase, track_search
from services.langfuse_service import get_dashboard_data, get_evaluation_data_from_traces, record_latency, record_session_eval, reset_session_eval
from services.supabase_service import get_all_products_db as get_all_products, get_product_db as get_product_by_id, search_products_db as search_products
from services.embedding_service import create_embedding, create_embeddings_batch, prepare_product_chunks
from services.supabase_service import upsert_product, upsert_embedding, semantic_search, clear_all_data, get_client
from config import PROXY_BASE_URL, APP_HOST, APP_PORT, LANGFUSE_HOST


app = FastAPI(
    title="BagBuddy",
    description="AI-powered e-commerce chatbot with observability and evaluation",
    version="1.0.0"
)



# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_type: str = "normal"
    product_info: Optional[dict] = None


class PurchaseRequest(BaseModel):
    product_id: str
    user_email: str


class SearchRequest(BaseModel):
    query: str


class CartAddRequest(BaseModel):
    session_id: str
    product_id: str = ""
    items: Optional[list] = None
    user_email: Optional[str] = None


class CartRemoveRequest(BaseModel):
    session_id: str
    product_id: str


class CartUpdateRequest(BaseModel):
    session_id: str
    product_id: str
    quantity: int = 1


# API Endpoints

@app.get("/api/startup")
async def get_startup():
    return {"startup_time": SERVER_STARTUP_TIME}
@app.get("/")
async def root():
    """Root endpoint - serves the chat UI"""
    chat_path = os.path.join(os.path.dirname(__file__), "templates", "chat.html")
    return FileResponse(chat_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - processes user messages"""
    session_id = request.session_id or str(uuid.uuid4())
    session = get_session(session_id)

    # Process message with timing
    start = time.time()
    result = process_user_message(request.message, session)
    latency_ms = round((time.time() - start) * 1000, 1)
    response_text = result["text"] if isinstance(result, dict) else result
    msg_type = result.get("message_type", "normal") if isinstance(result, dict) else "normal"
    prod_info = result.get("product_info") if isinstance(result, dict) else None

    # Track in Langfuse
    track_conversation(
        session_id=session_id,
        user_message=request.message,
        assistant_response=response_text,
        metadata={"message_length": len(request.message), "latency_ms": latency_ms}
    )

    return ChatResponse(response=response_text, session_id=session_id, message_type=msg_type, product_info=prod_info)




@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint - SSE events for real-time frontend rendering"""
    import json as _js
    session_id = request.session_id or str(uuid.uuid4())
    session = get_session(session_id)
    from agent.commerce_agent import async_stream_agent_response
    
    async def gen():
        yield f"data: {_js.dumps({'type': 'meta', 'session_id': session_id, 'startup_time': SERVER_STARTUP_TIME})}\n\n"
        _start = time.time()
        _collected = ""
        async for event in async_stream_agent_response(request.message, session):
            yield f"data: {_js.dumps(event)}\n\n"
            if event.get("type") == "chunk":
                _collected += event.get("content", "")
            elif event.get("type") == "tool_result":
                _collected = event.get("text", "")
        try:
            latency_ms = int((time.time() - _start) * 1000)
            track_conversation(
                session_id=session_id,
                user_message=request.message,
                assistant_response=_collected,
                metadata={"latency_ms": latency_ms}
            )
            record_latency(latency_ms)
            record_session_eval(request.message, _collected)
        except Exception as _e_trace:
            import logging
            logging.getLogger("bagbuddy.tracking").warning(f"Tracking error: {_e_trace}")


    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
@app.post("/api/search")
async def search(request: SearchRequest):
    """Search for products"""
    results = search_products(request.query)

    # Track search
    track_search(
        session_id="api_search",
        query=request.query,
        results_count=len(results)
    )

    return {"query": request.query, "results": results[:10], "count": len(results)}


@app.post("/api/purchase")
async def initiate_purchase(request: PurchaseRequest):
    """Initiate a purchase with Stripe"""
    product = get_product_by_id(request.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        checkout_url = create_checkout_session(
            product_id=product["id"],
            product_name=product["name"],
            price=product["price"],
            user_email=request.user_email
        )

        # Track purchase
        track_purchase(
            session_id="api_purchase",
            product_id=product["id"],
            amount=product["price"],
            success=True
        )

        try:
            acct = get_stripe_account_info()
            merchant_name = acct.get("business_name", "Stripe Merchant")
        except:
            merchant_name = "Stripe Merchant"
        return {
            "checkout_url": checkout_url,
            "product_name": product["name"],
            "price": product["price"],
            "email": request.user_email,
            "merchant_name": merchant_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cart/add")
async def add_to_cart(request: CartAddRequest):
    """Add a product to user's cart"""
    product = get_product_by_id(request.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    sid = request.session_id
    if sid not in cart_store:
        cart_store[sid] = []

    # Check if already in cart
    for item in cart_store[sid]:
        if item["product_id"] == request.product_id:
            item["quantity"] += 1
            return {"cart": cart_store[sid], "message": f"{product['name']} quantity updated"}

    cart_store[sid].append({
        "product_id": product["id"],
        "name": product["name"],
        "price": product["price"],
        "quantity": 1
    })
    return {"cart": cart_store[sid], "message": f"{product['name']} added to cart"}


@app.post("/api/cart/remove")
async def remove_from_cart(request: CartRemoveRequest):
    """Remove a product from user's cart"""
    sid = request.session_id
    if sid not in cart_store:
        raise HTTPException(status_code=404, detail="Cart is empty")

    original_len = len(cart_store[sid])
    cart_store[sid] = [item for item in cart_store[sid] if item["product_id"] != request.product_id]

    if len(cart_store[sid]) == original_len:
        raise HTTPException(status_code=404, detail="Item not found in cart")

    return {"cart": cart_store[sid], "message": "Item removed from cart"}


@app.post("/api/cart/update")
async def update_cart_qty(request: CartUpdateRequest):
    """Update quantity for a product in cart (set exact quantity)"""
    sid = request.session_id
    if sid not in cart_store:
        raise HTTPException(status_code=404, detail="Cart is empty")

    new_qty = request.quantity

    for item in cart_store[sid]:
        if item["product_id"] == request.product_id:
            if new_qty <= 0:
                cart_store[sid] = [i for i in cart_store[sid] if i["product_id"] != request.product_id]
                return {"cart": cart_store[sid], "message": "Item removed"}
            item["quantity"] = new_qty
            return {"cart": cart_store[sid], "message": "Quantity updated"}

    raise HTTPException(status_code=404, detail="Item not found in cart")


@app.get("/api/cart")
async def get_cart(session_id: str):
    """Get current user's cart"""
    items = cart_store.get(session_id, [])
    total = sum(item["price"] * item["quantity"] for item in items)
    return {"items": items, "total": total, "count": sum(item["quantity"] for item in items)}


@app.post("/api/cart/checkout")
async def checkout_cart(request: CartAddRequest):  # reuse CartAddRequest which has session_id
    """Checkout cart items via proxy - single Stripe session with multiple line items"""
    sid = request.session_id
    # Accept items from request body first, fall back to server-side cart_store
    items = request.items if request.items else cart_store.get(sid, [])

    if not items:
        raise HTTPException(status_code=400, detail="Cart is empty")

    user_email = request.user_email or ""

    try:
        checkout_url = create_line_items_checkout(items, user_email=user_email if user_email else None)

        # Clear cart after checkout initiated
        cart_store[sid] = []

        # Get merchant name for display
        try:
            acct = get_stripe_account_info()
            merchant_name = acct.get("business_name", "Stripe Merchant")
        except:
            merchant_name = "Stripe Merchant"

        total = sum(item.get("price", 0) * item.get("quantity", 1) for item in items)
        product_names = ", ".join(item.get("name", "Product") for item in items)
        item_count = sum(item.get("quantity", 1) for item in items)

        return {
            "checkout_url": checkout_url,
            "product_name": f"{item_count} item(s) in cart",
            "product_names": product_names,
            "price": total,
            "merchant_name": merchant_name,
            "email": user_email or "",
            "message": "Checkout initiated"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/payment/status/{session_id}")
async def payment_status(session_id: str):
    """Check payment status"""
    try:
        status = get_stripe_session(session_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/evaluation/run")
async def run_evaluation():
    """Refresh evaluation metrics from latest Langfuse traces"""
    # Data comes from Langfuse cloud — always fresh, no background task needed
    return {"status": "refreshed", "source": "langfuse_traces"}


@app.get("/api/dashboard-data")
async def dashboard_data():
    """Get real observability data from Langfuse"""
    return get_dashboard_data()


@app.get("/api/evaluation-data")
async def evaluation_data():
    """Get evaluation data computed from real conversation traces (Langfuse)"""
    return get_evaluation_data_from_traces()


# Dashboard Pages
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Observability dashboard"""
    return HTMLResponse(content=get_dashboard_html())


@app.get("/evaluation-dashboard", response_class=HTMLResponse)
async def evaluation_dashboard():
    """Evaluation dashboard"""
    return HTMLResponse(content=get_evaluation_dashboard_html())


@app.get("/success")
async def payment_success():
    """Payment success page"""
    return HTMLResponse(content=get_success_html())


@app.get("/cancel")
async def payment_cancel():
    """Payment cancel page"""
    return HTMLResponse(content=get_cancel_html())


@app.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events"""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = httpx.post(
            PROXY_BASE_URL.rstrip(chr(47)) + chr(47) + "stripe/webhook",
            content=await request.body(),
            headers={"stripe-signature": sig_header},
            timeout=10
        ).json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Webhook error: " + str(e))

    # Handle the event
    if event.get("type", "") == "checkout.session.completed":
        session = event
        track_purchase(
            session_id=session.get("metadata", {}).get("session_id", "webhook"),
            product_id=session.get("metadata", {}).get("product_id", ""),
            amount=session.get("amount_total", 0) / 100,
            success=True
        )
    elif event.get("type", "") == "checkout.session.expired":
        session = event
        track_purchase(
            session_id=session.get("metadata", {}).get("session_id", "webhook"),
            product_id=session.get("metadata", {}).get("product_id", ""),
            amount=session.get("amount_total", 0) / 100,
            success=False
        )

    return {"status": "success"}


# HTML Templates
def get_dashboard_html() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BagBuddy - Observability Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=Instrument+Serif&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #faf9f7;
            --surface: #f0eeeb;
            --text: #1a1a1a;
            --text-secondary: #6b6b6b;
            --accent: #c8553d;
            --border: #e0ddd8;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'DM Sans', sans-serif;
            font-weight: 400;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }

        /* Navigation */
        .nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 32px;
            border-bottom: 1px solid var(--border);
        }

        .nav-brand {
            font-family: 'Instrument Serif', serif;
            font-size: 22px;
            color: var(--text);
            text-decoration: none;
        }

        .nav-links {
            display: flex;
            gap: 24px;
        }

        .nav-links a {
            font-family: 'DM Sans', sans-serif;
            font-size: 14px;
            font-weight: 500;
            color: var(--text-secondary);
            text-decoration: none;
        }

        .nav-links a:hover {
            color: var(--text);
        }

        .nav-links a.active {
            color: var(--accent);
        }

        /* Container */
        .container {
            max-width: 960px;
            margin: 0 auto;
            padding: 48px 32px;
        }

        /* Metrics Section */
        .metrics {
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
            gap: 0;
            margin-bottom: 64px;
        }

        .metric {
            flex: 1;
            text-align: center;
            padding: 0 24px;
        }

        .metric:first-child {
            padding-left: 0;
        }

        .metric:last-child {
            padding-right: 0;
        }

        .metric + .metric {
            border-left: 1px solid var(--border);
        }

        .metric-value {
            font-family: 'Instrument Serif', serif;
            font-size: 48px;
            font-weight: 300;
            color: var(--text);
            line-height: 1.1;
        }

        .metric-label {
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-top: 8px;
        }

        .loading {
            color: var(--text-secondary);
            font-style: italic;
            font-size: 20px;
        }

        .error-msg {
            color: var(--accent);
            font-size: 14px;
        }

        /* Section Title */
        .section-title {
            font-family: 'Instrument Serif', serif;
            font-size: 24px;
            font-weight: 400;
            color: var(--text);
            margin-bottom: 24px;
        }

        /* Chart Section */
        .chart-section {
            margin-bottom: 64px;
        }

        .bar-chart {
            display: flex;
            align-items: flex-end;
            gap: 16px;
            height: 160px;
            padding-top: 24px;
        }

        .bar-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100%;
            justify-content: flex-end;
        }

        .bar {
            width: 8px;
            background: rgba(200, 85, 61, 0.35);
            position: relative;
            min-height: 2px;
        }

        .bar-value {
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 6px;
            font-weight: 500;
        }

        .bar-label {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 8px;
        }

        /* Traces Section */
        .traces-section {
            margin-bottom: 48px;
        }

        .traces-table {
            width: 100%;
            border-collapse: collapse;
        }

        .traces-table th {
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }

        .traces-table td {
            font-size: 14px;
            color: var(--text);
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
        }

        .traces-table td.mono {
            font-family: 'DM Mono', 'SFMono-Regular', Consolas, monospace;
            font-size: 13px;
        }

        .status-success {
            color: var(--accent);
        }

        .traces-table .empty-row td {
            color: var(--text-secondary);
            font-style: italic;
        }

        /* Langfuse Link */
        .langfuse-link {
            color: var(--accent);
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
        }

        .langfuse-link:hover {
            text-decoration: underline;
        }

        /* Responsive */
        @media (max-width: 640px) {
            .metrics {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 32px 24px;
            }

            .metric + .metric {
                border-left: none;
            }

            .metric {
                padding: 0;
            }

            .container {
                padding: 32px 20px;
            }
        }
    </style>
</head>
<body>
    <nav class="nav">
        <a href="/" class="nav-brand">BagBuddy</a>
        <div class="nav-links">
            <a href="/">Chat</a>
            <a href="/dashboard" class="active">Dashboard</a>
            <a href="/evaluation-dashboard">Evaluation</a>
        </div>
    </nav>

    <div class="container">
        <!-- Metrics -->
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="total-conversations"><span class="loading">—</span></div>
                <div class="metric-label">Total Conversations</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="avg-response-time"><span class="loading">—</span></div>
                <div class="metric-label">Avg Response Time</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="conversion-rate"><span class="loading">—</span></div>
                <div class="metric-label">Purchase Conversion</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="search-success"><span class="loading">—</span></div>
                <div class="metric-label">Search Success Rate</div>
            </div>
        </div>

        <!-- Daily Volume Chart -->
        <div class="chart-section">
            <h2 class="section-title">Daily Volume</h2>
            <div class="bar-chart" id="daily-volume-chart"></div>
        </div>

        <!-- Recent Traces -->
        <div class="traces-section">
            <h2 class="section-title">Recent Traces</h2>
            <table class="traces-table">
                <thead>
                    <tr>
                        <th>Trace ID</th>
                        <th>Type</th>
                        <th>Input</th>
                        <th>Status</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody id="traces-table">
                    <tr class="empty-row"><td colspan="5">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Langfuse Link -->
        <a href="#" id="langfuse-link" target="_blank" class="langfuse-link">View in Langfuse →</a>
    </div>

    <script>
        async function loadDashboard() {
            try {
                const res = await fetch('/api/dashboard-data');
                const data = await res.json();

                if (data.error) {
                    document.querySelectorAll('.metric-value').forEach(el => {
                        el.innerHTML = '<span class="error-msg">' + data.error + '</span>';
                    });
                    return;
                }

                document.getElementById('total-conversations').textContent = data.total_conversations;
                document.getElementById('avg-response-time').textContent = data.avg_response_time;
                document.getElementById('conversion-rate').textContent = data.conversion_rate;
                document.getElementById('search-success').textContent = data.search_success_rate;
                document.getElementById('langfuse-link').href = data.langfuse_url;

                // Daily volume chart
                const chartEl = document.getElementById('daily-volume-chart');
                const maxCount = Math.max(...data.daily_volume.map(d => d.count), 1);
                chartEl.innerHTML = data.daily_volume.map(d => {
                    const h = Math.max((d.count / maxCount) * 100, 2);
                    return '<div class="bar-wrapper">' +
                        '<span class="bar-value">' + d.count + '</span>' +
                        '<div class="bar" style="height:' + h + '%"></div>' +
                        '<span class="bar-label">' + d.day + '</span>' +
                    '</div>';
                }).join('');

                // Recent traces table
                const tbody = document.getElementById('traces-table');
                if (data.recent_traces.length === 0) {
                    tbody.innerHTML = '<tr class="empty-row"><td colspan="5">No traces yet. Start chatting to see data here.</td></tr>';
                } else {
                    tbody.innerHTML = data.recent_traces.map(t =>
                        '<tr>' +
                        '<td class="mono">' + t.id + '</td>' +
                        '<td>' + t.name + '</td>' +
                        '<td>' + t.input + '</td>' +
                        '<td class="status-success">Success</td>' +
                        '<td>' + t.latency + '</td>' +
                        '</tr>'
                    ).join('');
                }
            } catch (e) {
                document.querySelectorAll('.metric-value').forEach(el => {
                    el.innerHTML = '<span class="error-msg">Failed to load data</span>';
                });
            }
        }
        loadDashboard();
        setInterval(loadDashboard, 30000);
    </script>
</body>
</html>
"""


def get_evaluation_dashboard_html() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BagBuddy - Evaluation Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=Instrument+Serif&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #faf9f7;
            --surface: #f0eeeb;
            --text: #1a1a1a;
            --text-secondary: #6b6b6b;
            --accent: #c8553d;
            --border: #e0ddd8;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'DM Sans', sans-serif;
            font-weight: 400;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }

        .nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 32px;
            border-bottom: 1px solid var(--border);
        }

        .nav-brand {
            font-family: 'Instrument Serif', serif;
            font-size: 22px;
            color: var(--text);
            text-decoration: none;
        }

        .nav-links {
            display: flex;
            gap: 24px;
        }

        .nav-links a {
            font-family: 'DM Sans', sans-serif;
            font-size: 14px;
            font-weight: 500;
            color: var(--text-secondary);
            text-decoration: none;
        }

        .nav-links a:hover {
            color: var(--text);
        }

        .nav-links a.active {
            color: var(--accent);
        }

        .container {
            max-width: 720px;
            margin: 0 auto;
            padding: 64px 32px;
        }

        .top-metrics {
            display: flex;
            gap: 80px;
            margin-bottom: 56px;
        }

        .metric-block {}

        .metric-value-lg {
            font-family: 'Instrument Serif', serif;
            font-weight: 300;
            font-size: 64px;
            line-height: 1;
            color: var(--text);
        }

        .metric-value-md {
            font-family: 'Instrument Serif', serif;
            font-weight: 300;
            font-size: 32px;
            line-height: 1;
            color: var(--text);
        }

        .metric-label {
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
            margin-top: 8px;
        }

        .ragas-section {
            margin-bottom: 48px;
        }

        .ragas-section-title {
            font-family: 'Instrument Serif', serif;
            font-weight: 400;
            font-size: 20px;
            color: var(--text);
            margin-bottom: 32px;
        }

        .ragas-metric {
            margin-bottom: 28px;
        }

        .ragas-metric:last-child {
            margin-bottom: 0;
        }

        .ragas-metric-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 8px;
        }

        .ragas-metric-name {
            font-size: 14px;
            color: var(--text);
        }

        .ragas-metric-value {
            font-size: 24px;
            font-weight: 500;
            color: var(--text);
        }

        .progress-line {
            width: 100%;
            height: 2px;
            background: var(--border);
        }

        .progress-line-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.5s ease, opacity 0.3s ease;
        }

        .actions {
            display: flex;
            align-items: center;
            gap: 32px;
            margin-top: 48px;
        }

        .link-btn {
            font-family: 'DM Sans', sans-serif;
            font-size: 14px;
            font-weight: 500;
            color: var(--accent);
            text-decoration: none;
            cursor: pointer;
            background: none;
            border: none;
            padding: 0;
        }

        .link-btn:hover {
            text-decoration: underline;
        }

        .link-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            text-decoration: none;
        }

        .langfuse-link {
            font-family: 'DM Sans', sans-serif;
            font-size: 14px;
            font-weight: 500;
            color: var(--accent);
            text-decoration: none;
        }

        .langfuse-link:hover {
            text-decoration: underline;
        }

        .loading {
            color: var(--text-secondary);
            font-style: italic;
            font-size: 14px;
        }

        .error-msg {
            color: var(--accent);
            font-size: 14px;
        }

        .metric-legend {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 40px;
            padding-top: 32px;
            border-top: 1px solid var(--border);
        }

        .legend-card {
            background: var(--surface);
            padding: 20px;
            border-radius: 4px;
        }

        .legend-name {
            font-size: 13px;
            font-weight: 500;
            color: var(--text);
            margin-bottom: 6px;
        }

        .legend-score {
            font-family: 'Instrument Serif', serif;
            font-weight: 300;
            font-size: 28px;
            color: var(--text);
            margin-bottom: 10px;
        }

        .legend-desc {
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <nav class="nav">
        <a href="/" class="nav-brand">BagBuddy</a>
        <div class="nav-links">
            <a href="/">Chat</a>
            <a href="/dashboard">Dashboard</a>
            <a href="/evaluation-dashboard" class="active">Evaluation</a>
        </div>
    </nav>

    <div class="container">
        <div class="top-metrics">
            <div class="metric-block">
                <div class="metric-value-lg" id="overall-score"><span class="loading">Loading...</span></div>
                <div class="metric-label">Overall Score</div>
            </div>
            <div class="metric-block">
                <div class="metric-value-md" id="total-evals"><span class="loading">Loading...</span></div>
                <div class="metric-label">Total Evaluations</div>
            </div>
            <div class="metric-block">
                <div class="metric-value-sm" id="judge-badge">LLM-as-Judge</div>
                <div class="metric-label">Judge Method</div>
            </div>
        </div>

        <div class="ragas-section">
            <div class="ragas-section-title">Ragas Metrics</div>

            <div class="ragas-metric">
                <div class="ragas-metric-header">
                    <span class="ragas-metric-name">Faithfulness</span>
                    <span class="ragas-metric-value" id="faithfulness"><span class="loading">Loading...</span></span>
                </div>
                <div class="progress-line"><div class="progress-line-fill" id="faithfulness-bar" style="width:0%;opacity:1"></div></div>
            </div>

            <div class="ragas-metric">
                <div class="ragas-metric-header">
                    <span class="ragas-metric-name">Answer Relevancy</span>
                    <span class="ragas-metric-value" id="answer-relevancy"><span class="loading">Loading...</span></span>
                </div>
                <div class="progress-line"><div class="progress-line-fill" id="answer-relevancy-bar" style="width:0%;opacity:1"></div></div>
            </div>

            <div class="ragas-metric">
                <div class="ragas-metric-header">
                    <span class="ragas-metric-name">Context Precision</span>
                    <span class="ragas-metric-value" id="context-precision"><span class="loading">Loading...</span></span>
                </div>
                <div class="progress-line"><div class="progress-line-fill" id="context-precision-bar" style="width:0%;opacity:1"></div></div>
            </div>

            <div class="ragas-metric">
                <div class="ragas-metric-header">
                    <span class="ragas-metric-name">Context Relevance</span>
                    <span class="ragas-metric-value" id="context-relevance"><span class="loading">Loading...</span></span>
                </div>
                <div class="progress-line"><div class="progress-line-fill" id="context-relevance-bar" style="width:0%;opacity:1"></div></div>
            </div>
        </div>

        <div class="metric-legend">
            <div class="legend-card">
                <div class="legend-name">Faithfulness</div>
                <div class="legend-score" id="legend-faith">—</div>
                <div class="legend-desc">Does the response contain real product names and prices from our catalog?</div>
            </div>
            <div class="legend-card">
                <div class="legend-name">Answer Relevancy</div>
                <div class="legend-score" id="legend-relev">—</div>
                <div class="legend-desc">Does the response accurately address what the user asked about?</div>
            </div>
            <div class="legend-card">
                <div class="legend-name">Context Precision</div>
                <div class="legend-score" id="legend-prec">—</div>
                <div class="legend-desc">Did the system produce a valid, error-free response?</div>
            </div>
            <div class="legend-card">
                <div class="legend-name">Context Relevance</div>
                <div class="legend-score" id="legend-rel">—</div>
                <div class="legend-desc">Does the response contain concrete, useful information (price, features, ratings)?</div>
            </div>
        </div>

        <div class="actions">
            
            <a href="#" id="langfuse-link" target="_blank" class="langfuse-link">View in Langfuse &rarr;</a>
        </div>
    </div>

    <script>
        function getAccentOpacity(val) {
            if (val >= 0.8) return 1.0;
            if (val >= 0.6) return 0.6;
            return 0.35;
        }

        async function loadEvaluation() {
            try {
                const res = await fetch('/api/evaluation-data');
                const data = await res.json();

                if (data.error) {
                    document.getElementById('overall-score').innerHTML = '<span class="error-msg">' + data.error + '</span>';
                    document.getElementById('total-evals').innerHTML = '<span class="error-msg">' + data.error + '</span>';
                    ['faithfulness', 'answer-relevancy', 'context-precision', 'context-relevance'].forEach(id => {
                        document.getElementById(id).innerHTML = '<span class="error-msg">' + data.error + '</span>';
                    });
                    return;
                }

                document.getElementById('overall-score').textContent = data.overall_score.toFixed(3);
                document.getElementById('total-evals').textContent = data.total_evaluations;
                document.getElementById('langfuse-link').href = data.langfuse_url;

                const metrics = data.metrics;
                ['faithfulness', 'answer_relevancy', 'context_precision', 'context_relevance'].forEach(key => {
                    const val = metrics[key];
                    const displayId = key.replace(/_/g, '-');
                    const el = document.getElementById(displayId);
                    if (el) {
                        el.textContent = val.toFixed(3);
                    }
                    const barEl = document.getElementById(displayId + '-bar');
                    if (barEl) {
                        barEl.style.width = (val * 100) + '%';
                        barEl.style.opacity = getAccentOpacity(val);
                    }
                });

                // Populate legend cards
                document.getElementById('legend-faith').textContent = metrics.faithfulness.toFixed(3);
                document.getElementById('judge-badge').textContent = data.judge || 'LLM-as-Judge';
                document.getElementById('legend-relev').textContent = metrics.answer_relevancy.toFixed(3);
                document.getElementById('legend-prec').textContent = metrics.context_precision.toFixed(3);
                document.getElementById('legend-rel').textContent = metrics.context_relevance.toFixed(3);
            } catch (e) {
                document.getElementById('overall-score').innerHTML = '<span class="error-msg">Failed to load data</span>';
                document.getElementById('total-evals').innerHTML = '<span class="error-msg">Failed to load data</span>';
                ['faithfulness', 'answer-relevancy', 'context-precision', 'context-relevance'].forEach(id => {
                    document.getElementById(id).innerHTML = '<span class="error-msg">Failed to load data</span>';
                });
            }
        }

        loadEvaluation();
    </script>
</body>
</html>
"""


def get_success_html() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Confirmed</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=Instrument+Serif&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #faf9f7;
            --surface: #f0eeeb;
            --text: #1a1a1a;
            --text-secondary: #6b6b6b;
            --accent: #c8553d;
            --border: #e0ddd8;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'DM Sans', sans-serif;
            background: var(--bg);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            text-align: center;
            padding: 64px 48px;
            max-width: 480px;
        }
        h1 {
            font-family: 'Instrument Serif', serif;
            font-size: 36px;
            font-weight: 400;
            color: var(--text);
            margin-bottom: 16px;
        }
        p {
            font-size: 16px;
            font-weight: 400;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 40px;
        }
        a.back {
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            font-weight: 500;
            color: var(--accent);
            text-decoration: none;
        }
        a.back:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Payment Confirmed</h1>
        <p>Your order has been processed successfully.</p>
        <a href="/" class="back">&larr; Back to Chat</a>
    </div>
</body>
</html>
"""


def get_cancel_html() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Cancelled</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=Instrument+Serif&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #faf9f7;
            --surface: #f0eeeb;
            --text: #1a1a1a;
            --text-secondary: #6b6b6b;
            --accent: #c8553d;
            --border: #e0ddd8;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'DM Sans', sans-serif;
            background: var(--bg);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            text-align: center;
            padding: 64px 48px;
            max-width: 480px;
        }
        h1 {
            font-family: 'Instrument Serif', serif;
            font-size: 36px;
            font-weight: 400;
            color: var(--text);
            margin-bottom: 16px;
        }
        p {
            font-size: 16px;
            font-weight: 400;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 40px;
        }
        a.back {
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            font-weight: 500;
            color: var(--accent);
            text-decoration: none;
        }
        a.back:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Payment Cancelled</h1>
        <p>Your payment was not completed. You can try again.</p>
        <a href="/" class="back">&larr; Back to Chat</a>
    </div>
</body>
</html>
"""



# ===== RAG & Text-to-SQL Endpoints =====

@app.post("/api/products/seed")
async def seed_database():
    """Seed the Supabase database with 30 digital products and embeddings"""
    import time as ttime
    success_count = 0
    embed_count = 0
    errors = []

    # Clear existing data
    try:
        clear_all_data()
    except Exception as e:
        errors.append(f"Clear warning: {e}")

    try:
        from data.digital_products import DIGITAL_PRODUCTS
    except ImportError:
        return {"error": "Seed data not available", "detail": "Run BagBuddy from the full repository with data/ directory"}
    for product in DIGITAL_PRODUCTS:
        try:
            upsert_product(product)
            success_count += 1
            chunks = prepare_product_chunks(product)
            for chunk in chunks:
                embedding = create_embedding(chunk["chunk_text"])
                if embedding and len(embedding) == 512:
                    upsert_embedding(
                        product_id=product["id"],
                        chunk_text=chunk["chunk_text"],
                        chunk_type=chunk["chunk_type"],
                        embedding=embedding,
                    )
                    embed_count += 1
                ttime.sleep(0.05)
        except Exception as e:
            errors.append(f"{product['id']}: {e}")

    return {
        "products_seeded": success_count,
        "embeddings_created": embed_count,
        "errors": errors,
        "categories": list(set(p["category"] for p in DIGITAL_PRODUCTS)),
    }


@app.get("/api/rag/health")
async def rag_health():
    """Check Supabase connection and product count"""
    try:
        from services.supabase_service import get_all_products_db
        products = get_all_products_db()
        return {"status": "ok", "products_in_db": len(products), "connection": "proxy"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)


