"""Stripe payment service - proxied through Cloudflare Worker"""
import httpx
from config import PROXY_BASE_URL


def create_checkout_session(
    product_id: str,
    product_name: str,
    price: float,
    user_email: str = None
) -> str:
    """Create a Stripe Checkout session via proxy and return the payment URL"""
    if not PROXY_BASE_URL:
        raise Exception("PROXY_BASE_URL not configured. Set it in .env")

    payload = {
        "product_id": product_id,
        "product_name": product_name,
        "price": str(price),
        "success_url": "http://localhost:9000/success?session_id={CHECKOUT_SESSION_ID}",
        "cancel_url": "http://localhost:9000/cancel",
    }
    if user_email:
        payload["user_email"] = user_email

    try:
        resp = httpx.post(f"{PROXY_BASE_URL}/stripe/checkout", json=payload,
            timeout=httpx.Timeout(connect=4.0, read=10.0, write=4.0, pool=4.0))
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise Exception(data["error"])
        return data.get("url", "")
    except httpx.HTTPStatusError as e:
        raise Exception(f"Stripe proxy error: {e.response.text}")
    except Exception as e:
        raise Exception(f"Stripe proxy error: {str(e)}")


def get_session(session_id: str) -> dict:
    """Retrieve a Checkout session by ID via proxy"""
    if not PROXY_BASE_URL:
        return {"id": session_id, "status": "unknown"}
    try:
        resp = httpx.post(f"{PROXY_BASE_URL}/stripe/session",
            json={"session_id": session_id},
            timeout=httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0))
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"id": session_id, "status": "unknown"}


def get_stripe_account_info() -> dict:
    """Get Stripe account info via proxy"""
    if not PROXY_BASE_URL:
        return {"business_name": "Stripe Merchant", "email": "", "country": ""}
    try:
        resp = httpx.get(f"{PROXY_BASE_URL}/stripe/account",
            timeout=httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0))
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"business_name": "Stripe Merchant", "email": "", "country": ""}
