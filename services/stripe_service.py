"""Stripe payment service - proxied through Cloudflare Worker"""
import httpx
from config import PROXY_BASE_URL

_proxy = None

def _get_proxy():
    global _proxy
    if _proxy is None:
        _proxy = PROXY_BASE_URL
    return _proxy


def create_checkout_session(
    product_id: str,
    product_name: str,
    price: float,
    user_email: str = None
) -> str:
    """Create a Stripe Checkout session via proxy and return the payment URL"""
    proxy = _get_proxy()
    if not proxy:
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
        resp = httpx.post(f"{proxy}/stripe/checkout", json=payload, timeout=15)
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
    proxy = _get_proxy()
    if not proxy:
        return {"id": session_id, "status": "unknown"}

    try:
        resp = httpx.post(f"{proxy}/stripe/session", json={"session_id": session_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"id": session_id, "status": "unknown"}


def get_stripe_account_info() -> dict:
    """Get Stripe account info via proxy"""
    proxy = _get_proxy()
    if not proxy:
        return {"business_name": "Stripe Merchant", "email": "", "country": ""}

    try:
        resp = httpx.get(f"{proxy}/stripe/account", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"business_name": "Stripe Merchant", "email": "", "country": ""}
