import json, uuid, time
from datetime import datetime, timezone
from langfuse.api.client import LangfuseAPI
from langfuse.api.ingestion.types.ingestion_event import IngestionEvent_TraceCreate
from langfuse.api.ingestion.types.trace_body import TraceBody
from config import PROXY_BASE_URL, LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

_lf_api = None

def _get_api():
    global _lf_api
    if _lf_api is None:
        # 1) Direct credentials preferred (bypass blocked proxy)
        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            try:
                _lf_api = LangfuseAPI(
                    base_url=LANGFUSE_HOST.rstrip("/"),
                    username=LANGFUSE_PUBLIC_KEY,
                    password=LANGFUSE_SECRET_KEY,
                )
                return _lf_api
            except Exception:
                pass
        # 2) Proxy fallback (holds credentials for users without direct keys)
        if PROXY_BASE_URL:
            try:
                _lf_api = LangfuseAPI(base_url=PROXY_BASE_URL)
            except Exception:
                pass
    return _lf_api


def track_conversation(session_id, user_message, assistant_response, metadata=None):
    api = _get_api()
    if not api: return
    try:
        now = datetime.now(timezone.utc)
        trace_id = str(uuid.uuid4())
        meta = dict(metadata or {})
        meta["session_id"] = session_id
        api.ingestion.batch(batch=[IngestionEvent_TraceCreate(
            id=str(uuid.uuid4()),
            timestamp=now.isoformat(),
            body=TraceBody(
                id=trace_id,
                name="commerce_agent_conversation",
                timestamp=now,
                input={"message": user_message},
                output={"response": assistant_response},
                metadata=meta,
            )
        )])
    except Exception:
        pass


def track_purchase(session_id, product_id, amount, success):
    api = _get_api()
    if not api: return
    try:
        now = datetime.now(timezone.utc)
        api.ingestion.batch(batch=[IngestionEvent_TraceCreate(
            id=str(uuid.uuid4()),
            timestamp=now.isoformat(),
            body=TraceBody(
                id=str(uuid.uuid4()),
                name="purchase_flow",
                timestamp=now,
                input={"action": "purchase", "product_id": product_id},
                output={"success": success, "amount": amount},
                metadata={"session_id": session_id, "product_id": product_id},
            )
        )])
    except Exception:
        pass


def track_search(session_id, query, results_count):
    api = _get_api()
    if not api: return
    try:
        now = datetime.now(timezone.utc)
        api.ingestion.batch(batch=[IngestionEvent_TraceCreate(
            id=str(uuid.uuid4()),
            timestamp=now.isoformat(),
            body=TraceBody(
                id=str(uuid.uuid4()),
                name="product_search",
                timestamp=now,
                input={"query": query},
                output={"results_count": results_count},
                metadata={"session_id": session_id, "query": query},
            )
        )])
    except Exception:
        pass


def get_dashboard_url():
    return LANGFUSE_HOST + "/projects"
