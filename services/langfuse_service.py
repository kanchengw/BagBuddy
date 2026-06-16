import time
from datetime import datetime, timedelta, timezone
from langfuse.api.client import LangfuseAPI
from config import PROXY_BASE_URL, LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

_lf_api = None

def _get_lf_api():
    global _lf_api
    if _lf_api is None:
        # 1) Direct credentials preferred
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
        # 2) Proxy fallback
        if PROXY_BASE_URL:
            try:
                _lf_api = LangfuseAPI(base_url=PROXY_BASE_URL)
            except Exception:
                pass
    return _lf_api


_latencies = []
_cache = {"data": None, "timestamp": 0}
_session_evals = []
CACHE_TTL = 30
SERVER_START_TIME = datetime.now(timezone.utc)
def record_latency(latency_ms):
    _latencies.append(latency_ms / 1000.0)


def record_session_eval(question, answer):
    if not (question and answer and len(answer) > 10):
        return
    try:
        from services.llm_judge import evaluate_conversation
        score = evaluate_conversation(question, answer)
        if isinstance(score, dict) and "faithfulness" in score:
            _session_evals.append(score)
    except Exception:
        pass


def reset_session_eval():
    global SERVER_START_TIME
    _latencies.clear()
    _session_evals.clear()
    _cache["data"] = None
    _cache["timestamp"] = 0
    SERVER_START_TIME = datetime.now(timezone.utc)


def get_session_eval_count():
    return len(_session_evals)


def _empty_eval_result():
    return {
        "overall_score": 0,
        "metrics": {
            "faithfulness": 0,
            "answer_relevancy": 0,
            "context_precision": 0,
            "context_relevance": 0,
        },
        "total_evaluations": 0,
        "langfuse_url": LANGFUSE_HOST,
        "judge": "LLM-as-Judge",
    }


def get_dashboard_data(hours=168):
    now_ts = time.time()
    if _cache["data"] and (now_ts - _cache["timestamp"]) < CACHE_TTL:
        return _cache["data"]

    try:
        api = _get_lf_api()
        now = datetime.now(timezone.utc)
        from_ts = max(now - timedelta(hours=hours), SERVER_START_TIME)

        traces_data = []
        _use_local = False
        if api:
            try:
                resp = api.trace.list(
                    from_timestamp=from_ts,
                    to_timestamp=now,
                    limit=100,
                    order_by="timestamp.desc",
                )
                for t in resp.data:
                    traces_data.append({
                        "id": t.id,
                        "name": t.name or "",
                        "timestamp": str(t.timestamp) if t.timestamp else "",
                        "input": t.input,
                        "output": t.output,
                        "metadata": t.metadata or {},
                        "latency": t.latency,
                    })
            except Exception:
                pass

        total_conversations = len(traces_data)
        purchase_count = 0
        search_count = 0
        successful_searches = 0
        latencies = []

        for t_obj in traces_data:
            name = t_obj.get("name", "") or ""
            inp = str(t_obj.get("input", "") or "")
            inp_lower = inp.lower()
            out = str(t_obj.get("output", "") or "")

            if "purchase" in name.lower() or "buy" in inp_lower or "checkout" in inp_lower:
                purchase_count += 1

            search_keywords = ["show me", "search", "what do you have", "products under", "compare"]
            if any(kw in inp_lower for kw in search_keywords):
                search_count += 1
                if any(kw in out.lower() for kw in ["$", "price", "product"]):
                    successful_searches += 1

            meta = t_obj.get("metadata") or {}
            if isinstance(meta, dict) and "latency_ms" in meta:
                try:
                    latencies.append(float(meta["latency_ms"]) / 1000.0)
                except Exception:
                    pass

        conversion_rate = (purchase_count / total_conversations * 100) if total_conversations > 0 else 0
        search_success_rate = (successful_searches / search_count * 100) if search_count > 0 else 0
        all_lats = latencies + _latencies
        avg_latency = sum(all_lats) / len(all_lats) if all_lats else 0

        recent_traces = []
        for t_obj in traces_data[:10]:
            inp_text = str(t_obj.get("input", "") or "")[:50]
            if len(str(t_obj.get("input", "") or "")) > 50:
                inp_text += "..."
            meta = t_obj.get("metadata") or {}
            trace_latency = None
            if isinstance(meta, dict) and "latency_ms" in meta:
                try:
                    trace_latency = float(meta["latency_ms"]) / 1000.0
                except Exception:
                    pass
            if trace_latency is None and t_obj.get("latency") is not None:
                trace_latency = float(t_obj["latency"])
            if trace_latency is None and avg_latency > 0:
                trace_latency = avg_latency
            latency_str = f"{trace_latency:.1f}s" if trace_latency else "N/A"
            recent_traces.append({
                "id": (t_obj.get("id", "") or "")[:12] + "..." if t_obj.get("id") else "N/A",
                "name": t_obj.get("name", "") or "chat",
                "input": inp_text,
                "latency": latency_str,
            })

        daily_volume = []
        for i in range(7, 0, -1):
            day_from = now - timedelta(days=i)
            day_to = day_from + timedelta(days=1)
            count = 0
            for t in traces_data:
                ts = t.get("timestamp", "")
                if ts and day_from.isoformat()[:19] <= str(ts)[:19] < day_to.isoformat()[:19]:
                    count += 1
            daily_volume.append({"day": day_from.strftime("%a"), "count": count})

        result = {
            "total_conversations": total_conversations,
            "total_conversations_display": str(total_conversations),
            "avg_response_time": f"{avg_latency:.1f}s" if avg_latency > 0 else "N/A",
            "conversion_rate": f"{conversion_rate:.1f}%",
            "search_success_rate": f"{search_success_rate:.1f}%",
            "purchase_count": purchase_count,
            "daily_volume": daily_volume,
            "recent_traces": recent_traces,
            "langfuse_url": LANGFUSE_HOST,
            "avg_latency": avg_latency,
        }
        _cache["data"] = result
        _cache["timestamp"] = now_ts
        return result
    except Exception:
        fallback = {"total_conversations": 0, "total_conversations_display": "0",
                    "avg_response_time": "N/A", "conversion_rate": "0.0%",
                    "search_success_rate": "0.0%", "purchase_count": 0,
                    "daily_volume": [], "recent_traces": [],
                    "langfuse_url": LANGFUSE_HOST, "avg_latency": 0}
        _cache["data"] = fallback
        _cache["timestamp"] = now_ts
        return fallback


def get_evaluation_data_from_traces(hours=168):
    scores = list(_session_evals)
    if not scores:
        return _empty_eval_result()
    n = len(scores)
    result = {
        "overall_score": round((
            sum(s["faithfulness"] for s in scores) / n +
            sum(s["answer_relevancy"] for s in scores) / n +
            sum(s["context_precision"] for s in scores) / n +
            sum(s["context_relevance"] for s in scores) / n
        ) / 4, 3),
        "metrics": {
            "faithfulness": round(sum(s["faithfulness"] for s in scores) / n, 3),
            "answer_relevancy": round(sum(s["answer_relevancy"] for s in scores) / n, 3),
            "context_precision": round(sum(s["context_precision"] for s in scores) / n, 3),
            "context_relevance": round(sum(s["context_relevance"] for s in scores) / n, 3),
        },
        "total_evaluations": n,
        "langfuse_url": LANGFUSE_HOST,
        "judge": "LLM-as-Judge",
    }
    return result
