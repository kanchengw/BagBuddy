"""LLM-as-Judge evaluation using CNLLM.
Evaluates conversation traces using CNLLM as the judge.
Supports batch evaluation (multiple conversations per API call).
"""
import json
import sys

sys.path.insert(0, ".")
from cnllm import CNLLM
from config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL

_judge_client = None


def _get_judge():
    global _judge_client
    if _judge_client is None:
        _judge_client = CNLLM(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _judge_client


# Single conversation evaluation prompt
SINGLE_JUDGE_PROMPT = """You are an expert evaluator for a shopping assistant. The assistant should answer with relevant product info. Rate the assistant's response.

Question: {question}
Answer: {answer}

Rate each metric from 0.0 to 1.0:
- faithfulness: Is the product info factually accurate (real product names/prices)? (1.0 = accurate)
- answer_relevancy: Does it directly answer what was asked? (1.0 = directly relevant)
- context_precision: Is it concise without unnecessary fluff? (1.0 = precise)
- context_relevance: Is the detail level appropriate for the question? (1.0 = just right)

Return ONLY JSON: {{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_relevance": 0.0}}"""


# Batch evaluation prompt (multiple conversations in one call)
BATCH_JUDGE_PROMPT = """You are an expert AI evaluation judge. Evaluate each conversation below.

{conversations}

For each conversation, rate from 0.0 to 1.0:
- faithfulness: Factual accuracy (1.0 = fully accurate)
- answer_relevancy: Relevance to question (1.0 = perfectly relevant)
- context_precision: Concise without fluff (1.0 = perfectly precise)
- context_relevance: Appropriate detail level (1.0 = perfect)

Return ONLY a JSON array where each element corresponds to one conversation:
[{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_relevance": 0.0}}]"""


def _build_batch_text(conversations):
    """Build numbered conversation list for batch prompt."""
    parts = []
    for i, (question, answer) in enumerate(conversations, 1):
        parts.append(f"Conversation {i}:")
        parts.append(f"Question: {question[:300]}")
        parts.append(f"Answer: {answer[:500]}")
        parts.append("")
    return "\n".join(parts)


def _parse_json_response(text):
    """Extract and parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    return json.loads(text)


def evaluate_conversation(question: str, answer: str) -> dict:
    """Evaluate a single conversation turn using LLM-as-Judge."""
    if not question or not answer or len(answer) < 5:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_relevance": 0.0}

    prompt = SINGLE_JUDGE_PROMPT.format(question=question[:500], answer=answer[:1000])
    client = _get_judge()

    try:
        resp = client.chat.create(messages=[{"role": "user", "content": prompt}], stream=False)
        text = resp.still
        scores = _parse_json_response(text)
        return {
            "faithfulness": float(scores.get("faithfulness", 0.5)),
            "answer_relevancy": float(scores.get("answer_relevancy", 0.5)),
            "context_precision": float(scores.get("context_precision", 0.5)),
            "context_relevance": float(scores.get("context_relevance", 0.5)),
        }
    except Exception as e:
        print(f"Judge error: {e}", file=sys.stderr)
        return {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_relevance": 0.5}


def evaluate_batch(conversations: list, batch_size: int = 5) -> dict:
    """Evaluate conversations in batches using LLM-as-Judge.

    Args:
        conversations: List of (question, answer) tuples.
        batch_size: Number of conversations per API call.

    Returns:
        Dict with aggregated overall_score, faithfulness, answer_relevancy,
        context_precision, context_relevance.
    """
    if not conversations:
        return {"overall_score": 0, "faithfulness": 0, "answer_relevancy": 0,
                "context_precision": 0, "context_relevance": 0}

    all_scores = []
    client = _get_judge()
    total = len(conversations)

    for start in range(0, total, batch_size):
        batch = conversations[start:start + batch_size]

        if len(batch) == 1:
            # Single conversation - use simpler prompt
            q, a = batch[0]
            if not q or not a or len(a) < 5:
                all_scores.append({"faithfulness": 0.0, "answer_relevancy": 0.0,
                                   "context_precision": 0.0, "context_relevance": 0.0})
                continue
            prompt = SINGLE_JUDGE_PROMPT.format(question=q[:500], answer=a[:1000])
        else:
            # Multiple conversations - use batch prompt
            prompt = BATCH_JUDGE_PROMPT.format(conversations=_build_batch_text(batch))

        try:
            resp = client.chat.create(messages=[{"role": "user", "content": prompt}], stream=False)
            text = resp.still

            result = _parse_json_response(text)
            if isinstance(result, list):
                for item in result:
                    all_scores.append({
                        "faithfulness": float(item.get("faithfulness", 0.5)),
                        "answer_relevancy": float(item.get("answer_relevancy", 0.5)),
                        "context_precision": float(item.get("context_precision", 0.5)),
                        "context_relevance": float(item.get("context_relevance", 0.5)),
                    })
            elif isinstance(result, dict):
                all_scores.append({
                    "faithfulness": float(result.get("faithfulness", 0.5)),
                    "answer_relevancy": float(result.get("answer_relevancy", 0.5)),
                    "context_precision": float(result.get("context_precision", 0.5)),
                    "context_relevance": float(result.get("context_relevance", 0.5)),
                })
        except Exception as e:
            print(f"Batch judge error at offset {start}: {e}", file=sys.stderr)
            # Fallback: score individually
            for q, a in batch:
                result = evaluate_conversation(q, a)
                all_scores.append(result)

    n = len(all_scores) or 1
    avg_faith = sum(s["faithfulness"] for s in all_scores) / n
    avg_relev = sum(s["answer_relevancy"] for s in all_scores) / n
    avg_prec = sum(s["context_precision"] for s in all_scores) / n
    avg_rel = sum(s["context_relevance"] for s in all_scores) / n
    overall = (avg_faith + avg_relev + avg_prec + avg_rel) / 4

    return {
        "overall_score": round(overall, 3),
        "faithfulness": round(avg_faith, 3),
        "answer_relevancy": round(avg_relev, 3),
        "context_precision": round(avg_prec, 3),
        "context_relevance": round(avg_rel, 3),
    }


if __name__ == "__main__":
    test = [
        ("Show me wireless headphones",
         "We have the Sony WH-1000XM5 at $349.99 with 4.8/5 rating."),
        ("hello",
         "I can help you search, compare, and purchase products."),
        ("What products under $50?",
         "We have the Portable Power Bank at $45.99 and Organic Green Tea Set at $34.99."),
    ]
    result = evaluate_batch(test, batch_size=5)
    print(json.dumps(result, indent=2))
