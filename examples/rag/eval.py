"""A tiny RAG pipeline: keyword retriever + answer model, scored with recall@k and an LLM judge.

    evalbar run examples/rag/eval.py                    # offline: mock answers, no judge
    EVALBAR_LIVE=1 evalbar run examples/rag/eval.py     # Groq answers + LLM judge (needs GROQ_API_KEY)

The retriever matches words only, so questions phrased with synonyms ("money back" for
"refund") miss the right document. recall@k shows the retrieval miss; the judge shows what
it does to the answer.
"""

import os
import re
from pathlib import Path

from evalbar import Eval, completion, load_dataset, traced_tool
from evalbar.scorers import contains, llm_judge, recall_at_k

LIVE = os.environ.get("EVALBAR_LIVE") == "1"
MODEL = os.environ.get("EVALBAR_RAG_MODEL", "groq/openai/gpt-oss-120b")
JUDGE_MODEL = os.environ.get("EVALBAR_JUDGE_MODEL", MODEL)
K = 2

DOCS = {
    "refunds": "Refunds: you can request a full refund within 30 days of purchase from Settings > Billing.",
    "export": "Export: notes can be exported as Markdown, PDF or HTML from the share menu.",
    "sso": "Single sign-on: Business plans support SAML SSO with Okta, Azure AD and Google Workspace.",
    "offline": "Offline mode: the desktop and mobile apps keep working without internet and sync when you reconnect.",
    "retention": "Data retention: deleted notes stay in the trash for 30 days, then they are erased permanently.",
    "api_limits": "API limits: the REST API allows 60 requests per minute per token.",
    "pricing": "Pricing: the Free plan has 100 notes; Pro costs $8 per user per month with unlimited notes.",
    "mobile": "Mobile apps: Acme Notes is available for iOS 16+ and Android 10+.",
}
STOPWORDS = set("a an and are as at be can do does for from get how i in is it me my of on or the there to what "
                "which with you your".split())


def words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS}


@traced_tool
def search(question: str, k: int = K) -> list[str]:
    query = words(question)
    ranked = sorted(DOCS, key=lambda doc_id: len(query & words(DOCS[doc_id])), reverse=True)
    return [doc_id for doc_id in ranked[:k] if query & words(DOCS[doc_id])]


def answer(case_input: dict) -> dict:
    question = case_input["question"]
    retrieved = search(question)
    context = [DOCS[doc_id] for doc_id in retrieved]
    messages = [
        {"role": "system", "content": "Answer in one sentence using only the context. "
                                      "If the context does not contain the answer, say you don't know."},
        {"role": "user", "content": "Context:\n" + "\n".join(context) + f"\n\nQuestion: {question}"},
    ]
    if LIVE:
        reply = completion(model=MODEL, messages=messages, temperature=0, max_tokens=300, reasoning_effort="low")
    else:
        reply = completion(model="gpt-4o-mini", messages=messages,
                           mock_response=context[0] if context else "I don't know.")
    return {"answer": reply.choices[0].message.content, "retrieved": retrieved, "context": context}


SCORERS = [
    recall_at_k(k=K),
    contains("answer", expected_key="must_include"),
]
if LIVE:
    SCORERS.append(llm_judge(
        "The OUTPUT's answer correctly answers the question in INPUT, is supported by the OUTPUT's context, "
        "and does not invent facts. An answer saying it does not know scores 0.",
        model=JUDGE_MODEL, field=None, name="judge:answered", pass_threshold=0.7,
        max_tokens=300, reasoning_effort="low",
    ))

rag = Eval(
    name="rag-live" if LIVE else "rag-offline",
    pipeline=answer,
    dataset=load_dataset(Path(__file__).parent / "cases.yaml"),
    scorers=SCORERS,
    min_interval_s=2.0 if LIVE else 0.0,
    metadata={"retriever": f"keyword top-{K}", "model": MODEL if LIVE else "mock",
              "judge": JUDGE_MODEL if LIVE else "off"},
)
