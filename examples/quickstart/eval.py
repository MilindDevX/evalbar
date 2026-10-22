"""Runs offline: LiteLLM's mock_response stands in for a real model, so no API key is needed.

    evalbar run examples/quickstart/eval.py

Swap `mock_response=...` out and set a real `model` to evaluate a live pipeline.
"""

from pathlib import Path

from evalbar import Eval, completion, load_dataset, traced_tool
from evalbar.scorers import contains, exact, regex

CAPITALS = {"france": "Paris", "japan": "Tokyo", "kenya": "Nairobi", "peru": "Lima"}


@traced_tool
def lookup_capital(country: str) -> str | None:
    return CAPITALS.get(country.lower())


def answer(question: dict) -> dict:
    country = question["country"]
    fact = lookup_capital(country)
    reply = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Using this fact: {fact}. What is the capital of {country}?"}],
        mock_response=f"The capital of {country} is {fact}." if fact else "I don't know.",
    )
    text = reply.choices[0].message.content
    return {"answer": text, "city": fact or ""}


quickstart = Eval(
    name="capitals",
    pipeline=answer,
    dataset=load_dataset(Path(__file__).parent / "cases.yaml"),
    scorers=[
        exact("city"),
        contains("answer"),
        regex("answer", r"\.$", name="ends with a period"),
    ],
)
