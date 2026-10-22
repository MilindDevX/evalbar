from evalbar.cli import main
from evalbar.dataset import Case
from evalbar.html import render_html
from evalbar.report import save_run
from evalbar.runner import Eval, run
from evalbar.scorers import exact
from evalbar.trace import record_llm_call, traced_tool


@traced_tool
def lookup(term):
    return f"<b>{term}</b>"


def pipeline(text):
    lookup(text)
    record_llm_call(model="fake-model", prompt_tokens=12, completion_tokens=3, latency_s=0.3, cost_usd=0.002)
    if text == "boom":
        raise RuntimeError("<img src=x onerror=alert(1)>")
    return {"label": "<script>alert(1)</script>"}


def make_run():
    dataset = [
        Case(id="ok", input="<script>alert(1)</script>", expected={"label": "<script>alert(1)</script>"},
             metadata={"group": "x"}),
        Case(id="wrong", input="cat", expected={"label": "dog"}, metadata={}),
        Case(id="crash", input="boom", expected={"label": "x"}, metadata={}),
    ]
    return run(Eval(name="html-demo", pipeline=pipeline, dataset=dataset, scorers=[exact("label")],
                    metadata={"prompt": "v2"}))


def test_html_is_one_self_contained_page_with_summary_and_cases():
    page = render_html(make_run())

    assert page.startswith("<!doctype html>")
    assert "<title>evalbar: html-demo</title>" in page
    assert "exact:label" in page and "33.3%" in page
    assert "prompt" in page and "v2" in page
    for case_id in ("ok", "wrong", "crash"):
        assert f'id="case-{case_id}"' in page
    assert "fake-model" in page and "lookup" in page
    for tag in ("<script", "<link", "<img", "<iframe"):  # nothing external, nothing executable
        assert tag not in page
    assert "http://" not in page and "https://" not in page


def test_html_escapes_everything_that_came_from_the_pipeline():
    page = render_html(make_run())

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<img src=x" not in page
    assert "<b>" not in page


def test_failing_cases_are_marked_and_listed_first():
    page = render_html(make_run())

    assert page.index('id="case-crash"') < page.index('id="case-ok"')
    assert page.index('id="case-wrong"') < page.index('id="case-ok"')
    assert "RuntimeError" in page


def test_cli_html_writes_next_to_the_run_file(tmp_path, capsys):
    run_file = save_run(make_run(), tmp_path / "demo.json")

    assert main(["html", str(run_file)]) == 0
    assert (tmp_path / "demo.html").read_text().startswith("<!doctype html>")
    assert main(["html", str(run_file), "--out", str(tmp_path / "custom.html")]) == 0
    assert (tmp_path / "custom.html").exists()
    assert main(["html", str(tmp_path / "missing.json")]) == 2
