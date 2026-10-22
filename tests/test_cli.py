import json
import textwrap

from evalbar.cli import main

EVAL_FILE = textwrap.dedent(
    """
    from evalbar import Eval, Case
    from evalbar.scorers import exact

    DATA = [
        Case(id="a", input="x", expected={"output": "X"}, metadata={}),
        Case(id="b", input="y", expected={"output": "nope"}, metadata={}),
    ]

    upper = Eval(name="upper", pipeline=str.upper, dataset=DATA, scorers=[exact()])
    lower = Eval(name="lower", pipeline=str.lower, dataset=DATA, scorers=[exact()])
    """
)


def write_eval(tmp_path, source=EVAL_FILE):
    path = tmp_path / "my_eval.py"
    path.write_text(source)
    return path


def test_run_named_eval_writes_json_and_prints_report(tmp_path, capsys):
    out = tmp_path / "run.json"

    code = main(["run", f"{write_eval(tmp_path)}:upper", "--out", str(out)])

    assert code == 0
    data = json.loads(out.read_text())
    assert data["name"] == "upper" and len(data["cases"]) == 2
    printed = capsys.readouterr().out
    assert "exact" in printed and "50.0%" in printed


def test_run_picks_the_only_eval_when_no_name_given(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = EVAL_FILE.replace('lower = Eval(name="lower", pipeline=str.lower, dataset=DATA, scorers=[exact()])', "")

    assert main(["run", str(write_eval(tmp_path, source)), "--limit", "1"]) == 0
    assert "upper" in capsys.readouterr().out


def test_run_with_several_evals_and_no_name_is_a_usage_error(tmp_path, capsys):
    assert main(["run", str(write_eval(tmp_path))]) == 2
    assert "upper, lower" in capsys.readouterr().err


def test_show_reprints_a_saved_run(tmp_path, capsys):
    out = tmp_path / "run.json"
    main(["run", f"{write_eval(tmp_path)}:upper", "--out", str(out)])
    capsys.readouterr()

    assert main(["show", str(out)]) == 0
    assert "upper" in capsys.readouterr().out


def test_missing_file_is_a_usage_error(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.py")]) == 2
    assert "not found" in capsys.readouterr().err


def test_env_file_sets_missing_variables_only(tmp_path, monkeypatch):
    from evalbar.cli import load_env_file

    env = tmp_path / ".env"
    env.write_text('# comment\nEVALBAR_T_A="from file"\nexport EVALBAR_T_B=b\nEVALBAR_T_C=file\n')
    monkeypatch.setenv("EVALBAR_T_C", "shell")
    monkeypatch.delenv("EVALBAR_T_A", raising=False)
    monkeypatch.delenv("EVALBAR_T_B", raising=False)

    load_env_file(str(env))

    import os
    assert os.environ["EVALBAR_T_A"] == "from file"
    assert os.environ["EVALBAR_T_B"] == "b"
    assert os.environ["EVALBAR_T_C"] == "shell"
    monkeypatch.delenv("EVALBAR_T_A")
    monkeypatch.delenv("EVALBAR_T_B")
