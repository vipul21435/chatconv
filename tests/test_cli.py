import json
from pathlib import Path

from chatconv.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def mixed(tmp_path):
    path = tmp_path / "mixed.jsonl"
    lines = (FIXTURES / "sharegpt.jsonl").read_text() + (FIXTURES / "openai.jsonl").read_text()
    path.write_text(lines + "\n" + '{"conversations": [{"from": "alien", "value": "x"}]}\n{"x": \n')
    return path


def test_convert_writes_output_rejects_and_losses(tmp_path, capsys):
    src = mixed(tmp_path)
    out, rej, log = tmp_path / "out.jsonl", tmp_path / "rej.jsonl", tmp_path / "loss.jsonl"
    code = main(
        [
            "convert",
            str(src),
            "--to",
            "sharegpt",
            "-o",
            str(out),
            "--rejects",
            str(rej),
            "--losses",
            str(log),
        ]
    )
    assert code == 1
    assert len(out.read_text().splitlines()) == 7
    rejects = [json.loads(line) for line in rej.read_text().splitlines()]
    assert [r["line"] for r in rejects] == [9, 10]
    assert "unknown speaker" in rejects[0]["error"]
    assert rejects[1]["error"].startswith("invalid JSON")
    logged = [json.loads(line) for line in log.read_text().splitlines()]
    assert {e["line"] for e in logged} == {5, 6, 7}
    err = capsys.readouterr().err
    assert "line 9: entry 0: unknown speaker 'alien'" in err
    assert "records 9  converted 7  lossless 4  rejected 2" in err


def test_convert_clean_input_exits_zero(tmp_path):
    out = tmp_path / "out.jsonl"
    code = main(["convert", str(FIXTURES / "openai.jsonl"), "--to", "openai", "-o", str(out)])
    assert code == 0
    assert len(out.read_text().splitlines()) == 4


def test_strict_rejects_lossy_records(tmp_path, capsys):
    out = tmp_path / "out.jsonl"
    code = main(
        ["convert", str(FIXTURES / "openai.jsonl"), "--to", "sharegpt", "--strict", "-o", str(out)]
    )
    assert code == 1
    assert len(out.read_text().splitlines()) == 1
    assert "strict: lossy" in capsys.readouterr().err


def test_losses_json_summary(capsys):
    main(["losses", str(FIXTURES / "sharegpt.jsonl"), "--to", "openai", "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["records"] == 3
    assert summary["lossless"] == 1
    assert summary["records_with_loss"] == {"tool_ids_invented": 2}


def test_losses_per_record_and_text_report(capsys):
    main(["losses", str(FIXTURES / "openai.jsonl"), "--to", "sharegpt", "--per-record"])
    out = capsys.readouterr().out
    assert "4\tnames_dropped,tool_result_names_dropped,turn_split" in out
    assert "images_dropped" in out


def test_detect_counts_and_per_record(tmp_path, capsys):
    src = mixed(tmp_path)
    main(["detect", str(src)])
    counts = dict(line.split("\t") for line in capsys.readouterr().out.splitlines())
    assert counts == {"openai": "4", "sharegpt": "4", "invalid-json": "1"}
    main(["detect", str(src), "--per-record"])
    assert capsys.readouterr().out.splitlines()[0] == "1\tsharegpt"


def test_roundtrip_lists_lossy_records(capsys):
    code = main(["roundtrip", str(FIXTURES / "openai.jsonl"), "--to", "sharegpt"])
    captured = capsys.readouterr()
    assert code == 0
    assert "4\tlossy\tnames_dropped,tool_result_names_dropped,turn_split" in captured.out
    assert "records 4  lossless 1  lossy 3  rejected 0  mismatch 0" in captured.err


def test_roundtrip_reports_rejects_and_refuses_one_way(tmp_path, capsys):
    src = tmp_path / "x.jsonl"
    src.write_text('{"text": "plain"}\nnot json\n')
    main(["roundtrip", str(src), "--to", "openai"])
    out = capsys.readouterr().out
    assert "1\trejected\tformat not recognised" in out and "2\trejected\tinvalid JSON" in out
    assert main(["roundtrip", str(src), "--to", "llama3"]) == 2
    assert "write-only" in capsys.readouterr().err


def test_tools_policy_flag(capsys):
    main(
        ["losses", str(FIXTURES / "sharegpt.jsonl"), "--to", "alpaca", "--tools", "drop", "--json"]
    )
    summary = json.loads(capsys.readouterr().out)
    assert "tool_calls_dropped" in summary["records_with_loss"]
