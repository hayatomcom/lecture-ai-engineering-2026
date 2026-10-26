from types import SimpleNamespace

from app import ask


def test_measure_returns_elapsed_time_and_token_counts(monkeypatch):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=8),
        text="回答",
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    times = iter([10.0, 11.25])
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.time, "perf_counter", lambda: next(times))

    result = ask.measure("質問", model="test-model")

    assert result.text == "回答"
    assert result.elapsed_seconds == 1.25
    assert result.input_tokens == 12
    assert result.output_tokens == 8


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=8),
        text="回答",
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    result = ask.ask("質問")

    assert result == "回答"
    output = capsys.readouterr().out
    assert "[応答時間]" in output
    assert "入力: 12" in output
    assert "出力: 8" in output


def test_ask_reports_missing_usage_metadata(monkeypatch, capsys):
    response = SimpleNamespace(usage_metadata=None, text="回答")
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "[応答時間]" in output
    assert "入力: None" in output
    assert "出力: None" in output