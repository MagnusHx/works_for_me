from audio_emotion import cli


def test_cli_download_function(tmp_path, monkeypatch, capsys):
    """Call the download function directly to cover its logic."""

    called = {}
    monkeypatch.setattr(
        cli, "download_audio_emotions", lambda out: called.setdefault("path", out)
    )

    cli.download(output_dir=tmp_path)

    out = capsys.readouterr().out
    assert "Dataset available at" in out
    assert called.get("path") == tmp_path
    assert tmp_path.exists()
