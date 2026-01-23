from audio_emotion import download as download_module


def test_download_audio_emotions_short_circuit(tmp_path, capsys, monkeypatch):
    # Precreate a file to force the early return path
    tmp_path.mkdir(parents=True, exist_ok=True)
    existing = tmp_path / "dummy.txt"
    existing.write_text("ok")

    # Mock KaggleApi to ensure it is not called
    class DummyApi:
        def authenticate(self):
            raise AssertionError("authenticate should not be called when data exists")

        def dataset_download_files(self, *_, **__):
            raise AssertionError("download should not run when data exists")

    monkeypatch.setattr(download_module, "KaggleApi", DummyApi)

    out = download_module.download_audio_emotions(tmp_path)
    captured = capsys.readouterr().out
    assert "Dataset already exists" in captured
    assert out == tmp_path.resolve()
