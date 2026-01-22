from pathlib import Path

def test_raw_data_folder_integrity():
    root = Path("data/raw")
    assert root.exists() and root.is_dir()

    files = [p for p in root.rglob("*") if p.is_file()]
    assert len(files) == 12798

    total_bytes = sum(p.stat().st_size for p in files)
    assert total_bytes > 0
