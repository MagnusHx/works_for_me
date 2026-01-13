import kagglehub
from pathlib import Path
import os


def download_kaggle_dataset(dataset: str, target_dir: Path) -> Path:
    """
    Download Kaggle dataset to a deterministic location.
    
    Args:
        dataset: Kaggle dataset identifier (e.g., 'username/dataset-name')
        target_dir: Base directory where dataset cache will be stored
        
    Returns:
        Path to the downloaded dataset
        
    Note:
        Requires Kaggle API credentials in ~/.kaggle/kaggle.json or 
        KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    """
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Set kagglehub cache directory
    os.environ['KAGGLEHUB_CACHE_DIR'] = str(target_dir)

    print(f"Downloading {dataset}...")
    # Don't use 'path' parameter - let kagglehub manage the structure
    path = kagglehub.dataset_download(dataset)

    print(f"✓ Dataset downloaded to: {path}")
    return Path(path)
