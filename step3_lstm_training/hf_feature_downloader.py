import os
import pickle
import json
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download


class HFFeatureDownloader:
    """Download feature engineering results from HuggingFace Hub."""

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize HuggingFace downloader.

        Args:
            hf_token: HF API token. If None, attempts to read from HF_TOKEN env var.
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.repo_id = "zongowo111/zigzag-lstm-predictor"
        self.repo_type = "dataset"

    def _load_file(self, file_path: str) -> Any:
        """
        Load file based on extension.

        Args:
            file_path: Path to file.

        Returns:
            Loaded content (numpy array, pickle object, or dict).
        """
        if file_path.endswith(".npy"):
            return np.load(file_path)
        elif file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                return pickle.load(f)
        elif file_path.endswith(".json"):
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def download_features(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        local_cache_dir: str = "./step2_cache",
    ) -> Dict[str, Any]:
        """
        Download feature files from HuggingFace for a specific timeframe.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTCUSDT").
            timeframe: Timeframe ("15m" or "1h").
            local_cache_dir: Local directory to cache downloaded files.

        Returns:
            Dictionary containing loaded features:
            {
                'X_sequences': np.ndarray,
                'y_class': np.ndarray,
                'y_reg': np.ndarray,
                'scaler': sklearn.preprocessing.StandardScaler,
                'zigzag_points': dict,
                'statistics': dict
            }
        """
        Path(local_cache_dir).mkdir(parents=True, exist_ok=True)

        file_specs = [
            f"{timeframe}_X_sequences.npy",
            f"{timeframe}_y_class.npy",
            f"{timeframe}_y_reg.npy",
            f"{timeframe}_scaler.pkl",
            f"{timeframe}_zigzag_points.json",
            f"{timeframe}_statistics.json",
        ]

        print(f"[HF DOWNLOADER] Downloading {timeframe} features for {symbol}...")

        loaded_data = {}
        for filename in file_specs:
            remote_path = f"v2_model/{symbol}/{timeframe}/{filename}"
            print(f"  Downloading {filename}...", end=" ")

            try:
                file_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_path,
                    repo_type=self.repo_type,
                    cache_dir=local_cache_dir,
                    token=self.hf_token,
                )
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"OK ({file_size_mb:.2f} MB)")

                # Load content
                data = self._load_file(file_path)
                key = filename.replace(f"{timeframe}_", "").replace(
                    ".npy", ""
                ).replace(".pkl", "").replace(".json", "")
                loaded_data[key] = data

            except Exception as e:
                print(f"FAILED: {e}")
                if "statistics" in filename or "zigzag" in filename:
                    print(f"    Warning: Optional file missing, continuing...")
                    continue
                else:
                    raise RuntimeError(f"Failed to download critical file: {filename}")

        print(f"[HF DOWNLOADER] Download complete for {timeframe}")
        return loaded_data

    def download_all_features(
        self,
        symbol: str = "BTCUSDT",
        timeframes: list = None,
        local_cache_dir: str = "./step2_cache",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Download features for all timeframes.

        Args:
            symbol: Cryptocurrency symbol.
            timeframes: List of timeframes to download (default: ["15m", "1h"]).
            local_cache_dir: Local cache directory.

        Returns:
            Dictionary with timeframes as keys: {"15m": {...}, "1h": {...}}
        """
        if timeframes is None:
            timeframes = ["15m", "1h"]

        all_features = {}
        for tf in timeframes:
            all_features[tf] = self.download_features(
                symbol=symbol, timeframe=tf, local_cache_dir=local_cache_dir
            )

        return all_features


def download_training_features(
    symbol: str = "BTCUSDT",
    timeframes: list = None,
    hf_token: Optional[str] = None,
    local_cache_dir: str = "./step2_cache",
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to download training features.

    Args:
        symbol: Cryptocurrency symbol.
        timeframes: List of timeframes.
        hf_token: HuggingFace token.
        local_cache_dir: Local cache directory.

    Returns:
        Dictionary of downloaded features by timeframe.
    """
    if timeframes is None:
        timeframes = ["15m", "1h"]

    downloader = HFFeatureDownloader(hf_token=hf_token)
    return downloader.download_all_features(
        symbol=symbol, timeframes=timeframes, local_cache_dir=local_cache_dir
    )


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "15m"
    cache_dir = sys.argv[3] if len(sys.argv) > 3 else "./step2_cache"

    print(f"Downloading {timeframe} features for {symbol}...")
    downloader = HFFeatureDownloader()
    features = downloader.download_features(symbol, timeframe, cache_dir)
    print(f"Successfully downloaded {len(features)} feature files")
    for key, val in features.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape}")
        elif isinstance(val, dict):
            print(f"  {key}: dict with {len(val)} keys")
        else:
            print(f"  {key}: {type(val).__name__}")
