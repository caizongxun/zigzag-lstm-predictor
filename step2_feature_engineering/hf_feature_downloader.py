import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from huggingface_hub import hf_hub_download


class HFFeatureDownloader:
    """
    Handles downloading feature files from HuggingFace dataset repository for Step 3 model training.
    """

    def __init__(self, repo_id: str = "zongowo111/zigzag-lstm-predictor", cache_dir: str = None):
        """
        Initialize HuggingFace feature downloader.

        Args:
            repo_id (str): HuggingFace dataset repository ID
            cache_dir (str): Local cache directory for downloaded files
        """
        self.repo_id = repo_id
        self.repo_type = "dataset"
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "huggingface" / "zigzag_lstm")
        print(f"HuggingFace downloader initialized for repo: {repo_id}")
        print(f"Cache directory: {self.cache_dir}")

    def _format_file_size(self, bytes_size: int) -> str:
        """
        Format file size to human-readable format.

        Args:
            bytes_size (int): File size in bytes

        Returns:
            str: Formatted size (KB, MB, GB)
        """
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"

    def download_feature_file(
        self,
        symbol: str,
        timeframe: str,
        filename: str,
    ) -> str:
        """
        Download a single feature file from HuggingFace.

        Args:
            symbol (str): Symbol identifier (e.g., 'BTCUSDT')
            timeframe (str): Timeframe identifier (e.g., '15m', '1h')
            filename (str): Filename to download

        Returns:
            str: Local path to downloaded file
        """
        remote_path = f"v2_model/{symbol}/{timeframe}/{filename}"
        print(f"  Downloading {filename}...", end="")

        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type=self.repo_type,
                cache_dir=self.cache_dir,
            )
            print(f" [OK]")
            return local_path
        except Exception as e:
            print(f" [FAILED]")
            raise Exception(f"Failed to download {filename}: {e}")

    def download_training_features(
        self,
        symbol: str,
        timeframes: List[str],
    ) -> Dict:
        """
        Download all training features for specified timeframes.

        Args:
            symbol (str): Symbol identifier (e.g., 'BTCUSDT')
            timeframes (List[str]): List of timeframes to download (e.g., ['15m', '1h'])

        Returns:
            Dict: Training features with structure:
                {
                    '15m': {
                        'X_sequences': numpy array (n_samples, 30, 13),
                        'y_class': numpy array (n_samples,),
                        'y_reg': numpy array (n_samples,),
                        'scaler': StandardScaler object,
                    },
                    '1h': { ... }
                }
        """
        print(f"\nDownloading {symbol} training features from HuggingFace...")
        print(f"{'='*70}")

        features = {}

        for timeframe in timeframes:
            print(f"\nTimeframe: {timeframe}")
            timeframe_features = {}

            try:
                local_X_seq = self.download_feature_file(
                    symbol, timeframe, f"{timeframe}_X_sequences.npy"
                )
                timeframe_features["X_sequences"] = np.load(local_X_seq)

                local_y_class = self.download_feature_file(
                    symbol, timeframe, f"{timeframe}_y_class.npy"
                )
                timeframe_features["y_class"] = np.load(local_y_class)

                local_y_reg = self.download_feature_file(
                    symbol, timeframe, f"{timeframe}_y_reg.npy"
                )
                timeframe_features["y_reg"] = np.load(local_y_reg)

                local_scaler = self.download_feature_file(
                    symbol, timeframe, f"{timeframe}_scaler.pkl"
                )
                with open(local_scaler, "rb") as f:
                    timeframe_features["scaler"] = pickle.load(f)

                print(f"  Features loaded:")
                print(f"    X_sequences shape: {timeframe_features['X_sequences'].shape}")
                print(f"    y_class shape: {timeframe_features['y_class'].shape}")
                print(f"    y_reg shape: {timeframe_features['y_reg'].shape}")
                print(f"    scaler: StandardScaler object")

                features[timeframe] = timeframe_features

            except Exception as e:
                print(f"  ERROR: {e}")
                features[timeframe] = None

        print(f"\n{'='*70}")
        print(f"Download complete")
        return features

    def download_statistics(
        self,
        symbol: str,
        timeframe: str,
    ) -> dict:
        """
        Download statistics JSON for a timeframe.

        Args:
            symbol (str): Symbol identifier
            timeframe (str): Timeframe identifier

        Returns:
            dict: Statistics dictionary
        """
        import json

        try:
            local_path = self.download_feature_file(
                symbol, timeframe, f"{timeframe}_statistics.json"
            )
            with open(local_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to download statistics: {e}")
            return None


def download_training_features(
    symbol: str = "BTCUSDT",
    timeframes: List[str] = None,
    cache_dir: str = None,
) -> Dict:
    """
    Convenience function to download training features from HuggingFace.

    This function is designed for Step 3 model training in Colab.

    Args:
        symbol (str): Symbol identifier (default: 'BTCUSDT')
        timeframes (List[str]): Timeframes to download (default: ['15m', '1h'])
        cache_dir (str): Local cache directory

    Returns:
        Dict: Training features

    Example:
        from hf_feature_downloader import download_training_features
        features = download_training_features(symbol='BTCUSDT', timeframes=['15m', '1h'])
        X_train_15m = features['15m']['X_sequences']
        y_class_15m = features['15m']['y_class']
        scaler_15m = features['15m']['scaler']
    """
    if timeframes is None:
        timeframes = ["15m", "1h"]

    downloader = HFFeatureDownloader(cache_dir=cache_dir)
    return downloader.download_training_features(symbol, timeframes)
