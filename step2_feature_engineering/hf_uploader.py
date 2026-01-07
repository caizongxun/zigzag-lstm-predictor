import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from huggingface_hub import HfApi, repo_exists


class HFFeatureUploader:
    """
    Handles uploading feature engineering outputs to HuggingFace dataset repository.
    """

    def __init__(self, hf_token: str, repo_id: str = "zongowo111/zigzag-lstm-predictor"):
        """
        Initialize HuggingFace uploader.

        Args:
            hf_token (str): HuggingFace API token
            repo_id (str): Target dataset repository ID
        """
        self.api = HfApi(token=hf_token)
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.repo_type = "dataset"
        print(f"HuggingFace uploader initialized for repo: {repo_id}")

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

    def _validate_local_files(self, local_output_dir: str, timeframe: str) -> Tuple[bool, List[str]]:
        """
        Validate that all required feature files exist locally.

        Args:
            local_output_dir (str): Local output directory path
            timeframe (str): Timeframe identifier (e.g., '15m', '1h')

        Returns:
            Tuple[bool, List[str]]: (validation_success, missing_files)
        """
        required_files = [
            f"{timeframe}_X_sequences.npy",
            f"{timeframe}_y_class.npy",
            f"{timeframe}_y_reg.npy",
            f"{timeframe}_scaler.pkl",
            f"{timeframe}_zigzag_points.json",
            f"{timeframe}_statistics.json",
        ]

        local_dir = Path(local_output_dir)
        missing_files = []

        for filename in required_files:
            file_path = local_dir / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            return False, missing_files
        return True, []

    def upload_features(
        self,
        symbol: str,
        timeframe: str,
        local_output_dir: str,
    ) -> Dict:
        """
        Upload all feature files for a timeframe to HuggingFace.

        Args:
            symbol (str): Symbol identifier (e.g., 'BTCUSDT')
            timeframe (str): Timeframe identifier (e.g., '15m', '1h')
            local_output_dir (str): Local directory containing feature files

        Returns:
            Dict: Upload results containing success status, messages, and file list
        """
        print(f"\n{'='*90}")
        print(f"Uploading {symbol} {timeframe} features to HuggingFace")
        print(f"{'='*90}")

        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "message": "",
            "files_uploaded": [],
            "files_failed": [],
            "remote_dir": f"v2_model/{symbol}/{timeframe}",
            "total_size": 0,
        }

        try:
            is_valid, missing_files = self._validate_local_files(local_output_dir, timeframe)
            if not is_valid:
                results["message"] = f"Missing local files: {', '.join(missing_files)}"
                print(f"  ERROR: {results['message']}")
                return results

            local_dir = Path(local_output_dir)
            required_files = [
                f"{timeframe}_X_sequences.npy",
                f"{timeframe}_y_class.npy",
                f"{timeframe}_y_reg.npy",
                f"{timeframe}_scaler.pkl",
                f"{timeframe}_zigzag_points.json",
                f"{timeframe}_statistics.json",
            ]

            print(f"\n  Uploading {len(required_files)} files...")
            total_size = 0

            for filename in required_files:
                local_file = local_dir / filename
                file_size = local_file.stat().st_size
                total_size += file_size

                remote_path = f"{results['remote_dir']}/{filename}"

                try:
                    print(f"    Uploading {filename} ({self._format_file_size(file_size)})...", end="")

                    self.api.upload_file(
                        path_or_fileobj=str(local_file),
                        path_in_repo=remote_path,
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        commit_message=f"Upload {symbol} {timeframe} features: {filename}",
                    )

                    results["files_uploaded"].append(
                        {
                            "filename": filename,
                            "size": file_size,
                            "remote_path": remote_path,
                        }
                    )
                    print(" [OK]")

                except Exception as e:
                    print(f" [FAILED]")
                    results["files_failed"].append(
                        {
                            "filename": filename,
                            "error": str(e),
                        }
                    )
                    print(f"      Error: {e}")

            results["total_size"] = total_size

            if len(results["files_failed"]) == 0:
                results["success"] = True
                results["message"] = f"Successfully uploaded {len(results['files_uploaded'])} files ({self._format_file_size(total_size)})"
            else:
                results["success"] = False
                results["message"] = f"Uploaded {len(results['files_uploaded'])} files, but {len(results['files_failed'])} failed"

            print(f"\n  {results['message']}")
            print(f"  Remote directory: {results['remote_dir']}")

        except Exception as e:
            results["message"] = f"Upload failed: {str(e)}"
            print(f"  ERROR: {results['message']}")

        return results

    def verify_upload(self, symbol: str, timeframe: str) -> Dict:
        """
        Verify that all expected files were uploaded successfully.

        Args:
            symbol (str): Symbol identifier
            timeframe (str): Timeframe identifier

        Returns:
            Dict: Verification results with file status
        """
        print(f"\n  Verifying {symbol} {timeframe} upload...")

        expected_files = [
            f"{timeframe}_X_sequences.npy",
            f"{timeframe}_y_class.npy",
            f"{timeframe}_y_reg.npy",
            f"{timeframe}_scaler.pkl",
            f"{timeframe}_zigzag_points.json",
            f"{timeframe}_statistics.json",
        ]

        verification = {
            "symbol": symbol,
            "timeframe": timeframe,
            "remote_dir": f"v2_model/{symbol}/{timeframe}",
            "all_present": False,
            "files_status": {},
            "missing_files": [],
        }

        try:
            repo_files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )

            for filename in expected_files:
                expected_path = f"v2_model/{symbol}/{timeframe}/{filename}"
                is_present = expected_path in repo_files
                verification["files_status"][filename] = is_present

                if not is_present:
                    verification["missing_files"].append(filename)

            verification["all_present"] = len(verification["missing_files"]) == 0

            if verification["all_present"]:
                print(f"    All {len(expected_files)} files verified successfully")
            else:
                print(f"    Missing {len(verification['missing_files'])} files: {', '.join(verification['missing_files'])}")

        except Exception as e:
            print(f"    Verification failed: {e}")

        return verification


def upload_step2_features(
    symbol: str,
    timeframe: str,
    local_output_dir: str,
    hf_token: str,
) -> Dict:
    """
    Convenience function to upload Step 2 features to HuggingFace.

    Args:
        symbol (str): Symbol identifier (e.g., 'BTCUSDT')
        timeframe (str): Timeframe identifier (e.g., '15m', '1h')
        local_output_dir (str): Local directory containing feature files
        hf_token (str): HuggingFace API token

    Returns:
        Dict: Upload and verification results
    """
    uploader = HFFeatureUploader(hf_token)
    upload_result = uploader.upload_features(symbol, timeframe, local_output_dir)

    if upload_result["success"]:
        verification = uploader.verify_upload(symbol, timeframe)
        upload_result["verification"] = verification

    return upload_result
