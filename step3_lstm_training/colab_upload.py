import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

from huggingface_hub import HfApi, HfFolder


def upload_to_hf_from_colab(
    model_path: str,
    scaler_path: str,
    config_path: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    hf_token: Optional[str] = None,
    repo_id: str = "zongowo111/v2-crypto-ohlcv-data",
) -> Dict[str, Any]:
    """
    Upload trained model, scaler, and config to Hugging Face Hub from Colab.

    Args:
        model_path: Local path to model.h5 file.
        scaler_path: Local path to scaler.pkl file.
        config_path: Local path to config.json file.
        symbol: Cryptocurrency symbol (e.g., "BTCUSDT").
        timeframe: Timeframe (e.g., "15m" or "1h").
        hf_token: Hugging Face API token. If None, reads from HF_TOKEN env or ~/.huggingface/token.
        repo_id: HuggingFace dataset/model repo ID.

    Returns:
        Dictionary with upload status and details.
    """
    try:
        # Get HF token
        if hf_token is None:
            hf_token = HfFolder.get_token()
            if hf_token is None:
                hf_token = os.environ.get("HF_TOKEN")

        if hf_token is None:
            return {
                "status": "error",
                "message": "HF token not found. Set HF_TOKEN env var or login to HF.",
            }

        # Initialize API
        api = HfApi(token=hf_token)

        # Verify files exist
        for fpath in [model_path, scaler_path, config_path]:
            if not os.path.exists(fpath):
                return {"status": "error", "message": f"File not found: {fpath}"}

        # Target directory structure: v2_model/BTCUSDT/15m/
        target_dir = f"v2_model/{symbol}/{timeframe}"

        print(
            f"[HF UPLOAD] Uploading to {repo_id}/{target_dir}/"
        )

        # Upload model
        print(f"[HF UPLOAD] Uploading model.h5...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"{target_dir}/model.h5",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Upload scaler
        print(f"[HF UPLOAD] Uploading scaler.pkl...")
        api.upload_file(
            path_or_fileobj=scaler_path,
            path_in_repo=f"{target_dir}/scaler.pkl",
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Upload config
        print(f"[HF UPLOAD] Uploading config.json...")
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo=f"{target_dir}/config.json",
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"[HF UPLOAD] All files uploaded successfully to {repo_id}/{target_dir}/")

        return {
            "status": "success",
            "repo_id": repo_id,
            "target_dir": target_dir,
            "files": ["model.h5", "scaler.pkl", "config.json"],
        }

    except Exception as e:
        print(f"[HF UPLOAD] Error during upload: {str(e)}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python colab_upload.py <model_path> <scaler_path> <config_path> [symbol] [timeframe] [hf_token]"
        )
        sys.exit(1)

    model_p = sys.argv[1]
    scaler_p = sys.argv[2]
    config_p = sys.argv[3]
    sym = sys.argv[4] if len(sys.argv) > 4 else "BTCUSDT"
    tf = sys.argv[5] if len(sys.argv) > 5 else "15m"
    token = sys.argv[6] if len(sys.argv) > 6 else None

    result = upload_to_hf_from_colab(model_p, scaler_p, config_p, sym, tf, token)
    print(json.dumps(result, indent=2))
