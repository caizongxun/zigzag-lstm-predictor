#!/usr/bin/env python3
"""
STEP 2 Data Upload System - BTCUSDT Only

This script organizes and uploads BTCUSDT data from step2_output to Hugging Face.

Resource: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
"""

import os
import time
import shutil
from pathlib import Path
from typing import List, Tuple

try:
    from huggingface_hub import HfApi, upload_folder, upload_file, whoami, repo_info
except ImportError:
    raise ImportError("Please install: pip install huggingface-hub")


class Step2BtcustdUploader:
    """
    Upload BTCUSDT step2 data to Hugging Face.
    
    Rules:
    - 1h_ prefix files -> klines/BTCUSDT/15m/
    - 15m_ prefix files -> klines/BTCUSDT/1h/
    - execution_summary.json -> root
    - Only process BTCUSDT
    """
    
    def __init__(self):
        self.api = HfApi()
        self.repo_id = "zongowo111/v2-crypto-ohlcv-data"
        self.repo_type = "dataset"
        self.source_folder = "./step2_output"
        self.organized_folder = "./organized_step2_output"
        self.symbol = "BTCUSDT"
        self.api_delay = 5  # seconds
    
    def verify_prerequisites(self):
        """
        Step 1: Verify prerequisites
        """
        print("\n[VERIFICATION] Checking prerequisites...")
        
        # Check source folder
        if not os.path.exists(self.source_folder):
            raise FileNotFoundError(f"Source folder not found: {self.source_folder}")
        print(f"  OK: Source folder exists: {self.source_folder}")
        
        # Check HuggingFace login
        try:
            user = whoami()
            print(f"  OK: Logged in to HuggingFace as: {user['name']}")
        except Exception as e:
            raise PermissionError(f"Not logged in to HuggingFace. Run: huggingface-cli login")
        
        # Check dataset accessibility
        try:
            repo = repo_info(
                repo_id=self.repo_id,
                repo_type=self.repo_type
            )
            print(f"  OK: Dataset accessible: {self.repo_id}")
        except Exception as e:
            raise PermissionError(f"Cannot access dataset: {e}")
    
    def organize_files(self) -> Tuple[int, int]:
        """
        Step 2: Organize files locally
        
        Returns: (organized_count, skipped_count)
        """
        print(f"\n[ORGANIZATION] Organizing files for {self.symbol}...")
        
        # Clean organized folder
        if os.path.exists(self.organized_folder):
            shutil.rmtree(self.organized_folder)
        os.makedirs(self.organized_folder)
        
        # Scan source folder
        source_path = Path(self.source_folder)
        all_items = list(source_path.rglob("*"))
        files = [f for f in all_items if f.is_file()]
        
        print(f"  Found {len(files)} files in source folder")
        
        organized_count = 0
        skipped_count = 0
        
        for file_path in files:
            file_name = file_path.name
            
            # Rule 1: 1h_ prefix -> 15m folder
            if file_name.startswith("1h_"):
                timeframe = "15m"
            # Rule 2: 15m_ prefix -> 1h folder
            elif file_name.startswith("15m_"):
                timeframe = "1h"
            # Rule 3: execution_summary.json -> root
            elif file_name == "execution_summary.json":
                timeframe = None  # root
            else:
                skipped_count += 1
                continue
            
            # Organize file
            if timeframe:
                target_dir = Path(self.organized_folder) / self.symbol / timeframe
            else:
                target_dir = Path(self.organized_folder)
            
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / file_name
            shutil.copy2(file_path, target_path)
            organized_count += 1
            
            if timeframe:
                print(f"    Organized: {self.symbol}/{timeframe}/{file_name}")
            else:
                print(f"    Organized: {file_name}")
        
        print(f"  Summary: {organized_count} organized, {skipped_count} skipped")
        return organized_count, skipped_count
    
    def check_remote_folders(self):
        """
        Step 3a: Check remote folder structure
        """
        print(f"\n[REMOTE CHECK] Checking remote structure...")
        print(f"  Target symbol: {self.symbol}")
        print(f"  Target paths:")
        print(f"    - klines/{self.symbol}/15m/")
        print(f"    - klines/{self.symbol}/1h/")
    
    def upload_symbol(self):
        """
        Step 3b: Upload BTCUSDT folder
        """
        print(f"\n[UPLOAD] Uploading {self.symbol}...")
        
        symbol_path = Path(self.organized_folder) / self.symbol
        
        if not symbol_path.exists():
            print(f"  WARNING: {self.symbol} folder not found in organized output")
            return
        
        try:
            upload_folder(
                folder_path=str(symbol_path),
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                path_in_repo=f"klines/{self.symbol}",
                commit_message=f"Upload {self.symbol} step2 features - 15m and 1h organized"
            )
            print(f"  OK: {self.symbol} uploaded successfully")
        except Exception as e:
            print(f"  ERROR: Failed to upload {self.symbol}: {e}")
            raise
    
    def upload_summary(self):
        """
        Step 3c: Upload execution_summary.json
        """
        summary_path = Path(self.organized_folder) / "execution_summary.json"
        
        if not summary_path.exists():
            print(f"  WARNING: execution_summary.json not found")
            return
        
        print(f"\n[UPLOAD] Uploading execution_summary.json...")
        
        try:
            upload_file(
                path_or_fileobj=str(summary_path),
                path_in_repo="execution_summary.json",
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message="Upload step2 execution summary"
            )
            print(f"  OK: execution_summary.json uploaded successfully")
        except Exception as e:
            print(f"  ERROR: Failed to upload execution_summary.json: {e}")
            raise
    
    def run(self):
        """
        Execute complete upload process
        """
        try:
            print("="*70)
            print("STEP 2 DATA UPLOAD SYSTEM - BTCUSDT ONLY")
            print("="*70)
            
            # Step 1: Verify
            self.verify_prerequisites()
            
            # Step 2: Organize
            organized_count, skipped_count = self.organize_files()
            
            if organized_count == 0:
                print("\nWARNING: No files organized. Aborting upload.")
                return
            
            # Step 3a: Check remote
            self.check_remote_folders()
            
            # Step 3b: Upload BTCUSDT
            self.upload_symbol()
            
            # Wait for rate limit
            print(f"\n[RATE LIMIT] Waiting {self.api_delay} seconds...")
            time.sleep(self.api_delay)
            
            # Step 3c: Upload summary
            self.upload_summary()
            
            print("\n" + "="*70)
            print("SUCCESS: Upload completed")
            print("="*70)
            print(f"\nVerify at: https://huggingface.co/datasets/{self.repo_id}")
            
        except Exception as e:
            print(f"\n" + "="*70)
            print(f"ERROR: {e}")
            print("="*70)
            raise


if __name__ == "__main__":
    uploader = Step2BtcustdUploader()
    uploader.run()
