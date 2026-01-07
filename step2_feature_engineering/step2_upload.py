import os
import time
import shutil
from pathlib import Path
from typing import Set, Dict, List
from huggingface_hub import HfApi, upload_folder, upload_file


class Step2DataUploader:
    def __init__(self):
        self.api = HfApi()
        self.repo_id = "zongowo111/v2-crypto-ohlcv-data"
        self.repo_type = "dataset"
        self.source_folder = "./step2_output"
        self.organized_folder = "./organized_step2_output"
        self.allowed_symbols = ["BTCUSDT", "ETHUSDT"]

    def verify_prerequisites(self) -> None:
        """
        Verify all prerequisites for upload:
        - Source folder exists
        - HuggingFace authentication
        - Dataset accessible
        """
        print("Verifying prerequisites...")

        if not os.path.exists(self.source_folder):
            raise Exception(f"Error: {self.source_folder} does not exist")
        print(f"Success: Found source folder: {self.source_folder}")

        try:
            user = self.api.whoami()
            print(f"Success: Logged in to HuggingFace as: {user['name']}")
        except Exception as e:
            raise Exception(f"Error: Not logged in to HuggingFace: {e}")

        try:
            repo = self.api.repo_info(
                repo_id=self.repo_id,
                repo_type=self.repo_type
            )
            print(f"Success: Dataset accessible: {self.repo_id}")
        except Exception as e:
            raise Exception(f"Error: Cannot access dataset: {e}")

    def extract_symbol(self, file_path: Path, file_name: str) -> str:
        """
        Extract symbol (BTCUSDT, ETHUSDT) from file path or name.
        
        Args:
            file_path: Full file path
            file_name: File name
            
        Returns:
            Symbol string or None
        """
        path_parts = str(file_path).split(os.sep)
        for part in path_parts:
            if part in self.allowed_symbols:
                return part

        parent_name = file_path.parent.name
        if parent_name in self.allowed_symbols:
            return parent_name

        return None

    def organize_files(self) -> bool:
        """
        Organize files from source folder into target structure:
        organized_step2_output/
        ├── BTCUSDT/
        │   ├── 15m/
        │   │   └── 1h_* files
        │   └── 1h/
        │       └── 15m_* files
        ├── ETHUSDT/
        │   ├── 15m/
        │   │   └── 1h_* files
        │   └── 1h/
        │       └── 15m_* files
        └── execution_summary.json
        
        Returns:
            True if files were organized, False otherwise
        """
        print("\nOrganizing files...")

        if os.path.exists(self.organized_folder):
            shutil.rmtree(self.organized_folder)
        os.makedirs(self.organized_folder)

        source_path = Path(self.source_folder)
        all_files = list(source_path.rglob("*"))
        files = [f for f in all_files if f.is_file()]

        print(f"Scanned {len(files)} files")

        organized_count = 0
        skipped_count = 0

        for file_path in files:
            relative_name = file_path.name

            symbol = self.extract_symbol(file_path, relative_name)

            if symbol and symbol not in self.allowed_symbols:
                print(
                    f"  Skip: {relative_name} (symbol {symbol} not in allowed list)"
                )
                skipped_count += 1
                continue

            if relative_name.startswith("1h_"):
                timeframe = "15m"
            elif relative_name.startswith("15m_"):
                timeframe = "1h"
            else:
                timeframe = None

            if symbol and timeframe:
                target_dir = (
                    Path(self.organized_folder) / symbol / timeframe
                )
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / relative_name

                shutil.copy2(file_path, target_path)
                organized_count += 1
                print(f"  Organize: {symbol}/{timeframe}/{relative_name}")
            elif relative_name == "execution_summary.json":
                target_path = Path(self.organized_folder) / relative_name
                shutil.copy2(file_path, target_path)
                organized_count += 1
                print(f"  Organize: {relative_name}")

        print(f"\nSuccess: Organized {organized_count} files, skipped {skipped_count} files")
        return organized_count > 0

    def check_remote_folders(self) -> None:
        """
        Check which symbols are present in organized folder.
        """
        print("\nChecking organized structure...")

        organized_symbols: Set[str] = set()
        for symbol_dir in Path(self.organized_folder).iterdir():
            if symbol_dir.is_dir() and symbol_dir.name in self.allowed_symbols:
                organized_symbols.add(symbol_dir.name)

        print(f"Symbols to upload: {sorted(organized_symbols)}")

        for symbol in organized_symbols:
            if symbol not in self.allowed_symbols:
                raise Exception(
                    f"Error: Detected unauthorized symbol {symbol}"
                )

    def upload_by_symbol(self) -> None:
        """
        Upload files by symbol with delays to respect rate limits.
        """
        print("\nStarting symbol-by-symbol upload...")

        symbols: List[str] = []
        for symbol_dir in Path(self.organized_folder).iterdir():
            if symbol_dir.is_dir() and symbol_dir.name in self.allowed_symbols:
                symbols.append(symbol_dir.name)

        symbols.sort()
        total_symbols = len(symbols)

        for idx, symbol in enumerate(symbols, 1):
            print(f"\n[{idx}/{total_symbols}] Uploading {symbol}...")

            try:
                upload_folder(
                    folder_path=str(
                        Path(self.organized_folder) / symbol
                    ),
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    path_in_repo=f"klines/{symbol}",
                    commit_message=f"Upload {symbol} step2 data - organized by timeframe",
                    allow_patterns=["*.npy", "*.pkl", "*.json"],
                )
                print(f"Success: {symbol} uploaded successfully")
            except Exception as e:
                print(f"Warning: {symbol} upload failed: {e}")

            if idx < total_symbols:
                print(f"Waiting 5 seconds ({idx}/{total_symbols} completed)...")
                time.sleep(5)

        summary_path = Path(self.organized_folder) / "execution_summary.json"
        if summary_path.exists():
            print(f"\nUploading execution summary...")
            try:
                upload_file(
                    path_or_fileobj=str(summary_path),
                    path_in_repo="execution_summary.json",
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    commit_message="Upload execution summary",
                )
                print(f"Success: execution_summary.json uploaded")
            except Exception as e:
                print(f"Warning: execution_summary.json upload failed: {e}")

    def verify_upload(self) -> None:
        """
        Verify that uploaded files exist on HuggingFace.
        """
        print("\nVerifying upload...")
        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )

            file_list = [f for f in files]

            expected_patterns = [
                "klines/BTCUSDT/",
                "klines/ETHUSDT/",
                "execution_summary.json",
            ]

            for pattern in expected_patterns:
                found = any(pattern in f for f in file_list)
                if found:
                    print(f"Success: Found {pattern}")
                else:
                    print(f"Warning: Missing {pattern}")

        except Exception as e:
            print(f"Warning: Could not verify upload: {e}")

    def run(self) -> None:
        """
        Execute the complete upload process.
        """
        try:
            print("="*72)
            print("STEP 2 DATA UPLOAD SYSTEM")
            print("="*72)

            self.verify_prerequisites()

            if not self.organize_files():
                print(
                    "Warning: No files were organized, terminating upload"
                )
                return

            self.check_remote_folders()

            self.upload_by_symbol()

            self.verify_upload()

            print("\n" + "="*72)
            print("SUCCESS: Upload completed")
            print("="*72)

        except Exception as e:
            print(f"\nError: {e}")
            raise


if __name__ == "__main__":
    uploader = Step2DataUploader()
    uploader.run()
