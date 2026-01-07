#!/usr/bin/env python3
"""
BTCUSBT Upload Verification Script

Verifies that BTCUSDT data has been correctly uploaded to Hugging Face.
"""

import json
from pathlib import Path
from typing import Dict, List

try:
    from huggingface_hub import list_repo_files
except ImportError:
    raise ImportError("Please install: pip install huggingface-hub")


class BtcustdUploadVerifier:
    """
    Verify BTCUSDT upload to Hugging Face.
    """
    
    def __init__(self):
        self.repo_id = "zongowo111/v2-crypto-ohlcv-data"
        self.repo_type = "dataset"
        self.symbol = "BTCUSDT"
        self.expected_files = {
            f"klines/{self.symbol}/15m/1h_X_sequences.npy": "1h_X_sequences.npy",
            f"klines/{self.symbol}/15m/1h_y_class.npy": "1h_y_class.npy",
            f"klines/{self.symbol}/15m/1h_y_reg.npy": "1h_y_reg.npy",
            f"klines/{self.symbol}/15m/1h_scaler.pkl": "1h_scaler.pkl",
            f"klines/{self.symbol}/15m/1h_zigzag_points.json": "1h_zigzag_points.json",
            f"klines/{self.symbol}/15m/1h_statistics.json": "1h_statistics.json",
            f"klines/{self.symbol}/1h/15m_X_sequences.npy": "15m_X_sequences.npy",
            f"klines/{self.symbol}/1h/15m_y_class.npy": "15m_y_class.npy",
            f"klines/{self.symbol}/1h/15m_y_reg.npy": "15m_y_reg.npy",
            f"klines/{self.symbol}/1h/15m_scaler.pkl": "15m_scaler.pkl",
            f"klines/{self.symbol}/1h/15m_zigzag_points.json": "15m_zigzag_points.json",
            f"klines/{self.symbol}/1h/15m_statistics.json": "15m_statistics.json",
            "execution_summary.json": "execution_summary.json",
        }
    
    def verify_remote_structure(self) -> Dict:
        """
        Verify remote file structure on Hugging Face.
        """
        print("\n[REMOTE VERIFICATION] Fetching file list from Hugging Face...")
        
        try:
            files = list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.repo_type
            )
            print(f"  OK: Retrieved {len(files)} files from repository")
        except Exception as e:
            print(f"  ERROR: Failed to fetch files: {e}")
            return {"status": "error", "message": str(e)}
        
        # Check expected files
        results = {
            "total_files": len(files),
            "btcusdt_found": False,
            "15m_folder_found": False,
            "1h_folder_found": False,
            "missing_files": [],
            "found_files": [],
            "execution_summary_found": False,
        }
        
        for expected_path in self.expected_files.keys():
            found = any(expected_path in f for f in files)
            
            if found:
                results["found_files"].append(expected_path)
                if "BTCUSDT" in expected_path:
                    results["btcusdt_found"] = True
                if "15m/" in expected_path:
                    results["15m_folder_found"] = True
                if "1h/" in expected_path:
                    results["1h_folder_found"] = True
                if "execution_summary" in expected_path:
                    results["execution_summary_found"] = True
            else:
                results["missing_files"].append(expected_path)
        
        return results
    
    def print_verification_report(self, results: Dict):
        """
        Print detailed verification report.
        """
        print("\n" + "="*70)
        print("BTCUSDT UPLOAD VERIFICATION REPORT")
        print("="*70)
        
        if "error" in results:
            print(f"\nERROR: {results['message']}")
            return
        
        print(f"\n[REPOSITORY INFO]")
        print(f"  Repo ID: {self.repo_id}")
        print(f"  Total files: {results['total_files']}")
        
        print(f"\n[STRUCTURE CHECK]")
        print(f"  BTCUSDT folder found: {results['btcusdt_found']}")
        print(f"  15m folder found: {results['15m_folder_found']}")
        print(f"  1h folder found: {results['1h_folder_found']}")
        print(f"  execution_summary.json found: {results['execution_summary_found']}")
        
        print(f"\n[FILE DETAILS]")
        print(f"  Found files: {len(results['found_files'])}/13")
        for file_path in sorted(results["found_files"]):
            print(f"    OK: {file_path}")
        
        if results["missing_files"]:
            print(f"\n  Missing files: {len(results['missing_files'])}")
            for file_path in sorted(results["missing_files"]):
                print(f"    MISSING: {file_path}")
        
        print(f"\n[VERIFICATION RESULT]")
        all_found = len(results["missing_files"]) == 0
        if all_found:
            print("  SUCCESS: All files uploaded correctly")
        else:
            print(f"  WARNING: {len(results['missing_files'])} files missing")
        
        print("\n" + "="*70)
    
    def save_report(self, results: Dict, filename: str = "btcusdt_verification_report.json"):
        """
        Save verification report to JSON file.
        """
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to: {filename}")
    
    def run(self):
        """
        Execute verification process.
        """
        print("="*70)
        print("BTCUSDT UPLOAD VERIFICATION SYSTEM")
        print("="*70)
        
        # Verify remote structure
        results = self.verify_remote_structure()
        
        # Print report
        self.print_verification_report(results)
        
        # Save report
        self.save_report(results)
        
        return results


if __name__ == "__main__":
    verifier = BtcustdUploadVerifier()
    verifier.run()
