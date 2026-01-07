import os
import json
from pathlib import Path
from typing import Dict, List, Tuple


class Step2UploadVerifier:
    def __init__(self):
        self.repo_id = "zongowo111/v2-crypto-ohlcv-data"
        self.repo_type = "dataset"
        self.allowed_symbols = ["BTCUSDT", "ETHUSDT"]
        self.timeframes = ["15m", "1h"]
        self.expected_files = [
            "X_sequences.npy",
            "y_class.npy",
            "y_reg.npy",
            "scaler.pkl",
            "zigzag_points.json",
            "statistics.json",
        ]

    def verify_local_organization(self) -> Tuple[bool, str]:
        """
        Verify that local organized folder has correct structure.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        print("Verifying local organization...")
        organized_folder = "./organized_step2_output"

        if not os.path.exists(organized_folder):
            return False, f"Organized folder not found: {organized_folder}"

        organized_path = Path(organized_folder)

        for symbol in self.allowed_symbols:
            symbol_path = organized_path / symbol
            if not symbol_path.exists():
                return False, f"Symbol folder missing: {symbol}"

            for timeframe in self.timeframes:
                timeframe_path = symbol_path / timeframe
                if not timeframe_path.exists():
                    return False, f"Timeframe folder missing: {symbol}/{timeframe}"

                files_in_folder = list(timeframe_path.glob("*"))
                if not files_in_folder:
                    return False, f"Empty timeframe folder: {symbol}/{timeframe}"

        summary_path = organized_path / "execution_summary.json"
        if not summary_path.exists():
            return False, "execution_summary.json not found"

        print("Success: Local organization verified")
        return True, "Local organization is correct"

    def verify_remote_structure(self) -> Tuple[bool, Dict]:
        """
        Verify that remote HuggingFace dataset has expected structure.
        
        Returns:
            Tuple of (success: bool, details: dict)
        """
        print("\nVerifying remote HuggingFace structure...")
        try:
            from huggingface_hub import list_repo_files
        except ImportError:
            return False, {"error": "huggingface_hub not installed"}

        try:
            files = list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )
            file_list = list(files)
        except Exception as e:
            return False, {"error": str(e)}

        results = {"total_files": len(file_list), "symbols": {}}
        all_present = True

        for symbol in self.allowed_symbols:
            results["symbols"][symbol] = {"timeframes": {}}

            for timeframe in self.timeframes:
                prefix = self._get_prefix(timeframe)
                path_prefix = f"klines/{symbol}/{timeframe}/"
                timeframe_files = [
                    f
                    for f in file_list
                    if f.startswith(path_prefix)
                    and f.endswith(".npy")
                    or f.endswith(".pkl")
                    or f.endswith(".json")
                ]

                results["symbols"][symbol]["timeframes"][timeframe] = {
                    "files_found": len(timeframe_files),
                    "files": timeframe_files,
                }

                if len(timeframe_files) == 0:
                    all_present = False

        has_summary = any("execution_summary.json" in f for f in file_list)
        results["execution_summary_present"] = has_summary

        if not has_summary:
            all_present = False

        return all_present, results

    def _get_prefix(self, timeframe: str) -> str:
        """
        Get file prefix based on timeframe.
        
        Args:
            timeframe: "15m" or "1h"
            
        Returns:
            File prefix: "1h_" for 15m timeframe, "15m_" for 1h timeframe
        """
        if timeframe == "15m":
            return "1h_"
        elif timeframe == "1h":
            return "15m_"
        return ""

    def verify_file_count(self, organized_folder: str = "./organized_step2_output") -> Tuple[bool, Dict]:
        """
        Count and verify files in organized folder.
        
        Args:
            organized_folder: Path to organized folder
            
        Returns:
            Tuple of (success: bool, counts: dict)
        """
        print("\nCounting files in organized folder...")
        
        if not os.path.exists(organized_folder):
            return False, {"error": "Organized folder not found"}

        counts = {"symbols": {}, "total_files": 0}
        expected_per_timeframe = len(self.expected_files)

        for symbol in self.allowed_symbols:
            symbol_path = Path(organized_folder) / symbol
            counts["symbols"][symbol] = {"timeframes": {}}

            if not symbol_path.exists():
                counts["symbols"][symbol]["status"] = "missing"
                continue

            for timeframe in self.timeframes:
                timeframe_path = symbol_path / timeframe
                if timeframe_path.exists():
                    files = list(timeframe_path.glob("*"))
                    file_count = len(files)
                    counts["symbols"][symbol]["timeframes"][timeframe] = {
                        "count": file_count,
                        "expected": expected_per_timeframe,
                        "complete": file_count == expected_per_timeframe,
                        "files": [f.name for f in files],
                    }
                    counts["total_files"] += file_count
                else:
                    counts["symbols"][symbol]["timeframes"][timeframe] = {
                        "status": "missing"
                    }

        return True, counts

    def generate_verification_report(
        self, output_file: str = "upload_verification_report.json"
    ) -> None:
        """
        Generate comprehensive verification report.
        
        Args:
            output_file: Path to save report
        """
        print("\n" + "="*72)
        print("GENERATING VERIFICATION REPORT")
        print("="*72)

        report = {
            "timestamp": str(__import__("datetime").datetime.now()),
            "dataset": self.repo_id,
            "repo_type": self.repo_type,
            "checks": {},
        }

        local_ok, local_msg = self.verify_local_organization()
        report["checks"]["local_organization"] = {
            "success": local_ok,
            "message": local_msg,
        }
        print(f"Local Organization: {'OK' if local_ok else 'FAILED'}")

        if local_ok:
            count_ok, counts = self.verify_file_count()
            report["checks"]["file_counts"] = {
                "success": count_ok,
                "details": counts,
            }
            print(f"File Counts: Total {counts['total_files']} files")

        remote_ok, remote_details = self.verify_remote_structure()
        report["checks"]["remote_structure"] = {
            "success": remote_ok,
            "details": remote_details,
        }
        print(
            f"Remote Structure: {'OK' if remote_ok else 'INCOMPLETE'}"
        )

        if remote_ok and remote_details.get("symbols"):
            for symbol, data in remote_details["symbols"].items():
                total_in_symbol = sum(
                    v.get("files_found", 0) for v in data["timeframes"].values()
                )
                print(f"  {symbol}: {total_in_symbol} files")

        report["overall_status"] = "OK" if (local_ok and remote_ok) else "INCOMPLETE"

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_file}")

        print("\n" + "="*72)
        print(f"OVERALL STATUS: {report['overall_status']}")
        print("="*72)


if __name__ == "__main__":
    verifier = Step2UploadVerifier()
    verifier.generate_verification_report()
