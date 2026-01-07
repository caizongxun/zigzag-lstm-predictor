"""
Test Script: Quick validation and debugging of data extraction pipeline.
Use this to test individual components or run a quick pipeline test.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

from config import Config
from data_loader import DataLoader
from data_cleaner import clean_ohlcv_data
from validator import check_data_quality


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_loader():
    """
    Test data loader functionality.
    """
    logger.info("Testing DataLoader...")
    config = Config()
    loader = DataLoader(config.loader_config)
    
    try:
        df = loader.load_ohlcv_data('BTCUSDT', '15m')
        if df is not None:
            logger.info(f"DataLoader OK: Loaded {len(df):,} rows")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return True
        else:
            logger.error("DataLoader FAILED: No data returned")
            return False
    except Exception as e:
        logger.error(f"DataLoader ERROR: {str(e)}")
        return False


def test_data_cleaner():
    """
    Test data cleaner functionality.
    """
    logger.info("Testing DataCleaner...")
    config = Config()
    loader = DataLoader(config.loader_config)
    
    try:
        df_raw = loader.load_ohlcv_data('BTCUSDT', '15m')
        if df_raw is None:
            logger.warning("Cannot test DataCleaner: No data loaded")
            return False
        
        initial_rows = len(df_raw)
        df_clean, report = clean_ohlcv_data(df_raw, config.cleaner_config)
        final_rows = len(df_clean)
        
        logger.info(f"DataCleaner OK: {initial_rows:,} -> {final_rows:,} rows")
        logger.info(f"Cleaning report: {report}")
        return True
    except Exception as e:
        logger.error(f"DataCleaner ERROR: {str(e)}")
        return False


def test_validator():
    """
    Test data validator functionality.
    """
    logger.info("Testing DataValidator...")
    config = Config()
    loader = DataLoader(config.loader_config)
    
    try:
        df = loader.load_ohlcv_data('BTCUSDT', '15m')
        if df is None:
            logger.warning("Cannot test DataValidator: No data loaded")
            return False
        
        df_clean, _ = clean_ohlcv_data(df, config.cleaner_config)
        report = check_data_quality(df_clean, config.validator_config)
        
        score = report.get('overall_quality_score', 0)
        logger.info(f"DataValidator OK: Quality score {score:.1f}/100")
        logger.info(f"Validation report: {report}")
        return True
    except Exception as e:
        logger.error(f"DataValidator ERROR: {str(e)}")
        return False


def test_pipeline():
    """
    Run quick pipeline test with BTC 15m data.
    """
    logger.info("Starting pipeline test...")
    logger.info("=" * 70)
    
    from main import DataExtractionPipeline
    
    try:
        config = Config()
        pipeline = DataExtractionPipeline(config)
        
        result = pipeline.run(
            symbols=['BTCUSDT'],
            timeframes=['15m']
        )
        
        logger.info("=" * 70)
        logger.info("Pipeline test completed successfully")
        logger.info(f"Result: {result}")
        return True
    except Exception as e:
        logger.error(f"Pipeline test FAILED: {str(e)}")
        return False


def print_system_info():
    """
    Print system and configuration information.
    """
    logger.info("=" * 70)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 70)
    
    try:
        import pandas
        logger.info(f"Pandas version: {pandas.__version__}")
    except:
        logger.warning("Pandas not available")
    
    try:
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
    except:
        logger.warning("NumPy not available")
    
    try:
        import pyarrow
        logger.info(f"PyArrow version: {pyarrow.__version__}")
    except:
        logger.warning("PyArrow not available")
    
    try:
        from huggingface_hub import __version__
        logger.info(f"HuggingFace Hub version: {__version__}")
    except:
        logger.warning("HuggingFace Hub not available")
    
    logger.info("=" * 70)


def main():
    """
    Main test runner.
    """
    print_system_info()
    
    logger.info("\nRunning component tests...\n")
    
    tests = [
        ("DataLoader", test_data_loader),
        ("DataCleaner", test_data_cleaner),
        ("DataValidator", test_validator),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n>>> {test_name}")
        results[test_name] = test_func()
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPONENT TEST RESULTS")
    logger.info("=" * 70)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\nAll component tests passed.")
        logger.info("Run pipeline test? (y/n): ", end='')
        
        response = input().strip().lower()
        if response == 'y':
            logger.info("\n>>> Pipeline Test")
            test_pipeline()
    else:
        logger.error("\nSome component tests failed. Fix issues before running pipeline.")
        sys.exit(1)


if __name__ == '__main__':
    main()
