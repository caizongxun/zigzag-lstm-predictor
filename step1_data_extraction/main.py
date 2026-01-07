"""
Main Execution Script: Complete OHLCV data extraction and cleaning pipeline.
Downloads data from HuggingFace, cleans it, validates quality, and exports to CSV.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm

from config import Config
from data_loader import DataLoader
from data_cleaner import clean_ohlcv_data
from validator import check_data_quality


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step1_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataExtractionPipeline:
    """
    Main pipeline for data extraction, cleaning, and validation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the pipeline.
        
        Args:
            config (Config): Configuration object
        """
        self.config = config
        self.data_loader = DataLoader(config.loader_config)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.execution_log = []
        self.pipeline_results = {}
    
    def run(self, symbols: List[str] = None, timeframes: List[str] = None) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            symbols (List[str]): Symbols to process (default from config)
            timeframes (List[str]): Timeframes to process (default from config)
            
        Returns:
            Dict: Pipeline execution results
        """
        symbols = symbols or self.config.symbols
        timeframes = timeframes or self.config.timeframes
        
        logger.info(f"Starting pipeline: {len(symbols)} symbols x {len(timeframes)} timeframes")
        logger.info(f"Output directory: {self.output_dir}")
        
        start_time = datetime.now()
        self.log_step(f"Pipeline started at {start_time.isoformat()}")
        
        for symbol in tqdm(symbols, desc="Processing symbols", unit="symbol"):
            for timeframe in tqdm(timeframes, desc=f"  {symbol} timeframes", unit="tf", leave=False):
                result = self._process_symbol_timeframe(symbol, timeframe)
                if result:
                    self.pipeline_results[f"{symbol}_{timeframe}"] = result
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.log_step(f"Pipeline completed in {duration:.2f} seconds")
        
        summary = self._generate_summary()
        self._save_reports(summary, start_time, end_time, duration)
        
        return summary
    
    def _process_symbol_timeframe(self, symbol: str, timeframe: str) -> Dict:
        """
        Process a single symbol-timeframe combination.
        
        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe
            
        Returns:
            Dict: Processing results
        """
        logger.info(f"Processing {symbol} {timeframe}...")
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'processing'
        }
        
        try:
            df = self._download_and_load(symbol, timeframe)
            if df is None:
                result['status'] = 'download_failed'
                return result
            
            result['rows_before_cleaning'] = len(df)
            result['download_time'] = datetime.now().isoformat()
            file_size_mb = self.data_loader.get_file_info(df)['size_mb'] if hasattr(self.data_loader, 'get_file_info') else 0
            result['file_size_mb'] = file_size_mb
            
            logger.info(f"Downloaded: {len(df)} rows, {file_size_mb:.2f}MB")
            
            df, clean_report = self._clean_data(df)
            result['rows_after_cleaning'] = len(df)
            result['cleaning_report'] = clean_report
            logger.info(f"Cleaned: {len(df)} rows remaining")
            
            quality_report = self._validate_data(df)
            result['quality_report'] = quality_report
            logger.info(f"Quality Score: {quality_report['overall_quality_score']}")
            
            csv_file = self._save_to_csv(df, symbol, timeframe)
            result['csv_file'] = str(csv_file)
            result['status'] = 'success'
            
            self.log_step(f"Completed {symbol} {timeframe}: {result['rows_after_cleaning']} rows, Score: {quality_report['overall_quality_score']}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol} {timeframe}: {str(e)}")
            result['status'] = 'error'
            result['error'] = str(e)
            self.log_step(f"ERROR {symbol} {timeframe}: {str(e)}")
        
        return result
    
    def _download_and_load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Download and load OHLCV data.
        
        Args:
            symbol (str): Symbol to download
            timeframe (str): Timeframe
            
        Returns:
            pd.DataFrame: Loaded data or None
        """
        df = self.data_loader.load_ohlcv_data(symbol, timeframe)
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean OHLCV data.
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Cleaned data and report
        """
        return clean_ohlcv_data(df, self.config.cleaner_config)
    
    def _validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate data quality.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            Dict: Quality report
        """
        return check_data_quality(df)
    
    def _save_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        """
        Save DataFrame to CSV file.
        
        Args:
            df (pd.DataFrame): Data to save
            symbol (str): Symbol
            timeframe (str): Timeframe
            
        Returns:
            Path: Path to saved file
        """
        filename = f"{symbol}_{timeframe}.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved to {filepath}")
        return filepath
    
    def _generate_summary(self) -> Dict:
        """
        Generate execution summary.
        
        Returns:
            Dict: Summary statistics
        """
        summary = {
            'total_processed': len(self.pipeline_results),
            'successful': sum(1 for r in self.pipeline_results.values() if r.get('status') == 'success'),
            'failed': sum(1 for r in self.pipeline_results.values() if r.get('status') != 'success'),
            'details': self.pipeline_results
        }
        
        total_rows_before = sum(r.get('rows_before_cleaning', 0) for r in self.pipeline_results.values())
        total_rows_after = sum(r.get('rows_after_cleaning', 0) for r in self.pipeline_results.values())
        
        summary['total_rows_before_cleaning'] = total_rows_before
        summary['total_rows_after_cleaning'] = total_rows_after
        summary['total_rows_removed'] = total_rows_before - total_rows_after
        
        avg_quality = np.mean([
            r.get('quality_report', {}).get('overall_quality_score', 0)
            for r in self.pipeline_results.values()
            if r.get('status') == 'success'
        ])
        summary['average_quality_score'] = round(avg_quality, 2)
        
        return summary
    
    def _save_reports(self, summary: Dict, start_time: datetime, end_time: datetime, duration: float):
        """
        Save execution reports.
        
        Args:
            summary (Dict): Execution summary
            start_time (datetime): Start time
            end_time (datetime): End time
            duration (float): Duration in seconds
        """
        report = {
            'execution_start': start_time.isoformat(),
            'execution_end': end_time.isoformat(),
            'duration_seconds': duration,
            'summary': summary,
            'execution_log': self.execution_log
        }
        
        report_file = self.output_dir / 'validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved validation report to {report_file}")
        
        self._save_execution_log()
    
    def _save_execution_log(self):
        """
        Save execution log to file.
        """
        log_file = self.output_dir / 'EXECUTION_LOG.md'
        with open(log_file, 'w') as f:
            f.write('# Step 1 Data Extraction Execution Log\n\n')
            f.write(f'Execution Time: {datetime.now().isoformat()}\n\n')
            f.write('## Events\n\n')
            for event in self.execution_log:
                f.write(f'- {event}\n')
        logger.info(f"Saved execution log to {log_file}")
    
    def log_step(self, message: str):
        """
        Log a step in execution.
        
        Args:
            message (str): Step message
        """
        self.execution_log.append(f"{datetime.now().isoformat()}: {message}")


def main():
    """
    Main entry point.
    """
    try:
        config = Config()
        logger.info("Configuration loaded")
        logger.info(f"Testing symbols: {config.symbols}")
        logger.info(f"Testing timeframes: {config.timeframes}")
        
        pipeline = DataExtractionPipeline(config)
        
        logger.info("Starting data extraction pipeline...")
        summary = pipeline.run()
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Processed: {summary['total_processed']} symbol-timeframe pairs")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Total rows before cleaning: {summary['total_rows_before_cleaning']}")
        logger.info(f"Total rows after cleaning: {summary['total_rows_after_cleaning']}")
        logger.info(f"Rows removed: {summary['total_rows_removed']}")
        logger.info(f"Average quality score: {summary['average_quality_score']}/100")
        logger.info("="*60)
        
        return summary
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    import numpy as np
    main()
