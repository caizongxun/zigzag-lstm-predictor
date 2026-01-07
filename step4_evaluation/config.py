"""
Configuration for Step 4 Evaluation
"""

EVALUATION_CONFIG = {
    'test_split': 0.3,
    'random_state': 42,
    'save_plots': True,
    'plot_format': 'png',
    'dpi': 150
}

CLASSIFICATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'auc_roc',
    'confusion_matrix'
]

REGRESSION_METRICS = [
    'mse',
    'mae',
    'rmse',
    'r2_score'
]

BACKTEST_CONFIG = {
    'calculate_hit_rate': True,
    'calculate_lead_time': True,
    'calculate_confidence_stats': True,
    'confidence_threshold': 0.7
}

REPORT_CONFIG = {
    'generate_html': True,
    'generate_json': True,
    'include_plots': True,
    'summary_metrics': True
}

OUTPUT_CONFIG = {
    'output_dir': '../step4_output',
    'eval_results_file': 'evaluation_results.json',
    'backtest_results_file': 'backtest_results.json',
    'html_report_file': 'performance_report.html',
    'execution_log_file': 'EXECUTION_LOG.md'
}
