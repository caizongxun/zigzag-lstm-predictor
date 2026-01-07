# Step 4: Model Evaluation

Comprehensive model evaluation, backtesting analysis, and performance visualization.

## Status

Framework structure prepared. Ready for model evaluation after Step 3 training completion.

## Components

- **evaluator.py**: Classification and regression evaluation metrics
- **backtester.py**: Trading simulation and performance analysis
- **visualizer.py**: Result visualization and charts
- **reporter.py**: Report generation (HTML and JSON)
- **main.py**: Pipeline orchestration

## Input

Step 3 Output:
- `../step3_output/test_model_BTC_15m.h5`
- `../step2_output/BTC_15m_X_sequences.npy`
- `../step2_output/BTC_15m_y_class.npy`
- `../step2_output/BTC_15m_y_reg.npy`
- `../step2_output/BTC_15m_scaler.pkl`

## Output

Evaluation Reports:
- `step4_output/evaluation_results.json`
- `step4_output/backtest_results.json`
- `step4_output/performance_report.html`

Visualizations:
- `step4_output/confusion_matrix.png`
- `step4_output/roc_curve.png`
- `step4_output/predictions_vs_actual.png`
- `step4_output/residuals_plot.png`

## Evaluation Metrics

### Classification (HH vs LL prediction)
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
- Confusion Matrix

### Regression (Bars to next turning point)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared Score

### Backtesting
- Hit Rate: Correct turning point predictions %
- Average Lead Time: Bars ahead of actual turning point
- Confidence Distribution
- Signal Quality

## Usage

### Run Full Evaluation
```bash
python main.py
```

### Generate Specific Reports
```python
from evaluator import evaluate_classification
from backtester import backtest_predictions
from visualizer import plot_confusion_matrix
```

## Output Formats

### evaluation_results.json
```json
{
  "classification": {
    "accuracy": 0.87,
    "precision": 0.85,
    "recall": 0.89,
    "f1": 0.87,
    "auc": 0.92
  },
  "regression": {
    "mse": 15.2,
    "mae": 3.1,
    "rmse": 3.9,
    "r2": 0.78
  }
}
```

### performance_report.html
- Interactive visualizations
- Summary metrics
- Conclusion and recommendations

## Next Steps

After evaluation, options include:
1. Train models for other symbols/timeframes
2. Hyperparameter tuning
3. Ensemble methods
4. Production deployment
