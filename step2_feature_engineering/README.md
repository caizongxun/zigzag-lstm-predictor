# Step 2: Feature Engineering

This module computes Zigzag turning points, extracts technical indicators, and builds LSTM training sequences.

## Status

Framework structure prepared. Implementation ready for execution after Step 1 completion.

## Components

- **zigzag.py**: Zigzag turning point calculation
- **indicators.py**: Technical indicators (RSI, MACD, ATR)
- **feature_extractor.py**: Feature extraction and normalization
- **sequence_builder.py**: LSTM sequence construction
- **main.py**: Pipeline orchestration

## Input

Step 1 Output:
- `../step1_output/BTC_15m.csv`
- `../step1_output/BTC_1h.csv`

## Output

Step 2 Output:
- `BTC_15m_X_sequences.npy`: Feature sequences
- `BTC_15m_y_class.npy`: Classification labels
- `BTC_15m_y_reg.npy`: Regression labels
- `BTC_15m_scaler.pkl`: StandardScaler object
- `BTC_15m_zigzag_points.json`: Turning point details
- `BTC_15m_statistics.json`: Data statistics

## Next Steps

Implementation begins after Step 1 validation.
