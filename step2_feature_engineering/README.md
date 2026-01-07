STEP 2: Feature Engineering and Zigzag Calculation

Responsibility: Calculate Zigzag turning points, extract features, and create labeled training sequences

Input:
- CSV files from STEP 1

Output:
- Labeled sequences for LSTM training
- Feature-scaled data
- Zigzag point annotations (HH, HL, LL, LH)

Operations:
1. Calculate Zigzag turning points based on threshold
2. Generate classification labels (HH/LL) and regression labels (bars to next)
3. Calculate technical indicators (RSI, MACD, ATR, etc.)
4. Create price relative position features
5. Calculate momentum and volatility features
6. Standardize all features
7. Create sequences for LSTM input

Output Structure:
- X_sequences: Feature sequences (shape: n_samples, 30, n_features)
- y_classification: Turning point type (0=HH, 1=LL)
- y_regression: Bars to next turning point

Before implementing:
1. Research Zigzag indicator calculation algorithms
2. Research technical indicator formulas (RSI, MACD, ATR)
3. Research feature scaling for time series
4. Research sequence generation for LSTM
5. Research handling imbalanced classification in financial data
