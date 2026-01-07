STEP 3: LSTM Model Training (Colab Environment)

Responsibility: Build, train, and upload LSTM model. Must run in Google Colab and provide upload utilities.

Input:
- Processed sequences from STEP 2
- Feature scaler object

Output:
- Trained LSTM model (model.h5)
- Scaler pickle file (scaler.pkl)
- Config JSON file
- Colab upload script
- Local upload script

Model Architecture:
- Multi-task LSTM: Classification + Regression heads
- Classification: Predict zigzag type (HH vs LL)
- Regression: Predict bars to next turning point

Training Output Files:
1. test_model_{symbol}_{timeframe}.h5 - Test model
2. test_scaler_{symbol}_{timeframe}.pkl - Test scaler
3. test_config_{symbol}_{timeframe}.json - Test config
4. colab_upload_{symbol}_{timeframe}.py - Colab upload utility
5. local_upload_{symbol}_{timeframe}.py - Local upload utility

Colab Cell Code Requirements:
- No full repo clone
- Direct file uploads
- HF token integration
- Progress tracking

Before implementing:
1. Research LSTM multi-task learning architecture
2. Research TensorFlow/Keras best practices for time series
3. Research model serialization (h5, SavedModel formats)
4. Research HF upload API and authentication
5. Research Colab environment constraints and file handling
6. Research progress bar and status reporting in Colab
