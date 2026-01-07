Zigzag LSTM Predictor - Multi-threaded ML Training System

Project Overview

A comprehensive multi-step machine learning pipeline for predicting Zigzag turning points (HH, HL, LL, LH) in cryptocurrency trading using LSTM neural networks.

Version: v1 (Testing Phase)

Target Cryptocurrency: BTC (Testing), then 22 other cryptos

Timeframes: 15m, 1h

Data Source: Hugging Face Dataset (zongowo111/v2-crypto-ohlcv-data)

---

Architecture

This project uses a multi-threaded AI assistant system where each step is handled independently:

STEP 1: Data Extraction (HF Parquet -> Clean Data)
   |
   v
STEP 2: Feature Engineering + Zigzag Calculation
   |
   v
STEP 3: LSTM Training (Colab) + Model Upload
   |
   v
STEP 4: Model Evaluation + Backtesting

---

Directory Structure

step1_data_extraction/
- Parquet files from HF
- Data cleaning and validation
- Output: CSV format training data

step2_feature_engineering/
- Zigzag calculation and labeling
- Technical indicators (RSI, MACD, ATR, etc.)
- Output: Labeled sequences for LSTM

step3_lstm_training/
- LSTM model architecture
- Training pipeline (with Colab support)
- Colab upload utility
- Local upload utility

step4_evaluation/
- Model performance metrics
- Backtesting framework

---

Key Components

HuggingFace Data Structure

Dataset URL: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
Data Location: klines/{SYMBOL}/{SYMBOL}_{TIMEFRAME}.parquet
Example: klines/BTCUSDT/BTC_15m.parquet

Model Output Structure

v2_model/
- BTC/
  - 15m/
    - model.h5
    - scaler.pkl
    - config.json
  - 1h/
    - model.h5
    - scaler.pkl
    - config.json

---

Configuration

Each step has its own config.py with:
- Data paths
- Model hyperparameters
- Training settings
- Upload credentials

---

Important Notes

1. No emoji usage in code or documentation
2. Each step can be executed independently
3. All file updates use git push (no manual approval needed)
4. Colab training requires cell code (no full repo clone)
5. Model uploads to single HF dataset endpoint

Status: In Development

Currently testing with BTC 15m timeframe. Full pipeline rollout pending.
