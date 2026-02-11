# QuantIQ: AI-Driven Stock Market Analysis 📈

This project is a sophisticated **Algorithmic Trading System** that leverages **Machine Learning (ML)** and **Deep Learning** techniques to predict stock price movements and execute optimal trading strategies.

It combines traditional technical analysis (RSI, Bollinger Bands, Moving Averages) with advanced AI models to achieve robust profitability, specifically optimized for the **Daily Timeframe**.

---

## 🚀 Key Features & Strategy

### 1. Hybrid "Desi Trader" Strategy
We combine the wisdom of technical traders with the power of AI:
-   **Trend Confirmation (SuperTrend)**: The model utilizes the `SuperTrend` indicator (7, 3) as a core feature. It learns to identify high-probability setups by respecting the underlying market trend, mimicking a seasoned trader who "trends with the trend."
-   **Momentum & Volatility**: Incorporates RSI, MACD, Bollinger Bands, ADX, and ATR to capture market dynamics.
-   **AI Decision Making**: Instead of rigid `if/else` rules, the AI learns complex non-linear relationships between these indicators to generate buy/sell signals.

### 2. Multi-Model Ensemble (The "Senior Trader" Logic)
The system does not rely on a single model. It uses a **Voting Ensemble** of 5 distinct algorithms to ensure robustness and reduce overfitting:
1.  **XGBoost (Gradient Boosting)**: Excellent for structured data and capturing complex patterns (~39% Return).
2.  **Gradient Boosting (GBM)**: A second boosting algorithm to add diversity and stability (~30% Return).
3.  **Random Forest**: Captures non-linear interactions and is resistant to noise (~23% Return).
4.  **Support Vector Machine (SVM)**: Finds the optimal hyperplane for classification.
5.  **Multi-Layer Perceptron (Neural Network)**: A deep learning model that learns abstract feature representations (~15% Return).

**Why this is huge:** By combining these models, the system filters out false positives. If one model is wrong, the others can correct it, leading to a higher win rate and lower drawdown.

### 3. Advanced Backtesting Engine
-   **Realistic Simulation**: Accounts for transaction costs, slippage, and market liquidity.
-   **Dynamic Timeframe Support**: Can analyze 1-Minute, 5-Minute, Hourly, or Daily data.
-   **Performance Metrics**: Automatically calculates Total Return, Win Rate, Max Drawdown, Sharpe Ratio, and more.

---

## 📊 Performance (Daily Timeframe)

After rigorous optimization (Phase 16), the system achieved the following on HDFC Bank data:

| Model | Total Return | Win Rate | Verdict |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **39.42%** | **62.50%** | 🏆 **Best Performer** |
| **Gradient Boosting** | **30.21%** | **50.00%** | 🥈 Strong Runner-up |
| **Ensemble (All 5)** | **37.22%** | **50.00%** | 🛡️ **Most Robust** |
| Random Forest | 23.79% | 45.45% | Reliable Baseline |
| Neural Network (MLP) | 15.00% | 66.67% | High Precision, Low Frequency |

*> Note: These results significantly outperform a standard Buy & Hold strategy during choppy market periods.*

---

## 🛠️ Project Architecture & Workflow

The system follows a modular, production-grade pipeline:

```mermaid
flowchart LR
    %% =========================
    %% Data Pipeline
    %% =========================
    subgraph Data Pipeline
        A[Raw Market Data<br/>(CSV / API)]
        B[Data Loader]
        C[Preprocessing<br/>Cleaning & Resampling]
        D[Feature Engineering]
    end

    %% =========================
    %% Modeling Pipeline
    %% =========================
    subgraph Modeling Pipeline
        E[Model Training]
        F[Model Evaluation]
    end

    %% =========================
    %% Strategy Pipeline
    %% =========================
    subgraph Strategy & Analysis
        G[Backtesting Engine]
        H[Final Reports & Visualizations]
    end

    %% Flow
    A --> B --> C --> D --> E --> F --> G --> H
```



### Step-by-Step Workflow:

1.  **Data Loading (`data_loader.py`)**:
    -   Reads raw CSV data (Open, High, Low, Close, Volume).
    -   Handles date parsing and filtering.

2.  **Preprocessing (`preprocessing.py`)**:
    -   Cleans missing values.
    -   **Resampling**: Converts raw minute data into Daily (1D), Hourly (1H), or customized timeframes.

3.  **Feature Engineering (`features.py`)**:
    -   Calculates Technical Indicators:
        -   **Trend**: SMA_50, SMA_200, EMA_12, EMA_26, MACD, SuperTrend.
        -   **Momentum**: RSI (Relative Strength Index).
        -   **Volatility**: Bollinger Bands, ATR (Average True Range).
        -   **Volume**: OBV (On-Balance Volume), Volume MA.
    -   **Lag Features**: Adds past returns to give the model context of recent history.

4.  **Model Training (`models.py`)**:
    -   Splits data into Training and Testing sets (keeps time order intact).
    -   Scales features (StandardScaler) for Neural Networks.
    -   Trains all 5 models (RF, XGB, SVM, GBM, MLP).
    -   Optimizes hyperparameters for maximum performance.

5.  **Evaluation & Backtesting (`evaluation.py`, `backtesting.py`)**:
    -   Evaluates models using Accuracy, Precision, Recall, and F1-Score.
    -   Simulates real trading:
        -   **Entry**: Buy when the Ensemble or Model predicts "BUY".
        -   **Exit**: Sell when the Model predicts "SELL" or Stop-Loss/Take-Profit hits.
        -   **Risk Management**: Uses ATR-based stops and dynamic position sizing.

---

## 💻 How to Run

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
To train the models and run the backtest on Daily data:

```bash
python main.py --data_path stock_prediction/data/raw/HDFCBANK_minute.csv --timeframe 1d
```

### 3. View Results
-   **Logs**: Check `stock_prediction/logs/` for detailed execution logs.
-   **Plots**: Check `stock_prediction/data/plots/` for:
    -   `backtest_results_table.png`: Summary of returns.
    -   `confusion_matrix_*.png`: Model accuracy visualization.
    -   `roc_curve_comparison.png`: Model comparison.

---

## 🌟 Why This Project is Beneficial

1.  **Data-Driven Decisions**: Removes emotional bias from trading. The AI strictly follows patterns it has learned from historical data.
2.  **Adaptability**: The "Hybrid" strategy adapts to changing market conditions (Bullish, Bearish, or Sideways) by using dynamic indicators like SuperTrend and volatility bands.
3.  **Scalability**: The architecture is ready for **Deep Learning**. We can easily swap the current models for LSTMs or Transformers to leverage high-frequency (1-minute) data in the future.

---
**Developed for Advanced Algorithmic Trading Research.**
