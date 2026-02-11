
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import pandas as pd
import numpy as np
from stock_prediction.config import settings
from stock_prediction.src.utils import setup_logger

logger = setup_logger("visualization")

def save_plot(fig, filename):
    """Helper to save a matplotlib figure."""
    filepath = settings.PLOTS_DIR / filename
    fig.savefig(filepath)
    plt.close()
    logger.info(f"Saved plot: {filepath}")

def save_backtest_table(metrics_list: list):
    """
    Save backtest metrics as a table image.
    
    Args:
        metrics_list: List of dictionaries containing model metrics.
                      Example: [{'Model': 'RF', 'Return': '20%', ...}, ...]
    """
    if not metrics_list:
        logger.warning("No metrics to plot table.")
        return

    logger.info("Generating backtest result table...")
    
    df = pd.DataFrame(metrics_list)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 4)) # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    
    # Style logic
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2) # Scale width, height
    
    # Colors
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if key[0] == 0: # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4C72B0')
        else:
            cell.set_facecolor('#F5F5F5')
            
    output_path = settings.PLOTS_DIR / "backtest_results_table.png"
    save_plot(fig, "backtest_results_table.png")
    logger.info(f"Saved backtest table to {output_path}")

def plot_close_price(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['time'], df['close'], label="Close Price", color="dodgerblue")
    plt.title("Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.xticks(rotation=45)
    plt.legend()
    save_plot(plt.gcf(), "close_price.png")

def plot_candlestick(df, n=80):
    """Plot candlestick chart for last n periods."""
    ohlc_data = df[['time', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'Volume': 'volume'})
    ohlc_data = ohlc_data.tail(n).copy()
    ohlc_data.set_index('time', inplace=True)
    
    filepath = settings.PLOTS_DIR / "candlestick.png"
    mpf.plot(ohlc_data, type='candle', style='charles', title="Candlestick Chart - OHLC Data",
             ylabel="Price", ylabel_lower="Volume", volume=True, savefig=dict(fname=filepath, dpi=100))
    logger.info(f"Saved plot: {filepath}")

def plot_volume_bar(df, n=80):
    sampled_df = df.tail(n) # Using tail instead of sample for time continuity
    plt.figure(figsize=(14, 6))
    plt.bar(sampled_df['time'], sampled_df['volume'], color='purple', alpha=0.6)
    plt.title("Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), "volume_bar.png")

def plot_moving_averages(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['time'], df['close'], label="Close Price", color="blue")
    if 'ma_50' in df.columns:
        plt.plot(df['time'], df['ma_50'], label="50-day MA", color="orange", linestyle='--')
    if 'ma_200' in df.columns:
        plt.plot(df['time'], df['ma_200'], label="200-day MA", color="red", linestyle='--')
    plt.title("Close Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), "moving_averages.png")

def plot_high_low_area(df):
    plt.figure(figsize=(14, 6))
    plt.fill_between(df['time'], df['high'], df['low'], color='skyblue', alpha=0.4)
    plt.plot(df['time'], df['high'], color='lightcoral', label='High Price')
    plt.plot(df['time'], df['low'], color='darkslateblue', label='Low Price')
    plt.title("High-Low Price Range Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price Range")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), "high_low_area.png")

def plot_scatter_close_volume(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='volume', y='close', data=df, color="green", s=10)
    plt.title("Closing Price vs. Volume")
    plt.xlabel("Volume")
    plt.ylabel("Closing Price")
    save_plot(plt.gcf(), "scatter_close_volume.png")

def plot_rsi(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['time'], df['rsi'], label="RSI", color="purple")
    plt.axhline(settings.RSI_OVERBOUGHT, linestyle='--', alpha=0.5, color='red', label=f"Overbought ({settings.RSI_OVERBOUGHT})")
    plt.axhline(settings.RSI_OVERSOLD, linestyle='--', alpha=0.5, color='green', label=f"Oversold ({settings.RSI_OVERSOLD})")
    plt.title("RSI Over Time")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), "rsi.png")

def plot_macd(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['time'], df['macd_diff'], label="MACD Difference", color="brown")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.title("MACD Difference Over Time")
    plt.xlabel("Date")
    plt.ylabel("MACD Difference")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), "macd.png")

def plot_bollinger_bands(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['time'], df['close'], label="Close Price", color="blue")
    plt.plot(df['time'], df['bollinger_mavg'], label="Bollinger MAVG", color="black")
    plt.plot(df['time'], df['bollinger_hband'], label="Bollinger Upper Band", color="orange", linestyle='--')
    plt.plot(df['time'], df['bollinger_lband'], label="Bollinger Lower Band", color="orange", linestyle='--')
    plt.fill_between(df['time'], df['bollinger_lband'], df['bollinger_hband'], color='orange', alpha=0.1)
    plt.title("Bollinger Bands with Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), "bollinger_bands.png")

def plot_correlation_heatmap(df):
    corr_cols = ['close', 'rsi', 'ma_50', 'ma_200', 'ema_12', 'ema_26', 'macd_diff', 'bollinger_mavg', 'bollinger_hband', 'bollinger_lband', 'price_change_15m']
    # Filter only columns that exist
    corr_cols = [c for c in corr_cols if c in df.columns]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Price Indicators")
    save_plot(plt.gcf(), "correlation_heatmap.png")

def plot_rsi_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['rsi'], bins=30, color="purple", alpha=0.7)
    plt.axvline(settings.RSI_OVERBOUGHT, color='red', linestyle='--', label="Overbought")
    plt.axvline(settings.RSI_OVERSOLD, color='green', linestyle='--', label="Oversold")
    plt.title("Distribution of RSI Values")
    plt.xlabel("RSI")
    plt.ylabel("Frequency")
    plt.legend()
    save_plot(plt.gcf(), "rsi_distribution.png")

def plot_macd_histogram(df):
    plt.figure(figsize=(14, 6))
    # Simple bar plot logic, might be slow for full df, use tail if needed
    plt.bar(df['time'], df['macd_diff'], color=['green' if val >= 0 else 'red' for val in df['macd_diff']], alpha=0.7)
    plt.title("MACD Histogram")
    plt.xlabel("Date")
    plt.ylabel("MACD Difference")
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), "macd_histogram.png")

def plot_combined_analysis(df):
    """Plot subplots for Close, RSI, MACD."""
    fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Close Price
    ax[0].plot(df['time'], df['close'], label="Close Price", color="blue")
    if 'ma_50' in df.columns:
        ax[0].plot(df['time'], df['ma_50'], label="50-day MA", color="orange")
    if 'ma_200' in df.columns:
        ax[0].plot(df['time'], df['ma_200'], label="200-day MA", color="red")
    ax[0].set_title("Close Price with Moving Averages")
    ax[0].legend()
    
    # RSI
    ax[1].plot(df['time'], df['rsi'], label="RSI", color="purple")
    ax[1].axhline(settings.RSI_OVERBOUGHT, linestyle='--', alpha=0.5, color='red')
    ax[1].axhline(settings.RSI_OVERSOLD, linestyle='--', alpha=0.5, color='green')
    ax[1].set_title("RSI")
    ax[1].legend()
    
    # MACD
    ax[2].plot(df['time'], df['macd_diff'], label="MACD Difference", color="brown")
    ax[2].axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax[2].set_title("MACD Difference")
    ax[2].legend()
    
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    save_plot(fig, "combined_analysis.png")

def plot_ema(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['time'], df['close'], label="Close Price", color="blue")
    if 'ema_12' in df.columns:
        plt.plot(df['time'], df['ema_12'], label="EMA 12", color="purple", linestyle='--')
    if 'ema_26' in df.columns:
        plt.plot(df['time'], df['ema_26'], label="EMA 26", color="brown", linestyle='--')
    plt.title("Close Price with EMA 12 and EMA 26")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot(plt.gcf(), "ema_12_26.png")

def plot_price_change_distribution(df): # Distribution of Price Changes
    if 'past_return' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['past_return'], bins=50, kde=True)
        plt.title('Distribution of Price Changes (Past Return)')
        plt.xlabel('Price Change (%)')
        plt.ylabel('Frequency') # Added ylabel for clarity
        save_plot(plt.gcf(), 'price_change_distribution.png') # Changed to plt.gcf()
    # The following lines seem to be remnants from the old plot_price_change function
    # and are not relevant for a distribution plot.
    # plt.xlabel("Date")
    # plt.ylabel("Price Change (%)")
    # plt.legend()
    # plt.xticks(rotation=45)
    # save_plot(plt.gcf(), "price_change.png")

def plot_candlestick_mav(df, n=200):
    """Candlestick with 50 & 200 MA"""
    ohlc_data = df[['time', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'Volume': 'volume'})
    ohlc_data = ohlc_data.tail(n).copy()
    ohlc_data.set_index('time', inplace=True)
    
    filepath = settings.PLOTS_DIR / "candlestick_mav.png"
    # Ensure mav lengths don't exceed data length
    mav = (50, 200)
    if len(ohlc_data) < 200:
        mav = (10, 20) # Fallback for small data samples
        
    mpf.plot(ohlc_data, type='candle', style='yahoo',
             title=f"Candlestick Chart with MA (Last {n})",
             ylabel="Price", volume=True,
             mav=mav, savefig=dict(fname=filepath, dpi=100))
    logger.info(f"Saved plot: {filepath}")

def plot_close_rsi_dual(df, n=200):
    data = df.tail(n)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot Close Price
    ax1.plot(data['time'], data['close'], color="blue", label="Close Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    
    # Plot RSI on Secondary Axis
    ax2 = ax1.twinx()
    ax2.plot(data['time'], data['rsi'], color="purple", label="RSI")
    ax2.axhline(settings.RSI_OVERBOUGHT, color='red', linestyle='--', linewidth=0.5, label="Overbought")
    ax2.axhline(settings.RSI_OVERSOLD, color='green', linestyle='--', linewidth=0.5, label="Oversold")
    ax2.set_ylabel("RSI", color="purple")
    ax2.tick_params(axis='y', labelcolor="purple")
    
    fig.suptitle(f"Close Price with RSI (Last {n})", fontsize=14)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.xticks(rotation=45)
    save_plot(fig, "close_rsi_dual.png")

def plot_volume_close_dual(df, n=200):
    data = df.tail(n)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Area plot for Volume
    ax1.fill_between(data['time'], data['volume'], color="lightblue", alpha=0.5, label="Volume")
    ax1.set_ylabel("Volume", color="lightblue")
    ax1.tick_params(axis='y', labelcolor="lightblue")
    
    # Overlay line plot for Close Price
    ax2 = ax1.twinx()
    ax2.plot(data['time'], data['close'], color="blue", label="Close Price")
    ax2.set_ylabel("Price", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    
    fig.suptitle(f"Volume and Close Price (Last {n})", fontsize=14)
    save_plot(fig, "volume_close_dual.png")

def plot_macd_signals(df, n=200):
    data = df.tail(n).reset_index(drop=True)
    fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot Close Price
    ax[0].plot(data['time'], data['close'], label="Close Price", color="blue")
    ax[0].set_ylabel("Price")
    ax[0].legend(loc="upper left")
    
    # Plot MACD
    ax[1].plot(data['time'], data['macd_diff'], label="MACD Diff", color="darkred")
    ax[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # Annotate signals (basic crossover logic from original script)
    # Note: Original script iterates manually. We'll do same for specific annotation.
    for idx, value in enumerate(data['macd_diff']):
        if idx > 0:
            prev_val = data['macd_diff'].iloc[idx-1]
            current_time = data['time'].iloc[idx]
            
            # Crossover Buy (Negative to Positive)
            if prev_val < 0 < value:
                ax[1].annotate('Buy', (current_time, value), color="green", xytext=(0, 10),
                               textcoords='offset points', arrowprops=dict(arrowstyle="->", color='green'))
            
            # Crossover Sell (Positive to Negative)
            elif prev_val > 0 > value:
                ax[1].annotate('Sell', (current_time, value), color="red", xytext=(0, 10),
                               textcoords='offset points', arrowprops=dict(arrowstyle="->", color='red'))
                               
    ax[1].set_ylabel("MACD Diff")
    ax[1].legend(loc="upper left")
    
    plt.xticks(rotation=45)
    fig.suptitle(f"Close Price and MACD with Signals (Last {n})")
    save_plot(fig, "macd_signals.png")

def generate_all_eda_plots(df):
    logger.info("Generating EDA plots...")
    # Using a subset for heavy plots if needed, but 20k rows is fine for matplotlib
    
    try:
        plot_close_price(df)
        plot_candlestick(df)
        plot_volume_bar(df)
        plot_moving_averages(df)
        plot_high_low_area(df)
        plot_scatter_close_volume(df)
        plot_rsi(df)
        plot_macd(df)
        plot_bollinger_bands(df)
        plot_correlation_heatmap(df)
        plot_rsi_distribution(df)
        plot_macd_histogram(df)
        plot_combined_analysis(df)
        
        # New Plots
        plot_ema(df)
        plot_price_change(df)
        plot_candlestick_mav(df)
        plot_close_rsi_dual(df)
        plot_volume_close_dual(df)
        plot_macd_signals(df)
        
        logger.info("EDA plots completed.")
    except Exception as e:
        logger.error(f"Error generating EDA plots: {e}")

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sell/Hold", "Buy"], yticklabels=["Sell/Hold", "Buy"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot(plt.gcf(), f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")

def plot_model_accuracy_comparison(results):
    """
    Plot bar chart comparing model accuracies.
    Args:
        results: Dictionary with model names as keys and dict of metrics as values.
    """
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color=['blue', 'green', 'red'])
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    
    # Add text labels
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
    save_plot(plt.gcf(), "model_accuracy_comparison.png")

def plot_model_metrics_comparison(results):
    """
    Plot Recall and F1-Score comparison.
    """
    models = list(results.keys())
    
    # Extract macro avg call/f1 from classification report dict
    # Note: report is stored as a string or dict? In evaluation.py it return classification_report as string usually unless output_dict=True
    # I need to ensure evaluation.py returns dict for report.
    
    recalls = []
    f1_scores = []
    
    for m in models:
        report = results[m]['report_dict'] # Expecting dict
        recalls.append(report['macro avg']['recall'])
        f1_scores.append(report['macro avg']['f1-score'])
        
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, recalls, width, label='Recall', color='skyblue')
    plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='salmon')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Recall and F1-Score for Each Classifier')
    plt.xticks(x, models)
    plt.legend()
    
    save_plot(plt.gcf(), "model_metrics_comparison.png")

def plot_roc_curve_comparison(models, X_test, y_test):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize labels if multiclass, but here we likely have binary (Buy vs Hold/Sell) 
    # or we map to 0/1. features.py maps 'BUY':1, 'SELL':0, 'HOLD':0. So it is binary.
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue
            
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    save_plot(plt.gcf(), "roc_curve_comparison.png")
